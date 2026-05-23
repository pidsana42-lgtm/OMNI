"""
Phase 4: Native Speech Output Training
---------------------------------------
Trains the AudioLMHead to map LLM hidden states to EnCodec audio tokens.

Freeze  : Whisper + Qwen LLM Backbone + Projector (100% frozen)
Unfreeze: AudioLMHead (only ~5.2M parameters trained)

Run:
    python -m training.phase4_audio_output \\
        --config configs/phase4_audio_output.yaml \\
        --phase2_checkpoint outputs/phase2/phase2_final
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniModalModel, OmniProcessor
from src.audio_decoder import AudioCodec
from data.dataset_audio_output import AudioOutputDataset, AudioOutputCollator


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Audio Output SFT")
    parser.add_argument("--config", default="configs/phase4_audio_output.yaml")
    parser.add_argument("--phase2_checkpoint", required=True, help="Path to trained Phase 2 final checkpoint")
    parser.add_argument("--resume_from", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="wandb" if cfg.logging.use_wandb else None,
    )

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name or "phase4-audio-output",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── 1. Load Model from Phase 2 checkpoint ────────────────────────────
    print(f"[Phase4] Loading Phase 2 model: {args.phase2_checkpoint}")
    model = OmniModalModel.from_pretrained(args.phase2_checkpoint)
    
    # ── 2. Add AudioLMHead and setup Phase 4 freezes ─────────────────────
    if model.audio_lm_head is None:
        model.add_audio_output_head(
            codebook_size=cfg.model.audio_codebook_size,
            num_codebooks=cfg.model.audio_num_codebooks,
            bandwidth=cfg.model.audio_codec_bandwidth,
        )
    
    model.setup_phase4_audio_output()
    model.print_trainable_parameters()
    model = model.to(accelerator.device)

    # ── 3. Load Tokenizer & Processors ───────────────────────────────────
    processor = OmniProcessor.from_pretrained(
        llm_name=args.phase2_checkpoint,
        audio_encoder_name=model.config.audio_encoder_name,
    )

    # ── 4. Dataset Loader ────────────────────────────────────────────────
    print("[Phase4] Initializing Audio Output SFT Dataset...")
    train_dataset = AudioOutputDataset(
        processor=processor,
        codec=model.audio_codec,
        manifest_file=cfg.data.get("manifest_file"),
        hf_dataset_name=cfg.data.get("hf_dataset_name"),
        hf_dataset_config=cfg.data.get("hf_dataset_config"),
        hf_split=cfg.data.get("hf_split", "train"),
        target_audio_column=cfg.data.get("target_audio_column", "audio"),
        prompt_text_column=cfg.data.get("prompt_text_column", "text"),
        prompt_audio_column=cfg.data.get("prompt_audio_column"),
        max_prompt_length=cfg.data.max_prompt_length,
        max_target_seconds=cfg.data.max_target_seconds,
    )

    collator = AudioOutputCollator(
        pad_token_id=processor.pad_token_id or 0,
        ignore_index=-100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # ── 5. Optimizer & Scheduler ──────────────────────────────────────────
    # Only optimizing AudioLMHead parameters
    optimizer = AdamW(
        model.audio_lm_head.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.num_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 6. Prepare with Accelerator ───────────────────────────────────────
    # Since LLM is frozen, we wrap LLM in eval mode and only model.audio_lm_head is trained.
    model.audio_projector.eval()
    model.llm.eval()
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if args.resume_from:
        accelerator.load_state(args.resume_from)

    # ── 7. Training Loop ──────────────────────────────────────────────────
    global_step = 0
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase4] 🚀 Starting SFT Audio Output Alignment...")
    for epoch in range(cfg.training.num_epochs):
        model.audio_lm_head.train()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                audio_input_features = batch["audio_input_features"]
                has_audio_mask = batch["has_audio_mask"]
                targets = batch["target_tokens"]  # [B, T_target]

                # Run frozen LLM forward pass to get hidden states
                # Use sub-indexing to avoid shape mismatch (same fix as Phase 2)
                with torch.no_grad():
                    # We need the last layer's hidden states to feed our AudioLMHead
                    hidden_list = []
                    
                    # Split into audio & text sub-batches for correct inputs
                    if has_audio_mask.any():
                        audio_out = model.llm(
                            input_ids=None,
                            inputs_embeds=model._encode_audio(audio_input_features[has_audio_mask]),
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        # hidden_states: tuple of [B, T, hidden_size] per layer
                        # Get the last token's hidden state (which acts as the start for response generation)
                        # we take the last layer index -1
                        h_aud = audio_out.hidden_states[-1][:, -1:, :]
                        hidden_list.append((h_aud, has_audio_mask))

                    text_mask = ~has_audio_mask
                    if text_mask.any():
                        text_out = model.llm(
                            input_ids=input_ids[text_mask],
                            attention_mask=attention_mask[text_mask],
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        h_txt = text_out.hidden_states[-1][:, -1:, :]
                        hidden_list.append((h_txt, text_mask))

                    # Reassemble hidden states in batch order
                    hidden_states = torch.zeros(
                        input_ids.shape[0], 1, model.config.llm_hidden_size,
                        device=input_ids.device, dtype=torch.bfloat16
                    )
                    for h, mask in hidden_list:
                        hidden_states[mask] = h

                # Predict logits via trainable AudioLMHead
                # Shape: [B, 1, codebook_size] -> We replicate context along sequence length of targets
                T_target = targets.shape[1]
                # Context broadcast: [B, T_target, hidden_size]
                context_seq = hidden_states.expand(-1, T_target, -1)
                
                logits = model.module.audio_lm_head(context_seq) if hasattr(model, "module") else model.audio_lm_head(context_seq)
                # logits shape: [B, T_target, codebook_size]

                # Compute SFT cross-entropy loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                if cfg.logging.use_wandb and global_step % cfg.logging.log_every == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step
                    })

            # Checkpoint save
            if global_step % cfg.training.save_steps == 0 and accelerator.is_main_process:
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(ckpt_path))
                print(f"\n[Phase4] Checkpoint saved at step {global_step}")

    # ── 8. Final Save ─────────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_path = output_dir / "phase4_final"
        # Unwrap and save weights
        raw_model = accelerator.unwrap_model(model)
        raw_model.save_pretrained(str(final_path))
        processor.save_pretrained(str(final_path))
        print(f"\n[Phase4] SFT Audio Output completed successfully! Saved to {final_path}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
