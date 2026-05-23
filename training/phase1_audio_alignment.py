"""
Phase 1: Audio Modality Alignment
------------------------------------
Objective: Teach AudioProjector to map Whisper → Qwen space.

Freeze  : Whisper (audio encoder) + Qwen3.5-0.8B (LLM)
Unfreeze: AudioProjector ONLY

Dataset : Thai speech-to-text pairs
          Default: mozilla-foundation/common_voice_17_0 (th)

Duration: ~1–3 hours on H200 (141GB VRAM)

Run:
    python -m training.phase1_audio_alignment \
        --config configs/phase1_alignment.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniConfig, OmniModalModel, OmniProcessor
from data import AudioTextDataset, OmniDataCollator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_alignment.yaml")
    parser.add_argument("--resume_from", default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    # ── Accelerator (H200 bf16) ───────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="wandb" if cfg.logging.use_wandb else None,
    )

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name or "phase1-audio-alignment",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Load Processor ────────────────────────────────────────────────────
    processor = OmniProcessor.from_pretrained(
        llm_name=cfg.model.llm_model,
        audio_encoder_name=cfg.model.audio_encoder,
    )

    # ── Load Model ────────────────────────────────────────────────────────
    omni_cfg = OmniConfig(
        llm_model_name=cfg.model.llm_model,
        audio_encoder_name=cfg.model.audio_encoder,
        projector_hidden_size=cfg.model.projector_hidden,
        projector_num_layers=cfg.model.projector_layers,
        enable_thinking=False,  # ALWAYS OFF
    )
    model = OmniModalModel(omni_cfg)

    # Resize LLM embeddings for new special tokens
    model.llm.resize_token_embeddings(len(processor.tokenizer))
    print(f"[Phase1] Resized embeddings to {len(processor.tokenizer):,} tokens")

    # Enable gradient checkpointing to save VRAM
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        print("[Phase1] Enabled gradient checkpointing on LLM backbone")

    # ── Setup Phase 1 freeze strategy ────────────────────────────────────
    model.setup_phase1()
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────
    train_dataset = AudioTextDataset(
        processor=processor,
        hf_dataset_name=cfg.data.hf_dataset_name,
        hf_dataset_config=cfg.data.get("hf_dataset_config", None),
        hf_split=cfg.data.get("hf_split", "train"),
        text_column=cfg.data.get("text_column", None),
        max_audio_seconds=cfg.data.max_audio_seconds,
        max_text_length=cfg.data.max_text_length,
    )

    eval_dataset = AudioTextDataset(
        processor=processor,
        hf_dataset_name=cfg.data.hf_dataset_name,
        hf_dataset_config=cfg.data.get("hf_dataset_config", None),
        hf_split=cfg.data.get("eval_split", "validation"),
        text_column=cfg.data.get("text_column", None),
        max_audio_seconds=cfg.data.max_audio_seconds,
        max_text_length=cfg.data.max_text_length,
    )

    # Subsample if requested to prevent cloud timeout and speed up Phase 1
    import random
    max_train_samples = cfg.training.get("max_train_samples", None)
    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), max_train_samples)
        indices = list(range(len(train_dataset)))
        random.Random(42).shuffle(indices)
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:max_train_samples])
        print(f"[Phase1] Subsampled train dataset to {max_train_samples:,} samples")

    max_eval_samples = cfg.training.get("max_eval_samples", None)
    if max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        indices = list(range(len(eval_dataset)))
        random.Random(42).shuffle(indices)
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices[:max_eval_samples])
        print(f"[Phase1] Subsampled eval dataset to {max_eval_samples:,} samples")

    collator = OmniDataCollator(processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # ── Optimizer (high LR for projector-only training) ───────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.training.lr * 0.1)

    # ── Prepare with Accelerator ──────────────────────────────────────────
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    resume_step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        print(f"[Phase1] Resumed accelerator state from {args.resume_from}")
        try:
            global_step = int(Path(args.resume_from).name.split("-")[-1])
            resume_step = global_step % len(train_loader)
            start_epoch = global_step // len(train_loader)
            print(f"[Phase1] Calculated starting point: global_step={global_step}, start_epoch={start_epoch}, resume_step={resume_step}")
        except Exception as e:
            print(f"[Phase1] Warning: Could not parse step number from checkpoint. Starting from step 0. Error: {e}")

    # ── Training Loop ─────────────────────────────────────────────────────
    best_eval_loss = float("inf")
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(pbar):
            # Skip steps if resuming
            if epoch == start_epoch and step < resume_step:
                if step % 100 == 0 and accelerator.is_main_process:
                    print(f"[Phase1] Skipping step {step} to resume training...")
                continue

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_input_features=batch.get("audio_input_features"),
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

                if cfg.logging.use_wandb and global_step % cfg.logging.log_every == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": global_step,
                    })

            # Save checkpoint every N steps
            if global_step % cfg.training.save_steps == 0 and accelerator.is_main_process:
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(ckpt_path))
                print(f"\n[Phase1] ✅ Saved accelerator state at step {global_step}")

                # Save standard HF model weights and processor inside the checkpoint directory
                hf_ckpt_path = output_dir / f"checkpoint-{global_step}-hf"
                accelerator.unwrap_model(model).save_pretrained(str(hf_ckpt_path))
                processor.save_pretrained(str(hf_ckpt_path))
                print(f"[Phase1] ✅ Saved HF model weights at step {global_step} to {hf_ckpt_path}")

                # Push this step checkpoint to Hugging Face Hub
                if cfg.training.get("push_to_hub", False):
                    try:
                        from scripts.push_to_hub import push_to_hub_direct
                        push_to_hub_direct(
                            local_path=str(hf_ckpt_path),
                            repo_name=cfg.training.get("hub_model_id", "thai-omni-modal-0.8b-phase1"),
                            private=cfg.training.get("hub_private", True),
                        )
                    except Exception as e:
                        print(f"[Phase1] HF push failed at step {global_step}: {e}")

        # ── Eval per epoch ────────────────────────────────────────────────
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_main_process):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_input_features=batch.get("audio_input_features"),
                    labels=batch["labels"],
                )
                eval_loss += outputs.loss.item()

        eval_loss /= len(eval_loader)
        avg_train_loss = epoch_loss / len(train_loader)

        if accelerator.is_main_process:
            print(f"\n[Phase1] Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, eval_loss={eval_loss:.4f}")
            if cfg.logging.use_wandb:
                wandb.log({"eval/loss": eval_loss, "epoch": epoch + 1})

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_path = output_dir / "best_phase1"
                accelerator.unwrap_model(model).save_pretrained(str(best_path))
                processor.save_pretrained(str(best_path))
                print(f"[Phase1] 🏆 New best! Saved to {best_path}")

                if cfg.training.get("push_to_hub", False):
                    try:
                        from scripts.push_to_hub import push_to_hub_direct
                        push_to_hub_direct(
                            local_path=str(best_path),
                            repo_name=cfg.training.get("hub_model_id", "thai-omni-modal-0.8b-phase1"),
                            private=cfg.training.get("hub_private", True),
                        )
                    except Exception as e:
                        print(f"[Phase1] HF push failed: {e}")

    # ── Final save ────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_path = output_dir / "phase1_final"
        accelerator.unwrap_model(model).save_pretrained(str(final_path))
        processor.save_pretrained(str(final_path))
        print(f"\n[Phase1] ✅ Training complete! Final model at {final_path}")

        if cfg.training.get("push_to_hub", False):
            try:
                from scripts.push_to_hub import push_to_hub_direct
                push_to_hub_direct(
                    local_path=str(final_path),
                    repo_name=cfg.training.get("hub_model_id", "thai-omni-modal-0.8b-phase1"),
                    private=cfg.training.get("hub_private", True),
                )
            except Exception as e:
                print(f"[Phase1] HF push failed: {e}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
