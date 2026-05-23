"""
Phase 2: Omni-Modal Full Fine-Tuning
--------------------------------------
Objective: Teach Qwen3.5-0.8B to reason across Audio + Vision + Text.

Freeze  : Whisper (audio encoder) + Qwen Vision Encoder
Unfreeze: Qwen LLM backbone + AudioProjector

Dataset : Interleaved mixture:
          35% Audio  (Common Voice TH / Gowajee)
          35% Text   (Thai instruction following)
          30% Vision (BidirLM/colpali_train_retrieval or LLaVA-style)

Duration: ~8–15 hours on H200 (141GB VRAM)

Run:
    python -m training.phase2_omni_finetune \
        --config configs/phase2_finetune.yaml \
        --phase1_checkpoint outputs/phase1_final
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniConfig, OmniModalModel, OmniProcessor
from data import (
    AudioTextDataset,
    VisionTextDataset,
    TextOnlyDataset,
    OmniInterleavedDataset,
    OmniDataCollator,
)


def dynamic_push_to_hub(local_path: str, repo_name: str, private: bool = True):
    import importlib.util
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    push_to_hub_path = project_root / "scripts" / "push_to_hub.py"
    if not push_to_hub_path.exists():
        raise FileNotFoundError(f"Could not find push_to_hub.py at {push_to_hub_path}")
    spec = importlib.util.spec_from_file_location("push_to_hub_dynamic", str(push_to_hub_path))
    push_to_hub_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(push_to_hub_module)
    push_to_hub_module.push_to_hub_direct(
        local_path=local_path,
        repo_name=repo_name,
        private=private,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_finetune.yaml")
    parser.add_argument("--phase1_checkpoint", required=True,
                        help="Path to Phase 1 final checkpoint")
    parser.add_argument("--resume_from", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    # ── Accelerator ───────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="wandb" if cfg.logging.use_wandb else None,
    )

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name or "phase2-omni-finetune",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Load model from Phase 1 checkpoint ───────────────────────────────
    print(f"[Phase2] Loading Phase 1 checkpoint from {args.phase1_checkpoint}")
    model = OmniModalModel.from_pretrained(args.phase1_checkpoint)
    processor = OmniProcessor.from_pretrained(
        llm_name=args.phase1_checkpoint,       # Phase 1 saved tokenizer here
        audio_encoder_name=model.config.audio_encoder_name,
    )

    # ── Setup Phase 2 freeze strategy ────────────────────────────────────
    model.setup_phase2()
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────
    print("[Phase2] Loading interleaved datasets...")

    audio_ds = AudioTextDataset(
        processor=processor,
        hf_dataset_name=cfg.data.audio.hf_dataset_name,
        hf_dataset_config=cfg.data.audio.get("hf_dataset_config"),
        hf_split=cfg.data.audio.get("hf_split", "train"),
        text_column=cfg.data.audio.get("text_column"),
        max_audio_seconds=cfg.data.max_audio_seconds,
    )

    text_ds = TextOnlyDataset(
        processor=processor,
        hf_dataset_name=cfg.data.text.hf_dataset_name,
        hf_split=cfg.data.text.get("hf_split", "train"),
        max_length=cfg.data.max_text_length,
    )

    vision_ds = VisionTextDataset(
        processor=processor,
        hf_dataset_name=cfg.data.vision.hf_dataset_name,
        hf_split=cfg.data.vision.get("hf_split", "train"),
        data_files=cfg.data.vision.get("data_files"),
        question_column=cfg.data.vision.get("question_column", "anchor"),
        answer_column=cfg.data.vision.get("answer_column", None),
        colpali_mode=cfg.data.vision.get("colpali_mode", False),
    )

    train_dataset = OmniInterleavedDataset(
        audio_dataset=audio_ds,
        text_dataset=text_ds,
        vision_dataset=vision_ds,
        audio_ratio=cfg.data.audio_ratio,
        text_ratio=cfg.data.text_ratio,
        vision_ratio=cfg.data.vision_ratio,
        total_samples=cfg.data.get("total_samples"),
    )

    collator = OmniDataCollator(
        processor=processor,
        max_length=cfg.data.max_text_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # ── Optimizer with differential LR ────────────────────────────────────
    # Higher LR for projector (needs to adapt fast), lower for LLM
    param_groups = [
        {
            "params": model.audio_projector.parameters(),
            "lr": cfg.training.projector_lr,
            "name": "projector",
        },
        {
            "params": [p for p in model.llm.parameters() if p.requires_grad],
            "lr": cfg.training.llm_lr,
            "name": "llm",
        },
    ]
    optimizer = AdamW(param_groups, weight_decay=cfg.training.weight_decay)

    total_steps = len(train_loader) * cfg.training.num_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Prepare ───────────────────────────────────────────────────────────
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if args.resume_from:
        accelerator.load_state(args.resume_from)

    # ── Training Loop ─────────────────────────────────────────────────────
    global_step = 0
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.num_epochs):
        model.train()
        modality_losses = {"audio": [], "vision": [], "text": []}

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                modalities = batch.get("modality", [])
                has_audio_mask = batch.get("has_audio_mask")   # [B] bool
                has_vision_mask = batch.get("has_vision_mask") # [B] bool

                # ── BUG FIX (Bug 1): Sub-index per modality before forward ──
                # Each model() call receives tensors with consistent batch dim.
                # Losses are accumulated then averaged before backward.
                loss_parts = []

                # ── Audio sub-batch ──────────────────────────────────────────
                if has_audio_mask is not None and has_audio_mask.any():
                    audio_out = model(
                        input_ids=batch["input_ids"][has_audio_mask],
                        attention_mask=batch["attention_mask"][has_audio_mask],
                        audio_input_features=batch["audio_input_features"],  # [n_audio, 128, 3000]
                        labels=batch["labels"][has_audio_mask],
                    )
                    if audio_out.loss is not None:
                        loss_parts.append(("audio", audio_out.loss, has_audio_mask.sum()))

                # ── Vision sub-batch ─────────────────────────────────────────
                if has_vision_mask is not None and has_vision_mask.any():
                    vision_out = model(
                        input_ids=batch["input_ids"][has_vision_mask],
                        attention_mask=batch["attention_mask"][has_vision_mask],
                        pixel_values=batch.get("pixel_values"),
                        image_grid_thw=batch.get("image_grid_thw"),
                        labels=batch["labels"][has_vision_mask],
                    )
                    if vision_out.loss is not None:
                        loss_parts.append(("vision", vision_out.loss, has_vision_mask.sum()))

                # ── Text sub-batch ───────────────────────────────────────────
                text_mask = ~has_audio_mask & ~has_vision_mask if (
                    has_audio_mask is not None and has_vision_mask is not None
                ) else None
                if text_mask is not None and text_mask.any():
                    text_out = model(
                        input_ids=batch["input_ids"][text_mask],
                        attention_mask=batch["attention_mask"][text_mask],
                        labels=batch["labels"][text_mask],
                    )
                    if text_out.loss is not None:
                        loss_parts.append(("text", text_out.loss, text_mask.sum()))

                if not loss_parts:
                    continue

                # Weighted average loss (by sub-batch size)
                total_items = sum(n for _, _, n in loss_parts)
                loss = sum(l * (n / total_items) for _, l, n in loss_parts)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            main_modality = modalities[0] if modalities else "text"
            modality_losses[main_modality].append(loss.item())

            if accelerator.is_main_process:
                postfix = {"loss": f"{loss.item():.4f}", "lr_llm": f"{optimizer.param_groups[1]['lr']:.2e}"}
                for mod, l, _ in loss_parts:
                    postfix[f"l_{mod[:3]}"] = f"{l.item():.3f}"
                pbar.set_postfix(postfix)

                if cfg.logging.use_wandb and global_step % cfg.logging.log_every == 0:
                    log_dict = {"train/loss": loss.item(), "train/step": global_step}
                    for mod, l, _ in loss_parts:
                        log_dict[f"train/loss_{mod}"] = l.item()
                    for mod, losses in modality_losses.items():
                        if losses:
                            log_dict[f"train/loss_{mod}_avg"] = sum(losses[-10:]) / min(len(losses), 10)
                    wandb.log(log_dict)

            # Save checkpoint
            if global_step % cfg.training.save_steps == 0 and accelerator.is_main_process:
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(ckpt_path))
                print(f"\n[Phase2] ✅ Checkpoint saved at step {global_step}")

                # Save HF weights and processor in the checkpoint directory
                hf_ckpt_path = output_dir / f"checkpoint-{global_step}-hf"
                accelerator.unwrap_model(model).save_pretrained(str(hf_ckpt_path))
                processor.save_pretrained(str(hf_ckpt_path))

                # Bundle accelerator states (optimizer/scheduler) into the HF directory
                import shutil
                for file_path in ckpt_path.glob("*"):
                    dest_file = hf_ckpt_path / file_path.name
                    if not dest_file.exists():
                        if file_path.is_dir():
                            shutil.copytree(file_path, dest_file)
                        else:
                            shutil.copy(file_path, dest_file)

                # Push step checkpoint to HF Hub
                if cfg.training.get("push_to_hub", False):
                    try:
                        dynamic_push_to_hub(
                            local_path=str(hf_ckpt_path),
                            repo_name=cfg.training.get("hub_model_id", "thai-omni-modal-0.8b-phase2"),
                            private=cfg.training.get("hub_private", True),
                        )
                    except Exception as e:
                        print(f"[Phase2] HF push failed: {e}")

    # ── Final save ────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_path = output_dir / "phase2_final"
        accelerator.unwrap_model(model).save_pretrained(str(final_path))
        processor.save_pretrained(str(final_path))
        print(f"\n[Phase2] ✅ Complete! Model at {final_path}")

        if cfg.training.get("push_to_hub", False):
            try:
                dynamic_push_to_hub(
                    local_path=str(final_path),
                    repo_name=cfg.training.get("hub_model_id", "thai-omni-modal-0.8b-phase2"),
                    private=cfg.training.get("hub_private", True),
                )
            except Exception as e:
                print(f"[Phase2] HF push failed: {e}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
