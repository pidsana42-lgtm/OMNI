"""
Phase 3: Knowledge Distillation — Streaming Head
--------------------------------------------------
Objective: Train typhoon-asr-realtime to produce fast, streaming
output using the Phase 2 Omni model as Teacher.

Teacher (Frozen): Phase 2 OmniModalModel
Student (Training): scb10x/typhoon-asr-realtime (or similar)

Loss: KL Divergence (Teacher logits → Student) + Cross-Entropy (labels)

NOTE: This phase is optional and can be skipped for MVP.
      Only needed for real-time streaming inference.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import wandb
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniModalModel, OmniProcessor
from data import AudioTextDataset, OmniDataCollator


def kl_div_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """
    KL Divergence loss between student and teacher distributions.
    Temperature scaling softens the teacher's distribution.
    """
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (T ** 2)  # Scale by T^2 to maintain gradient magnitude


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_distillation.yaml")
    parser.add_argument("--teacher_checkpoint", required=True,
                        help="Path to Phase 2 final checkpoint (teacher)")
    parser.add_argument("--student_model", default=None,
                        help="HF model name for student (overrides config)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    )

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name="phase3-distillation",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Load Teacher (Phase 2 model, fully frozen) ────────────────────────
    print(f"[Phase3] Loading teacher from {args.teacher_checkpoint}")
    teacher = OmniModalModel.from_pretrained(args.teacher_checkpoint)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("[Phase3] ❄️  Teacher frozen (100%)")

    # ── Load Teacher Processor (Qwen tokenizer) ───────────────────────────
    teacher_processor = OmniProcessor.from_pretrained(
        llm_name=args.teacher_checkpoint,
        audio_encoder_name=teacher.config.audio_encoder_name,
    )

    # ── Load Student (streaming model) ───────────────────────────────────
    student_name = args.student_model or cfg.model.student_model
    print(f"[Phase3] Loading student: {student_name}")
    student = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)

    # ── BUG FIX (Bug 2): Dual Dataset / Dual Collator ────────────────────
    # Teacher uses Qwen tokenizer (for KL divergence on logits).
    # Student uses its own tokenizer to produce labels for CE loss.
    # Mixing vocab IDs causes IndexError if Qwen IDs > student vocab size.
    print("[Phase3] Creating DUAL datasets (teacher + student tokenizers)...")

    # Teacher dataset: tokenized with Qwen tokenizer
    teacher_dataset = AudioTextDataset(
        processor=teacher_processor,
        hf_dataset_name=cfg.data.hf_dataset_name,
        hf_dataset_config=cfg.data.get("hf_dataset_config"),
        hf_split="train",
    )
    teacher_collator = OmniDataCollator(processor=teacher_processor)

    # Student dataset: same audio, different tokenizer for labels
    # Wrap student_tokenizer in a minimal processor-compatible shim
    class StudentProcessorShim:
        """Minimal shim to make student_tokenizer compatible with OmniDataCollator."""
        def __init__(self, tok):
            self.tokenizer = tok
            self.sample_rate = teacher_processor.sample_rate
            self.audio_processor = teacher_processor.audio_processor
            self.llm_processor = teacher_processor.llm_processor
            self.audio_start_id = getattr(teacher_processor, 'audio_start_id', None)
            self.audio_end_id = getattr(teacher_processor, 'audio_end_id', None)
            self.audio_pad_id = getattr(teacher_processor, 'audio_pad_id', None)
        @property
        def pad_token_id(self):
            return self.tokenizer.pad_token_id
        @property
        def eos_token_id(self):
            return self.tokenizer.eos_token_id
        def process_audio(self, audio, sr):
            return teacher_processor.process_audio(audio, sr)
        def apply_chat_template(self, messages, **kwargs):
            # Use student tokenizer for encoding
            text = student_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=kwargs.get('add_generation_prompt', True)
            )
            return student_tokenizer(text, return_tensors=kwargs.get('return_tensors', 'pt'))
        def build_audio_instruction(self, transcript, system_prompt=""):
            return teacher_processor.build_audio_instruction(transcript, system_prompt)

    student_shim = StudentProcessorShim(student_tokenizer)
    student_dataset = AudioTextDataset(
        processor=student_shim,
        hf_dataset_name=cfg.data.hf_dataset_name,
        hf_dataset_config=cfg.data.get("hf_dataset_config"),
        hf_split="train",
    )
    student_collator = OmniDataCollator(processor=student_shim)

    # Zip both datasets so they are always in sync
    from torch.utils.data import DataLoader
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # zip requires same order
        collate_fn=teacher_collator,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    student_loader = DataLoader(
        student_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=student_collator,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = AdamW(student.parameters(), lr=cfg.training.lr, weight_decay=0.01)
    total_steps = len(teacher_loader) * cfg.training.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)

    teacher, student, optimizer, teacher_loader, student_loader, scheduler = accelerator.prepare(
        teacher, student, optimizer, teacher_loader, student_loader, scheduler
    )

    # ── Training Loop ─────────────────────────────────────────────────────
    alpha = cfg.training.get("kl_weight", 0.5)   # Weight for KL loss
    temperature = cfg.training.get("temperature", 4.0)
    global_step = 0
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.num_epochs):
        student.train()
        pbar = tqdm(
            zip(teacher_loader, student_loader),
            total=len(teacher_loader),
            desc=f"Phase3 Epoch {epoch+1}",
            disable=not accelerator.is_main_process,
        )

        for teacher_batch, student_batch in pbar:
            with accelerator.accumulate(student):
                # ── Teacher forward with QWEN tokens (KL loss) ─────────────
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=teacher_batch["input_ids"],
                        attention_mask=teacher_batch["attention_mask"],
                        audio_input_features=teacher_batch.get("audio_input_features"),
                    )
                    teacher_logits = teacher_out.logits  # [B, T_teacher, qwen_vocab]

                # ── Student forward with STUDENT tokens (CE loss on student labels) ─
                # BUG FIX (Bug 2): student_batch is tokenized by student_tokenizer
                # so labels are valid indices within student vocab
                student_out = student(
                    input_ids=student_batch["input_ids"],
                    attention_mask=student_batch["attention_mask"],
                )
                student_logits = student_out.logits  # [B, T_student, student_vocab]

                # ── KL Loss: align sequence lengths first (may differ slightly) ─
                T = min(student_logits.shape[1], teacher_logits.shape[1])
                min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
                s_kl = student_logits[:, :T, :min_vocab]
                t_kl = teacher_logits[:, :T, :min_vocab]
                kl_loss = kl_div_loss(s_kl, t_kl, temperature)

                # ── CE Loss: student logits vs student labels ─────────────────
                ce_loss = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.shape[-1]),
                    student_batch["labels"].reshape(-1),  # ← student vocab IDs
                    ignore_index=-100,
                )

                loss = alpha * kl_loss + (1 - alpha) * ce_loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "kl": f"{kl_loss.item():.4f}"})
                if cfg.logging.use_wandb:
                    wandb.log({"loss": loss.item(), "kl_loss": kl_loss.item(), "ce_loss": ce_loss.item()})

    # Save student
    if accelerator.is_main_process:
        final_path = output_dir / "phase3_student_final"
        accelerator.unwrap_model(student).save_pretrained(str(final_path))
        student_tokenizer.save_pretrained(str(final_path))
        print(f"[Phase3] ✅ Student saved to {final_path}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
