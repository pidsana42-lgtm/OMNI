"""
Vision-Text Dataset
--------------------
Loads image-text pairs for the Vision pathway of Phase 2.

Supported:
  - HuggingFace datasets with 'image' + instruction columns
  - BidirLM/colpali_train_retrieval  (image=positive, text=anchor)
  - LLaVA-style instruction-following datasets
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import numpy as np

from datasets import load_dataset

from ..src.processing_omni import OmniProcessor


class VisionTextDataset(Dataset):
    """
    Dataset for Vision-Text pairs.

    Each item returns tokenized instruction + image features
    compatible with Qwen3.5-0.8B's native image handling.
    """

    def __init__(
        self,
        processor: OmniProcessor,
        hf_dataset_name: str,
        hf_dataset_config: Optional[str] = None,
        hf_split: str = "train",
        image_column: str = "image",
        question_column: str = "anchor",
        answer_column: Optional[str] = None,
        max_image_pixels: int = 1_280_000,  # ~1.28M pixels before tiling
        max_text_length: int = 512,
        system_prompt: str = "คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการวิเคราะห์ภาพ",
        colpali_mode: bool = False,  # Special mode for BidirLM/colpali_train_retrieval
    ):
        self.processor = processor
        self.max_text_length = max_text_length
        self.system_prompt = system_prompt
        self.image_col = image_column
        self.question_col = question_column
        self.answer_col = answer_column
        self.colpali_mode = colpali_mode

        print(f"[VisionDataset] Loading {hf_dataset_name} [{hf_split}]")
        self.data = load_dataset(
            hf_dataset_name,
            hf_dataset_config,
            split=hf_split,
            trust_remote_code=True,
        )
        print(f"[VisionDataset] Loaded {len(self.data):,} samples")

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, image_data) -> Image.Image:
        """Handle various image formats from HF datasets."""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, dict):
            # HF format: {'bytes': b'...', 'path': '...'}
            if "bytes" in image_data and image_data["bytes"]:
                return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
            elif "path" in image_data and image_data["path"]:
                return Image.open(image_data["path"]).convert("RGB")
        raise ValueError(f"Cannot load image from: {type(image_data)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # ── Load image ────────────────────────────────────────────────────
        image = self._load_image(item[self.image_col])

        # ── Build question/answer ─────────────────────────────────────────
        if self.colpali_mode:
            # BidirLM/colpali_train_retrieval: anchor is the question
            question = str(item.get("anchor", "อธิบายเนื้อหาในภาพนี้"))
            # No ground-truth answer in colpali dataset — use Qwen to generate
            # For training: we treat it as image captioning
            answer = item.get("answer_type", "")  # fallback
        else:
            question = str(item.get(self.question_col, "อธิบายภาพนี้"))
            answer = str(item.get(self.answer_col, "")) if self.answer_col else ""

        # ── Build message with image ──────────────────────────────────────
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        if answer:
            messages.append({"role": "assistant", "content": answer})

        # ── Apply Qwen chat template (with image processing) ──────────────
        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=(not bool(answer)),
            enable_thinking=False,   # ← Always off
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")

        # Labels
        labels = input_ids.clone()
        if answer:
            # Mask prompt part
            answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            mask_len = max(0, len(input_ids) - len(answer_tokens))
            labels[:mask_len] = -100
        else:
            labels[:] = -100  # No answer — no loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "modality": "vision",
        }
