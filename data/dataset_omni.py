"""
Omni-Modal Interleaved Dataset (Phase 2)
-----------------------------------------
Combines Audio, Vision, and Text datasets with a specified mixing ratio.
Ensures each batch contains diverse modality examples.

Default ratio (Phase 2):
  35% Audio  (ASR + Thai speech understanding)
  35% Text   (preserve language reasoning)
  30% Vision (image QA with colpali data)
"""

from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, Any, List, Optional

from ..src.processing_omni import OmniProcessor
from .dataset_audio import AudioTextDataset
from .dataset_vision import VisionTextDataset


class TextOnlyDataset(Dataset):
    """
    Simple text instruction-following dataset.
    Uses Qwen's chat template directly — no audio, no image.

    Useful for preventing catastrophic forgetting of language ability.
    """

    def __init__(
        self,
        processor: OmniProcessor,
        hf_dataset_name: str = "mlabonne/FineTome-100k",
        hf_split: str = "train",
        max_length: int = 1024,
        system_prompt: str = "คุณเป็น AI ผู้ช่วยภาษาไทยที่ฉลาดและเป็นประโยชน์",
    ):
        self.processor = processor
        self.max_length = max_length
        self.system_prompt = system_prompt

        from datasets import load_dataset
        self.data = load_dataset(hf_dataset_name, split=hf_split, trust_remote_code=True)
        print(f"[TextDataset] Loaded {len(self.data):,} samples from {hf_dataset_name}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Handle common formats: 'messages' list, 'conversations' list, or 'instruction'/'output'
        if "messages" in item:
            import json
            msgs = item["messages"]
            if isinstance(msgs, str):
                try:
                    msgs = json.loads(msgs)
                except Exception:
                    pass
            messages = []
            for m in msgs:
                messages.append({
                    "role": m.get("role", "user"),
                    "content": m.get("content", "")
                })
        elif "conversations" in item:
            messages = [
                {"role": ("user" if m["from"] == "human" else "assistant"),
                 "content": m["value"]}
                for m in item["conversations"]
            ]
        else:
            question = item.get("instruction", item.get("input", ""))
            answer = item.get("output", item.get("response", ""))
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        input_ids = encoded["input_ids"].squeeze(0)[:self.max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)[:self.max_length]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "modality": "text",
        }


class OmniInterleavedDataset(Dataset):
    """
    Interleaved dataset mixing audio, text, and vision.

    Implements weighted sampling without replacement per epoch,
    maintaining the specified ratio across the full training dataset.
    """

    def __init__(
        self,
        audio_dataset: Optional[AudioTextDataset] = None,
        text_dataset: Optional[TextOnlyDataset] = None,
        vision_dataset: Optional[VisionTextDataset] = None,
        audio_ratio: float = 0.35,
        text_ratio: float = 0.35,
        vision_ratio: float = 0.30,
        total_samples: Optional[int] = None,
        seed: int = 42,
    ):
        assert abs(audio_ratio + text_ratio + vision_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.datasets = {}
        self.ratios = {}

        if audio_dataset is not None:
            self.datasets["audio"] = audio_dataset
            self.ratios["audio"] = audio_ratio

        if text_dataset is not None:
            self.datasets["text"] = text_dataset
            self.ratios["text"] = text_ratio

        if vision_dataset is not None:
            self.datasets["vision"] = vision_dataset
            self.ratios["vision"] = vision_ratio

        # Renormalize if some datasets are missing
        total_ratio = sum(self.ratios.values())
        self.ratios = {k: v / total_ratio for k, v in self.ratios.items()}

        # Determine total samples
        max_ds = max(len(d) for d in self.datasets.values())
        self._total = total_samples or max_ds

        # Build index: list of (modality, local_idx) pairs
        random.seed(seed)
        self._index = self._build_index()
        print(
            f"[OmniDataset] Total {self._total:,} samples | "
            f"Ratios: {self.ratios}"
        )

    def _build_index(self) -> List[tuple]:
        index = []
        for modality, ratio in self.ratios.items():
            n = int(self._total * ratio)
            ds_len = len(self.datasets[modality])
            local_indices = [i % ds_len for i in random.sample(
                range(ds_len * max(1, n // ds_len + 1)), n
            )]
            index.extend([(modality, li) for li in local_indices])
        random.shuffle(index)
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        modality, local_idx = self._index[idx]
        return self.datasets[modality][local_idx]
