"""
Omni-Modal Data Collator
--------------------------
Custom collator that handles batches with mixed modalities.

Challenge: a single batch may contain:
  - Audio items  → need input_features + input_ids
  - Vision items → need pixel_values + input_ids
  - Text items   → need only input_ids

This collator pads everything correctly and builds unified tensors.

BUG FIX (Bug 1 — Shape Mismatch):
  audio_input_features shape is [n_audio, 128, 3000] (only real audio rows).
  has_audio_mask [B] bool tells the training loop WHICH indices in the full
  batch are audio, so it can sub-index before calling model().
  DO NOT pad to full batch size — that would silently encode silence tokens
  and prepend meaningless embeddings to text/vision items.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OmniDataCollator:
    """
    Collates mixed-modality batches.

    Padding strategy:
      - input_ids / attention_mask / labels: right-pad to max length in batch
      - audio_input_features: [n_audio, 128, 3000] — only real audio rows
      - has_audio_mask: [B] bool — which batch rows have real audio
      - has_vision_mask: [B] bool — which batch rows have vision
      - pixel_values: kept as list (Qwen handles variable sizes internally)
    """

    processor: Any
    max_length: int = 2048
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # ── Text tensors (present for ALL modalities) ─────────────────────
        all_input_ids = [f["input_ids"] for f in features]
        all_masks = [f["attention_mask"] for f in features]
        all_labels = [f["labels"] for f in features]

        batch["input_ids"] = self._pad_sequence(
            all_input_ids,
            pad_value=self.processor.pad_token_id or 0,
        )
        batch["attention_mask"] = self._pad_sequence(all_masks, pad_value=0)
        batch["labels"] = self._pad_sequence(all_labels, pad_value=self.label_pad_token_id)

        # ── BUG FIX: has_audio_mask [B] + audio sub-batch ────────────────
        # The training loop MUST use has_audio_mask to sub-index before
        # calling model() with audio_input_features, so all tensor batch
        # dims stay consistent (n_audio == n_audio, never n_audio vs B).
        has_audio = [f.get("modality") == "audio" for f in features]
        batch["has_audio_mask"] = torch.tensor(has_audio, dtype=torch.bool)  # [B]

        audio_feats = [
            features[i]["input_features"]
            for i, flag in enumerate(has_audio) if flag
        ]
        batch["audio_input_features"] = (
            torch.stack(audio_feats, dim=0) if audio_feats else None
        )  # [n_audio, 128, 3000]  — NOT [B, 128, 3000]

        # ── BUG FIX: has_vision_mask [B] + vision sub-batch ──────────────
        has_vision = [f.get("modality") == "vision" for f in features]
        batch["has_vision_mask"] = torch.tensor(has_vision, dtype=torch.bool)  # [B]

        pixel_values_list = [
            features[i].get("pixel_values")
            for i, flag in enumerate(has_vision)
            if flag and features[i].get("pixel_values") is not None
        ]
        if pixel_values_list:
            try:
                batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
                thw_list = [
                    features[i].get("image_grid_thw")
                    for i, flag in enumerate(has_vision)
                    if flag and features[i].get("image_grid_thw") is not None
                ]
                if thw_list:
                    batch["image_grid_thw"] = torch.cat(thw_list, dim=0)
            except Exception:
                batch["pixel_values"] = pixel_values_list
        else:
            batch["pixel_values"] = None

        # ── Modality labels (for logging) ─────────────────────────────────
        batch["modality"] = [f.get("modality", "text") for f in features]

        return batch

    @staticmethod
    def _pad_sequence(
        sequences: List[torch.Tensor],
        pad_value: int,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Right-pad all sequences to max length in batch."""
        max_len = max_length or max(s.shape[0] for s in sequences)
        padded = []
        for s in sequences:
            pad_size = max_len - s.shape[0]
            if pad_size > 0:
                padded.append(F.pad(s, (0, pad_size), value=pad_value))
            else:
                padded.append(s[:max_len])
        return torch.stack(padded, dim=0)
