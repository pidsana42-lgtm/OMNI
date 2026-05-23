"""
Audio Dataset for Phase 1 & Phase 2
--------------------------------------
Loads speech-to-text pairs and formats them for training.

Expects a dataset with columns:
  - 'audio': dict with {'array': np.ndarray, 'sampling_rate': int}
             OR path string to audio file
  - 'sentence' / 'transcription' / 'text': Thai transcript

Supported sources (auto-detected):
  - Mozilla Common Voice TH  (mozilla-foundation/common_voice_17_0, 'th')
  - VISTEC Thai Speech        (custom local)
  - Gowajee                   (custom local)
  - Any HuggingFace dataset with 'audio' + text column
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset, Audio as HFAudio

from src.processing_omni import OmniProcessor


TRANSCRIPT_COLUMNS = ["response", "sentence", "transcription", "text", "transcript", "label", "instruction"]


class AudioTextDataset(Dataset):
    """
    Dataset for Audio-Text pairs (ASR-style).

    Each item returns:
        input_features    : [128, 3000] log-mel spectrogram
        input_ids         : [T_text]    tokenized instruction + label
        attention_mask    : [T_text]
        labels            : [T_text]    same as input_ids but audio prefix = -100
        modality          : "audio"
    """

    def __init__(
        self,
        processor: OmniProcessor,
        hf_dataset_name: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
        hf_split: str = "train",
        audio_column: str = "audio",
        text_column: Optional[str] = None,   # auto-detected if None
        local_audio_dir: Optional[str] = None,
        manifest_file: Optional[str] = None,  # jsonl: {"audio": "path.wav", "text": "..."}
        max_audio_seconds: float = 30.0,
        max_text_length: int = 256,
        system_prompt: str = "คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการถอดเสียง",
    ):
        self.processor = processor
        self.max_audio_seconds = max_audio_seconds
        self.max_text_length = max_text_length
        self.sample_rate = processor.sample_rate
        self.system_prompt = system_prompt

        # ── Load data ─────────────────────────────────────────────────────
        if hf_dataset_name:
            print(f"[AudioDataset] Loading {hf_dataset_name} ({hf_dataset_config or ''}) [{hf_split}]")
            raw = load_dataset(
                hf_dataset_name,
                hf_dataset_config,
                split=hf_split,
                trust_remote_code=True,
            )
            # Cast audio column to standard format
            raw = raw.cast_column(audio_column, HFAudio(sampling_rate=self.sample_rate))
            self.data = raw
            self.audio_col = audio_column
            self.text_col = text_column or self._detect_text_column(raw.column_names)

        elif manifest_file:
            self.data = self._load_manifest(manifest_file)
            self.audio_col = "audio"
            self.text_col = "text"

        else:
            raise ValueError("Provide either hf_dataset_name or manifest_file")

        print(f"[AudioDataset] Loaded {len(self.data):,} samples. Text column: '{self.text_col}'")

    @staticmethod
    def _detect_text_column(columns: List[str]) -> str:
        for c in TRANSCRIPT_COLUMNS:
            if c in columns:
                return c
        raise ValueError(
            f"Cannot find transcript column. Got: {columns}\n"
            f"Set text_column= explicitly."
        )

    @staticmethod
    def _load_manifest(path: str) -> List[Dict]:
        import json
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # ── Load audio ────────────────────────────────────────────────────
        if isinstance(item[self.audio_col], dict):
            # HuggingFace audio dict format
            waveform = item[self.audio_col]["array"].astype(np.float32)
            sr = item[self.audio_col]["sampling_rate"]
        elif isinstance(item[self.audio_col], (str, Path)):
            waveform, sr = sf.read(str(item[self.audio_col]))
            waveform = waveform.astype(np.float32)
        else:
            raise ValueError(f"Unexpected audio format: {type(item[self.audio_col])}")

        # Resample if needed
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Clip to max length
        max_samples = int(self.max_audio_seconds * self.sample_rate)
        waveform = waveform[:max_samples]

        # ── Extract mel features ──────────────────────────────────────────
        audio_features = self.processor.process_audio(waveform, self.sample_rate)
        input_features = audio_features["input_features"].squeeze(0)  # [128, 3000]

        # ── Tokenize text ─────────────────────────────────────────────────
        transcript = str(item[self.text_col]).strip()
        messages = self.processor.build_audio_instruction(
            transcript=transcript,
            system_prompt=self.system_prompt,
        )
        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            enable_thinking=False,   # ← Always off
        )
        input_ids = encoded["input_ids"].squeeze(0)        # [T]
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: copy input_ids, but -100 for the prompt part
        # (only compute loss on assistant response)
        labels = self._mask_prompt_labels(
            input_ids, messages, transcript
        )

        return {
            "input_features": input_features,      # [128, 3000]
            "input_ids": input_ids,                 # [T]
            "attention_mask": attention_mask,       # [T]
            "labels": labels,                       # [T]
            "modality": "audio",
        }

    def _mask_prompt_labels(
        self,
        input_ids: torch.Tensor,
        messages: List[Dict],
        assistant_text: str,
    ) -> torch.Tensor:
        """
        Create labels tensor where everything before assistant response = -100.

        BUG FIX (Bug 3): The previous approach subtracted len(answer_tokens) from
        total len, which ignores chat template special tokens like:
          <|im_start|>assistant\n ... <|im_end|>
        causing the first tokens of the assistant answer to be masked as -100.

        Correct approach: tokenize the PROMPT portion (all messages except the
        last assistant turn) via apply_chat_template with add_generation_prompt=True.
        The length of that encoded sequence is the exact mask boundary.
        """
        labels = input_ids.clone()

        # Build the prompt-only portion (everything before assistant response)
        # Exclude the last message (the assistant turn we want to learn)
        prompt_messages = messages[:-1]

        try:
            # Tokenize just the prompt with generation prompt appended
            # This captures ALL chat template tokens accurately
            prompt_encoded = self.processor.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            prompt_len = prompt_encoded["input_ids"].shape[-1]
        except Exception:
            # Fallback: if apply_chat_template fails for any reason,
            # use a conservative estimate (mask 75% of sequence)
            prompt_len = max(0, int(len(input_ids) * 0.75))

        # Clamp to valid range — at minimum leave 1 token unmasked
        prompt_len = min(prompt_len, len(input_ids) - 1)
        labels[:prompt_len] = -100

        return labels

