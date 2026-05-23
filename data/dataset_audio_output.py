"""
Audio Output Dataset for Phase 4 Training
-----------------------------------------
Loads training pairs of (input_prompt, target_audio_response).
Target audio response is processed using EnCodec to extract target discrete tokens
which our AudioLMHead will learn to predict.

Supported Input Prompts:
  - Text prompts (Text-to-Speech alignment)
  - Audio prompts (Speech-to-Speech alignment)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import soundfile as sf
import numpy as np
import librosa

from src.processing_omni import OmniProcessor
from src.audio_decoder import AudioCodec


class AudioOutputDataset(Dataset):
    """
    Dataset for native audio generation training.
    Maps: (Text/Audio prompt) -> Target Audio Tokens.
    """

    def __init__(
        self,
        processor: OmniProcessor,
        codec: AudioCodec,
        manifest_file: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
        hf_split: str = "train",
        target_audio_column: str = "audio",
        prompt_text_column: str = "text",
        prompt_audio_column: Optional[str] = None,
        max_prompt_length: int = 512,
        max_target_seconds: float = 10.0,
        system_prompt: str = "คุณเป็น AI ผู้ช่วยภาษาไทยที่ตอบด้วยเสียงพูดภาษาไทยที่ชัดเจน",
    ):
        self.processor = processor
        self.codec = codec
        self.max_prompt_length = max_prompt_length
        self.max_target_seconds = max_target_seconds
        self.system_prompt = system_prompt
        self.target_audio_col = target_audio_column
        self.prompt_text_col = prompt_text_column
        self.prompt_audio_col = prompt_audio_column
        self.hf_dataset_name = hf_dataset_name

        # Load Manifest or HF Dataset
        if hf_dataset_name:
            print(f"[AudioOutputDS] Loading HF dataset: {hf_dataset_name} ({hf_dataset_config or ''}) [{hf_split}]")
            from datasets import load_dataset, Audio as HFAudio
            raw = load_dataset(
                hf_dataset_name,
                hf_dataset_config,
                split=hf_split,
                trust_remote_code=True,
            )
            # Cast target audio column to ensure correct sample rate for vocoder
            raw = raw.cast_column(target_audio_column, HFAudio(sampling_rate=self.codec.SAMPLE_RATE))
            if prompt_audio_column:
                raw = raw.cast_column(prompt_audio_column, HFAudio(sampling_rate=self.processor.sample_rate))
            self.samples = raw
        elif manifest_file:
            import json
            self.samples = []
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))
        else:
            raise ValueError("Provide either hf_dataset_name or manifest_file")

        print(f"[AudioOutputDS] Loaded {len(self.samples):,} samples for speech output training.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # ── 1. Target Audio processing (EnCodec tokens) ──────────────────
        target_val = item[self.target_audio_col]
        if isinstance(target_val, dict) and "array" in target_val:
            # Hugging Face Audio feature format
            target_wav = target_val["array"].astype(np.float32)
            sr = target_val["sampling_rate"]
        elif isinstance(target_val, (str, Path)):
            # Local file path format
            target_wav, sr = sf.read(str(target_val))
            target_wav = target_wav.astype(np.float32)
        else:
            raise ValueError(f"Unexpected target audio format: {type(target_val)}")

        # EnCodec expects 24kHz
        if sr != self.codec.SAMPLE_RATE:
            target_wav = librosa.resample(target_wav, orig_sr=sr, target_sr=self.codec.SAMPLE_RATE)

        # Clip to max duration
        max_samples = int(self.max_target_seconds * self.codec.SAMPLE_RATE)
        target_wav = target_wav[:max_samples]

        # Convert to tensor [T_samples]
        target_wav_tensor = torch.from_numpy(target_wav)

        # Extract tokens: shape [num_codebooks, T_frames]
        # We only train codebook 0 (primary head) for speed and simplicity
        audio_tokens = self.codec.encode(target_wav_tensor, device=torch.device("cpu"))
        target_tokens = audio_tokens[0]  # [T_frames]

        # ── 2. Input Prompt preparation ──────────────────────────────────
        # Check if we have audio prompt (Speech-to-Speech)
        prompt_audio_val = item.get(self.prompt_audio_col) if self.prompt_audio_col else None
        has_audio_prompt = prompt_audio_val is not None

        audio_features = None
        if has_audio_prompt:
            if isinstance(prompt_audio_val, dict) and "array" in prompt_audio_val:
                p_wav = prompt_audio_val["array"].astype(np.float32)
                p_sr = prompt_audio_val["sampling_rate"]
            else:
                p_wav, p_sr = sf.read(str(prompt_audio_val))
                p_wav = p_wav.astype(np.float32)

            if p_sr != self.processor.sample_rate:
                p_wav = librosa.resample(p_wav, orig_sr=p_sr, target_sr=self.processor.sample_rate)
            # Mel scale features for Whisper input
            audio_feats = self.processor.process_audio(p_wav, self.processor.sample_rate)
            audio_features = audio_feats["input_features"].squeeze(0)  # [128, 3000]

            messages = self.processor.build_audio_instruction(
                transcript="<|audio|>",
                system_prompt=self.system_prompt,
            )
        else:
            # Text prompt (Text-to-Speech)
            prompt_text = item[self.prompt_text_col]
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text},
            ]

        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Truncate prompt if too long
        if len(input_ids) > self.max_prompt_length:
            input_ids = input_ids[-self.max_prompt_length:]
            attention_mask = attention_mask[-self.max_prompt_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_input_features": audio_features,  # None for Text prompts
            "target_tokens": target_tokens,          # [T_frames] targets for AudioLMHead
            "modality": "audio_prompt" if has_audio_prompt else "text_prompt",
        }


class AudioOutputCollator:
    """Collates prompt-target pairs for Phase 4 training."""

    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        has_audio = [item["audio_input_features"] is not None for item in batch]

        # Pad text inputs
        max_prompt_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids = []
        attention_mask = []

        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_prompt_len - ids.shape[0]
            if pad_len > 0:
                # Right-pad
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros((pad_len,), dtype=mask.dtype)])
            input_ids.append(ids)
            attention_mask.append(mask)

        # Pad target audio tokens (right-pad with EOA/ignore index)
        max_target_len = max(item["target_tokens"].shape[0] for item in batch)
        target_tokens = []
        for item in batch:
            tokens = item["target_tokens"]
            pad_len = max_target_len - tokens.shape[0]
            if pad_len > 0:
                # Right pad with ignore index so we don't calculate loss on padding
                tokens = torch.cat([tokens, torch.full((pad_len,), self.ignore_index, dtype=tokens.dtype)])
            target_tokens.append(tokens)

        # Stack audio features if present
        audio_features = None
        if any(has_audio):
            feats = []
            for item in batch:
                if item["audio_input_features"] is not None:
                    feats.append(item["audio_input_features"])
                else:
                    # Dummy silence vector to keep batch dim aligned
                    feats.append(torch.zeros(128, 3000))
            audio_features = torch.stack(feats, dim=0)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "audio_input_features": audio_features,
            "target_tokens": torch.stack(target_tokens),  # [B, T_target]
            "has_audio_mask": torch.tensor(has_audio, dtype=torch.bool),
        }
