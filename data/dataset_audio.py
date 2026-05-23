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

import sys
# Block torchcodec to prevent Hugging Face datasets from attempting to load it
# and crashing due to missing system FFmpeg libraries.
sys.modules["torchcodec"] = None
sys.modules["torchcodec.decoders"] = None

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
        system_prompt: str = "คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการถอดเสียง /no_think",
    ):
        self.processor = processor
        self.max_audio_seconds = max_audio_seconds
        self.max_text_length = max_text_length
        self.local_audio_dir = local_audio_dir
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
            # Cast all Audio columns to plain dicts to completely bypass Hugging Face
            # audio decoding logic (which triggers torchcodec import and requirement).
            import datasets
            try:
                new_features = raw.features.copy()
                cast_cols = []
                for col_name, feat in raw.features.items():
                    if isinstance(feat, datasets.Audio) or (hasattr(feat, "__class__") and feat.__class__.__name__ == "Audio"):
                        new_features[col_name] = {"bytes": datasets.Value("binary"), "path": datasets.Value("string")}
                        cast_cols.append(col_name)
                
                if cast_cols:
                    raw = raw.cast(new_features)
                    print(f"[AudioDataset] Successfully cast audio columns {cast_cols} to struct format to bypass torchcodec requirement.")
            except Exception as e:
                print(f"[AudioDataset] Warning: failed to cast audio columns to struct. Error: {e}")
                # Fallback to decode=False on target column if cast fails
                try:
                    raw = raw.cast_column(audio_column, HFAudio(decode=False))
                except Exception:
                    pass
            self.data = raw
            self.audio_col = audio_column
            # Check if text_column is in the dataset columns. If not, we will detect or handle it dynamically in __getitem__
            if text_column and text_column in raw.column_names:
                self.text_col = text_column
            else:
                try:
                    self.text_col = text_column or self._detect_text_column(raw.column_names)
                except ValueError:
                    # Fallback placeholder, we will extract it dynamically in __getitem__
                    self.text_col = text_column or "transcription"

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

        # ── Determine audio and text columns dynamically ──────────────────
        is_spoken_arena = "conversation_a" in item and ("voice_a" in item or "voice_user" in item)
        
        audio_val = None
        transcript = ""

        if is_spoken_arena:
            # Use voice_a (model response) and match with conversation_a's assistant response
            audio_val = item.get("voice_a")
            conv = item.get("conversation_a", [])
            assistant_msgs = [m["content"] for m in conv if m.get("role") == "assistant"]
            transcript = assistant_msgs[-1] if assistant_msgs else ""
            
            # Fallback if voice_a is missing/empty
            if audio_val is None or (isinstance(audio_val, dict) and not audio_val.get("bytes") and not audio_val.get("path")):
                audio_val = item.get("voice_user")
                user_msgs = [m["content"] for m in conv if m.get("role") == "user"]
                transcript = user_msgs[0] if user_msgs else ""
        else:
            audio_val = item.get(self.audio_col)
            if self.text_col in item:
                transcript = str(item[self.text_col]).strip()
            else:
                # Dynamic fallback for text if self.text_col is not found
                for col in TRANSCRIPT_COLUMNS:
                    if col in item:
                        transcript = str(item[col]).strip()
                        break

        # ── Load audio ────────────────────────────────────────────────────
        waveform = None
        sr = self.sample_rate
        path_val = item.get("path")

        # 1. Try loading from HuggingFace audio dict/object if present
        if audio_val is not None:
            if isinstance(audio_val, dict):
                if audio_val.get("array") is not None:
                    # Already decoded (e.g. cast_column was used elsewhere)
                    waveform = np.array(audio_val["array"], dtype=np.float32)
                    sr = audio_val.get("sampling_rate", self.sample_rate)
                elif audio_val.get("bytes") is not None:
                    # Raw encoded bytes (datasets 4.x without torchcodec)
                    import io
                    try:
                        waveform, sr = sf.read(io.BytesIO(audio_val["bytes"]))
                        waveform = waveform.astype(np.float32)
                    except Exception:
                        pass
                if audio_val.get("path") is not None and waveform is None:
                    path_val = audio_val["path"]
            elif isinstance(audio_val, (str, Path)):
                path_val = str(audio_val)

        # Downmix stereo to mono if needed
        if waveform is not None and waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        # 2. Try loading from file path if waveform is still None
        if waveform is None and path_val is not None:
            try:
                p = Path(path_val)
                # If local_audio_dir is specified and path is relative, prefix it
                # We can also search in local_audio_dir if path is absolute but missing
                if self.local_audio_dir:
                    if not p.is_absolute():
                        p = Path(self.local_audio_dir) / p
                    elif not p.exists():
                        p = Path(self.local_audio_dir) / p.name

                if p.exists():
                    waveform, sr = sf.read(str(p))
                    waveform = waveform.astype(np.float32)
            except Exception:
                pass

        # 3. Fallback to dummy silent waveform if still None
        if waveform is None:
            sr = self.sample_rate
            waveform = np.zeros(int(self.max_audio_seconds * sr), dtype=np.float32)

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
        # Use the dynamically resolved transcript
        transcript = str(transcript).strip()
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

