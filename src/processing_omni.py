"""
OmniModal Processor (processing_omni.py)
-----------------------------------------
Unified processor wrapping:
  - WhisperProcessor  → audio log-mel spectrogram
  - AutoProcessor     → Qwen3.5 tokenizer + image processor

Also handles adding special audio tokens to the tokenizer and
applying the Qwen3.5 chat template with enable_thinking=False.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, List, Union, Dict, Any

from transformers import AutoProcessor, WhisperProcessor


AUDIO_SPECIAL_TOKENS = [
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
]

THINKING_SPECIAL_TOKENS = [
    "<think>",
    "</think>",
]



class OmniProcessor:
    """
    Wraps both processors into one unified API.

    Example usage:
        proc = OmniProcessor.from_pretrained(
            llm_name="Qwen/Qwen3.5-0.8B",
            audio_encoder_name="scb10x/typhoon-whisper-turbo",
        )

        # Audio + text
        batch = proc(
            text="ถอดความเสียงต่อไปนี้:",
            audio=waveform_array,
            return_tensors="pt",
        )

        # Vision + text (native Qwen path)
        batch = proc(
            messages=[{"role": "user", "content": [
                {"type": "image", "url": "..."},
                {"type": "text", "text": "อธิบายภาพนี้"},
            ]}],
            return_tensors="pt",
        )
    """

    def __init__(
        self,
        llm_processor: AutoProcessor,
        audio_processor: WhisperProcessor,
        sample_rate: int = 16000,
    ):
        self.llm_processor = llm_processor
        self.audio_processor = audio_processor
        self.tokenizer = getattr(llm_processor, "tokenizer", llm_processor)
        self.sample_rate = sample_rate

    @classmethod
    def from_pretrained(
        cls,
        llm_name: str = "Qwen/Qwen3.5-0.8B",
        audio_encoder_name: str = "scb10x/typhoon-whisper-turbo",
        sample_rate: int = 16000,
    ) -> "OmniProcessor":
        print(f"[OmniProcessor] Loading LLM processor from {llm_name}")
        llm_processor = AutoProcessor.from_pretrained(
            llm_name,
            trust_remote_code=True,
        )

        print(f"[OmniProcessor] Loading audio processor from {audio_encoder_name}")
        audio_processor = WhisperProcessor.from_pretrained(
            audio_encoder_name,
            trust_remote_code=True,
        )

        proc = cls(llm_processor, audio_processor, sample_rate)
        proc._add_audio_tokens()
        return proc

    def _add_audio_tokens(self):
        """
        Add special audio and thinking tokens to the tokenizer.
        Must be called once before training — then save the tokenizer
        alongside the model so embeddings stay consistent.
        """
        existing = set(self.tokenizer.all_special_tokens)
        new_tokens = [t for t in AUDIO_SPECIAL_TOKENS + THINKING_SPECIAL_TOKENS if t not in existing]

        if new_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            print(f"[OmniProcessor] Added {len(new_tokens)} special tokens: {new_tokens}")
        else:
            print("[OmniProcessor] Special tokens already present.")

        # Cache token ids for convenience
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    def process_audio(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw waveform(s) to log-mel spectrogram features
        expected by Whisper encoder.

        Args:
            audio: numpy array [T] or list of arrays
            sampling_rate: original sample rate (will resample to 16kHz)

        Returns:
            dict with 'input_features': [B, 128, 3000]
        """
        sr = sampling_rate or self.sample_rate
        result = self.audio_processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )
        return result  # keys: input_features, attention_mask

    def process_text(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        truncation: bool = True,
        padding: Union[bool, str] = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize plain text."""
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,   # ← Always False during training
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Qwen3.5 chat template directly via self.tokenizer.

        KEY: enable_thinking=False prevents 0.8B model from
        entering infinite thinking loops during training & inference.

        Calls self.tokenizer.apply_chat_template() directly (not through
        llm_processor) so that any chat_template we patch is always honoured.
        If the checkpoint tokenizer is missing a chat_template we auto-patch
        it from the Qwen3.5-0.8B base tokenizer.
        """
        # ── Auto-patch missing chat_template ─────────────────────────────
        if not getattr(self.tokenizer, "chat_template", None):
            print("[OmniProcessor] ⚠️  No chat_template found in tokenizer. "
                  "Auto-patching from Qwen/Qwen3.5-0.8B base...")
            from transformers import AutoTokenizer
            _base_tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-0.8B", trust_remote_code=True
            )
            self.tokenizer.chat_template = _base_tok.chat_template

        # ── Remove <think> blocks from text before tokenization ──────────
        if not enable_thinking:
            import copy
            import re
            cleaned_messages = []
            for m in messages:
                m_copy = copy.deepcopy(m)
                if isinstance(m_copy.get("content"), str):
                    m_copy["content"] = re.sub(r'<think>.*?</think>', '', m_copy["content"], flags=re.DOTALL).strip()
                    # Also clean lingering tags just in case
                    m_copy["content"] = m_copy["content"].replace("<think>", "").replace("</think>", "").strip()
                cleaned_messages.append(m_copy)
            messages = cleaned_messages

        # Call directly on the tokenizer so our patch is always used
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            **kwargs,
        )
        if not enable_thinking:
            input_ids = self._strip_thinking_tokens(input_ids)
        return input_ids

    def _strip_thinking_tokens(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        """Remove <think>...</think> block from tokenized output."""
        think_start_id = self.tokenizer.convert_tokens_to_ids("<think>")
        think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")

        think_starts = [think_start_id]
        think_ends = [think_end_id]

        alt_start = self.tokenizer.convert_tokens_to_ids("<|think|>")
        if alt_start != self.tokenizer.unk_token_id and alt_start is not None:
            think_starts.append(alt_start)
        alt_end = self.tokenizer.convert_tokens_to_ids("<|/think|>")
        if alt_end != self.tokenizer.unk_token_id and alt_end is not None:
            think_ends.append(alt_end)

        # Filter out invalid/unk ids
        think_starts = [t for t in think_starts if t != self.tokenizer.unk_token_id and t is not None]
        think_ends = [t for t in think_ends if t != self.tokenizer.unk_token_id and t is not None]

        if not think_starts:
            return encoded

        input_ids = encoded.get("input_ids", None)
        if input_ids is None:
            return encoded

        attention_mask = encoded.get("attention_mask", None)
        is_pt = isinstance(input_ids, torch.Tensor)

        # Convert to list for easier manipulation
        if is_pt:
            input_ids_list = input_ids.tolist()
            attention_mask_list = attention_mask.tolist() if attention_mask is not None else None
        else:
            input_ids_list = input_ids
            attention_mask_list = attention_mask

        new_input_ids_list = []
        new_attention_mask_list = [] if attention_mask_list is not None else None

        for seq_idx, seq in enumerate(input_ids_list):
            new_seq = []
            new_mask = []
            mask_seq = attention_mask_list[seq_idx] if attention_mask_list is not None else None

            in_think = False
            for i, token in enumerate(seq):
                if token in think_starts:
                    in_think = True
                    continue
                elif token in think_ends:
                    in_think = False
                    continue

                if not in_think:
                    new_seq.append(token)
                    if mask_seq is not None:
                        new_mask.append(mask_seq[i])

            new_input_ids_list.append(new_seq)
            if new_attention_mask_list is not None:
                new_attention_mask_list.append(new_mask)

        # Convert back to torch tensor if input was torch tensor
        if is_pt:
            encoded["input_ids"] = torch.tensor(new_input_ids_list, dtype=input_ids.dtype, device=input_ids.device)
            if attention_mask is not None:
                encoded["attention_mask"] = torch.tensor(new_attention_mask_list, dtype=attention_mask.dtype, device=attention_mask.device)
        else:
            encoded["input_ids"] = new_input_ids_list
            if attention_mask is not None:
                encoded["attention_mask"] = new_attention_mask_list

        return encoded

    def build_audio_instruction(
        self,
        transcript: str,
        system_prompt: str = "คุณเป็น AI ผู้ช่วยภาษาไทย",
    ) -> List[Dict[str, Any]]:
        """
        Build a chat message for audio-text training.
        Audio features will be prepended separately in the collator.

        Structure:
          <|audio_start|>...<|audio_end|>
          [Instruction: ถอดเสียงต่อไปนี้เป็นข้อความ]
          [Answer: transcript]
        """
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"<|audio_start|><|audio_end|>\n"
                    f"ถอดเสียงต่อไปนี้เป็นข้อความภาษาไทย:"
                ),
            },
            {"role": "assistant", "content": transcript},
        ]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, save_directory: str):
        # Save the full llm_processor (includes image processor config etc.)
        self.llm_processor.save_pretrained(save_directory)
        # Explicitly save the tokenizer with chat_template to tokenizer_config.json
        # This ensures apply_chat_template works when reloading the checkpoint
        self.tokenizer.save_pretrained(save_directory)
        # ONLY save the feature extractor from audio processor to prevent overwriting LLM's tokenizer.json
        self.audio_processor.feature_extractor.save_pretrained(save_directory)
        print(f"[OmniProcessor] Saved to {save_directory}")

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def __repr__(self):
        return (
            f"OmniProcessor(\n"
            f"  llm={self.llm_processor.__class__.__name__},\n"
            f"  audio={self.audio_processor.__class__.__name__},\n"
            f"  vocab_size={len(self.tokenizer)},\n"
            f"  sample_rate={self.sample_rate},\n"
            f")"
        )
