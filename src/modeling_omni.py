"""
OmniModal Model (modeling_omni.py)
------------------------------------
The "glue" class that combines:
  1. typhoon-whisper-turbo  → Audio Encoder (INPUT)
  2. AudioProjector          → MLP bridge
  3. Qwen3.5-0.8B            → Core LLM + Vision (native)
  4. AudioLMHead + EnCodec   → Native Speech Output (OPTIONAL)

Forward Pass Logic:
  ┌─ Audio present?  → Whisper → Projector → prepend to LLM inputs_embeds
  └─ Image present?  → Qwen handles natively via pixel_values
  Text              → Qwen embedding lookup as usual

  All paths merge at Qwen's transformer layers.

Speech Output (Phase 4):
  LLM hidden states → AudioLMHead → Audio Tokens → EnCodec Decoder → WAV
  Supports streaming: decode every N tokens for near-real-time playback

Thinking Mode:
  ALWAYS disabled (enable_thinking=False) during training to prevent
  Qwen3.5-0.8B's known thinking-loop bug on small models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from transformers import (
    WhisperModel,
    AutoProcessor,
    WhisperProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# Qwen3.5-0.8B is a VLM — use AutoModelForVision2Seq on HF>=4.50
# For older local transformers: falls back gracefully
try:
    from transformers import AutoModelForImageTextToText as _QwenLoader
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as _QwenLoader
    except ImportError:
        from transformers import AutoModelForCausalLM as _QwenLoader

from .configuration_omni import OmniConfig
from .projector import AudioProjector
from .audio_decoder import AudioLMHead, AudioCodec


class OmniModalModel(nn.Module):
    """
    Thai Omni-Modal Model: Audio + Vision + Text → Thai response.

    Usage:
        config = OmniConfig()
        model = OmniModalModel(config)

        # Audio + text
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_features=whisper_mel_features,
            labels=labels,
        )

        # Vision + text (native Qwen path)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
    """

    config_class = OmniConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config

        # ── 1. Audio Encoder (Whisper) ────────────────────────────────────
        print(f"[OmniModal] Loading audio encoder: {config.audio_encoder_name}")
        self.audio_encoder = WhisperModel.from_pretrained(
            config.audio_encoder_name,
            torch_dtype=torch.bfloat16,
        ).encoder   # Only need encoder part, decoder not used

        # ── 2. Audio Projector (MLP bridge) ──────────────────────────────
        # Auto-detect whisper hidden size from loaded model
        whisper_hidden = self.audio_encoder.config.d_model
        if whisper_hidden != config.audio_encoder_hidden_size:
            print(
                f"[OmniModal] Auto-updating audio_encoder_hidden_size: "
                f"{config.audio_encoder_hidden_size} → {whisper_hidden}"
            )
            config.audio_encoder_hidden_size = whisper_hidden

        print(
            f"[OmniModal] AudioProjector: "
            f"{whisper_hidden} → {config.projector_hidden_size} → {config.llm_hidden_size}"
        )
        self.audio_projector = AudioProjector(
            input_dim=whisper_hidden,
            output_dim=config.llm_hidden_size,
            hidden_dim=config.projector_hidden_size,
            num_layers=config.projector_num_layers,
            dropout=config.projector_dropout,
        )

        # ── 3. Core LLM: Qwen3.5-0.8B (Vision already built-in) ─────────
        print(f"[OmniModal] Loading core LLM: {config.llm_model_name}")
        self.llm = _QwenLoader.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",   # Efficient SDPA attention on H200
            trust_remote_code=True,
        )

        # Expose lm_head for generation compatibility
        self.lm_head = self.llm.lm_head if hasattr(self.llm, "lm_head") else None

        # ── 4. Audio LM Head (Speech Output) — optional ───────────────────
        # Initialized to None; call add_audio_output_head() to enable.
        # Trained in Phase 4 (audio output alignment).
        self.audio_lm_head: Optional[AudioLMHead] = None
        self.audio_codec: Optional[AudioCodec] = None

        if config.enable_audio_output:
            self.add_audio_output_head(
                codebook_size=config.audio_codebook_size,
                num_codebooks=config.audio_num_codebooks,
                bandwidth=config.audio_codec_bandwidth,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Freeze / Unfreeze helpers (called per training phase)
    # ─────────────────────────────────────────────────────────────────────

    def freeze_audio_encoder(self):
        """Freeze Whisper — protect pretrained Thai ASR capability."""
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        print("[OmniModal] ❄️  Audio encoder frozen.")

    def unfreeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = True
        print("[OmniModal] 🔥  Audio encoder unfrozen.")

    def freeze_llm(self):
        """Freeze Qwen3.5-0.8B — protect pretrained language/vision knowledge."""
        for p in self.llm.parameters():
            p.requires_grad = False
        print("[OmniModal] ❄️  LLM frozen.")

    def unfreeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = True
        print("[OmniModal] 🔥  LLM unfrozen.")

    def freeze_projector(self):
        for p in self.audio_projector.parameters():
            p.requires_grad = False
        print("[OmniModal] ❄️  Audio projector frozen.")

    def unfreeze_projector(self):
        for p in self.audio_projector.parameters():
            p.requires_grad = True
        print("[OmniModal] 🔥  Audio projector unfrozen.")

    def setup_phase1(self):
        """
        Phase 1: Audio Modality Alignment
        Freeze everything except AudioProjector.
        Goal: teach the projector to map Whisper → Qwen space.
        """
        print("\n[OmniModal] 📌 Setting up Phase 1: Audio Alignment")
        self.freeze_audio_encoder()
        self.freeze_llm()
        self.unfreeze_projector()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[OmniModal] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def setup_phase2(self):
        """
        Phase 2: Omni-Modal Full Fine-Tuning
        Freeze: Whisper + Qwen Vision Encoder
        Unfreeze: Qwen LLM backbone + AudioProjector
        """
        print("\n[OmniModal] 📌 Setting up Phase 2: Omni Fine-Tuning")
        self.freeze_audio_encoder()
        self.unfreeze_projector()
        self.unfreeze_llm()
        # Re-freeze just the vision encoder inside Qwen (if accessible)
        if hasattr(self.llm, "visual"):
            for p in self.llm.visual.parameters():
                p.requires_grad = False
            print("[OmniModal] ❄️  Qwen vision encoder frozen (within LLM).")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[OmniModal] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def setup_phase4_audio_output(self):
        """
        Phase 4: Native Speech Output Training
        Freeze: Everything except AudioLMHead
        Unfreeze: AudioLMHead only

        Goal: teach the AudioLMHead to predict EnCodec audio tokens
        from LLM hidden states, enabling native speech output.
        """
        print("\n[OmniModal] 📌 Setting up Phase 4: Audio Output Alignment")
        if self.audio_lm_head is None:
            raise RuntimeError(
                "AudioLMHead not initialized. Call add_audio_output_head() first."
            )
        self.freeze_audio_encoder()
        self.freeze_llm()
        self.freeze_projector()
        # Unfreeze ONLY the audio output head
        for p in self.audio_lm_head.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[OmniModal] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────
    # Core Forward Pass
    # ─────────────────────────────────────────────────────────────────────

    def _encode_audio(
        self, audio_input_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Run Whisper encoder and project output into LLM hidden space.

        Args:
            audio_input_features: [B, 128, T_mel] log-mel spectrogram
                                  (from WhisperProcessor)

        Returns:
            projected: [B, T_audio, llm_hidden]
        """
        with torch.no_grad() if not self.audio_encoder.training else torch.enable_grad():
            encoder_out = self.audio_encoder(audio_input_features)
        audio_hidden = encoder_out.last_hidden_state     # [B, T, whisper_hidden]
        projected = self.audio_projector(audio_hidden)   # [B, T, llm_hidden]
        return projected

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Audio inputs (present when modality == "audio")
        audio_input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        # Vision inputs (passed directly to Qwen)
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        # Misc
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Unified forward pass for all 3 modalities.

        Audio path:
          - Encodes audio → projects → prepends to text embeddings
          - Adjusts attention_mask and labels accordingly

        Vision path:
          - Passes pixel_values directly to Qwen (native handling)

        Text path:
          - Passes input_ids directly to Qwen
        """
        # ── If audio is present: build inputs_embeds by prepending audio ──
        if audio_input_features is not None:
            # 1. Get projected audio features: [B, T_audio, D]
            audio_embeds = self._encode_audio(audio_input_features)
            B, T_audio, D = audio_embeds.shape

            # 2. Get text token embeddings via Qwen's embed_tokens
            embed_fn = self._get_embed_tokens()
            text_embeds = embed_fn(input_ids)   # [B, T_text, D]
            _, T_text, _ = text_embeds.shape

            # 3. Concatenate: [audio | text]
            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            # 4. Extend attention mask
            if attention_mask is not None:
                audio_mask = torch.ones(
                    B, T_audio,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

            # 5. Extend labels: ignore audio prefix in loss (-100)
            if labels is not None:
                audio_labels = torch.full(
                    (B, T_audio),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([audio_labels, labels], dim=1)

            # 6. Forward to Qwen (no input_ids — we provide inputs_embeds)
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        # ── Text-only or Vision-Text path: delegate directly to Qwen ──────
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _get_embed_tokens(self):
        """Retrieve Qwen's token embedding layer regardless of model structure."""
        # Try standard locations in Qwen3.5 structure
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
            return self.llm.model.embed_tokens
        if hasattr(self.llm, "language_model") and hasattr(
            self.llm.language_model, "model"
        ):
            return self.llm.language_model.model.embed_tokens
        raise AttributeError(
            "Cannot find embed_tokens in LLM. "
            "Please inspect self.llm structure and update _get_embed_tokens()."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Save / Load helpers
    # ─────────────────────────────────────────────────────────────────────

    def add_audio_output_head(
        self,
        codebook_size: int = 1024,
        num_codebooks: int = 1,
        bandwidth: float = 6.0,
    ):
        """
        Attach AudioLMHead and AudioCodec to enable speech output.
        Call this BEFORE Phase 4 training or loading a Phase 4 checkpoint.
        """
        self.audio_lm_head = AudioLMHead(
            hidden_size=self.config.llm_hidden_size,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
        )
        self.audio_codec = AudioCodec(bandwidth=bandwidth)
        print(
            f"[OmniModal] 🔊 AudioLMHead added: "
            f"{self.config.llm_hidden_size} → vocab({codebook_size}) "
            f"× {num_codebooks} codebook(s)"
        )
        print(
            f"[OmniModal]    Params: {self.audio_lm_head.num_parameters():,}"
        )

    def save_pretrained(self, save_directory: str):
        """Save all components to disk."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)
        torch.save(
            self.audio_projector.state_dict(),
            os.path.join(save_directory, "audio_projector.pt"),
        )
        torch.save(
            self.audio_encoder.state_dict(),
            os.path.join(save_directory, "audio_encoder.pt"),
        )
        # Save AudioLMHead if present (Phase 4)
        if self.audio_lm_head is not None:
            torch.save(
                self.audio_lm_head.state_dict(),
                os.path.join(save_directory, "audio_lm_head.pt"),
            )
            print("[OmniModal] 🔊 AudioLMHead weights saved.")
        print(f"[OmniModal] ✅ Saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load all components from disk."""
        import os
        config = OmniConfig.from_pretrained(load_directory)
        model = cls(config)
        model.audio_projector.load_state_dict(
            torch.load(os.path.join(load_directory, "audio_projector.pt"), weights_only=True)
        )
        model.audio_encoder.load_state_dict(
            torch.load(os.path.join(load_directory, "audio_encoder.pt"), weights_only=True)
        )
        # Load AudioLMHead if checkpoint has one (Phase 4)
        audio_head_path = os.path.join(load_directory, "audio_lm_head.pt")
        if os.path.exists(audio_head_path):
            model.add_audio_output_head(
                codebook_size=config.audio_codebook_size,
                num_codebooks=config.audio_num_codebooks,
                bandwidth=config.audio_codec_bandwidth,
            )
            model.audio_lm_head.load_state_dict(
                torch.load(audio_head_path, weights_only=True)
            )
            print("[OmniModal] 🔊 AudioLMHead loaded from checkpoint.")
        print(f"[OmniModal] ✅ Loaded from {load_directory}")
        return model

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"Trainable params: {trainable:,} || "
            f"All params: {total:,} || "
            f"Trainable%: {100 * trainable / total:.4f}%"
        )


    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load all components from disk."""
        import os
        config = OmniConfig.from_pretrained(load_directory)
        model = cls(config)
        model.audio_projector.load_state_dict(
            torch.load(os.path.join(load_directory, "audio_projector.pt"), weights_only=True)
        )
        model.audio_encoder.load_state_dict(
            torch.load(os.path.join(load_directory, "audio_encoder.pt"), weights_only=True)
        )
        print(f"[OmniModal] ✅ Loaded from {load_directory}")
        return model

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"Trainable params: {trainable:,} || "
            f"All params: {total:,} || "
            f"Trainable%: {100 * trainable / total:.4f}%"
        )
