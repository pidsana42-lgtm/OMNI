"""
Audio Decoder — Native Speech Output Head
------------------------------------------
Adds a native Audio LM Head to the Thai Omni-Modal model,
enabling the model to generate speech output natively (not via pipeline TTS).

Architecture:
  LLM Hidden States [B, T, 1024]
        ↓
  AudioLMHead (Linear 1024 → codebook_size)
        ↓
  Audio Tokens [B, T_audio]
        ↓
  EnCodec Decoder → Waveform [B, T_samples]

Near-real-time streaming:
  - EnCodec 24kHz: 75 tokens/second per codebook
  - Decode every CHUNK_TOKENS tokens (default: 25 = ~333ms audio)
  - On H200: LLM generates 25 tokens in ~30-50ms → 6-10x faster than audio plays
  - Effective first-chunk latency: ~300-500ms

Requirements:
  pip install encodec
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterator, Tuple
import numpy as np


# ── EnCodec wrapper ────────────────────────────────────────────────────────────

class AudioCodec(nn.Module):
    """
    Wraps facebook/encodec_24khz for audio tokenization and detokenization.
    Provides:
      - encode(waveform) → audio_tokens  [codebooks, T]
      - decode(audio_tokens) → waveform  [T_samples]
    """

    SAMPLE_RATE = 24_000
    TOKENS_PER_SECOND = 75   # EnCodec 24kHz frame rate
    NUM_CODEBOOKS = 8        # EnCodec uses 8 RVQ codebooks
    CODEBOOK_SIZE = 1024     # Each codebook has 1024 entries

    def __init__(self, bandwidth: float = 6.0):
        """
        Args:
            bandwidth: EnCodec bandwidth in kbps. Higher = better quality.
                       Options: 1.5, 3.0, 6.0, 12.0, 24.0
                       6.0 kbps = 8 codebooks active (balanced quality/speed)
        """
        super().__init__()
        self.bandwidth = bandwidth
        self._model = None   # Lazy load to avoid import at module level

    def _load(self, device):
        if self._model is None:
            try:
                from encodec import EncodecModel
                from encodec.utils import convert_audio
                self._model = EncodecModel.encodec_model_24khz()
                self._model.set_target_bandwidth(self.bandwidth)
                self._model = self._model.to(device)
                self._model.eval()
                self._convert_audio = convert_audio
            except ImportError:
                raise ImportError(
                    "EnCodec not installed. Run: pip install encodec"
                )
        return self._model

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Encode waveform to discrete audio tokens.

        Args:
            waveform: [T_samples] or [1, T_samples] at SAMPLE_RATE Hz
        Returns:
            tokens: [num_active_codebooks, T_frames]
        """
        model = self._load(device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)   # [1, 1, T]
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)                 # [1, C, T]

        waveform = waveform.to(device)
        encoded_frames = model.encode(waveform)
        # encoded_frames: list of (codes, scale) where codes: [B, K, T]
        codes = torch.cat([frames for frames, _ in encoded_frames], dim=-1)
        return codes.squeeze(0)   # [K, T]

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Decode discrete audio tokens back to waveform.

        Args:
            tokens: [K, T_frames] or [1, K, T_frames]
        Returns:
            waveform: [T_samples] numpy array (float32, [-1, 1])
        """
        model = self._load(device)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)   # [1, K, T]
        tokens = tokens.to(device).long()

        # EnCodec expects list of (codes, None) frames
        encoded_frames = [(tokens, None)]
        decoded = model.decode(encoded_frames)   # [1, 1, T_samples]
        waveform = decoded.squeeze().cpu().numpy()
        return waveform.astype(np.float32)


# ── Audio LM Head ──────────────────────────────────────────────────────────────

class AudioLMHead(nn.Module):
    """
    Predicts the next audio token from LLM hidden states.

    Takes the hidden state at each position and maps it to a distribution
    over the EnCodec codebook vocabulary.

    For multi-codebook prediction, we predict one codebook at a time
    (flat/single-head approach: predict codebook 0 first, then use a
    separate residual head for 1-7 — kept simple for MVP).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        codebook_size: int = 1024,
        num_codebooks: int = 1,   # Start with 1 for speed; expand later
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        # One projection head per codebook
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, codebook_size),
            )
            for _ in range(num_codebooks)
        ])

        # Special tokens for audio generation
        # BOA = Begin Of Audio, EOA = End Of Audio
        self.BOA_TOKEN = codebook_size      # Index just outside codebook
        self.EOA_TOKEN = codebook_size + 1

        self._init_weights()

    def _init_weights(self):
        for head in self.heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        codebook_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hidden_size]
            codebook_idx: which codebook head to use (0 for primary)
        Returns:
            logits: [B, T, codebook_size]
        """
        return self.heads[codebook_idx](hidden_states)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Streaming Audio Generator ──────────────────────────────────────────────────

class StreamingAudioGenerator:
    """
    Handles near-real-time streaming audio generation.

    Generates audio tokens chunk by chunk and decodes them immediately,
    allowing audio playback to start before generation is complete.

    Usage:
        gen = StreamingAudioGenerator(model, codec, chunk_tokens=25)
        for audio_chunk_np in gen.stream(hidden_states):
            play_audio(audio_chunk_np)  # ~333ms chunks
    """

    CHUNK_TOKENS = 25          # tokens per decode chunk (~333ms at 75 tok/s)
    MAX_AUDIO_TOKENS = 750     # max 10 seconds of audio output
    TEMPERATURE = 0.8
    TOP_K = 50

    def __init__(
        self,
        audio_lm_head: AudioLMHead,
        codec: AudioCodec,
        chunk_tokens: int = CHUNK_TOKENS,
        max_tokens: int = MAX_AUDIO_TOKENS,
        temperature: float = TEMPERATURE,
    ):
        self.audio_lm_head = audio_lm_head
        self.codec = codec
        self.chunk_tokens = chunk_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature

    @torch.no_grad()
    def stream(
        self,
        hidden_states: torch.Tensor,    # [1, T_prompt, hidden_size] — last LLM states
        device: torch.device,
    ) -> Iterator[np.ndarray]:
        """
        Yields audio waveform numpy arrays (float32, 24kHz) in streaming chunks.
        Each chunk is ~333ms of audio (at default chunk_tokens=25).
        """
        # Use the last hidden state as the starting context
        context = hidden_states[:, -1:, :]   # [1, 1, hidden_size]

        accumulated_tokens = []   # List of predicted token tensors
        current_chunk = []        # Tokens in current chunk (for decode)

        for step in range(self.max_tokens):
            # Predict next audio token distribution
            logits = self.audio_lm_head(context, codebook_idx=0)  # [1, 1, vocab]
            logits = logits[:, -1, :] / self.temperature           # [1, vocab]

            # Top-K sampling for diversity
            if self.TOP_K > 0:
                top_k_logits, top_k_indices = torch.topk(logits, self.TOP_K, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Check EOA stop condition
            if next_token.item() >= self.audio_lm_head.codebook_size:
                if current_chunk:
                    yield self._decode_chunk(current_chunk, device)
                break

            current_chunk.append(next_token.squeeze())
            accumulated_tokens.append(next_token.squeeze())

            # Decode and yield when chunk is full
            if len(current_chunk) >= self.chunk_tokens:
                yield self._decode_chunk(current_chunk, device)
                current_chunk = []

            # Update context: use the predicted token embedding
            # (simple AR: feed predicted token ID back as position embedding)
            # In practice: look up embedding from codec codebook embedding table
            # Here we reuse the same context (stateless head for simplicity)
            # For full AR quality: attach audio token to LLM and re-run

    def _decode_chunk(
        self, tokens: list, device: torch.device
    ) -> np.ndarray:
        """Decode a list of audio token indices to a waveform chunk."""
        token_tensor = torch.stack(tokens).unsqueeze(0)   # [1, T_chunk]
        waveform = self.codec.decode(token_tensor, device)
        return waveform
