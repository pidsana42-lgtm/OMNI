"""
Audio Projector MLP
--------------------
Bridges the Whisper audio encoder's hidden space to Qwen3.5's hidden space.

Design:
  Input  : [B, T_audio, whisper_hidden]  e.g. [B, 1500, 1280]
  Output : [B, T_audio, qwen_hidden]     e.g. [B, 1500, 1024]

Architecture (2-layer MLP with GELU):
  Linear(in → mid) → LayerNorm → GELU → Dropout → Linear(mid → out) → LayerNorm
"""

import torch
import torch.nn as nn


class AudioProjector(nn.Module):
    """
    Multi-layer perceptron that projects Whisper audio features
    into the LLM embedding space.

    Kept separate from the main model so it can be:
    - Frozen/unfrozen independently during each training phase
    - Saved/loaded independently for debugging
    """

    def __init__(
        self,
        input_dim: int,       # Whisper encoder hidden size  (e.g. 1280)
        output_dim: int,      # Qwen3.5 hidden size           (e.g. 1024)
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(current_dim, out, bias=True))
            layers.append(nn.LayerNorm(out))

            if not is_last:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

            current_dim = out

        self.projection = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: Tensor [B, T, whisper_hidden]
                            Output of WhisperEncoder.last_hidden_state

        Returns:
            projected: Tensor [B, T, qwen_hidden]
        """
        return self.projection(audio_features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
