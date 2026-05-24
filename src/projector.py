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


class AudioQFormerProjector(nn.Module):
    """
    Q-Former + MLP Projector.
    Uses learnable query tokens to cross-attend to Whisper audio features,
    compressing the sequence to a fixed length (e.g. 64 tokens),
    and then maps the compressed features to the LLM space using an MLP.
    """

    def __init__(
        self,
        input_dim: int,           # Whisper hidden size (e.g. 1280)
        output_dim: int,          # Qwen LLM hidden size (e.g. 1024)
        num_query_tokens: int = 64,
        num_layers: int = 2,
        nhead: int = 8,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        
        # Learnable queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, output_dim))
        
        # Encoder projection: Maps Whisper hidden dim to Q-Former (output_dim) dimension
        self.encoder_proj = nn.Linear(input_dim, output_dim)
        
        # Q-Former Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # MLP bridge (similar to standard AudioProjector, e.g. 3M params)
        self.output_mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize queries
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Xavier uniform init for linear weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: Tensor [B, T, whisper_hidden]
        
        Returns:
            projected: Tensor [B, num_query_tokens, qwen_hidden]
        """
        B = audio_features.shape[0]
        
        # 1. Project Whisper features to match Q-Former's hidden dimension
        # encoder_features: [B, T_audio, output_dim]
        encoder_features = self.encoder_proj(audio_features)
        
        # 2. Expand learnable queries to batch size
        # queries: [B, num_query_tokens, output_dim]
        queries = self.query_tokens.expand(B, -1, -1)
        
        # 3. Cross attention through Transformer Decoder
        # queries: target (queries)
        # encoder_features: memory (keys, values)
        # qformer_out: [B, num_query_tokens, output_dim]
        qformer_out = self.qformer(tgt=queries, memory=encoder_features)
        
        # 4. Feedforward projection (MLP)
        # out: [B, num_query_tokens, output_dim]
        out = self.output_mlp(qformer_out)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

