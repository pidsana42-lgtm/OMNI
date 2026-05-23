"""
OmniModal Configuration Class
-------------------------------
Stores all sub-model configs and hyperparams for Thai-Omni-Modal.

Architecture:
  Audio Pathway:  typhoon-whisper-turbo → AudioProjector (MLP)  ─┐
  Vision Pathway: Qwen3.5-0.8B Vision Encoder (native) ──────────┤→ Qwen3.5-0.8B LLM
  Text Pathway:   Qwen3.5-0.8B Embedding Layer ──────────────────┘
"""

from transformers import PretrainedConfig


class OmniConfig(PretrainedConfig):
    """
    Configuration for the Thai-Omni-Modal model.
    Inherits from PretrainedConfig so it can be saved/loaded
    with model.save_pretrained() / from_pretrained().
    """

    model_type = "thai_omni"

    def __init__(
        self,
        # ── Sub-model identifiers ─────────────────────────────────────────
        llm_model_name: str = "Qwen/Qwen3.5-0.8B",
        audio_encoder_name: str = "typhoon-ai/typhoon-whisper-turbo",

        # ── Audio Projector dims ──────────────────────────────────────────
        # Whisper large-v3 hidden = 1280; medium = 1024
        # Will be auto-detected at model init if set to None
        audio_encoder_hidden_size: int = 1280,
        llm_hidden_size: int = 1024,         # Qwen3.5-0.8B hidden dim
        projector_hidden_size: int = 2048,   # Intermediate MLP dim
        projector_num_layers: int = 2,       # Depth of projector MLP
        projector_dropout: float = 0.0,

        # ── Special tokens ────────────────────────────────────────────────
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
        audio_pad_token: str = "<|audio_pad|>",

        # ── Audio preprocessing ───────────────────────────────────────────
        sample_rate: int = 16000,
        max_audio_length_seconds: float = 30.0,

        # ── Thinking mode ─────────────────────────────────────────────────
        # ALWAYS False during training — prevents thinking loops in 0.8B
        enable_thinking: bool = False,

        # ── Misc ──────────────────────────────────────────────────────────
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Sub-model names
        self.llm_model_name = llm_model_name
        self.audio_encoder_name = audio_encoder_name

        # Projector architecture
        self.audio_encoder_hidden_size = audio_encoder_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.projector_num_layers = projector_num_layers
        self.projector_dropout = projector_dropout

        # Special tokens
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.audio_pad_token = audio_pad_token

        # Audio settings
        self.sample_rate = sample_rate
        self.max_audio_length_seconds = max_audio_length_seconds

        # Critical: disable thinking during training
        self.enable_thinking = enable_thinking
