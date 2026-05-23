"""
scripts/setup_tokens.py
------------------------
One-time setup script:
1. Loads Qwen3.5-0.8B tokenizer
2. Adds audio special tokens
3. Saves updated tokenizer for use in all training phases

Run ONCE before any training:
    python scripts/setup_tokens.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniProcessor, OmniConfig

print("=" * 60)
print("Thai Omni-Modal: Token Setup")
print("=" * 60)

cfg = OmniConfig()
processor = OmniProcessor.from_pretrained(
    llm_name=cfg.llm_model_name,
    audio_encoder_name=cfg.audio_encoder_name,
)

print(f"\nVocab size before: {processor.tokenizer.vocab_size:,}")
print(f"Vocab size after adding audio tokens: {len(processor.tokenizer):,}")
print(f"\nSpecial audio token IDs:")
print(f"  <|audio_start|> → {processor.audio_start_id}")
print(f"  <|audio_end|>   → {processor.audio_end_id}")
print(f"  <|audio_pad|>   → {processor.audio_pad_id}")

# Save tokenizer to a setup directory
save_path = Path("outputs/tokenizer_setup")
save_path.mkdir(parents=True, exist_ok=True)
processor.save_pretrained(str(save_path))
print(f"\n✅ Saved to {save_path}")
print("Use this path as llm_name in training configs.")
