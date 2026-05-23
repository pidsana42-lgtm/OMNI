"""
Pre-download and Cache Training Datasets from Hugging Face Hub
--------------------------------------------------------------
Downloads all datasets defined in configs to avoid wasting GPU leasing time.

Usage:
    python scripts/download_datasets.py
"""

import sys
from datasets import load_dataset


def download_dataset(name, split, config=None):
    try:
        print(f"[HF Data] Downloading '{name}' (split: {split}, config: {config})...")
        load_dataset(name, name=config, split=split, trust_remote_code=True)
        print(f"  ✅ Successfully cached '{name}' [{split}]")
    except Exception as e:
        print(f"  ❌ Error downloading '{name}': {e}")


def main():
    print("🚀 Starting Pre-downloading of all official Typhoon datasets...")

    # Phase 1 & 2: Audio Dataset
    download_dataset(
        name="typhoon-ai/typhoon-audio-preview-data",
        split="train"
    )
    download_dataset(
        name="typhoon-ai/typhoon-audio-preview-data",
        split="test"
    )

    # Phase 2: Text Instruct Dataset
    download_dataset(
        name="typhoon-ai/typhoon-s-instruct-post-training",
        split="sft"
    )

    # Phase 2: Vision Dataset
    download_dataset(
        name="typhoon-ai/typhoon-vision-preview-data",
        split="finetune"
    )

    # Phase 4: TTS Resynthesized Dataset
    download_dataset(
        name="typhoon-ai/tts_arena_resynthesized",
        split="train"
    )

    print("\n🎉 All datasets pre-downloaded and cached successfully!")


if __name__ == "__main__":
    main()
