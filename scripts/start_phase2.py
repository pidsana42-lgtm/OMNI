"""
start_phase2.py — Pull Phase 1 checkpoint from HF Hub, pre-cache datasets, then start Phase 2.

Usage (on new cloud session):
    git pull origin main
    pip install -r requirements.txt
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python scripts/start_phase2.py --hf_token YOUR_TOKEN
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--hub_model_id", default="Phonsiri/thai-omni-modal-0.8b-phase1",
                        help="HF Hub repo ID for Phase 1 checkpoint")
    parser.add_argument("--local_dir", default="outputs/phase1/hf_checkpoint",
                        help="Local directory to save the pulled checkpoint")
    parser.add_argument("--phase2_config", default="configs/phase2_finetune.yaml")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip checkpoint download if already exists locally")
    parser.add_argument("--skip_datasets", action="store_true",
                        help="Skip dataset pre-caching")
    return parser.parse_args()


def precache_datasets(token: str):
    """Pre-download all Phase 2 datasets into HF cache before training starts."""
    from datasets import load_dataset
    from datasets import Audio as HFAudio

    datasets_to_cache = [
        # (name, config, split, description)
        ("google/fleurs",              "th_th",  "train",      "FLEURS Thai (audio)"),
        ("google/fleurs",              "th_th",  "validation", "FLEURS Thai (audio eval)"),
        ("typhoon-ai/chatbot-arena-spoken-voices", None, "train", "Typhoon Spoken Voices"),
        ("mlabonne/FineTome-100k",     None,     "train",      "FineTome text instruction"),
        ("patomp/thai-mscoco-2014-captions", None, "train",    "Thai MSCOCO image captions"),
    ]

    for ds_name, config, split, desc in datasets_to_cache:
        print(f"\n📥 Pre-caching: {desc} ({ds_name}/{split})...")
        try:
            kwargs = dict(split=split, token=token)
            if config:
                ds = load_dataset(ds_name, config, **kwargs)
            else:
                ds = load_dataset(ds_name, **kwargs)

            # For audio datasets, disable auto-decode to avoid torchcodec
            if ds_name in ["google/fleurs", "typhoon-ai/chatbot-arena-spoken-voices"]:
                ds = ds.cast_column("audio", HFAudio(decode=False))

            print(f"   ✅ {len(ds):,} samples cached.")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not cache {ds_name}: {e}")
            print(f"   → Training will attempt to download it on-the-fly.")


def main():
    args = parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF token provided. Use --hf_token or set HF_TOKEN env var.")
        sys.exit(1)

    # ── Step 1: Login ─────────────────────────────────────────────────────
    print("=" * 60)
    print("[Step 1] Logging into HuggingFace Hub...")
    from huggingface_hub import login, snapshot_download
    login(token=token)
    print("✅ Logged in.\n")

    # ── Step 2: Pull Phase 1 checkpoint from HF Hub ───────────────────────
    local_dir = Path(args.local_dir)
    print("=" * 60)
    repo_id = args.hub_model_id
    if "/" not in repo_id:
        repo_id = f"Phonsiri/{repo_id}"
    print(f"[Step 2] Downloading Phase 1 checkpoint: {repo_id}")

    if args.skip_download and local_dir.exists() and any(local_dir.iterdir()):
        print(f"✅ Skipping download — checkpoint already at {local_dir}")
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            token=token,
        )
        print(f"✅ Phase 1 checkpoint saved to: {local_dir}\n")

    # ── Step 3: Validate Phase 1 checkpoint ──────────────────────────────
    print("=" * 60)
    print("[Step 3] Validating Phase 1 checkpoint with inference_demo.py...")
    val_cmd = [
        sys.executable, "scripts/inference_demo.py",
        "--model_path", str(local_dir),
        "--test_dataset",
        "--max_new_tokens", "64",
    ]
    val_result = subprocess.run(val_cmd)
    if val_result.returncode != 0:
        print("\n⚠️  Phase 1 validation had errors. Review output above.")
        ans = input("Continue to Phase 2 anyway? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted. Fix the checkpoint and try again.")
            sys.exit(1)
    else:
        print("✅ Phase 1 validation passed!\n")

    # ── Step 4: Pre-cache all Phase 2 datasets ────────────────────────────
    if not args.skip_datasets:
        print("=" * 60)
        print("[Step 4] Pre-caching Phase 2 datasets...")
        precache_datasets(token)
        print("\n✅ All datasets cached.\n")
    else:
        print("[Step 4] Skipping dataset pre-cache (--skip_datasets)\n")

    # ── Step 5: Run Phase 2 ───────────────────────────────────────────────
    print("=" * 60)
    print(f"[Step 5] Starting Phase 2 training...")
    cmd = [
        sys.executable, "-m", "training.phase2_omni_finetune",
        "--config", args.phase2_config,
        "--phase1_checkpoint", str(local_dir),
    ]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
