"""
start_phase2.py — Pull Phase 1 checkpoint from HF Hub then start Phase 2.

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
    parser.add_argument("--hub_model_id", default="thai-omni-modal-0.8b-phase1",
                        help="HF Hub repo ID for Phase 1 checkpoint")
    parser.add_argument("--local_dir", default="outputs/phase1/hf_checkpoint",
                        help="Local directory to save the pulled checkpoint")
    parser.add_argument("--phase2_config", default="configs/phase2_finetune.yaml")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if checkpoint already exists locally")
    return parser.parse_args()


def main():
    args = parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF token provided. Use --hf_token or set HF_TOKEN env var.")
        sys.exit(1)

    # ── Step 1: Login ─────────────────────────────────────────────────────
    print(f"[Setup] Logging into HuggingFace Hub...")
    from huggingface_hub import login, snapshot_download
    login(token=token)
    print("✅ Logged in.")

    # ── Step 2: Pull Phase 1 checkpoint from HF Hub ───────────────────────
    local_dir = Path(args.local_dir)

    if args.skip_download and local_dir.exists() and any(local_dir.iterdir()):
        print(f"[Setup] Skipping download — checkpoint already at {local_dir}")
    else:
        print(f"[Setup] Downloading Phase 1 checkpoint: {args.hub_model_id} → {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.hub_model_id,
            repo_type="model",
            local_dir=str(local_dir),
            token=token,
        )
        print(f"✅ Phase 1 checkpoint downloaded to {local_dir}")

    # ── Step 3: Run Phase 2 ───────────────────────────────────────────────
    print(f"\n[Setup] Starting Phase 2 training...")
    cmd = [
        sys.executable, "-m", "training.phase2_omni_finetune",
        "--config", args.phase2_config,
        "--phase1_checkpoint", str(local_dir),
    ]
    print(f"[Setup] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
