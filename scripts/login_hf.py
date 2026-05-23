"""
Login to Hugging Face Hub Programmatically
-----------------------------------------
Logs in using python script instead of Hugging Face CLI.
Saves credentials locally in ~/.cache/huggingface/token for downstream access.

Usage:
    python scripts/login_hf.py --token "your_hf_token_here"
"""

import argparse
import sys
from huggingface_hub import login


def main():
    parser = argparse.ArgumentParser(description="Login to Hugging Face Hub")
    parser.add_argument(
        "--token",
        required=True,
        help="Hugging Face write token",
    )
    args = parser.parse_args()

    try:
        print("[HF Login] Logging in using token...")
        login(token=args.token, add_to_git_credential=False)
        print("  ✅ Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"  ❌ Login failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
