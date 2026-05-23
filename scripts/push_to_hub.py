"""
Push Model to Hugging Face Hub
-------------------------------
Uploads local checkpoints to Hugging Face Hub under the 'Phonsiri' namespace.

Usage:
    python scripts/push_to_hub.py \
        --local_path outputs/phase4/phase4_final \
        --repo_name thai-omni-modal-0.8b \
        --private          # Optional: upload as private repository

Requirements:
    pip install huggingface_hub
    # Make sure you have logged in via: python scripts/login_hf.py --token "..."
    # Or export HF_TOKEN="your_token"
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--local_path",
        required=True,
        help="Path to the local directory containing model weights and configs",
    )
    parser.add_argument(
        "--repo_name",
        default="thai-omni-modal-0.8b",
        help="Name of the repository on Hugging Face Hub",
    )
    parser.add_argument(
        "--username",
        default="Phonsiri",
        help="Hugging Face username/organization",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face write token (optional if logged in via CLI)",
    )
    return parser.parse_args()


def prepare_custom_model_code(local_dir: Path):
    import shutil
    import json

    src_dir = Path(__file__).parent.parent / "src"
    if not src_dir.exists():
        print(f"[HF Hub] Warning: 'src' directory not found at {src_dir}. Skipping code copying.")
        return

    print(f"[HF Hub] Copying custom model code files from {src_dir} to {local_dir}...")
    for py_file in src_dir.glob("*.py"):
        shutil.copy(py_file, local_dir / py_file.name)

    # Update config.json with auto_map
    config_json_path = local_dir / "config.json"
    if config_json_path.exists():
        try:
            with open(config_json_path, "r") as f:
                config_data = json.load(f)

            config_data["auto_map"] = {
                "AutoConfig": "configuration_omni.OmniConfig",
                "AutoModel": "modeling_omni.OmniModalModel",
                "AutoModelForCausalLM": "modeling_omni.OmniModalModel"
            }

            with open(config_json_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print("[HF Hub]   ✅ Injected AutoModel auto_map into config.json")
        except Exception as e:
            print(f"[HF Hub]   ❌ Failed to update config.json: {e}")

    # Update preprocessor_config.json with auto_map
    preprocessor_config_path = local_dir / "preprocessor_config.json"
    if preprocessor_config_path.exists():
        try:
            with open(preprocessor_config_path, "r") as f:
                prep_data = json.load(f)

            prep_data["auto_map"] = {
                "AutoProcessor": "processing_omni.OmniProcessor"
            }

            with open(preprocessor_config_path, "w") as f:
                json.dump(prep_data, f, indent=2)
            print("[HF Hub]   ✅ Injected AutoProcessor auto_map into preprocessor_config.json")
        except Exception as e:
            print(f"[HF Hub]   ❌ Failed to update preprocessor_config.json: {e}")


def main():
    args = parse_args()
    local_dir = Path(args.local_path)

    if not local_dir.exists():
        print(f"[Error] Local path '{local_dir}' does not exist.")
        return

    # Automatically prepare checkpoint with custom architecture code and auto_map configurations
    prepare_custom_model_code(local_dir)

    repo_id = f"{args.username}/{args.repo_name}"
    print(f"[HF Hub] Preparing to push {local_dir} to {repo_id}...")

    # Detect optimizer and scheduler files for cloud training resume
    has_optimizer = any(local_dir.glob("**/optimizer*")) or any(local_dir.glob("**/optimizer*.bin"))
    has_scheduler = any(local_dir.glob("**/scheduler*")) or any(local_dir.glob("**/scheduler*.bin"))
    if has_optimizer or has_scheduler:
        print("[HF Hub] 🔄 Detected active training state (optimizer and/or scheduler files).")
        print("[HF Hub] Pushing full checkpoint including optimizer states to Hugging Face Hub...")
    else:
        print("[HF Hub] 📦 Pushing standard model weights and configurations to Hugging Face Hub...")

    # Initialize Hugging Face API
    api = HfApi(token=args.token)

    # Step 1: Create repository on Hugging Face if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=args.token,
            private=args.private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"[HF Hub] Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"[HF Hub] Warning during repository creation: {e}")
        print("[HF Hub] Proceeding with upload attempt...")

    # Step 2: Upload all files in the directory
    try:
        print(f"[HF Hub] Uploading files from {local_dir}...")
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload model checkpoint from local path: {local_dir.name}",
        )
        print(f"\n[HF Hub] 🎉 Success! Model uploaded to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[HF Hub] ❌ Upload failed: {e}")
        print("[HF Hub] Hint: Ensure you are logged in (python scripts/login_hf.py) or HF_TOKEN is exported.")


if __name__ == "__main__":
    main()
