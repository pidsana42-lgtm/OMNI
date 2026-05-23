"""
Push Checkpoint 2500 to Hugging Face Hub
----------------------------------------
Loads the accelerator state from step 2500, converts it to standard Hugging Face weights,
bundles the optimizer/scheduler states, and uploads it to the Hugging Face Hub.

Run this command in a new terminal:
    python scripts/push_checkpoint_2500.py
"""

import sys
from pathlib import Path
import shutil
from accelerate import Accelerator

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import OmniConfig, OmniModalModel, OmniProcessor
from training.phase1_audio_alignment import dynamic_push_to_hub

def main():
    checkpoint_dir = project_root / "outputs" / "phase1" / "checkpoint-2500"
    hf_ckpt_dir = project_root / "outputs" / "phase1" / "checkpoint-2500-hf"

    if not checkpoint_dir.exists():
        print(f"[Error] Checkpoint directory not found at {checkpoint_dir}")
        sys.exit(1)

    print(f"[1/4] Initializing model and accelerator...")
    accelerator = Accelerator()
    config = OmniConfig(
        llm_model_name="Qwen/Qwen3.5-0.8B",
        audio_encoder_name="typhoon-ai/typhoon-whisper-turbo",
        projector_hidden_size=2048,
        projector_num_layers=2,
    )
    processor = OmniProcessor.from_pretrained(
        llm_name="Qwen/Qwen3.5-0.8B",
        audio_encoder_name="typhoon-ai/typhoon-whisper-turbo",
    )
    model = OmniModalModel(config)
    model = accelerator.prepare(model)

    print(f"[2/4] Loading state from {checkpoint_dir}...")
    accelerator.load_state(str(checkpoint_dir))

    print(f"[3/4] Exporting to HF format at {hf_ckpt_dir}...")
    hf_ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(str(hf_ckpt_dir))
    processor.save_pretrained(str(hf_ckpt_dir))

    # Copy optimizer and scheduler files
    for file_path in checkpoint_dir.glob("*"):
        dest_file = hf_ckpt_dir / file_path.name
        if not dest_file.exists():
            if file_path.is_dir():
                shutil.copytree(file_path, dest_file)
            else:
                shutil.copy(file_path, dest_file)
    print(f"[Info] Bundled optimizer and scheduler states.")

    print(f"[4/4] Pushing to Hugging Face Hub...")
    try:
        dynamic_push_to_hub(
            local_path=str(hf_ckpt_dir),
            repo_name="thai-omni-modal-0.8b-phase1",
            private=True,
        )
        print("\n🎉 Checkpoint 2500 has been successfully pushed to Hugging Face Hub!")
    except Exception as e:
        print(f"\n❌ Push failed: {e}")

if __name__ == "__main__":
    main()
