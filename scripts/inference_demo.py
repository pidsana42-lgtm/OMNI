"""
scripts/inference_demo.py
--------------------------
Quick demo script to test the trained Thai-Omni-Modal model.

Usage:
    # Audio input
    python scripts/inference_demo.py \
        --model_path outputs/phase2_final \
        --audio_file sample_thai_speech.wav

    # Image + text input
    python scripts/inference_demo.py \
        --model_path outputs/phase2_final \
        --image_file sample_document.png \
        --prompt "อธิบายเนื้อหาในเอกสารนี้"

    # Text only
    python scripts/inference_demo.py \
        --model_path outputs/phase2_final \
        --prompt "อธิบายวิธีการทำต้มยำกุ้ง"
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniModalModel, OmniProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Thai Omni-Modal Inference Demo")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--audio_file", default=None, help="Path to .wav audio file")
    parser.add_argument("--image_file", default=None, help="Path to image file")
    parser.add_argument("--prompt", default="ถอดเสียงต่อไปนี้:", help="Text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def run_inference(args):
    print(f"[Demo] Loading model from {args.model_path}...")
    model = OmniModalModel.from_pretrained(args.model_path)
    processor = OmniProcessor.from_pretrained(
        llm_name=args.model_path,
        audio_encoder_name=model.config.audio_encoder_name,
    )
    model = model.to(args.device, dtype=torch.bfloat16)
    model.eval()
    print(f"[Demo] Model loaded on {args.device} ✅")

    # ── Determine modality ─────────────────────────────────────────────────
    audio_features = None
    pixel_values = None
    image_grid_thw = None

    if args.audio_file:
        print(f"[Demo] Loading audio: {args.audio_file}")
        waveform, sr = sf.read(args.audio_file)
        waveform = waveform.astype(np.float32)
        audio_batch = processor.process_audio(waveform, sr)
        audio_features = audio_batch["input_features"].to(args.device, dtype=torch.bfloat16)

        # Build instruction prompt
        messages = processor.build_audio_instruction(
            transcript="",    # Empty — model will generate
            system_prompt="คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการถอดเสียง",
        )
        # Remove the empty assistant turn so model generates freely
        messages = messages[:2]

        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,   # ← Critical
        )

    elif args.image_file:
        print(f"[Demo] Loading image: {args.image_file}")
        image = Image.open(args.image_file).convert("RGB")
        messages = [
            {"role": "system", "content": "คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการวิเคราะห์ภาพ"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if "pixel_values" in encoded:
            pixel_values = encoded["pixel_values"].to(args.device, dtype=torch.bfloat16)
        if "image_grid_thw" in encoded:
            image_grid_thw = encoded["image_grid_thw"].to(args.device)

    else:
        # Text only
        messages = [
            {"role": "system", "content": "คุณเป็น AI ผู้ช่วยภาษาไทยที่เป็นประโยชน์"},
            {"role": "user", "content": args.prompt},
        ]
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # Move inputs to device
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    # ── Generate ───────────────────────────────────────────────────────────
    print("[Demo] Generating response...")
    print("=" * 60)

    with torch.autocast(args.device, dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_features=audio_features,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
        )

    # For demonstration: use model.llm.generate for actual text generation
    # (OmniModal forward gives loss; use .generate for inference)
    with torch.autocast(args.device, dtype=torch.bfloat16):
        generated = model.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.5,
            pad_token_id=processor.pad_token_id,
            eos_token_id=processor.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = generated[0][input_ids.shape[1]:]
    response = processor.decode(new_tokens)
    print(response)
    print("=" * 60)
    return response


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
