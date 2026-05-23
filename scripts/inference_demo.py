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
    parser.add_argument("--test_dataset", action="store_true", help="Pull a sample from the dataset to test")
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
    ground_truth = None

    # Auto-trigger test_dataset if no inputs are provided
    use_dataset = args.test_dataset or (not args.audio_file and not args.image_file and args.prompt == "ถอดเสียงต่อไปนี้:")

    if use_dataset:
        print("[Demo] Loading 1 sample from 'google/fleurs' (th_th) [split: validation]...")
        from datasets import load_dataset, Audio as HFAudio
        import io
        import soundfile as sf
        
        ds = load_dataset("google/fleurs", "th_th", split="validation", streaming=True)
        ds = ds.cast_column("audio", HFAudio(decode=False))
        sample = next(iter(ds))

        audio_dict = sample.get("audio", {})
        audio_bytes = audio_dict.get("bytes")
        if audio_bytes is not None:
            waveform, sr = sf.read(io.BytesIO(audio_bytes))
            waveform = waveform.astype(np.float32)
        else:
            raise ValueError("Audio bytes not found in dataset sample.")

        audio_batch = processor.process_audio(waveform, sr)
        audio_features = audio_batch["input_features"].to(args.device, dtype=torch.bfloat16)

        ground_truth = sample.get("transcription", "")
        print(f"[Demo] Ground Truth Transcription: {ground_truth}")

        # Build instruction prompt
        messages = processor.build_audio_instruction(
            transcript="",
            system_prompt="คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการถอดเสียง",
        )
        messages = messages[:2]
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    elif args.audio_file:
        print(f"[Demo] Loading audio: {args.audio_file}")
        waveform, sr = sf.read(args.audio_file)
        waveform = waveform.astype(np.float32)
        audio_batch = processor.process_audio(waveform, sr)
        audio_features = audio_batch["input_features"].to(args.device, dtype=torch.bfloat16)

        messages = processor.build_audio_instruction(
            transcript="",
            system_prompt="คุณเป็น AI ผู้ช่วยภาษาไทยที่เชี่ยวชาญด้านการถอดเสียง",
        )
        messages = messages[:2]
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
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

    # Retrieve bad words ids to prevent thinking loops if model is not fully SFT aligned yet
    tokenizer = processor.tokenizer
    think_end_tokens = tokenizer.convert_tokens_to_ids(["</think>", "<|/think|>"])
    think_end_tokens = [t for t in think_end_tokens if t is not None and t != tokenizer.unk_token_id]
    stop_ids = [processor.eos_token_id] + think_end_tokens

    # We DO NOT use suppress_tokens here.
    # If we block the <think> token, reasoning models will try to spell it out using raw text tokens
    # (e.g. " here" + "think" + ">") which causes infinite loops.
    # Instead, we let it output the <think> block naturally, and then strip it using regex later.
    suppress_tokens = None

    # Use model.llm.generate for text generation
    with torch.autocast(args.device, dtype=torch.bfloat16):
        if audio_features is not None:
            # For audio, we MUST construct inputs_embeds because Qwen's native generate doesn't know about audio
            audio_embeds = model._encode_audio(audio_features)
            embed_fn = model._get_embed_tokens()
            text_embeds = embed_fn(input_ids)
            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            # Extend attention mask
            B, T_audio, _ = audio_embeds.shape
            audio_mask = torch.ones(B, T_audio, dtype=attention_mask.dtype, device=attention_mask.device)
            extended_attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

            generated = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.pad_token_id,
                eos_token_id=stop_ids,
                suppress_tokens=suppress_tokens if suppress_tokens else None,
            )
            new_tokens = generated[0]
        else:
            generated = model.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.pad_token_id,
                eos_token_id=stop_ids,
                suppress_tokens=suppress_tokens if suppress_tokens else None,
            )
            new_tokens = generated[0][input_ids.shape[1]:]

    import re
    def strip_thinking(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("</think>", "").replace("<think>", "")
        return text.strip()

    response = strip_thinking(processor.decode(new_tokens))
    print(f"Prediction: {response}")
    print("=" * 60)

    if ground_truth:
        print(f"Ground Truth: {ground_truth}")
        print("=" * 60)

    return response


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
