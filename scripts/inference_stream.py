"""
Streaming Audio Inference
--------------------------
Demonstrates near-real-time audio-in → speech-out using the
Thai Omni-Modal model with its native AudioLMHead.

Usage:
    python scripts/inference_stream.py \\
        --model_path outputs/phase2/phase2_final \\
        --audio_file input.wav \\
        --output_wav response.wav \\
        --stream          # Play audio chunks in real-time (requires sounddevice)

Pipeline:
    input.wav → [Whisper Encoder] → [AudioProjector] → [Qwen3.5 LLM]
                                                               ↓
                                              [AudioLMHead] + [EnCodec Decoder]
                                                               ↓
                                              response.wav (streaming chunks)

Latency budget on H200 (estimated):
    - Whisper encoding:        ~20ms
    - LLM prefill (512 tok):   ~50ms
    - First 25 audio tokens:   ~30ms
    - EnCodec decode (25 tok): ~5ms
    ─────────────────────────────────
    First audio chunk ready:   ~105ms  ← Near-real-time!
    Chunk duration:            ~333ms (25 tokens at 75 tok/s)
"""

import argparse
import sys
import time
import wave
import struct
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import OmniModalModel, OmniProcessor
from src.audio_decoder import AudioCodec, StreamingAudioGenerator

SAMPLE_RATE = 24_000


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming Audio Inference")
    parser.add_argument("--model_path", required=True, help="Path to trained OmniModal model")
    parser.add_argument("--audio_file", default=None, help="Input WAV file (Thai speech)")
    parser.add_argument("--image_file", default=None, help="Input image file (optional)")
    parser.add_argument("--prompt", default=None, help="Text prompt (alternative to audio)")
    parser.add_argument("--output_wav", default="outputs/response.wav", help="Output WAV file path")
    parser.add_argument("--stream", action="store_true", help="Play audio in real-time via sounddevice")
    parser.add_argument("--chunk_tokens", type=int, default=25, help="Tokens per audio chunk (25=~333ms)")
    parser.add_argument("--max_text_tokens", type=int, default=256, help="Max tokens for text response")
    parser.add_argument("--bandwidth", type=float, default=6.0, help="EnCodec bandwidth kbps (1.5/3/6/12/24)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import librosa
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform.astype(np.float32)


def play_chunk_realtime(chunk: np.ndarray, stream_player):
    """Write a chunk to the sounddevice output stream."""
    if stream_player is not None:
        stream_player.write(chunk)


def save_wav(waveform: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save waveform as WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)
    print(f"[Stream] 💾 Saved to {path}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── Load Model & Processor ────────────────────────────────────────────
    print(f"[Stream] Loading model from {args.model_path}...")
    t0 = time.perf_counter()
    model = OmniModalModel.from_pretrained(args.model_path)
    model = model.to(device).eval()

    processor = OmniProcessor.from_pretrained(
        llm_name=args.model_path,
        audio_encoder_name=model.config.audio_encoder_name,
    )
    print(f"[Stream] Model loaded in {time.perf_counter()-t0:.1f}s")

    # ── Check AudioLMHead is available ────────────────────────────────────
    if not hasattr(model, "audio_lm_head") or model.audio_lm_head is None:
        print("[Stream] ⚠️  Model does not have AudioLMHead yet.")
        print("         Run Phase 4 audio output training first.")
        print("         Falling back to TEXT-ONLY output mode.")
        use_audio_output = False
    else:
        use_audio_output = True

    # ── Load EnCodec ──────────────────────────────────────────────────────
    codec = AudioCodec(bandwidth=args.bandwidth)
    streaming_gen = StreamingAudioGenerator(
        audio_lm_head=model.audio_lm_head if use_audio_output else None,
        codec=codec,
        chunk_tokens=args.chunk_tokens,
    ) if use_audio_output else None

    # ── Prepare input ─────────────────────────────────────────────────────
    from PIL import Image

    encode_kwargs = {}
    pixel_values = None
    image_grid_thw = None

    if args.image_file:
        print(f"[Stream] 🖼️ Loading image: {args.image_file}")
        image = Image.open(args.image_file).convert("RGB")
        prompt_text = args.prompt or "อธิบายภาพนี้ให้ฉันฟังหน่อย"
        
        # Build Qwen VL message structure
        messages = [
            {"role": "system", "content": "คุณเป็น AI ผู้ช่วยภาษาไทยที่สามารถเห็นภาพและสื่อสารด้วยเสียงพูดภาษาไทยที่ชัดเจน"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]
    elif args.audio_file:
        print(f"[Stream] 🎤 Processing audio: {args.audio_file}")
        waveform = load_audio(args.audio_file, target_sr=processor.sample_rate)
        audio_feats = processor.process_audio(waveform, processor.sample_rate)
        encode_kwargs["audio_input_features"] = audio_feats["input_features"].to(device)

        messages = processor.build_audio_instruction(
            transcript="<|audio|>",
            system_prompt="คุณเป็น AI ผู้ช่วยภาษาไทยที่ตอบด้วยเสียงพูดภาษาไทยที่ชัดเจนและเป็นธรรมชาติ",
        )
    elif args.prompt:
        messages = [
            {"role": "system", "content": "คุณเป็น AI ผู้ช่วยภาษาไทย"},
            {"role": "user", "content": args.prompt},
        ]
    else:
        print("[Error] Provide --audio_file, --image_file, or --prompt")
        sys.exit(1)

    # Process template and extract inputs
    encoded = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,     # ← Always off
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Extract vision tensors if they were processed by llm_processor
    if "pixel_values" in encoded:
        pixel_values = encoded["pixel_values"].to(device)
    if "image_grid_thw" in encoded:
        image_grid_thw = encoded["image_grid_thw"].to(device)

    # ── Step 1: Text Response Generation ─────────────────────────────────
    print(f"\n[Stream] 🧠 Generating text response...")
    t1 = time.perf_counter()

    with torch.no_grad():
        # Setup generation kwargs based on input modality
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_text_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": processor.pad_token_id,
            "eos_token_id": processor.eos_token_id,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw

        text_outputs = model.llm.generate(**gen_kwargs)

    text_tokens = text_outputs.sequences[0, input_ids.shape[1]:]
    text_response = processor.tokenizer.decode(text_tokens, skip_special_tokens=True)
    print(f"[Stream] 📝 Text: {text_response}")
    print(f"[Stream] Text generation: {time.perf_counter()-t1:.2f}s")

    if not use_audio_output:
        print("\n[Stream] ✅ Text-only mode complete.")
        return

    # ── Step 2: Stream Audio Token Generation ────────────────────────────
    # Use the final hidden states from the LLM as context for AudioLMHead
    # hidden_states shape: tuple of [B, T, hidden_size] per layer
    # We take the last layer's last position
    last_hidden = text_outputs.hidden_states[-1][-1]  # [B, T, hidden_size]

    print(f"\n[Stream] 🔊 Streaming audio output...")
    print(f"         Chunk size: {args.chunk_tokens} tokens (~{args.chunk_tokens/75*1000:.0f}ms audio)")

    # Setup real-time playback if requested
    stream_player = None
    if args.stream:
        try:
            import sounddevice as sd
            stream_player = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
            )
            stream_player.start()
            print("[Stream] 🎧 Real-time playback started")
        except ImportError:
            print("[Stream] ⚠️  sounddevice not found. Install: pip install sounddevice")

    all_chunks = []
    t2 = time.perf_counter()
    first_chunk_time = None

    for i, audio_chunk in enumerate(streaming_gen.stream(last_hidden, device)):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t2
            print(f"[Stream] ⚡ First audio chunk ready in {first_chunk_time*1000:.0f}ms")

        chunk_duration_ms = len(audio_chunk) / SAMPLE_RATE * 1000
        print(f"[Stream]    Chunk {i+1}: {len(audio_chunk)} samples ({chunk_duration_ms:.0f}ms audio)")

        all_chunks.append(audio_chunk)

        # Real-time playback
        if stream_player is not None:
            play_chunk_realtime(audio_chunk, stream_player)

    if stream_player is not None:
        stream_player.stop()
        stream_player.close()

    total_audio_time = time.perf_counter() - t2
    total_audio_dur = sum(len(c) for c in all_chunks) / SAMPLE_RATE

    print(f"\n[Stream] ✅ Audio generation complete!")
    print(f"         Audio duration: {total_audio_dur:.2f}s")
    print(f"         Generation time: {total_audio_time:.2f}s")
    print(f"         RTF: {total_audio_time/total_audio_dur:.2f}x (< 1.0 = faster than real-time)")

    # Save full waveform
    if all_chunks:
        full_waveform = np.concatenate(all_chunks)
        save_wav(full_waveform, args.output_wav)


if __name__ == "__main__":
    main()
