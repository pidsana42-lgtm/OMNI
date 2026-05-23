import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import OmniConfig, OmniProcessor
from data.dataset_audio import AudioTextDataset

def main():
    print("Initializing OmniProcessor...")
    processor = OmniProcessor.from_pretrained(
        llm_name="Qwen/Qwen3.5-0.8B",
        audio_encoder_name="typhoon-ai/typhoon-whisper-turbo",
    )

    print("Initializing AudioTextDataset...")
    dataset = AudioTextDataset(
        processor=processor,
        hf_dataset_name="typhoon-ai/typhoon-audio-preview-data",
        hf_split="pretrained",
        max_audio_seconds=30.0,
    )

    print(f"Total samples: {len(dataset)}")
    print("Checking first 5 samples...")

    for i in range(5):
        item = dataset.data[i]
        path_val = item.get("path")
        print(f"\n--- Sample {i} ---")
        print(f"Path: {path_val}")
        
        # Manually load using dataset's logic
        waveform = None
        sr = 16000
        
        if path_val:
            p = Path(path_val)
            print(f"File exists on disk: {p.exists()}")
            if p.exists():
                print(f"File size: {p.stat().st_size} bytes")
                try:
                    import soundfile as sf
                    waveform, sr = sf.read(str(p))
                    print(f"Successfully loaded with soundfile. Shape: {waveform.shape}, SR: {sr}")
                    print(f"Is all zeros: {np.all(waveform == 0)}")
                except Exception as e:
                    print(f"soundfile failed: {e}")
                    
                    try:
                        import librosa
                        waveform, sr = librosa.load(str(p), sr=16000)
                        print(f"Successfully loaded with librosa. Shape: {waveform.shape}")
                        print(f"Is all zeros: {np.all(waveform == 0)}")
                    except Exception as e2:
                        print(f"librosa failed: {e2}")

if __name__ == "__main__":
    main()
