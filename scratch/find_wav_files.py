import os
from pathlib import Path

def find_wav_files():
    search_paths = ["/teamspace", "/home", "/workspace2"]
    found_any = False
    
    print("Searching for any .wav files in common directories...")
    for p in search_paths:
        path = Path(p)
        if not path.exists():
            continue
        print(f"Scanning {p}...")
        count = 0
        for root, dirs, files in os.walk(str(path)):
            # Limit depth or count to avoid hanging
            wavs = [f for f in files if f.endswith(".wav") or f.endswith(".mp3")]
            if wavs:
                found_any = True
                print(f"Found {len(wavs)} audio files in: {root}")
                for w in wavs[:5]:
                    print(f"  - {os.path.join(root, w)}")
                count += len(wavs)
                if count > 50:
                    print("  ... truncated after 50 files")
                    break
    if not found_any:
        print("No .wav or .mp3 files found in /teamspace, /home, or /workspace2.")

if __name__ == "__main__":
    find_wav_files()
