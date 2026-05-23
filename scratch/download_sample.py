import urllib.request
import librosa
import soundfile as sf
import os

def main():
    print("Downloading Thai audio sample from Wikimedia Commons...")
    url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Th-%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%97%E0%B8%A8%E0%B9%84%E0%B8%composite.ogg"
    # Let's use a simpler URL to avoid encoding issues
    url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Th-%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%97%E0%B8%A8%E0%B9%84%E0%B8%97%E0%B8%A2.ogg"
    output_ogg = "sample_thai.ogg"
    output_wav = "sample_thai.wav"
    
    try:
        urllib.request.urlretrieve(url, output_ogg)
        print(f"Downloaded to {output_ogg}")
        y, sr = librosa.load(output_ogg, sr=16000)
        print(f"Loaded audio: shape {y.shape}, sr {sr}")
        sf.write(output_wav, y, sr)
        print(f"Successfully saved to {output_wav}")
        # Clean up ogg
        if os.path.exists(output_ogg):
            os.remove(output_ogg)
    except Exception as e:
        print(f"Error downloading/processing: {e}")

if __name__ == "__main__":
    main()
