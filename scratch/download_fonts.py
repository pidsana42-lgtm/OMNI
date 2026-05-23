import urllib.request
import os
from pathlib import Path

def download_fonts():
    font_dir = Path("assets/fonts")
    font_dir.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "Sarabun-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf",
        "Sarabun-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Bold.ttf",
        "Sarabun-Italic.ttf": "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Italic.ttf"
    }
    
    for name, url in urls.items():
        dest = font_dir / name
        if not dest.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded {name} successfully.")
        else:
            print(f"{name} already exists.")

if __name__ == "__main__":
    download_fonts()
