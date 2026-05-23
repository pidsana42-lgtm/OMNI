from datasets import load_dataset
try:
    ds = load_dataset("typhoon-ai/tts_arena_resynthesized", split="train", streaming=True, trust_remote_code=True)
    sample = next(iter(ds))
    print("Keys:", sample.keys())
    if "audio" in sample:
        print("Audio keys:", sample["audio"].keys())
except Exception as e:
    print("Error:", e)
