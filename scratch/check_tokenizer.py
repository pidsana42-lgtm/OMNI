from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("outputs/phase1/hf_checkpoint")
    print("Successfully loaded tokenizer from outputs/phase1/hf_checkpoint")
except Exception as e:
    print("Could not load from outputs/phase1/hf_checkpoint:", e)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    print("Loaded tokenizer from Qwen/Qwen3.5-0.8B instead")

vocab = tokenizer.get_vocab()
for word in ["<think>", "</think>", "<|thought|>", "</|thought|>"]:
    if word in vocab:
        print(f"Direct match: {word} -> ID: {vocab[word]}")
    else:
        print(f"No direct match for {word}")

matching = {k: v for k, v in vocab.items() if "think" in k.lower() or "thought" in k.lower()}
print(f"Total matching tokens: {len(matching)}")
for k, v in sorted(matching.items(), key=lambda item: item[1]):
    print(f"  {k!r} -> ID: {v}")
