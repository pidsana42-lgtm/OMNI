## 📌 Current Status: Transitioning to Phase 2 SFT (New GPU Session)

### ✅ Completed
- **Phase 1 Audio Alignment**: Completed/Nearing completion on the cloud. Trains on real Thai speech using `google/fleurs` (th_th) to align speech features with the LLM. Checkpoint pushed/saved to HF (`thai-omni-modal-0.8b-phase1`).
- **Dense Architecture & Pipeline**: Verified. (`python dry_run.py` → 6/6 Passed)
- **MoE Architecture & Adapter**: Verified. (`python dry_run_moe.py` → 3/3 Passed)
- **Token Setup**: Special audio tokens added to tokenizer (`outputs/tokenizer_setup/`)
- **HF Hub Sync**: Model auto-syncs every `save_steps` during training runs.
- **Multi-modality Fixes**: Avoided `torchcodec` dependency issues by implementing manual audio bytes decoding via `soundfile/io.BytesIO` and setting `Audio(decode=False)` in dataset mapping.

### 🔄 In Progress / Starting Next
- **Phase 2 SFT (Interleaved Audio + Vision + Text)**: Ready to run on a new GPU/session.
  - Setup script `scripts/start_phase2.py` fully configured.
  - Datasets updated to stable public endpoints:
    - **Audio**: `google/fleurs` th_th (voice data cached with manual decoding) or candidate `typhoon-ai/chatbot-arena-spoken-voices` (real spoken arena dataset).
    - **Text**: `mlabonne/FineTome-100k` (high quality conversations, bypasses permission locks).
    - **Vision**: `liuhaotian/LLaVA-Instruct-150K` (public multi-modal instructions).

### ⏳ Pending
- **Phase 3**: Distillation → Streaming Head
- **Phase 4**: Native Speech Output (AudioLMHead)
- **MoE Conversion**: Dense → 4-Expert MoE (Phase 2 SFT)

---

## 🐛 Critical Bugs Found & Fixed

| Bug | ไฟล์ | สถานะ |
|-----|------|--------|
| `ImportError: To support decoding audio data, please install 'torchcodec'` | `data/dataset_audio.py` | ✅ Fixed (Use `HFAudio(decode=False)` and decode using `soundfile/BytesIO`) |
| `local_audio_dir` ไม่ถูก save เป็น `self.local_audio_dir` → NameError | `data/dataset_audio.py` | ✅ Fixed |
| `_get_embed_tokens()` หา embed_tokens ใน LLM ไม่เจอ | `src/modeling_omni.py` | ✅ Fixed |
| `OmniConfig` ไม่มี `enable_audio_output` attribute | `src/configuration_omni.py` | ✅ Fixed |
| `Qwen2Tokenizer` ไม่มี `additional_special_tokens` attribute | `src/processing_omni.py` | ✅ Fixed |
| Dataset `typhoon-audio-preview-data` มีแค่ path strings ไม่มีเสียงจริง | `configs/phase1_alignment.yaml` | ✅ Fixed (switched to FLEURS / Spoken-Voices) |
| CUDA OOM ที่ batch_size=8 | `configs/phase1_alignment.yaml` | ✅ Fixed (batch=4, grad_accum=8) |

---

## 📂 Codebase Directory Map

```
omni/
├── STATUS.md                     # <-- รายงานสถานะโปรเจกต์ (อ่านที่แรก)
├── MoE.md                        # แผนภาพและรายละเอียดสถาปัตยกรรม MoE
├── dry_run.py                    # ตรวจสอบ Dense architecture (6/6 Passed)
├── dry_run_moe.py                # ตรวจสอบ MoE architecture (3/3 Passed)
├── requirements.txt              # แพ็คเกจที่ต้องติดตั้ง
│
├── configs/
│   ├── phase1_alignment.yaml     # FLEURS th_th, save_steps=100
│   ├── phase2_finetune.yaml      # ✅ ใช้งานอยู่ — FLEURS th_th + FineTome + LLaVA
│   ├── phase3_distillation.yaml  # Distill → Streaming Head
│   └── phase4_audio_output.yaml  # Native Speech Output (AudioLMHead)
│
├── src/
│   ├── configuration_omni.py     # OmniConfig
│   ├── modeling_omni.py          # OmniModalModel (forward, embed_tokens fix)
│   ├── processing_omni.py        # OmniProcessor (audio tokenizer)
│   ├── projector.py              # AudioProjector MLP (4.7M params, trainable Phase 1)
│   ├── moe.py                    # SparseMoELayer (Top-2 routing)
│   ├── modeling_moe_adapter.py   # Dense → MoE converter
│   └── audio_decoder.py          # EnCodec + AudioLMHead (Phase 4)
│
├── data/
│   ├── collator.py               # OmniDataCollator (mixed-modality padding)
│   ├── dataset_audio.py          # AudioTextDataset (fixed decode=False, no torchcodec ✅)
│   ├── dataset_vision.py         # Vision / LLaVA loading
│   ├── dataset_agent.py          # Agent Tool-use
│   ├── dataset_omni.py           # Mixed SFT (Audio 35%, Text 35%, Vision 30%)
│   └── dataset_audio_output.py   # Phase 4 speech pairs
│
├── training/
│   ├── phase1_audio_alignment.py # AudioProjector alignment
│   ├── phase2_omni_finetune.py   # LLM + Projector SFT
│   ├── phase3_distillation.py    # Streaming head distillation
│   └── phase4_audio_output.py    # AudioLMHead training
│
└── scripts/
    ├── setup_tokens.py           # เพิ่ม audio special tokens (รันครั้งเดียว) ✅ Done
    ├── start_phase2.py           # ✅ ใช้งานอยู่ — สคริปต์สปินเซสชันใหม่ ดึงโมเดล+เทส+เทรนต่อ
    ├── download_datasets.py      # Pre-cache HF datasets
    ├── inference_demo.py         # ทดสอบ inference (--test_dataset flag)
    ├── push_to_hub.py            # Push model → HF Hub
    └── login_hf.py               # HF + W&B login
```

---

## 🚀 Quick Start Phase 2 (บน GPU/Session ใหม่)

หากจะสลับไปรันบน GPU เครื่องใหม่ ให้ใช้คำสั่งนี้เพื่อดาวน์โหลด checkpoint ล่าสุดมาทดสอบและรันต่อเนื่องได้ทันที:

```bash
# 1. Clone & install
git clone https://github.com/pidsana42-lgtm/OMNI.git && cd OMNI
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. รันสคริปต์ดึง Phase 1 + ตรวจสอบ + เทรนต่อ
python scripts/start_phase2.py --hf_token "YOUR_HF_TOKEN"
```


**Expected Phase 1 behavior:**
- Loss เริ่มต้นประมาณ ~5.5 (random), ลดลงต่อเนื่อง → ดีกว่า 3.0 คือ alignment เริ่มทำงาน
- Speed: ประมาณ 1.5–2 it/s บน H200
- Checkpoints: `outputs/phase1/checkpoint-100/`, `checkpoint-200/`, ...

---

## 🦄 MoE Upgrade Path

รายละเอียดเต็มอยู่ใน [MoE.md](./MoE.md)

```python
# เปิดใช้หลัง Phase 1 เสร็จ (ก่อน Phase 2 SFT)
from src.modeling_moe_adapter import convert_mlp_to_moe
model = OmniModalModel.from_pretrained("outputs/phase1/best_phase1")
model = convert_mlp_to_moe(model, num_experts=4, top_k=2)
```

---

## 📊 Model Parameters

| Component | Parameters | Trainable Phase 1 |
|-----------|-----------|-------------------|
| Qwen3.5-0.8B LLM | ~890M | ❌ Frozen |
| typhoon-whisper-turbo (Audio Encoder) | ~600M | ❌ Frozen |
| AudioProjector MLP (1280→2048→1024) | **4.7M** | ✅ Training |
| **Total Trainable** | **4.7M / 1,494M** | **0.32%** |
