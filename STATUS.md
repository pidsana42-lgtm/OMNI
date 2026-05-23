## 📌 Current Status: Phase 1 ✅ Done → Phase 2 Ready to Run

### ✅ Completed
- **Phase 1 Audio Alignment**: เสร็จสิ้น บน Google FLEURS (th_th). AudioProjector ถูก Align เข้ากับ Qwen3.5-0.8B แล้ว. Checkpoint บันทึกไว้ที่ HF Hub (`Phonsiri/thai-omni-modal-0.8b-phase1`)
- **Phase 1 Inference Verified**: ทดสอบ Inference ผ่าน — โมเดลสามารถสร้างข้อความภาษาไทยจาก Audio input ได้สำเร็จ (ยังไม่ตรงคำถามเนื่องจากยังไม่ผ่าน Phase 2 SFT)
- **Dense Architecture & Pipeline**: ผ่านการตรวจสอบ (`python dry_run.py` → 6/6 Passed)
- **MoE Architecture & Adapter**: ผ่านการตรวจสอบ (`python dry_run_moe.py` → 3/3 Passed)
- **MoE Aux Loss**: เพิ่ม `_load_balance_loss` ใน `SparseMoELayer` และ integrate เข้า training loop (`0.01 * aux_loss`) สำเร็จ
- **Think Token Bias Fix**: แก้ปัญหา Qwen3.5 มี Bias พ่น `<think>` สำเร็จ
  - ล้างบล็อก `<think>...</think>` ออกจากทุก Label ในระดับ Text ก่อน Tokenize (ใน `processing_omni.py`)
  - ลบ `</think>` ออกจาก `stop_ids` ใน inference (เป็นสาเหตุหลักของ `herethink>` loop)
  - ใช้ `repetition_penalty=1.3` แทนเพื่อตัด Repetition loops
  - ปรับปรุง `strip_thinking()` ให้จับ text-level pattern `\w*think>` ด้วย
- **Tokenizer Overwrite Bug Fix**: แก้ปัญหา `OmniProcessor.save_pretrained()` เขียนทับ Qwen Tokenizer ด้วย Whisper Tokenizer
  - `save_pretrained()` เปลี่ยนจาก `audio_processor.save_pretrained()` เป็น `audio_processor.feature_extractor.save_pretrained()` เท่านั้น
  - Inference & Training scripts บังคับโหลด Tokenizer จาก `Qwen/Qwen3.5-0.8B` โดยตรง เพื่อ bypass tokenizer พังใน Phase 1 checkpoint
- **Duplicate `from_pretrained` Bug Fix**: ลบ method ซ้ำใน `modeling_omni.py` ที่ขาด AudioLMHead loading logic

### 🔄 พร้อมรัน
- **Phase 2 SFT (Omni-Modal)**: โค้ดทุกส่วนพร้อมแล้ว รัน `python scripts/start_phase2.py --hf_token YOUR_TOKEN` ได้เลย
  - Audio: `google/fleurs` th_th + `typhoon-ai/chatbot-arena-spoken-voices`
  - Text: `mlabonne/FineTome-100k`
  - Vision: `patomp/thai-mscoco-2014-captions` (ปิดอยู่ `vision_ratio: 0.00` ใน config)

### ⏳ Pending
- **Phase 3**: Distillation → Streaming Head
- **Phase 4**: Native Speech Output (AudioLMHead + Unit Vocoder)

---

## 🐛 Critical Bugs Fixed (Full History)

| Bug | ไฟล์ | สถานะ |
|-----|------|--------|
| `ImportError: To support decoding audio data, please install 'torchcodec'` | `data/dataset_audio.py` | ✅ Fixed |
| `KeyError: 'response'` on `chatbot-arena-spoken-voices` | `data/dataset_audio.py` | ✅ Fixed |
| `RuntimeError: index_add_(): self (BFloat16) and source (Float)` | `src/moe.py` | ✅ Fixed |
| `local_audio_dir` ไม่ถูก save → NameError | `data/dataset_audio.py` | ✅ Fixed |
| `_get_embed_tokens()` หา embed_tokens ไม่เจอ | `src/modeling_omni.py` | ✅ Fixed |
| `OmniConfig` ไม่มี `enable_audio_output` attribute | `src/configuration_omni.py` | ✅ Fixed |
| MoE ไม่ return `gate_logits` → ไม่สามารถคำนวณ Aux Loss ได้ | `src/moe.py` | ✅ Fixed |
| `/no_think` ใน system prompt ทำให้โมเดลหลอน | `data/dataset_audio.py`, `scripts/inference_demo.py` | ✅ Fixed |
| `</think>` เป็น stop token ทำให้พ่น ` herethink>` วนลูป | `scripts/inference_demo.py` | ✅ Fixed |
| `suppress_tokens` ทำให้ LLM สะกด `<think>` เป็น text แทน | `scripts/inference_demo.py` | ✅ Fixed |
| CUDA OOM ที่ `batch_size=8` | `configs/phase1_alignment.yaml` | ✅ Fixed (batch=4, grad_accum=8) |
| **Whisper Tokenizer เขียนทับ Qwen Tokenizer** ใน `save_pretrained()` → Inference ออกภาษาต่างดาว | `src/processing_omni.py` | ✅ Fixed |
| **Phase 1 Checkpoint มี corrupted tokenizer** → Inference/Phase 2 โหลด tokenizer ผิด | `scripts/inference_demo.py`, `training/phase2_omni_finetune.py` | ✅ Fixed (hardcode `Qwen/Qwen3.5-0.8B`) |
| **Duplicate `from_pretrained()`** ทำให้ Phase 4 AudioLMHead ไม่โหลด | `src/modeling_omni.py` | ✅ Fixed |

---

## 📂 Codebase Directory Map

```
omni/
├── STATUS.md                     # <-- รายงานสถานะโปรเจกต์ (อ่านที่แรก)
├── MoE.md                        # แผนภาพและรายละเอียดสถาปัตยกรรม MoE
├── README.md                     # Quick Start Guide
├── dry_run.py                    # ตรวจสอบ Dense architecture (6/6 Passed)
├── dry_run_moe.py                # ตรวจสอบ MoE architecture (3/3 Passed)
├── requirements.txt
│
├── configs/
│   ├── phase1_alignment.yaml     # FLEURS th_th, batch=4, grad_accum=8
│   ├── phase2_finetune.yaml      # ✅ พร้อมรัน — FLEURS + Typhoon + FineTome (50/50 audio/text)
│   ├── phase3_distillation.yaml
│   └── phase4_audio_output.yaml
│
├── src/
│   ├── configuration_omni.py     # OmniConfig (llm_model, audio_encoder, etc.)
│   ├── modeling_omni.py          # OmniModalModel (forward, freeze/unfreeze, save/load)
│   ├── processing_omni.py        # OmniProcessor (strip_thinking, apply_chat_template, save fix)
│   ├── projector.py              # AudioProjector MLP (4.7M params)
│   ├── moe.py                    # SparseMoELayer (Top-2 routing + Aux Loss ✅)
│   ├── modeling_moe_adapter.py   # Dense → MoE converter
│   └── audio_decoder.py          # AudioCodec (EnCodec) + AudioLMHead (Phase 4)
│
├── data/
│   ├── collator.py               # OmniDataCollator (mixed-modality padding)
│   ├── dataset_audio.py          # AudioTextDataset (soundfile decode, no torchcodec, no /no_think)
│   ├── dataset_vision.py         # VisionTextDataset (Thai MSCOCO captions)
│   ├── dataset_agent.py          # Agent Tool-use dataset
│   ├── dataset_omni.py           # OmniInterleavedDataset (configurable ratio mix)
│   └── dataset_audio_output.py   # Phase 4 speech pairs
│
├── training/
│   ├── phase1_audio_alignment.py # ✅ Done — Projector-only training
│   ├── phase2_omni_finetune.py   # ✅ Ready — MoE SFT with Aux Loss (Qwen tokenizer hardcoded)
│   ├── phase3_distillation.py
│   └── phase4_audio_output.py
│
└── scripts/
    ├── setup_tokens.py           # ✅ Done — audio special tokens
    ├── start_phase2.py           # สปิน session ใหม่ + ดึง Phase 1 + เทรน
    ├── inference_demo.py         # ✅ Fixed — Qwen tokenizer, repetition_penalty, strip_thinking
    ├── inference_stream.py       # Streaming speech output demo
    ├── push_to_hub.py
    ├── push_checkpoint_2500.py
    └── login_hf.py
```

---

## 🚀 Quick Start Phase 2

```bash
# 1. Clone & install
git clone https://github.com/pidsana42-lgtm/OMNI.git && cd OMNI
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. รัน Phase 2 จาก Phase 1 checkpoint (ดาวน์โหลดจาก HF Hub อัตโนมัติ)
python scripts/start_phase2.py --hf_token "YOUR_HF_TOKEN"
```

**Expected Phase 2 logs:**
- Router entropy สูงตอนแรก (ดี — หมายถึง router กำลัง explore)
- `aux_loss` ค่อยๆ ลง → Expert balance ดีขึ้น
- `train/loss` < 2.0 ภายใน epoch แรก = alignment ทำงาน

---

## 📊 Model Parameters

| Component | Parameters | Phase 1 | Phase 2 |
|-----------|-----------|---------|---------|
| Qwen3.5-0.8B LLM | ~890M | ❌ Frozen | ✅ Partially Unfrozen (excl. Expert 2) |
| typhoon-whisper-turbo | ~600M | ❌ Frozen | ❌ Frozen |
| AudioProjector MLP | **4.7M** | ✅ Training | ✅ Training |
| MoE Routers (all layers) | ~1M | ❌ N/A (Dense) | ✅ Training |
| Expert 2 (Text/Lang) | ~varies | ❌ N/A (Dense) | ❌ Frozen (preserve Thai LM) |

---

## ⚠️ Known Limitations (Phase 1)

- **Phase 1 สามารถสร้างข้อความภาษาไทยได้** แต่คำตอบยังไม่ตรงคำถาม เพราะยังไม่ได้ผ่าน Instruction Fine-tuning (Phase 2)
- **Qwen3.5-0.8B (0.8B params)** มีแนวโน้ม Hallucinate สูงเมื่อเจอคำถามกว้างๆ จะดีขึ้นหลัง Phase 2 SFT
- **Phase 1 Checkpoint บน HF Hub** มี tokenizer ที่ถูก Whisper เขียนทับ → ต้องบังคับโหลด `Qwen/Qwen3.5-0.8B` tokenizer เสมอ (แก้ในโค้ดแล้ว)
