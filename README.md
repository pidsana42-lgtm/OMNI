# Thai Omni-Modal AI 🇹🇭

**Thai-Omni-Modal-0.8B** — โมเดล AI ไทยที่รับเข้าได้ทั้ง **เสียง + ภาพ + ข้อความ** และสามารถตอบสนองกลับแบบ Native ได้ทั้ง **ข้อความ + เสียงพูดตอบกลับภาษาไทย (Speech-out)** ในตัวเดียว

---

## 🏛️ สถาปัตยกรรม (Architecture & Options)

โมเดลนี้รองรับการทำงาน 2 รูปแบบหลักในการประมวลผลคำสั่ง:

### รูปแบบที่ 1: Dense Architecture (แบบปกติ)
```
เสียง (.wav) ──→ [typhoon-ai/typhoon-whisper-turbo] ──→ [AudioProjector MLP] ──┐
ภาพ  (.jpg) ──────────────────────────────────────────────────────────────────┤──→ [Qwen/Qwen3.5-0.8B] ──┬─→ ข้อความภาษาไทย
ข้อความ ──────────────────────────────────────────────────────────────────────┘                          └─→ [AudioLMHead] ─→ เสียงสด (Streaming)
```

### รูปแบบที่ 2: Multi-Modal MoE Architecture (แบบผสมผู้เชี่ยวชาญ)
เราเปลี่ยนเลเยอร์ FFN ให้เป็น **Sparse MoE Layer** โดยมี **Top-2 Router** คอยจัดสรรให้โทเค็นวิ่งเข้าไปยัง Experts 4 สายตรงงาน:
```
Token Inputs ─→ [Top-2 Router] ─┬─→ [Expert 1: Audio]        (ถนัดเสียงไทย/ASR)
                                ├─→ [Expert 2: Vision/OCR]   (ถนัดวิเคราะห์ภาพ/OCR)
                                ├─→ [Expert 3: Text/Lang]    (ถนัดพูดคุยทั่วไป/ภาษาไทย) ← ❄️ Frozen
                                └─→ [Expert 4: Agent]        (ถนัด Tool-calling/JSON/ReAct)
```

> **Note:** `enable_thinking=False` ถูกปิดไว้ตลอดทุกการรัน (ทั้ง Dense และ MoE) เพื่อป้องกันปัญหา Infinite Thinking Loops ในโมเดลขนาดเล็ก

---

## 📂 โครงสร้างโปรเจกต์ (Project Directory Map)

```
omni/
├── STATUS.md                     # <-- รายงานสถานะโปรเจกต์ ลำดับการรันด่วน (อ่านที่แรก)
├── MoE.md                        # รายละเอียดแผนสถาปัตยกรรม MoE & Agent Expert
├── requirements.txt              # รายการแพ็คเกจที่ต้องติดตั้ง
├── dry_run.py                    # ✅ 6/6 Passed — ตรวจสอบโครงสร้าง Dense
├── dry_run_moe.py                # ✅ 3/3 Passed — ตรวจสอบโครงสร้าง MoE
│
├── configs/                      # ค่าคอนฟิกสำหรับรันเทรน
│   ├── phase1_alignment.yaml     # Phase 1: FLEURS th_th, batch=4, grad_accum=8
│   ├── phase2_finetune.yaml      # Phase 2: FLEURS + Typhoon + FineTome (50/50 audio/text)
│   ├── phase3_distillation.yaml  # Phase 3: Distillation ไปยังโมเดล Streaming
│   └── phase4_audio_output.yaml  # Phase 4: เทรนหัวเสียงพูดภาษาไทย
│
├── src/                          # โค้ดโครงสร้างหลัก
│   ├── configuration_omni.py     # OmniConfig (enable_thinking=False by default)
│   ├── modeling_omni.py          # OmniModalModel (forward, freeze/unfreeze, save/load)
│   ├── processing_omni.py        # OmniProcessor (strip_thinking, chat_template, save fix)
│   ├── projector.py              # AudioProjector MLP (4.7M params)
│   ├── moe.py                    # SparseMoELayer (Top-2 routing + Aux Loss + dtype fix)
│   ├── modeling_moe_adapter.py   # Dense → MoE converter (convert_mlp_to_moe)
│   └── audio_decoder.py          # AudioCodec (EnCodec) + AudioLMHead (Phase 4)
│
├── data/                         # ตัวเตรียมข้อมูลสำหรับเทรน
│   ├── collator.py               # OmniDataCollator (mixed-modality padding)
│   ├── dataset_audio.py          # AudioTextDataset (soundfile, no torchcodec, no /no_think)
│   ├── dataset_vision.py         # VisionTextDataset (Thai MSCOCO captions)
│   ├── dataset_agent.py          # Agent Tool-use dataset
│   ├── dataset_omni.py           # OmniInterleavedDataset (configurable ratio mix)
│   └── dataset_audio_output.py   # Phase 4 speech pairs
│
├── training/                     # สคริปต์ขั้นตอนการเทรน
│   ├── phase1_audio_alignment.py # ✅ Done — Projector-only training
│   ├── phase2_omni_finetune.py   # ✅ Ready — MoE SFT + Aux Loss + Qwen tokenizer fix
│   ├── phase3_distillation.py    # ⏳ Pending
│   └── phase4_audio_output.py    # ⏳ Pending
│
└── scripts/                      # สคริปต์อำนวยความสะดวก
    ├── setup_tokens.py           # ✅ Done — สร้าง special tokens
    ├── start_phase2.py           # ✅ Ready — ดึง Phase 1 checkpoint + validate + เทรน Phase 2
    ├── inference_demo.py         # ✅ Fixed — Qwen tokenizer, repetition_penalty, strip_thinking
    ├── inference_stream.py       # Streaming speech output demo (Phase 4)
    ├── push_to_hub.py            # อัปโหลดโมเดลขึ้น HF Hub
    ├── push_checkpoint_2500.py   # อัปโหลด checkpoint เฉพาะ
    └── login_hf.py               # เข้าสู่ระบบ HF Hub
```

---

## 🚀 ลำดับขั้นตอนการรันโปรเจกต์ (Quick Start)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
pip install "transformers>=4.52"  # รองรับ Qwen3.5-0.8B
pip install encodec sounddevice   # พิเศษสำหรับการตอบกลับเป็นเสียงพูด
```

### 2. รันจำลองเพื่อตรวจความถูกต้อง (Dry Run - ตรวจผลใน 2 วินาที)
```bash
# ตรวจสอบโครงสร้าง Dense
python dry_run.py

# ตรวจสอบโครงสร้าง MoE
python dry_run_moe.py
```

### 3. เตรียมโทเค็นพิเศษ (รันก่อนเทรนครั้งเดียว)
```bash
python scripts/setup_tokens.py
```
*ตัวโมเดลพร้อมใช้งานจะถูกบันทึกไว้ใน `outputs/tokenizer_setup/`*

### 4. สตาร์ทการเทรนจริง
```bash
# Phase 1: จัดตำแหน่งเสียง (✅ เสร็จแล้ว — checkpoint อยู่บน HF Hub)
python -m training.phase1_audio_alignment --config configs/phase1_alignment.yaml

# Phase 2 (บน GPU/Session ใหม่): ดึงโมเดล Phase 1 อัตโนมัติและเทรน SFT ทันที
python scripts/start_phase2.py --hf_token "YOUR_HF_TOKEN"

# Phase 4: เทรนการออกเสียงพูดตอบกลับภาษาไทย (⏳ ยังไม่พร้อม — ต้องรอ Phase 2 เสร็จ)
python -m training.phase4_audio_output \
    --config configs/phase4_audio_output.yaml \
    --phase2_checkpoint outputs/phase2/phase2_final
```

### 5. ทดสอบใช้งาน (Inference & Streaming Demo)

**ตอบกลับเป็นข้อความไทย (Text Output):**
```bash
# ทดสอบเสียง → ข้อความ
python scripts/inference_demo.py --model_path outputs/phase1/phase1_final --audio_file sample.wav

# ทดสอบข้อความ → ข้อความ
python scripts/inference_demo.py --model_path outputs/phase1/phase1_final --prompt "สวัสดีครับ"

# ทดสอบจาก dataset โดยตรง (ไม่ต้องมีไฟล์เสียง)
python scripts/inference_demo.py --model_path outputs/phase1/phase1_final --test_dataset
```

**ตอบกลับเป็นเสียงพูดภาษาไทย (Speech Output - Streaming):**
```bash
# ⏳ ต้องรอ Phase 4 เสร็จก่อน
python scripts/inference_stream.py --model_path outputs/phase4/phase4_final --audio_file sample.wav --stream
```

### 6. อัปโหลดโมเดลขึ้น Hugging Face Hub

```bash
python scripts/push_to_hub.py \
    --local_path outputs/phase2/phase2_final \
    --repo_name thai-omni-modal-0.8b \
    --username Phonsiri
```
*(หากยังไม่ได้เข้าสู่ระบบของ HF สามารถล็อกอินด้วย: `python scripts/login_hf.py --token "your_hf_token"` หรือตั้งค่า `export HF_TOKEN="your_token"` ไว้ล่วงหน้าได้ครับ)*

---

## 🛠️ ความเสถียรของสถาปัตยกรรมและท่อข้อมูล (Stability & Fixes)

เราได้เสริมความแข็งแกร่งของโค้ดให้พร้อมสำหรับระบบคลาวด์และ GPU หลายรูปแบบด้วยการแก้ไขจุดวิกฤต:

1. **การเลี่ยงการใช้ `torchcodec`**: ลบการใช้ระบบถอดรหัสอัตโนมัติของ Hugging Face ที่เป็นปัญหาคอขวดบนเครื่องไม่มี FFmpeg โดยการ Cast คอลัมน์เสียงทั้งหมดเป็น plain struct format และถอดรหัสแบบ Manual ผ่าน `soundfile`
2. **การรองรับ Chatbot Spoken Voices**: ตรวจจับดาต้าเซ็ตสนทนาเสียงของ Typhoon อัตโนมัติและจัดคู่ข้อมูลระหว่าง `voice_a` (หรือ `voice_user`) กับประโยคสนทนาอย่างถูกต้อง
3. **การหลีกเลี่ยง Dtype Collision ใน MoE**: แปลงประเภทข้อมูลน้ำหนัก Routing ของ Softmax (Float32) ให้ตรงกับ BFloat16 ก่อน `index_add_`
4. **การแก้ปัญหา Tokenizer ถูกเขียนทับ**: `OmniProcessor.save_pretrained()` เปลี่ยนจาก `audio_processor.save_pretrained()` (ซึ่งเขียนทับ Qwen Tokenizer ด้วย Whisper Tokenizer) เป็น `audio_processor.feature_extractor.save_pretrained()` เท่านั้น
5. **Hardcoded Qwen Tokenizer**: `inference_demo.py` และ `phase2_omni_finetune.py` บังคับโหลด tokenizer จาก `Qwen/Qwen3.5-0.8B` โดยตรง เพื่อ bypass tokenizer พังใน Phase 1 checkpoint
6. **ลบ `/no_think` ออกจากทุก system prompt**: ป้องกัน Qwen3.5 จากการ hallucinate โดยไม่จำเป็น
7. **แก้ `strip_thinking()`**: จับ text-level pattern `\w*think>` ที่ LLM สร้างขึ้นเมื่อพยายามสะกด `<think>` เป็น BPE tokens

---

## 📊 Model Components

| Component | Parameters | HF Hub |
|-----------|-----------|--------|
| Qwen3.5-0.8B LLM | ~890M | `Qwen/Qwen3.5-0.8B` |
| typhoon-whisper-turbo | ~600M | `typhoon-ai/typhoon-whisper-turbo` |
| AudioProjector MLP | **4.7M** | (saved in checkpoint) |
| Phase 1 Checkpoint | — | `Phonsiri/thai-omni-modal-0.8b-phase1` |
