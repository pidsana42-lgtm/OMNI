# Thai Omni-Modal AI 🇹🇭

**Thai-Omni-Modal-0.8B** — โมเดล AI ไทยที่รับเข้าได้ทั้ง **เสียง + ภาพ + ข้อความ** และสามารถตอบสนองกลับแบบ Native ได้ทั้ง **ข้อความ + เสียงพูดตอบกลับภาษาไทย (Speech-out)** ในตัวเดียว

---

## 🏛️ สถาปัตยกรรม (Architecture & Options)

โมเดลนี้รองรับการทำงาน 2 รูปแบบหลักในการประมวลผลคำสั่ง:

### รูปแบบที่ 1: Dense Architecture (แบบปกติ)
```
เสียง (.wav) ──→ [scb10x/typhoon-whisper-turbo] ──→ [AudioProjector MLP] ──┐
ภาพ  (.jpg) ──────────────────────────────────────────────────────────────┤──→ [Qwen/Qwen3.5-0.8B] ──┬─→ ข้อความภาษาไทย
ข้อความ ──────────────────────────────────────────────────────────────────┘                          └─→ [AudioLMHead] ─→ เสียงสด (Streaming)
```

### รูปแบบที่ 2: Multi-Modal MoE Architecture (แบบผสมผู้เชี่ยวชาญ)
เราเปลี่ยนเลเยอร์ FFN ให้เป็น **Sparse MoE Layer** โดยมี **Top-2 Router** คอยจัดสรรให้โทเค็นวิ่งเข้าไปยัง Experts 4 สายตรงงาน:
```
Token Inputs ─→ [Top-2 Router] ─┬─→ [Expert 1: Audio]        (ถนัดเสียงไทย/ASR)
                                ├─→ [Expert 2: Vision/OCR]   (ถนัดวิเคราะห์ภาพ/OCR)
                                ├─→ [Expert 3: Text/Lang]    (ถนัดพูดคุยทั่วไป/ภาษาไทย)
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
├── dry_run.py                    # ตรวจสอบโครงสร้าง Dense (ไม่ต้องดาวน์โหลดน้ำหนักโมเดลจริง)
├── dry_run_moe.py                # ตรวจสอบโครงสร้าง MoE & Top-2 Routing (ไม่ต้องดาวน์โหลดน้ำหนัก)
│
├── configs/                      # ค่าคอนฟิกสำหรับรันเทรน
│   ├── phase1_alignment.yaml     # Phase 1: คอนฟิกการทำจัดตำแหน่งเสียง (Audio Projector)
│   ├── phase2_finetune.yaml      # Phase 2: คอนฟิก SFT รวม 3 ประสาทสัมผัส
│   ├── phase3_distillation.yaml  # Phase 3: คอนฟิกการทำ Distillation ไปยังโมเดล Streaming
│   └── phase4_audio_output.yaml  # Phase 4: คอนฟิกการเทรนหัวเสียงพูดภาษาไทย [NEW]
│
├── src/                          # โค้ดโครงสร้างหลัก
│   ├── configuration_omni.py     # โครงสร้างคอนฟิก (OmniConfig)
│   ├── modeling_omni.py          # คลาสโมเดลหลัก (OmniModalModel)
│   ├── processing_omni.py        # ตัวจัดการข้อมูลดิบ (OmniProcessor)
│   ├── projector.py              # ตัวเชื่อมโยงเสียง (AudioProjector MLP)
│   ├── moe.py                    # โครงสร้างตัวกระจายงาน MoE (SparseMoELayer)
│   ├── modeling_moe_adapter.py   # ตัวแปลงโมเดล Dense ไปเป็น MoE
│   └── audio_decoder.py          # ตัวจัดการ EnCodec และ AudioLMHead [NEW]
│
├── data/                         # ตัวเตรียมข้อมูลสำหรับเทรน (Data Loader)
│   ├── collator.py               # จัดการขนาด/Padding ของ Mixed batch ให้เหมาะสม
│   ├── dataset_audio.py          # โหลดข้อมูลเสียง (Audio Dataset)
│   ├── dataset_vision.py         # โหลดข้อมูลภาพ / ColPali
│   ├── dataset_agent.py          # โหลดข้อมูลสอน Agent (Tool-use / ReAct)
│   ├── dataset_omni.py           # ตัวผสมข้อมูล SFT (Audio 35%, Text 35%, Vision 30%)
│   └── dataset_audio_output.py   # โหลดคู่ข้อมูลเสียงพูดคำตอบ (Phase 4) [NEW]
│
├── training/                     # สคริปต์ขั้นตอนการเทรน
│   ├── phase1_audio_alignment.py # เทรนเฉพาะ AudioProjector เท่านั้น
│   ├── phase2_omni_finetune.py   # เทรนสมอง LLM + Projector (แช่แข็งเสียง/ภาพ)
│   ├── phase3_distillation.py    # Distill ความรู้ไปยัง Student (Streaming Head)
│   └── phase4_audio_output.py    # เทรน AudioLMHead เพื่อตอบกลับเป็นเสียงพูด [NEW]
│
└── scripts/                      # สคริปต์อำนวยความสะดวก
    ├── setup_tokens.py           # สร้าง Special tokens บน Tokenizer ก่อนเทรน (รันครั้งเดียว)
    ├── inference_demo.py         # รันเดโมทดสอบอินเฟอเรนซ์เสียง/ภาพ/ข้อความ -> ข้อความ
    └── inference_stream.py       # รันเดโมตอบกลับเป็นเสียงพูดสดแบบ Streaming [NEW]
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

### 4. สตาร์ทการเทรนจริง (Phase 1, Phase 2 และ Phase 4)
```bash
# Phase 1: จัดตำแหน่งเสียง
python -m training.phase1_audio_alignment --config configs/phase1_alignment.yaml

# Phase 2 (บน GPU/Session ใหม่): ดึงโมเดล Phase 1 อัตโนมัติและเทรน SFT ทันที
python scripts/start_phase2.py --hf_token "YOUR_HF_TOKEN"

# Phase 4: เทรนการออกเสียงพูดตอบกลับภาษาไทย (Speech Output)
python -m training.phase4_audio_output \
    --config configs/phase4_audio_output.yaml \
    --phase2_checkpoint outputs/phase2/phase2_final
```

### 5. ทดสอบใช้งาน (Inference & Streaming Demo)

**ตอบกลับเป็นข้อความไทย (Text Output):**
```bash
# ทดสอบเสียง + ข้อความ
python scripts/inference_demo.py --model_path outputs/phase2/phase2_final --audio_file sample.wav

# ทดสอบภาพ + ข้อความ
python scripts/inference_demo.py --model_path outputs/phase2/phase2_final --image_file doc.png --prompt "วิเคราะห์เอกสารนี้"
```

**ตอบกลับเป็นเสียงพูดภาษาไทย (Speech Output - Streaming):**
```bash
# ส่งเสียงเข้ามา -> โมเดลส่งเสียงพูดภาษาไทยกลับมาทันที
python scripts/inference_stream.py --model_path outputs/phase4/phase4_final --audio_file sample.wav --stream

# ส่งรูปภาพเข้ามา -> โมเดลอธิบายภาพด้วยเสียงพูดสดทันที
python scripts/inference_stream.py --model_path outputs/phase4/phase4_final --image_file chart.png --prompt "ภาพนี้แสดงอะไร" --stream
```

### 6. อัปโหลดโมเดลขึ้น Hugging Face Hub

รันคำสั่งเพื่ออัปโหลดโมเดลตัวเต็มขึ้นไปที่บัญชี Hugging Face ของคุณได้โดยตรง:
```bash
python scripts/push_to_hub.py \
    --local_path outputs/phase2/phase2_final \
    --repo_name thai-omni-modal-0.8b \
    --username your_username
```
*(หากยังไม่ได้เข้าสู่ระบบของ HF สามารถล็อกอินด้วย Python Script: `python scripts/login_hf.py --token "your_hf_token"` หรือตั้งค่า `export HF_TOKEN="your_token"` ไว้ล่วงหน้าได้ครับ)*

---

## 🛠️ ความเสถียรของสถาปัตยกรรมและท่อข้อมูล (Stability & Fixes)

เราได้เสริมความแข็งแกร่งของโค้ดให้พร้อมสำหรับระบบคลาวด์และ GPU หลายรูปแบบด้วยการแก้ไขจุดวิกฤต:
1. **การเลี่ยงการใช้ `torchcodec`**: ลบการใช้ระบบถอดรหัสอัตโนมัติของ Hugging Face ที่เป็นปัญหาคอขวดบนเครื่องไม่มี FFmpeg โดยการ Cast คอลัมน์เสียงทั้งหมดเป็น plain struct format และถอดรหัสแบบ Manual ผ่าน `soundfile`
2. **การรองรับ Chatbot Spoken Voices**: ตรวจจับดาต้าเซ็ตสนทนาเสียงของ Typhoon อัตโนมัติและจัดคู่ข้อมูลระหว่าง `voice_a` (หรือ `voice_user`) กับประโยคสนทนาอย่างถูกต้องโดยไม่มีปัญหาคอลัมน์ขาดหาย
3. **การหลีกเลี่ยง Dtype Collision ใน MoE**: แปลงประเภทข้อมูลน้ำหนัก Routing ของ Softmax (ที่เป็น Float32 เสมอ) ให้กลับมาตรงกับประเภทข้อมูลหลักของแบบจำลอง (BFloat16) ก่อนที่จะบวกสะสมลงในบล็อก `index_add_` ของสถาปัตยกรรม MoE กั้นความคลาดเคลื่อนของชนิดข้อมูลในช่วงเทรน SFT แบบผสมประสาทสัมผัส
