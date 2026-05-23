# Thai Omni-Modal AI — Project Status & Run Manifest
Last Updated: 2026-05-23 (Dry-Runs 100% Passed)

## 📌 Current Status: Ready for H200 Execution
- **Dense Architecture & Pipeline**: Scaffolded and verified. (`python dry_run.py` -> 6/6 Passed)
- **MoE Architecture & Adapter**: Scaffolded and verified. (`python dry_run_moe.py` -> 3/3 Passed)
- **Speech Output Support**: Added EnCodec wrapper, AudioLMHead, and chunked streaming inference. Ready for Phase 4.
- **Base Models Configured**:
  - LLM Backbone: `Qwen/Qwen3.5-0.8B` (Native Vision support)
  - Audio Encoder: `typhoon-ai/typhoon-whisper-turbo` (Outputs 1280-dim hidden states)
- **Dataset Setup**: Integrated official Typhoon datasets for all phases (ASR, Text, Vision, and TTS Resynthesized).
- **Hugging Face Hub**: Added `scripts/push_to_hub.py` utility for direct model uploads to `Phonsiri` HF account.
- **Cloud Compatibility**: Default attention implementation set to PyTorch SDPA (`attn_implementation="sdpa"`) to support cloud instances without Flash Attention.
- **Anti-Thinking Loop Guard**: Confirmed `enable_thinking=False` is hardcoded globally to prevent infinite loops in the 0.8B model.

---

## 📂 Codebase Directory Map

```
/Users/phonsirithabunsri/Desktop/omni/
├── STATUS.md                     # <-- รายงานสถานะโปรเจกต์ ลำดับการรันด่วน (อ่านที่แรก)
├── MoE.md                        # แผนภาพและรายละเอียดสถาปัตยกรรม MoE 
├── dry_run.py                    # สคริปต์ตรวจสอบการรันของรุ่นปกติ (Dense)
├── dry_run_moe.py                # สคริปต์ตรวจสอบการรันของรุ่นผสมผู้เชี่ยวชาญ (MoE)
├── requirements.txt              # แพ็คเกจที่ต้องติดตั้ง
├── README.md                     # เอกสารคำแนะนำคู่มือการใช้งานตัวเต็ม
│
├── configs/                      # ค่าคอนฟิกในการเทรน
│   ├── phase1_alignment.yaml     # คอนฟิกการแปลงประสาทสัมผัสเสียง
│   ├── phase2_finetune.yaml      # คอนฟิก SFT รวม 3 Modalities (Audio, Vision, Text)
│   ├── phase3_distillation.yaml  # คอนฟิกการ Distill ไปยัง Student Streaming Head
│   └── phase4_audio_output.yaml  # คอนฟิกการเทรนเสียงพูดตอบกลับ Native [NEW]
│
├── src/                          # โค้ดโครงสร้างหลัก
│   ├── configuration_omni.py     # โครงสร้างคอนฟิก (OmniConfig)
│   ├── modeling_omni.py          # คลาสโมเดลหลัก (OmniModalModel)
│   ├── processing_omni.py        # ตัวเตรียมข้อมูลดิบ (OmniProcessor)
│   ├── projector.py              # ตัวเชื่อมโยงเสียง (AudioProjector MLP)
│   ├── moe.py                    # [MoE] โครงสร้างตัวกระจายงาน MoE (SparseMoELayer)
│   ├── modeling_moe_adapter.py   # [MoE] ตัวแปลงโมเดล Dense ไปเป็น MoE
│   └── audio_decoder.py          # [Speech Output] ตัวจัดการ EnCodec และ AudioLMHead [NEW]
│
├── data/                         # ตัวเตรียมข้อมูลสำหรับเทรน (Data Loader)
│   ├── collator.py               # จัดการขนาด/Padding ของ Mixed batch ให้เหมาะสม
│   ├── dataset_audio.py          # โหลดข้อมูลเสียง (Audio Dataset)
│   ├── dataset_vision.py         # โหลดข้อมูลภาพ / ColPali
│   ├── dataset_agent.py          # [MoE] โหลดข้อมูลสอน Agent (Tool-use / ReAct)
│   ├── dataset_omni.py           # ตัวผสมข้อมูล SFT (Audio 35%, Text 35%, Vision 30%)
│   └── dataset_audio_output.py   # โหลดข้อมูลคู่เสียงพูดคำตอบเป้าหมาย (Phase 4) [NEW]
│
├── training/                     # สคริปต์ขั้นตอนการเทรน
│   ├── phase1_audio_alignment.py # เทรนเฉพาะ AudioProjector เท่านั้น
│   ├── phase2_omni_finetune.py   # เทรนสมอง LLM + Projector (แช่แข็งเสียง/ภาพ)
│   ├── phase3_distillation.py    # Distill ความรู้ไปยัง Student (Streaming Head)
│   └── phase4_audio_output.py    # เทรน AudioLMHead เพื่อตอบกลับเป็นเสียงพูด [NEW]
│
└── scripts/                      # สคริปต์อำนวยความสะดวก
    ├── setup_tokens.py           # สร้าง Special tokens บน Tokenizer ก่อนเทรน (รันครั้งเดียว)
    ├── inference_demo.py         # รันเดโมทดสอบอินเฟอเรนซ์เสียง/ภาพ/ข้อความ
    ├── inference_stream.py       # รันเดโมตอบกลับเป็นเสียงพูดแบบ Streaming เรียลไทม์ [NEW]
    └── push_to_hub.py            # พุชโมเดลขึ้น Hugging Face Hub (User: Phonsiri) [NEW]
```

---

## 🚀 Execution Guide (Next Steps)

รันคำสั่งเหล่านี้เมื่อนำไปใช้งานบนเครื่อง GPU H200:

### Step 0: Ensure correct PYTHONPATH
รันคำสั่งนี้ที่โฟลเดอร์หลักของโปรเจกต์ก่อนเริ่มรัน Python เสมอ
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install "transformers>=4.52"  # บังคับอัพเกรดเพื่อให้สามารถโหลด Qwen3.5-0.8B ได้
pip install encodec sounddevice   # พิเศษสำหรับการตอบกลับเป็นเสียงพูด
```

### Step 2: Initialize Special Tokens (Run Once)
```bash
python scripts/setup_tokens.py
```
*เก็บข้อมูลโทเค็นและโมเดลที่เพิ่มแล้วไว้ใน `outputs/tokenizer_setup/`*

### Step 3: Run Phase 1 — Audio Alignment
```bash
python -m training.phase1_audio_alignment --config configs/phase1_alignment.yaml
```
*เก็บผลลัพธ์ไว้ที่ `outputs/phase1/best_phase1/`*

### Step 4: Run Phase 2 — Omni Fine-Tuning (SFT)
```bash
python -m training.phase2_omni_finetune \
    --config configs/phase2_finetune.yaml \
    --phase1_checkpoint outputs/phase1/best_phase1
```
*เก็บผลลัพธ์โมเดลแบบปกติไว้ที่ `outputs/phase2/phase2_final/`*

### Step 5: Run Phase 4 — Speech Output Alignment SFT [NEW]
เทรนหัวเสียงพูดภาษาไทย เพื่อให้โมเดลตอบกลับเป็นเสียงสดเกือบเรียลไทม์ได้
```bash
python -m training.phase4_audio_output \
    --config configs/phase4_audio_output.yaml \
    --phase2_checkpoint outputs/phase2/phase2_final
```
*เก็บผลลัพธ์โมเดลแบบพูดได้ไว้ที่ `outputs/phase4/phase4_final/`*

และทดสอบอินเฟอเรนซ์เสียงพูดตอบกลับแบบเล่นสด (Real-time Stream):
```bash
python scripts/inference_stream.py \
    --model_path outputs/phase4/phase4_final \
    --audio_file sample.wav \
    --stream
```

### Step 6: Push Model to Hugging Face Hub [NEW]
เมื่อประมวลผลการเทรนเสร็จสิ้นเรียบร้อย สามารถอัปโหลดโมเดลขึ้น Hugging Face Hub ในโปรไฟล์ `Phonsiri` ได้ทันที:
```bash
python scripts/push_to_hub.py \
    --local_path outputs/phase4/phase4_final \
    --repo_name thai-omni-modal-0.8b \
    --username Phonsiri
```

---


## 🦄 MoE Upgrade (Optional)
หากต้องการรันฝึกสอนในสถาปัตยกรรม MoE (Mixture of Experts) ที่มี **Agent Expert**:

1. **สลับโมเดลเป็น MoE ในโค้ดเทรน**:
เปิดใช้งานตัวแปลงในสคริปต์ SFT ก่อนรันการเทรน (ดูรายละเอียดสัดส่วนโมเดลได้ใน [MoE.md](file:///Users/phonsirithabunsri/Desktop/omni/MoE.md))
```python
from src.modeling_moe_adapter import convert_mlp_to_moe
model = OmniModalModel.from_pretrained(phase1_path)
model = convert_mlp_to_moe(model, num_experts=4, top_k=2) # 4 Experts, Top-2 Routing
```
2. **รันตรวจสอบโครงสร้าง MoE**:
```bash
python dry_run_moe.py
```
