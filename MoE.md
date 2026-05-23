# Thai Omni-Modal AI — MoE Architecture Spec (MoE.md)

เอกสารนี้ระบุรายละเอียดการอัพเกรดสถาปัตยกรรมโมเดลจากแบบ Dense ไปเป็น **Multi-Modal Mixture-of-Experts (MoE)** โดยเน้นการสร้าง **Agent Expert** ร่วมกับอีก 3 Experts เฉพาะทาง

---

## 🏛️ สถาปัตยกรรม (Architecture Blueprint)

เราทำการเปลี่ยนชั้น FFN (Feed-Forward Network) ในแต่ละเลเยอร์ของ Qwen3.5-0.8B ให้เป็น **Sparse MoE Layer** โดยมี Experts ทั้งหมด 4 ตัว และมี **Top-2 Router** คอยควบคุมการแจกจ่าย Token

```
Token Embeddings (1024-dim)
         │
         ├──→ [Gating Router] ──→ คำนวณความเหมาะสม (Softmax) ──→ เลือก Top-2 Experts
         │
         ├───→ [Expert 1: Audio]        (ถนัดงานเสียงไทย/ASR)
         ├───→ [Expert 2: Vision/OCR]   (ถนัดอ่านลายมือ/วิเคราะห์โครงสร้างเอกสารไทย)
         ├───→ [Expert 3: Text/Lang]    (ถนัดพูดคุยทั่วไป/ไวยากรณ์ไทย) ← ❄️ FROZEN in Phase 2
         └───→ [Expert 4: Agent]        (ถนัดทำ Tool-calling/JSON/Reasoning Loop)
```

### การแบ่งหน้าที่ของ Experts ทั้ง 4:
| Expert | ฟังก์ชันการทำงาน | แหล่งข้อมูลการเทรน (Training Data) |
|---|---|---|
| **Audio** | ตีความสำเนียง คลื่นเสียง ไวยากรณ์คำพูด | `google/fleurs` (th_th) + `typhoon-ai/chatbot-arena-spoken-voices` |
| **Vision/OCR** | สแกนตัวอักษรไทย Layout ตาราง และรูปภาพทั่วไป | `patomp/thai-mscoco-2014-captions` |
| **Text/Lang** | ลื่นไหลด้านภาษาทั่วไป ตอบคำถามเชิงความรู้ | `mlabonne/FineTome-100k` |
| **Agent** | การใช้เครื่องมือ (Tool Use), คิดวิเคราะห์แบบ ReAct, JSON Format | `xverse/agent-sft-data` (แปลไทย) / `tool-bench` |

---

## 🛠️ โครงสร้างไฟล์โค้ด (Implementation Files)

```
omni/
├── src/
│   ├── moe.py                     # ✅ Done — SparseMoELayer & Top-2 Routing + Aux Loss
│   └── modeling_moe_adapter.py    # ✅ Done — convert_mlp_to_moe() แปลง Dense FFN → MoE
├── data/
│   └── dataset_agent.py           # Placeholder — ตัวโหลดข้อมูลฝึก Agent (Tool use/JSON parsing)
└── dry_run_moe.py                 # ✅ 3/3 Passed — สคริปต์รันแห้งทดสอบ MoE routing
```

### โครงสร้างบล็อก `SparseMoELayer` (`src/moe.py`)
```python
class SparseMoELayer(nn.Module):
    def __init__(self, experts, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, hidden_states):
        # 1. Compute gating logits → Softmax
        # 2. Select Top-K experts per token
        # 3. Route tokens, accumulate weighted expert outputs
        # 4. Compute aux_loss (load balancing) for training
        # Key fixes:
        #   - .to(final_output.dtype) ก่อน index_add_() (BFloat16 safety)
        #   - aux_loss stored as self.aux_loss for training loop access
        ...
```

### การแปลง Dense → MoE (`src/modeling_moe_adapter.py`)
```python
def convert_mlp_to_moe(model, num_experts=4, top_k=2):
    """
    สแกนทุก Transformer Layer ของ Qwen3.5-0.8B
    แล้วเปลี่ยน FFN (MLP) ให้เป็น SparseMoELayer
    Expert ตัวแรก = copy จาก original MLP weights (warm-start)
    """
```

---

## 📈 แผนการฝึกสอน (Training Stages)

### เฟส 1: Alignment (จัดตำแหน่งเสียง) ✅ Done
* **เป้าหมาย**: ให้ `AudioProjector` (MLP 4.7M params) เรียนรู้วิธีแมปเสียงเข้าสู่มิติหลัก (1024-dim)
* **จุดที่เทรน**: AudioProjector เท่านั้น
* **จุดที่ฟรีซ**: Whisper + Qwen LLM ทั้งหมด 100%
* **ผลลัพธ์**: สร้างข้อความภาษาไทยจากเสียงได้ (แต่ยังไม่ตรงคำถาม — ต้องการ Phase 2)

### เฟส 2: Router & Expert Co-Training (ฝึกสมอง MoE) 🔄 Ready to Run
* **เป้าหมาย**: ฝึกให้ Router รู้ว่าเมื่อไหร่ควรเบี่ยง Token ส่งไปหาใคร และเทรน Expert ปลายทางให้เก่งงานเฉพาะสาย
* **จุดที่เทรน**: `Gate (Router)` + `Audio Expert` + `Vision Expert` + `Agent Expert` + `AudioProjector`
* **จุดที่ฟรีซ**: `Text/Lang Expert` (Expert 2) เพื่อป้องกันภาษาไทยปกติพังหรือหลงลืมความรู้เดิม + `Whisper Encoder` + `Qwen Vision Encoder`
* **Config**: `configs/phase2_finetune.yaml` — Audio 50% + Text 50% (10,000 samples, batch=2, grad_accum=16)
* **⚠️ สำคัญ**: Tokenizer ต้องโหลดจาก `Qwen/Qwen3.5-0.8B` โดยตรง (ไม่ใช่จาก Phase 1 checkpoint)

### เฟส 4: Native Speech Output Alignment (ฝึกให้ออกเสียงพูดภาษาไทย) ⏳ Pending
* **เป้าหมาย**: ฝึกฝนโมเดลให้ตอบสนองด้วยเสียงสดธรรมชาติผ่าน `AudioLMHead` โดยรับการกระตุ้นจาก Hidden States ชุดสุดท้ายของเลเยอร์ MoE
* **จุดที่เทรน**: `AudioLMHead` เท่านั้น (~5.2M พารามิเตอร์)
* **จุดที่ฟรีซ**: ฟรีซโมเดลหลัก (LLM MoE, Audio Encoder, Projector) ทั้งหมด 100%

---

## ⚠️ Known Issues & Fixes

| ปัญหา | สาเหตุ | สถานะ |
|-------|--------|--------|
| `index_add_()` dtype mismatch | Softmax upcast weights เป็น Float32 | ✅ Fixed — `.to(final_output.dtype)` |
| Phase 1 tokenizer พัง | `audio_processor.save_pretrained()` เขียนทับ Qwen tokenizer | ✅ Fixed — ใช้ `feature_extractor.save_pretrained()` แทน |
| Phase 2 โหลด tokenizer ผิด | ใช้ `llm_name=phase1_checkpoint` ที่มี tokenizer พัง | ✅ Fixed — hardcode `Qwen/Qwen3.5-0.8B` |
