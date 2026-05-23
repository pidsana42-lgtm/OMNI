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
         ├───→ [Expert 3: Text/Lang]    (ถนัดพูดคุยทั่วไป/ไวยากรณ์ไทย)
         └───→ [Expert 4: Agent]        (ถนัดทำ Tool-calling/JSON/Reasoning Loop)
```

### การแบ่งหน้าที่ของ Experts ทั้ง 4:
| Expert | ฟังก์ชันการทำงาน | แหล่งข้อมูลการเทรน (Training Data) |
|---|---|---|
| **Audio** | ตีความสำเนียง คลื่นเสียง ไวยากรณ์คำพูด | `google/fleurs` (th_th) หรือ `typhoon-ai/chatbot-arena-spoken-voices` |
| **Vision/OCR** | สแกนตัวอักษรไทย Layout ตาราง และรูปภาพทั่วไป | `liuhaotian/LLaVA-Instruct-150K` / `BidirLM/colpali_train_retrieval` |
| **Text/Lang** | ลื่นไหลด้านภาษาทั่วไป ตอบคำถามเชิงความรู้ | `mlabonne/FineTome-100k` / `pythainlp/thai-instruction-sft` |
| **Agent** | การใช้เครื่องมือ (Tool Use), คิดวิเคราะห์แบบ ReAct, JSON Format | `xverse/agent-sft-data` (แปลไทย) / `tool-bench` |

---

## 🛠️ โครงสร้างไฟล์โค้ดที่จะเพิ่ม (Planned Files)

เมื่อต้องการเริ่มสร้าง ให้เพิ่มไฟล์ตามโครงสร้างนี้ในโปรเจกต์:

```
omni/
├── src/
│   ├── moe.py                     # [NEW] ตัวบล็อก MoE Layer & Top-2 Routing
│   └── modeling_moe_adapter.py    # [NEW] สคริปต์แก้ไข/เปลี่ยน FFN ของ Qwen เป็น MoE
├── data/
│   └── dataset_agent.py           # [NEW] ตัวโหลดข้อมูลฝึก Agent (Tool use/JSON parsing)
└── dry_run_moe.py                 # [NEW] สคริปต์รันแห้งทดสอบการกระจายโทเค็นใน MoE
```

### 1. โครงสร้างบล็อก `MoELayer` (`src/moe.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoELayer(nn.Module):
    def __init__(self, experts, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, orig_shape[-1])
        num_tokens = x.shape[0]

        # 1. Compute gating logits
        gate_logits = self.gate(x)
        
        # 2. Select Top-K experts
        weights = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Route tokens to respective experts
        final_output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            mask = (top_indices == expert_idx)
            if not mask.any():
                continue

            token_indices, k_positions = torch.where(mask)
            expert_inputs = x[token_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)
            expert_weights = top_weights[token_indices, k_positions].unsqueeze(-1)

            # [Dtype Fix]: Softmax upcasts routing weights to float32. 
            # We explicitly cast scaled outputs to match final_output dtype (e.g. BFloat16)
            # to avoid runtime index_add_ errors.
            scaled_outputs = (expert_outputs * expert_weights).to(final_output.dtype)
            final_output.index_add_(0, token_indices, scaled_outputs)

        return final_output.view(orig_shape)
```

---

## 📈 แผนการฝึกสอน (Training Stages)

### เฟส 1: Alignment (จัดตำแหน่งภาพและเสียง)
* **เป้าหมาย**: ให้ตัวแปลงสัญญาณ (`AudioProjector` และ `VisionProjector`) เรียนรู้วิธีแมปข้อมูลเข้าสู่มิติหลัก (1024-dim)
* **จุดที่เทรน**: เทรนแค่ Projectors
* **จุดที่ฟรีซ**: ฟรีซโมเดลหลัก (LLM MoE) ทั้งหมด 100%

### เฟส 2: Router & Expert Co-Training (ฝึกสมอง MoE)
* **เป้าหมาย**: ฝึกให้ Router รู้ว่าเมื่อไหร่ควรเบี่ยง Token ส่งไปหาใคร และเทรน Expert ปลายทางให้เก่งงานเฉพาะสาย
* **จุดที่เทรน**: `Gate (Router)` + `Audio Expert` + `Vision Expert` + `Agent Expert`
* **จุดที่ฟรีซ**: `Text/Lang Expert` (เพื่อป้องกันภาษาไทยปกติพังหรือหลงลืมความรู้เดิม)

### เฟส 4: Native Speech Output Alignment (ฝึกให้ออกเสียงพูดภาษาไทย) [NEW]
* **เป้าหมาย**: ฝึกฝนโมเดลให้ตอบสนองด้วยเสียงสดธรรมชาติผ่าน `AudioLMHead` โดยรับการกระตุ้นจาก Hidden States ชุดสุดท้ายของเลเยอร์ MoE (รองรับการตอบกลับเสียงพูดได้จากทุกประสาทสัมผัส เช่น เสียง ➔ เสียง, ภาพ ➔ เสียง, ข้อความ ➔ เสียง)
* **จุดที่เทรน**: `AudioLMHead` เท่านั้น (~5.2M พารามิเตอร์)
* **จุดที่ฟรีซ**: ฟรีซโมเดลหลัก (LLM MoE, Audio Encoder, Projector) ทั้งหมด 100% เพื่อรักษาเสถียรภาพและคุณภาพภาษาไทยดั้งเดิม

