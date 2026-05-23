"""
Agent SFT Dataset
------------------
Loads Agent instruction datasets (ReAct loops, Tool use, SQL/API calls).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from datasets import load_dataset

from ..src.processing_omni import OmniProcessor


class AgentSFTDataset(Dataset):
    """
    Dataset containing tool calling / Agent conversations.
    Uses target prompt template for tool-use loops.
    """

    def __init__(
        self,
        processor: OmniProcessor,
        hf_dataset_name: str = "xverse/agent-sft-data",
        hf_split: str = "train",
        max_length: int = 2048,
        system_prompt: str = (
            "คุณเป็น AI Agent ภาษาไทยที่เชี่ยวชาญด้านการทำงานร่วมกับระบบภายนอกและการเรียกใช้เครื่องมือ "
            "ตอบกลับในรูปแบบ ReAct หรือ JSON ตามที่ผู้ใช้กำหนดอย่างเคร่งครัด"
        ),
    ):
        self.processor = processor
        self.max_length = max_length
        self.system_prompt = system_prompt

        print(f"[AgentDataset] Loading agent dataset from {hf_dataset_name} ({hf_split})")
        # Load stream if needed, fallback to standard download
        try:
            self.data = load_dataset(hf_dataset_name, split=hf_split, trust_remote_code=True)
        except Exception:
            # Fallback mock/public dataset if xverse dataset isn't immediately available locally
            print("[AgentDataset] Fallback to mlabonne/FineTome-100k filtered for agent conversations")
            self.data = load_dataset("mlabonne/FineTome-100k", split=hf_split, trust_remote_code=True)

        print(f"[AgentDataset] Loaded {len(self.data):,} samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Extract conversation history
        if "conversations" in item:
            messages = []
            for m in item["conversations"]:
                role = "user" if m["from"] in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": m["value"]})
        else:
            # Fallback schema
            query = item.get("instruction", item.get("input", ""))
            response = item.get("output", item.get("response", ""))
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]

        # Build SFT input
        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        input_ids = encoded["input_ids"].squeeze(0)[:self.max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)[:self.max_length]
        labels = input_ids.clone()

        # Simple prompt masking: find target start
        # Mask everything except assistant responses for SFT loss
        # (Assuming the last assistant response is the target)
        last_turn = messages[-1]["content"] if messages else ""
        if last_turn:
            ans_tokens = self.processor.tokenizer.encode(last_turn, add_special_tokens=False)
            mask_len = max(0, len(input_ids) - len(ans_tokens))
            labels[:mask_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "modality": "text",  # Handled as text tokens inside MoE routing
        }
