"""
Sparse MoE Layer (moe.py)
--------------------------
Replaces the standard MLP/FFN layer in the transformer.
Routes tokens to Top-2 experts out of N available experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable


class SparseMoELayer(nn.Module):
    """
    A Top-2 Sparse Mixture-of-Experts (MoE) layer.
    Wraps multiple expert MLP modules and uses a gating network
    to route tokens dynamically.
    """

    def __init__(
        self,
        experts: List[nn.Module],
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        assert len(experts) == num_experts, "Number of experts must match num_experts"
        self.experts = nn.ModuleList(experts)
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network: maps token hidden states to expert logits
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self._init_gate_weights()

    def _init_gate_weights(self):
        # Small weights init to keep routing balanced initially
        nn.init.normal_(self.gate.weight, std=0.01)

    def _load_balance_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        # fraction of tokens ที่แต่ละ expert ได้รับ
        probs = F.softmax(gate_logits, dim=-1)          # [Tokens, E]
        density = probs.mean(dim=0)                      # [E]
        # ถ้าสมดุล density ทุกตัว = 1/E → loss ต่ำ
        aux = self.num_experts * (density * density).sum()
        return aux

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [Batch, SeqLen, HiddenDim] or flattened [Tokens, HiddenDim]
        """
        orig_shape = hidden_states.shape
        # Flatten batch and sequence dimensions: [Tokens, HiddenDim]
        x = hidden_states.view(-1, orig_shape[-1])
        num_tokens = x.shape[0]

        # 1. Compute gating logits
        gate_logits = self.gate(x)  # [Tokens, num_experts]
        
        # คำนวณ aux loss และเก็บไว้ในตัวแปรของ layer
        self.aux_loss = self._load_balance_loss(gate_logits)
        
        # 2. Select Top-K experts
        weights = F.softmax(gate_logits, dim=-1)  # [Tokens, num_experts]
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)  # [Tokens, top_k]

        # Normalize weights over selected experts
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Route tokens to respective experts
        # Output tensor accumulator
        final_output = torch.zeros_like(x)

        # We execute expert layers sequentially for memory simplicity,
        # masking out tokens not assigned to the current expert.
        for expert_idx in range(self.num_experts):
            # Find tokens where this expert is selected
            # top_indices shape is [Tokens, top_k]
            mask = (top_indices == expert_idx)  # [Tokens, top_k] (bool)
            
            # If no tokens route to this expert, skip it
            if not mask.any():
                continue

            # Get token indices and their corresponding K position (0 or 1 for top-2)
            token_indices, k_positions = torch.where(mask)

            # Extract matching token states
            expert_inputs = x[token_indices]

            # Pass through the selected expert
            expert_outputs = self.experts[expert_idx](expert_inputs)

            # Get respective weights: top_weights[token_index, k_position]
            expert_weights = top_weights[token_indices, k_positions].unsqueeze(-1)

            # Accumulate scaled outputs, ensuring the dtype matches final_output
            scaled_outputs = (expert_outputs * expert_weights).to(final_output.dtype)
            final_output.index_add_(0, token_indices, scaled_outputs)

        return final_output.view(orig_shape)
