"""
Modeling MoE Adapter
---------------------
Patches standard Qwen/LLaMA models, converting their FFN/MLP blocks
into SparseMoELayers with replicated experts.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, List

from .moe import SparseMoELayer


def convert_mlp_to_moe(
    model: nn.Module,
    num_experts: int = 4,
    top_k: int = 2,
    target_layers: Optional[List[int]] = None,
):
    """
    Traverses the model's transformer layers and replaces their MLP layers
    with SparseMoELayers.
    
    Replicates the existing pretrained weights into all experts to initialize them,
    preserving general language knowledge before fine-tuning starts.
    """
    # 1. Identify the transformer layers block inside LLM
    # In Qwen3.5: model.llm.model.layers
    # We navigate dynamically to handle wrappers
    llm_model = model
    if hasattr(model, "llm"):
        llm_model = model.llm
    if hasattr(llm_model, "model"):
        llm_model = llm_model.model

    if not hasattr(llm_model, "layers"):
        raise AttributeError("Cannot locate transformer layers in the model structure.")

    layers = llm_model.layers
    num_layers = len(layers)
    target_layers = target_layers or list(range(num_layers))

    print(f"[MoE Adapter] Upgrading {len(target_layers)}/{num_layers} layers to MoE (Experts={num_experts}, Top-{top_k})")

    for i in target_layers:
        layer = layers[i]
        
        # Check standard MLP attribute names (Qwen/Llama use 'mlp')
        if not hasattr(layer, "mlp"):
            print(f"  [Warning] Layer {i} does not have an 'mlp' attribute. Skipping.")
            continue

        original_mlp = layer.mlp
        hidden_dim = llm_model.config.hidden_size

        # 2. Replicate the original pretrained FFN into num_experts copies
        experts = []
        for e_idx in range(num_experts):
            # Deepcopy ensures independent weight adaptation
            expert_copy = copy.deepcopy(original_mlp)
            experts.append(expert_copy)

        # 3. Create the MoE wrapper layer
        moe_layer = SparseMoELayer(
            experts=experts,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # 4. Swap layer FFN block
        layer.mlp = moe_layer

    print("[MoE Adapter] ✅ MoE conversion completed successfully.")
    return model
