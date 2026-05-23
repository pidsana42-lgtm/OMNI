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
    # We navigate dynamically to handle different wrappers (e.g. Qwen3.5, Qwen2-VL)
    llm_model = model
    if hasattr(model, "llm"):
        llm_model = model.llm

    layers = None
    # Try common candidate paths
    for candidate_path in [
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "layers"
    ]:
        curr = llm_model
        parts = candidate_path.split(".")
        found = True
        for part in parts:
            if hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                found = False
                break
        if found and (isinstance(curr, nn.ModuleList) or isinstance(curr, nn.Sequential)):
            if len(curr) > 0 and hasattr(curr[0], "mlp"):
                layers = curr
                print(f"[MoE Adapter] Found layers block at: model.llm.{candidate_path}")
                break

    # Recursive fallback search
    if layers is None:
        for name, sub_mod in llm_model.named_modules():
            if isinstance(sub_mod, nn.ModuleList):
                if len(sub_mod) > 0 and hasattr(sub_mod[0], "mlp"):
                    if "visual" not in name:
                        layers = sub_mod
                        print(f"[MoE Adapter] Found layers block dynamically at: model.llm.{name}")
                        break

    if layers is None:
        raise AttributeError("Cannot locate transformer layers in the model structure.")
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

        # Robustly determine hidden_dim
        hidden_dim = None
        for config_attr in ["hidden_size", "text_config.hidden_size", "d_model"]:
            if "." in config_attr:
                part1, part2 = config_attr.split(".")
                if hasattr(llm_model.config, part1):
                    sub_cfg = getattr(llm_model.config, part1)
                    if hasattr(sub_cfg, part2):
                        hidden_dim = getattr(sub_cfg, part2)
                        break
            else:
                if hasattr(llm_model.config, config_attr):
                    hidden_dim = getattr(llm_model.config, config_attr)
                    break
        
        # MLP shape-based fallback if config attributes are missing or structured differently
        if hidden_dim is None:
            if hasattr(original_mlp, "gate_proj") and hasattr(original_mlp.gate_proj, "in_features"):
                hidden_dim = original_mlp.gate_proj.in_features
            elif hasattr(original_mlp, "up_proj") and hasattr(original_mlp.up_proj, "in_features"):
                hidden_dim = original_mlp.up_proj.in_features
            elif hasattr(original_mlp, "proj") and hasattr(original_mlp.proj, "weight"):
                hidden_dim = original_mlp.proj.weight.shape[-1]
            else:
                raise AttributeError("Cannot determine hidden dimension size for the FFN layer.")

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
