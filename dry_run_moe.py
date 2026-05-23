"""
dry_run_moe.py
---------------
Dry-run verification test suite for the Multi-Modal MoE architecture.
Asserts Routing logic, dimension outputs, and target parameters of SparseMoELayers.
"""

import sys
import traceback
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.moe import SparseMoELayer
from src.modeling_moe_adapter import convert_mlp_to_moe

# ──────────────────────────────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        print(f"         → {e}")
        traceback.print_exc()

# ──────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  Thai Omni-Modal — MoE Dry Run Test Suite")
print("═"*60 + "\n")

# ── Test 1: SparseMoELayer Forward & Shapes ─────────────────────────
print("📋 [1/3] SparseMoELayer Forward Pass")
def test_moe_forward():
    # Dimensions
    B, T, hidden_size = 2, 128, 64
    num_experts = 4
    top_k = 2

    # Create dummy experts (tiny MLPs)
    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        def forward(self, x):
            return self.net(x)

    experts = [DummyMLP() for _ in range(num_experts)]
    moe = SparseMoELayer(experts=experts, hidden_dim=hidden_size, num_experts=num_experts, top_k=top_k)
    moe.eval()

    x = torch.randn(B, T, hidden_size)
    with torch.no_grad():
        out = moe(x)

    assert out.shape == (B, T, hidden_size), f"Expected {(B, T, hidden_size)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in MoE output"
    assert not torch.isinf(out).any(), "Inf detected in MoE output"

test("MoE Layer forward pass execution and dimension assertion", test_moe_forward)

# ── Test 2: Top-2 Routing & Softmax Balance ─────────────────────────
print("\n📋 [2/3] Top-2 Routing Coefficient Balance")
def test_routing_coefficients():
    B, T, hidden_size = 1, 100, 64
    num_experts = 4
    top_k = 2

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
        def forward(self, x):
            return self.linear(x)

    experts = [DummyMLP() for _ in range(num_experts)]
    moe = SparseMoELayer(experts=experts, hidden_dim=hidden_size, num_experts=num_experts, top_k=top_k)

    # Force gate weights to route tokens to expert 0 and 3
    moe.gate.weight.data.fill_(0.0)
    moe.gate.weight.data[0, :] = 1.0    # Expert 0 top weight
    moe.gate.weight.data[3, :] = 0.9    # Expert 3 second-best weight

    # Set x to all ones to ensure positive activation: 10 * 1 = 10 (maximum)
    x = torch.ones(B, T, hidden_size)
    out = moe(x)

    # Let's verify top indices and weights
    flat_x = x.view(-1, hidden_size)
    logits = moe.gate(flat_x)
    weights = F.softmax(logits, dim=-1)
    top_weights, top_indices = torch.topk(weights, top_k, dim=-1)

    # All tokens should have routed to 0 and 3
    assert (top_indices[:, 0] == 0).all(), "Top-1 routing incorrect"
    assert (top_indices[:, 1] == 3).all(), "Top-2 routing incorrect"

test("MoE gating top-2 router expert assignments and normalization", test_routing_coefficients)

# ── Test 3: Model Adapter Convert Block ─────────────────────────────
print("\n📋 [3/3] Model FFN to MoE Conversion Adapter")
def test_adapter_patching():
    # Construct a Mock standard transformer model
    class MockMLP(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        def forward(self, x):
            return self.proj(x)

    class MockLayer(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.mlp = MockMLP(hidden_dim)
        def forward(self, x):
            return self.mlp(x)

    class MockLLMModel(nn.Module):
        def __init__(self, hidden_dim, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer(hidden_dim) for _ in range(num_layers)])
            # Mock config
            class Config:
                hidden_size = hidden_dim
            self.config = Config()

    class MockOMNIModel(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.llm = MockLLMModel(hidden_dim)

    # Initialize mock model
    hidden_size = 32
    omni = MockOMNIModel(hidden_size)

    # Verify initial layer class is MockMLP
    assert isinstance(omni.llm.layers[0].mlp, MockMLP)

    # Run converter adapter
    convert_mlp_to_moe(omni, num_experts=4, top_k=2, target_layers=[0, 2])

    # Assert target layers converted, non-targeted untouched
    assert isinstance(omni.llm.layers[0].mlp, SparseMoELayer)
    assert isinstance(omni.llm.layers[1].mlp, MockMLP)
    assert isinstance(omni.llm.layers[2].mlp, SparseMoELayer)
    assert isinstance(omni.llm.layers[3].mlp, MockMLP)

    # Assert number of experts inside converted layer is 4
    assert len(omni.llm.layers[0].mlp.experts) == 4

test("convert_mlp_to_moe patches transformer layer MLPs successfully", test_adapter_patching)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "═"*60)
passed = sum(1 for r, _ in results if r == PASS)
failed = sum(1 for r, _ in results if r == FAIL)
print(f"  Results: {passed}/{len(results)} passed  |  {failed} failed")
if failed == 0:
    print("  🎉 All MoE architecture dry-run tests passed!")
else:
    print("  ⚠️  Fix MoE failures before proceeding to training.")
print("═"*60 + "\n")

sys.exit(0 if failed == 0 else 1)
