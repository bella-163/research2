import torch
import torch.nn as nn

from adadrs.lora_kv import LoRAKVQKVLinear


def test_lora_kv_shape_and_merge():
    base = nn.Linear(16, 48)
    layer = LoRAKVQKVLinear(base, rank=4, alpha=8)
    x = torch.randn(2, 5, 16)
    y = layer(x)
    assert y.shape == (2, 5, 48)

    loss = y.sum()
    loss.backward()
    assert layer.B_k.grad is not None
    assert layer.B_v.grad is not None

    with torch.no_grad():
        layer.B_k.add_(0.01)
        layer.B_v.add_(0.01)
        before = layer.old_delta_k.clone()
        layer.merge_current_into_old()
        assert not torch.allclose(before, layer.old_delta_k)


def test_subtraction_mode_runs():
    base = nn.Linear(8, 24)
    layer = LoRAKVQKVLinear(base, rank=2, alpha=4)
    x = torch.randn(3, 7, 8)
    with torch.no_grad():
        layer.B_k.add_(0.1)
        layer.B_v.add_(0.1)
        layer.merge_current_into_old()
    layer.set_old_scale(-1.0)
    layer.enable_current(False)
    y = layer(x)
    assert y.shape == (3, 7, 24)
