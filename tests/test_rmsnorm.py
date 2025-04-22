import torch
import torch.nn as nn
from layers.norm import RMSNorm, ManualLayerNorm  # 你自定义的 RMSNorm 层

def test_rmsnorm_forward():
    torch.manual_seed(0)

    B, S, D = 2, 4, 8  # batch, seq_len, dim
    x = torch.randn(B, S, D)

    # 自定义 RMSNorm
    my_norm = RMSNorm(dim=D, eps=1e-6)

    # 参考：手动计算 RMSNorm
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    y_expected = x / rms * my_norm.weight  # [B, S, D]

    y_actual = my_norm.forward(x)

    assert torch.allclose(y_actual, y_expected, atol=1e-5), "❌ RMSNorm forward mismatch!"
    print("✅ test_rmsnorm_forward passed!")

def test_rmsnorm_shape():
    x = torch.randn(3, 6, 128)
    norm = RMSNorm(128)
    y = norm(x)
    assert y.shape == x.shape, "❌ Shape mismatch after RMSNorm"
    print("✅ test_rmsnorm_shape passed!")


def test_manual_layernorm():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)

    ref = nn.LayerNorm(8, eps=1e-5)
    test = ManualLayerNorm(8, eps=1e-5)

    # 对齐参数
    test.weight.data.copy_(ref.weight.data)
    test.bias.data.copy_(ref.bias.data)

    y_ref = ref(x)
    y_test = test(x)

    assert torch.allclose(y_ref, y_test, atol=1e-6), "❌ LayerNorm mismatch"
    print("✅ test_manual_layernorm passed!")


if __name__ == "__main__":
    test_manual_layernorm()
    # test_rmsnorm_forward()
    # test_rmsnorm_shape()
