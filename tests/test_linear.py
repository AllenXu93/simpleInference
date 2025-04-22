import torch
import torch.nn as nn
from layers.linear import Linear  # 你写的 Linear 层

def test_linear_forward():
    torch.manual_seed(0)  # 保证可复现

    # 定义输入
    B, S, D_in, D_out = 2, 4, 8, 16
    x = torch.randn(B, S, D_in)

    # 初始化 PyTorch 和自定义 Linear
    ref_linear = nn.Linear(D_in, D_out)
    my_linear = Linear(D_in, D_out)

    # 手动对齐权重
    my_linear.weight.data.copy_(ref_linear.weight.data)
    print(ref_linear.weight.shape)
    print(my_linear.weight.shape)
    print(ref_linear.bias.shape)
    print(my_linear.bias.shape)
    my_linear.bias.data.copy_(ref_linear.bias.data)

    # 前向输出
    y_ref = ref_linear(x)
    y_test = my_linear.forward(x)

    # 对比结果
    assert torch.allclose(y_ref, y_test, atol=1e-6), "Linear forward mismatch!"
    print("✅ test_linear_forward passed!")

def test_linear_no_bias():
    B, S, D_in, D_out = 1, 2, 4, 4
    x = torch.randn(B, S, D_in)

    # 无 bias 情况
    ref_linear = nn.Linear(D_in, D_out, bias=False)
    my_linear = Linear(D_in, D_out, bias=False)

    my_linear.weight.data.copy_(ref_linear.weight.data)
    y_ref = ref_linear(x)
    y_test = my_linear.forward(x)

    assert torch.allclose(y_ref, y_test, atol=1e-6), "Linear no-bias mismatch!"
    print("✅ test_linear_no_bias passed!")

if __name__ == "__main__":
    test_linear_forward()
    test_linear_no_bias()
