# test/test_attention.py
import torch
from layers.attention import CausalSelfAttention

def test_attention_forward():
    B, S, D, H = 2, 4, 128, 4  # batch, seq_len, embed_dim, heads

    attn = CausalSelfAttention(embed_dim=D, num_heads=H)
    x = torch.randn(B, S, D)

    y = attn.forward(x)

    assert y.shape == (B, S, D), f"Expected shape {(B, S, D)}, got {y.shape}"
    print("✅ test_attention_forward passed!")

if __name__ == "__main__":
    test_attention_forward()
