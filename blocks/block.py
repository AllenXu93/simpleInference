import torch
import torch.nn as nn

from layers.norm import RMSNorm, ManualLayerNorm
from layers.attention import CausalSelfAttention
from layers.linear import Linear


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=None, attn_pdrop=None, layer_norm_epsilon=None):
        super().__init__()
        self.ln_1 = ManualLayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.attn = CausalSelfAttention(embed_dim, num_heads, attn_pdrop=attn_pdrop)
        self.ln_2 = ManualLayerNorm(embed_dim)

        self.mlp = MLP(embed_dim, mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))
        return x

    
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))



def test_block():
    x = torch.randn(2, 8, 128)  # batch=2, seq_len=8, embed_dim=128
    block = TransformerBlock(embed_dim=128, num_heads=4)
    y = block(x)
    assert y.shape == x.shape
    print("✅ TransformerBlock passed:", y.shape)



if __name__ == "__main__":
    test_block()