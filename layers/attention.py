import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_pdrop):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 1024, 1024)))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).view(B, T, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.mask[:, :, :T, :T]
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(attn_output)


class CausalSelfAttention2(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_pdrop):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 统一 QKV 投影
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # 输出 projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        if attn_pdrop != None:
            self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        mask: [seq_len, seq_len] (optional)
        """
        # 👉 你来完成下面这些步骤
        # 1. QKV projection
        b, s, d = x.shape
        qkv = self.c_attn(x)
        # 2. 拆 heads
        qkv = qkv.view(b, s, self.num_heads, -1)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 3. attention score
        qk = q.matmul(k.transpose(2,3)) * self.scale

        # 4. causal mask
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool)).view(1, 1, s, s)
        # 5. softmax
        attn_probs = torch.softmax(qk.masked_fill(mask == 0, float('-inf')), dim=-1)

        if self.attn_dropout != None:
            attn_probs = self.attn_dropout(attn_probs)

        # 6. attention weighted sum
        attn_output = attn_probs.matmul(v)
        # 7. 合并 head
        context = attn_output.transpose(1,2).contiguous().view(b,s,d)
        # 8. output projection

        return self.c_proj(context)
