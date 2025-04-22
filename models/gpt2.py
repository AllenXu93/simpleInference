import torch
import torch.nn as nn

from blocks.block import TransformerBlock
from layers.norm import RMSNorm, ManualLayerNorm
from layers.attention import CausalSelfAttention
from layers.linear import Linear

from typing import Iterable, Optional, Tuple, Type


class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, mlp_hidden_dim=None, attn_pdrop=None, layer_norm_epsilon=None):
        super().__init__()
        hidden_dim = mlp_hidden_dim or (4 * embed_dim)
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)
        self.h = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_hidden_dim=hidden_dim, attn_pdrop=attn_pdrop, layer_norm_epsilon=layer_norm_epsilon)
            for i in range(num_layers)
        ])
        self.ln_f = ManualLayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # 权重共享
    
    def forward(self, input_ids):
        b, s = input_ids.shape
        tok_emb = self.wte(input_ids)
        pos_ids = torch.arange(s, device=input_ids.device).unsqueeze(0)  # [1, s]
        pos_emb = self.wpe(pos_ids)  # [1, s, d]
        # pos_emb = self.wpe(input_ids)
        emb = tok_emb + pos_emb
        for block in self.h:
            emb = block(emb)
        norm = self.ln_f(emb)
        logits = self.lm_head(norm)
        return logits
        

    def load_model(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Step 1: 加载 Hugging Face 模型的 state_dict
        new_state_dict = {}
        for k in weights.keys():
            v = weights[k]
            # 忽略 prefix
            if k.startswith("transformer."):
                k = k[len("transformer."):]

            if ".attn.bias" in k or ".attn.masked_bias" in k:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue

            # 替换规则
            # if k == "wpe.weight":
            #     k = k.replace("wpe.weight", "wpe")
            
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            # Note(zhuohan): the logic below might break quantized models.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in k:
                    continue
                if not k.endswith(".weight"):
                    continue
                v = v.t()

            if k == "lm_head.weight":
                if v.data_ptr() == weights["transformer.wte.weight"].data_ptr():
                    v = v.clone()

            new_state_dict[k] = v

        # 加载：尝试部分加载
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        print("✅ 权重加载完成")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)




def test_block():
    x = torch.randn(2, 8, 128)  # batch=2, seq_len=8, embed_dim=128
    block = TransformerBlock(embed_dim=128, num_heads=4)
    y = block(x)
    assert y.shape == x.shape
    print("✅ TransformerBlock passed:", y.shape)



if __name__ == "__main__":
    test_block()