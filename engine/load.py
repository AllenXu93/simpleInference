import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from models.gpt2 import GPT2  # 你实现的模型

def load_model(model_name: str, enable_compile=False):
    """根据模型名自动加载不同结构的模型"""
    cfg = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if cfg.architectures[0] == "GPT2LMHeadModel":
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = GPT2(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.n_embd,
            num_heads=cfg.n_head,
            num_layers=cfg.n_layer,
            max_seq_len=cfg.n_positions,
            attn_pdrop=cfg.attn_pdrop,
            layer_norm_epsilon=cfg.layer_norm_epsilon,
        )
        model.load_model(hf_model.state_dict())
        print("load my GPT2 model")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    max_length = 50
    if cfg.task_specific_params["text-generation"]["max_length"] != None:
        max_length = cfg.task_specific_params["text-generation"]["max_length"]
    model.eval()
    if enable_compile:
        model = torch.compile(model)
    return tokenizer, model, max_length


