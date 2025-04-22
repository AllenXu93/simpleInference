import torch

def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 50):
    """自回归生成"""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    return tokenizer.decode(generated[0])