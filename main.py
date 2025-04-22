from engine.generate import generate_text
from engine.load import load_model

if __name__ == "__main__":
    model_name = "gpt2"
    prompt = "Once upon a time"
    tokenizer, model, max_length = load_model(model_name)
    result = generate_text(tokenizer, model, prompt, max_new_tokens=max_length)

    print("\n🧠 Generated result:\n", result)