import torch
import time
from engine.load import load_model

def benchmark_model(model, input_ids, device, warmup=10, iters=50):
    model.eval()
    model.to(device)
    input_ids = input_ids.to(device)

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) * 1000 / iters  # ms
    return avg_latency


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "gpt2"
    tokenizer, model, max_length = load_model(model_name)

    input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids

    print("🚀 加载未优化模型")
    baseline_time = benchmark_model(model, input_ids, device)
    print(f"🕒 原始模型推理时间: {baseline_time:.2f} ms")

    print("⚙️  torch.compile 优化中...")
    compiled_model = torch.compile(model)
    compiled_time = benchmark_model(compiled_model, input_ids, device)
    print(f"🚀 编译模型推理时间: {compiled_time:.2f} ms")

    speedup = baseline_time / compiled_time
    print(f"⚡️ 推理加速比: {speedup:.2f}x")

if __name__ == "__main__":
    main()
