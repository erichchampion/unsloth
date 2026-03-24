# Chapter 10: Inference with vLLM — fast_inference Mode

> *"When you need more than just fast — you need production-grade throughput."*

---

## Introduction

By default, Unsloth uses Hugging Face's `model.generate()` for inference. This is fine for single-request development and testing, but for production workloads — serving many concurrent requests, maximizing GPU utilization, or running continuous batching — you need more. That's what `fast_inference=True` provides: it delegates inference to vLLM, a high-performance serving engine that implements PagedAttention, continuous batching, and optimized memory management.

This chapter explains how the vLLM integration works, what happens during `fast_inference_setup()`, and when you should (or should not) use it.

### What You'll Learn

- How `fast_inference` is activated and configured
- The `fast_inference_setup()` function and what it does
- vLLM's key features: PagedAttention, continuous batching, FP8 KV cache
- Hardware compatibility and restrictions
- When to use `fast_inference` vs. standard generation

### Prerequisites

- The loading pipeline from Chapter 7
- Basic understanding of inference serving concepts (batching, KV caching)

---

## 10.1 Enabling Fast Inference

Fast inference is activated via a single parameter in `from_pretrained()`:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    fast_inference = True,
    gpu_memory_utilization = 0.5,    # Fraction of GPU VRAM for vLLM
    float8_kv_cache = False,         # Use FP8 for KV cache
)
```

When this flag is set, `from_pretrained()` first checks that vLLM is installed:

```python
if importlib.util.find_spec("vllm") is None:
    raise ImportError(
        "Unsloth: vLLM is not installed. Install it with: pip install vllm"
    )
```

---

## 10.2 Hardware Restrictions

Not all hardware supports vLLM. The loader includes explicit restrictions:

- **DGX Spark (GB10)** — vLLM is disabled due to known compute compatibility issues. The code detects this by checking for a "GB10" substring in `torch.cuda.get_device_name()`.
- **AMD ROCm** — vLLM support depends on the ROCm version and GPU family.
- **Minimum GPU memory** — vLLM requires enough VRAM to hold the model weights plus the KV cache pool.

---

## 10.3 The fast_inference_setup Function

Located in `_utils.py` (line 2265), this function prepares the model's configuration for vLLM:

```python
def fast_inference_setup(model_name, model_config):
    # 1. Save model config to a temp directory vLLM can read
    # 2. Configure vLLM engine parameters:
    #    - max_model_len (from max_seq_length)
    #    - gpu_memory_utilization
    #    - dtype (matches training dtype)
    #    - quantization config (if using 4-bit or FP8)
    # 3. Handle LoRA adapter configuration
    #    - max_lora_rank = 64
    #    - enable_lora = True if LoRA adapter detected
    # 4. Return vLLM-compatible config dict
```

The function bridges the gap between Unsloth's model representation and vLLM's expected configuration format. It handles the translation of quantization configs, attention parameters, and LoRA settings.

---

## 10.4 vLLM Key Features

### PagedAttention

vLLM's core innovation is PagedAttention, which manages KV cache memory like virtual memory pages. Instead of pre-allocating a contiguous block for each sequence's KV cache, vLLM allocates fixed-size "pages" on demand. This eliminates the memory fragmentation that plagues naive KV cache allocation and enables near-100% GPU memory utilization.

### Continuous Batching

Rather than waiting for all sequences in a batch to finish generating before starting new ones, vLLM uses iteration-level scheduling. When one sequence in the batch reaches its stop token, a new sequence immediately takes its slot. This maximizes throughput for variable-length generations.

### FP8 KV Cache

When `float8_kv_cache=True`, vLLM stores attention KV cache values in FP8 instead of FP16/BF16, halving KV cache memory. This allows longer context lengths or larger batch sizes within the same GPU memory budget, at the cost of a small accuracy reduction.

---

## 10.5 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `fast_inference` | `False` | Enable vLLM backend |
| `gpu_memory_utilization` | `0.5` | Fraction of GPU VRAM allocated to vLLM |
| `float8_kv_cache` | `False` | Use FP8 for KV cache (saves memory) |
| `max_lora_rank` | `64` | Maximum LoRA rank for vLLM LoRA serving |
| `disable_log_stats` | `True` | Suppress vLLM statistics logging |

The `gpu_memory_utilization` parameter is particularly important. Setting it too high (>0.9) can cause out-of-memory errors because the system needs memory for intermediate computations. Setting it too low wastes GPU capacity. The default of 0.5 is conservative — for dedicated inference servers, 0.8-0.9 is typical.

---

## 10.6 Generation with vLLM

Once a model is loaded with `fast_inference=True`, generation uses a different code path than standard `model.generate()`. The typical pattern:

```python
# Standard generation (without fast_inference)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

# With fast_inference, use the chat template + generate pattern
messages = [{"role": "user", "content": "What is machine learning?"}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
outputs = model.fast_generate(
    [input_text],
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
```

The `fast_generate` method wraps vLLM's `SamplingParams` and engine interface, providing a clean API that mirrors Hugging Face's `generate()` but runs through vLLM's optimized pipeline.

### LoRA Serving with vLLM

When a LoRA adapter is loaded alongside `fast_inference=True`, vLLM serves the adapter dynamically. This means you can switch between different LoRA adapters without reloading the base model:

- The base model weights stay fixed in GPU memory
- LoRA adapter weights are loaded separately
- `max_lora_rank=64` sets the maximum rank vLLM will accept

---

## 10.7 When to Use fast_inference

| Scenario | Recommendation |
|----------|---------------|
| Single prompt, quick test | Standard `model.generate()` |
| Batch inference (many prompts) | `fast_inference=True` |
| Interactive chat serving | `fast_inference=True` |
| Training then inference | Load once with `fast_inference=True` |
| Low-memory GPU (<8GB) | Keep default (vLLM has memory overhead) |
| Unsupported hardware (DGX Spark) | Standard `model.generate()` |
| GGUF inference | Use llama.cpp directly |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Fast inference setup | `unsloth/models/_utils.py` → `fast_inference_setup()` |
| vLLM integration in loader | `unsloth/models/loader.py` |
| GPU detection for restrictions | `unsloth/device_type.py` |
