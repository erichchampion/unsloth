# Chapter 10: Inference with vLLM — fast_inference Mode

---

## Introduction

When `fast_inference=True` is set, Unsloth delegates inference to vLLM, a high-performance serving engine. This chapter explains how this integration works.

### What You'll Learn

- How `fast_inference` is set up in `from_pretrained()`
- The `fast_inference_setup()` function
- vLLM's role: PagedAttention, continuous batching
- FP8 KV cache and GPU memory utilization settings

---

## Notes & Key Points

### 10.1 Enabling Fast Inference

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    fast_inference = True,
    gpu_memory_utilization = 0.5,
    float8_kv_cache = False,
)
```

### 10.2 vLLM Detection

- Checks `importlib.util.find_spec("vllm")` — raises `ImportError` if not installed
- Special handling for DGX Spark (GB10): disables vLLM due to known issues
- `fast_inference_setup()` prepares model config for vLLM loading

### 10.3 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gpu_memory_utilization` | 0.5 | Fraction of GPU memory for vLLM |
| `float8_kv_cache` | False | Use FP8 for KV cache (saves memory) |
| `max_lora_rank` | 64 | Maximum LoRA rank for vLLM LoRA serving |
| `disable_log_stats` | True | Suppress vLLM statistics logging |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Fast inference setup | `unsloth/models/_utils.py` → `fast_inference_setup()` |
| vLLM integration | `unsloth/models/loader.py` |
