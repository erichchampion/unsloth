# Chapter 12: LoRA and QLoRA — Parameter-Efficient Fine-Tuning

---

## Introduction

LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are the default training modes in Unsloth. This chapter explains how Unsloth wraps PEFT's LoRA with optimized kernels for dramatic speed and memory improvements.

### What You'll Learn

- How `get_peft_model()` configures LoRA adapters
- QLoRA: 4-bit base weights + LoRA adapters
- Unsloth's fast LoRA kernels vs. standard PEFT
- Target modules and rank selection

---

## Notes & Key Points

### 12.1 Adding LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

### 12.2 QLoRA: 4-bit Quantization + LoRA

- Base model weights are loaded in 4-bit NF4 format via bitsandbytes
- LoRA adapters are trained in full precision (float16/bfloat16)
- `load_in_4bit=True` triggers QLoRA mode in `from_pretrained()`
- Dequantization uses Unsloth's `fast_dequantize()` kernel

### 12.3 Fast LoRA Kernel

- `kernels/fast_lora.py` (21K) — Custom Triton kernel for LoRA forward pass
- Fuses the base weight dequantization + LoRA addition into a single kernel
- `get_lora_parameters_bias()` extracts W, A, B, scaling factor, and bias

### 12.4 Gradient Checkpointing

- `use_gradient_checkpointing="unsloth"` enables Unsloth's custom strategy
- `apply_unsloth_gradient_checkpointing()` decides strategy based on seq length and dtype
- Trades compute for memory: re-computes activations during backward pass

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| get_peft_model | `unsloth/models/llama.py` |
| Fast LoRA kernel | `unsloth/kernels/fast_lora.py` |
| LoRA parameter extraction | `unsloth/kernels/__init__.py` → `get_lora_parameters_bias` |
| Gradient checkpointing | `unsloth/models/_utils.py` → `apply_unsloth_gradient_checkpointing` |
