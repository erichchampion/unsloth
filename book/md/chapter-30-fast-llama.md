# Chapter 30: FastLlamaModel — The Reference Implementation

> *"Llama is the standard — all other architectures are measured against it."*

---

## Introduction

`FastLlamaModel` is the cornerstone of Unsloth's architecture-specific optimizations. At 143K (approximately 3,500 lines), it is the most complete and well-developed `Fast*Model` class, serving as the reference implementation from which all other architecture classes inherit or delegate. This chapter provides a detailed walkthrough of its patching strategy, showing exactly how a standard Hugging Face Llama model is transformed into a "fast" model.

### What You'll Learn

- The complete `from_pretrained()` flow for Llama models
- How attention, MLP, and normalization layers are patched
- RoPE variants and their configuration
- The `get_peft_model()` and `patch_peft_model()` cycle
- Multi-GPU cos/sin caching strategy

### Prerequisites

- The model loading pipeline from Chapter 7
- Kernel internals from Part VI (Chapters 22–29)
- LoRA concepts from Chapter 12

---

## 30.1 from_pretrained() Flow

`FastLlamaModel.from_pretrained()` performs a carefully ordered sequence of operations:

```
1. Resolve model identity
   ├─ Map model name to HF repo
   ├─ Detect architecture (LlamaForCausalLM)
   └─ Check quantization compatibility

2. Load base model
   ├─ AutoModelForCausalLM.from_pretrained()
   ├─ Apply BitsAndBytesConfig (if 4-bit)
   └─ Set max_seq_length and dtype

3. Patch model layers
   ├─ For each transformer block:
   │   ├─ Replace self_attn.forward → optimized attention
   │   ├─ Replace mlp.forward → fused SwiGLU
   │   └─ Replace norms → Triton RMSNorm
   ├─ Replace loss function → fused cross-entropy
   └─ Configure gradient checkpointing

4. Load tokenizer
   ├─ load_correct_tokenizer()
   ├─ Fix slow/fast tokenizer mismatches
   └─ Set chat template (if applicable)

5. Return (model, tokenizer)
```

---

## 30.2 Layer Patching Deep Dive

### Attention Patching

The attention forward pass is replaced to inject:
- **RoPE kernel** — fused cos/sin rotation (Chapter 25)
- **Attention dispatch** — selects Flash Attention 2, FlexAttention, SDPA, or manual attention based on hardware
- **Grouped Query Attention (GQA)** — optimized for configurations where `n_kv_heads < n_heads`

```python
# Attention dispatch priority:
if flash_attention_2_available and ampere_or_newer:
    attention_fn = flash_attention_forward
elif flex_attention_available and pytorch_2_5:
    attention_fn = flex_attention_forward
else:
    attention_fn = sdpa_attention_forward     # PyTorch built-in
```

### MLP Patching

The MLP replaces the standard three-matmul SwiGLU with Unsloth's fused version:

```python
# Before patching:
#   gate = self.gate_proj(x)
#   gate = F.silu(gate)
#   up = self.up_proj(x)
#   output = gate * up
#   output = self.down_proj(output)

# After patching:
#   output = fast_swiglu_fg(self, x)  # Single fused operation
```

### Norm Patching

Both `input_layernorm` and `post_attention_layernorm` have their forward methods replaced with `fast_rms_layernorm()`.

---

## 30.3 RoPE Variants

`FastLlamaModel` supports every RoPE variant across the Llama family:

| Model | RoPE Type | Parameters |
|-------|-----------|-----------|
| Llama 2 | Standard | `base=10000` |
| CodeLlama | Dynamic NTK | Dynamically scaled frequencies |
| Llama 3 | Standard | `base=500000` |
| Llama 3.1 | `"llama3"` | `low_freq_wavelen`, `high_freq_wavelen`, `factor` |
| Llama 3.2 | `"llama3"` | Same as 3.1 with different factor |

The scaling configuration is read from the model's `config.json` → `rope_scaling` field.

---

## 30.4 get_peft_model() and Re-Patching

When applying LoRA adapters, PEFT wraps the model in a `PeftModelForCausalLM`. This wrapping can undo some of Unsloth's patches because PEFT replaces the original `nn.Linear` layers with its own `LoraLayer` wrappers:

```python
# The two-step process:
model = peft.get_peft_model(model, lora_config)    # PEFT wrapping (may break patches)
FastLlamaModel.patch_peft_model(model, ...)         # Restore Unsloth patches
```

`patch_peft_model()` walks through the now-wrapped model and re-applies the Triton kernel replacements to the PEFT `LoraLayer` objects.

---

## 30.5 Multi-GPU Cos/Sin Caching

For multi-GPU setups, RoPE cos/sin values must be cached per device to avoid expensive cross-device transfers:

```python
multi_gpu_cos_cached = {}  # {device_id: cos_tensor}
multi_gpu_sin_cached = {}  # {device_id: sin_tensor}

def get_cos_sin(position_ids, device):
    if device not in multi_gpu_cos_cached:
        multi_gpu_cos_cached[device] = cos_values.to(device)
        multi_gpu_sin_cached[device] = sin_values.to(device)
    return multi_gpu_cos_cached[device], multi_gpu_sin_cached[device]
```

---

## 30.6 Gradient Checkpointing

Unsloth's `"unsloth"` gradient checkpointing strategy is more selective than PyTorch's default:

| Strategy | What Gets Checkpointed | Memory Savings |
|----------|----------------------|----------------|
| `"none"` | Nothing | 0% |
| `True` (PyTorch default) | All layers uniformly | ~60% |
| `"unsloth"` | Selective by memory footprint | ~70% |

The Unsloth strategy identifies which layers use the most activation memory and selectively checkpoints those, achieving better memory-compute trade-offs.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastLlamaModel | `unsloth/models/llama.py` (143K) |
| Model utilities | `unsloth/models/_utils.py` (107K) |
| Attention dispatch | `unsloth/models/llama.py` |
| RoPE configuration | `unsloth/models/llama.py`, `unsloth/kernels/rope_embedding.py` |
