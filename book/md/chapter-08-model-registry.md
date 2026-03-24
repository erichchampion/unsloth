# Chapter 8: The Model Registry — Mapping Names to Weights

---

## Introduction

Unsloth maintains a registry of known model families, mapping user-friendly names to Hugging Face repository paths. This system enables automatic quantization selection, pre-quantized model routing, and consistent model naming.

### What You'll Learn

- The `ModelInfo` and `ModelMeta` data structures
- How `MODEL_REGISTRY` is populated by family modules
- The registration flow: `_register_models()` and `register_model()`
- How `QuantType` (BNB, GGUF, BF16, FP8) affects model path construction

---

## Notes & Key Points

### 8.1 Registry Data Structures

```python
class QuantType(Enum):
    BNB = "bnb"            # bitsandbytes 4-bit
    UNSLOTH = "unsloth"    # Unsloth dynamic 4-bit
    GGUF = "GGUF"          # GGUF format
    NONE = "none"          # No quantization
    BF16 = "bf16"          # Bfloat16

@dataclass
class ModelInfo:
    org: str               # e.g., "unsloth"
    base_name: str         # e.g., "Llama-3.2"
    version: str           # e.g., "3.2"
    size: int              # e.g., 1 (for 1B)
    quant_type: QuantType
    # → model_path = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
```

### 8.2 Family Modules

- `_llama.py` — Llama family (3, 3.1, 3.2, 3.3)
- `_gemma.py` — Gemma 1, 2, 3
- `_qwen.py` — Qwen 2, 2.5, 3
- `_mistral.py` — Mistral, Ministral
- `_phi.py` — Phi-3, Phi-4
- `_deepseek.py` — DeepSeek V2, V3, R1

### 8.3 Registration Flow

1. Each family module defines `ModelMeta` instances with sizes, instruct tags, and quant types
2. `_register_models()` generates all combinations and calls `register_model()`
3. The global `MODEL_REGISTRY` dict maps `"org/name"` → `ModelInfo`

### 8.4 Name Construction

```
{base_name}-{instruct_tag}-{quant_tag}
 Llama-3.2  -  Instruct   -  bnb-4bit
  org: unsloth → "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Registry core | `unsloth/registry/registry.py` |
| Llama family | `unsloth/registry/_llama.py` |
| Qwen family | `unsloth/registry/_qwen.py` |
| Gemma family | `unsloth/registry/_gemma.py` |
| Registry docs | `unsloth/registry/REGISTRY.md` |
