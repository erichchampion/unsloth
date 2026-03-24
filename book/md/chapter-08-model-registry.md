# Chapter 8: The Model Registry — Mapping Names to Weights

> *"A name is just a pointer to a quantized checkpoint."*

---

## Introduction

When a user writes `FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B-Instruct")`, how does Unsloth know where to find the weights? The answer is the **model registry** — a declarative system that maps user-friendly model names to their Hugging Face repository paths, tracks available quantization variants, and generates all valid name combinations automatically.

This chapter dissects the registry's data structures, shows how family modules declare models, and traces the registration flow from `ModelMeta` definitions to the global `MODEL_REGISTRY` dictionary.

### What You'll Learn

- The `ModelInfo` and `ModelMeta` data structures
- How `QuantType` controls model path construction
- The registration flow: `ModelMeta` → `_register_models()` → `MODEL_REGISTRY`
- How family modules (`_llama.py`, `_qwen.py`, etc.) declare model families
- The name construction algorithm and its output

### Prerequisites

- Understanding of Hugging Face model repositories and naming conventions
- Familiarity with quantization concepts (4-bit, GGUF, FP8)

---

## 8.1 Registry Data Structures

### QuantType

The `QuantType` enum defines the five quantization formats Unsloth supports:

```python
# registry/registry.py (lines 6-11)
class QuantType(Enum):
    BNB = "bnb"           # bitsandbytes 4-bit NF4
    UNSLOTH = "unsloth"   # Dynamic quantization (Unsloth's own)
    GGUF = "GGUF"         # GGUF format for llama.cpp
    NONE = "none"         # No quantization (original weights)
    BF16 = "bf16"         # Bfloat16 (used for DeepSeek V3)
```

Each `QuantType` maps to a tag that gets appended to the model name:

```python
QUANT_TAG_MAP = {
    QuantType.BNB:     "bnb-4bit",          # → Llama-3.2-1B-Instruct-bnb-4bit
    QuantType.UNSLOTH: "unsloth-bnb-4bit",  # → Llama-3.2-1B-Instruct-unsloth-bnb-4bit
    QuantType.GGUF:    "GGUF",              # → Llama-3.2-1B-Instruct-GGUF
    QuantType.NONE:    None,                # → Llama-3.2-1B-Instruct (no suffix)
    QuantType.BF16:    "bf16",              # → DeepSeek-R1-bf16
}
```

### ModelInfo

`ModelInfo` is a dataclass that represents a single registered model variant:

```python
# registry/registry.py (lines 30-75)
@dataclass
class ModelInfo:
    org: str               # e.g., "unsloth"
    base_name: str         # e.g., "Llama"
    version: str           # e.g., "3.2"
    size: int              # e.g., 1 (for 1B)
    name: str = None       # Full name, auto-constructed
    is_multimodal: bool = False
    instruct_tag: str = None    # e.g., "Instruct"
    quant_type: QuantType = None
    description: str = None

    @property
    def model_path(self) -> str:
        return f"{self.org}/{self.name}"
        # → "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
```

### ModelMeta

`ModelMeta` is the declarative template that describes an entire model family. A single `ModelMeta` can generate dozens of `ModelInfo` entries:

```python
# registry/registry.py (lines 78-89)
@dataclass
class ModelMeta:
    org: str                    # Original author (e.g., "meta-llama")
    base_name: str              # Family name (e.g., "Llama")
    model_version: str          # Version string (e.g., "3.2")
    model_info_cls: type        # ModelInfo subclass for name construction
    model_sizes: list[str]      # Available sizes ["1", "3", "8"]
    instruct_tags: list[str]    # [None, "Instruct"]
    quant_types: list[QuantType] | dict[str, list[QuantType]]
    is_multimodal: bool = False
```

---

## 8.2 Family Modules

Each model family has a dedicated Python file that defines its `ModelMeta` instances:

| File | Family | Sizes | Versions |
|------|--------|-------|----------|
| `_llama.py` | Llama | 1B, 3B, 8B, 11B, 90B | 3.1, 3.2, 3.2-Vision |
| `_qwen.py` | Qwen | 0.5B–72B | 2, 2.5, 3, 3-MoE |
| `_gemma.py` | Gemma | 2B–27B | 1, 2, 3 |
| `_mistral.py` | Mistral, Ministral | 3B, 7B, 8B, 24B | v0.1–v0.4, Nemo |
| `_phi.py` | Phi | 3.8B, 14B | 3, 4 |
| `_deepseek.py` | DeepSeek | 1.5B–236B | V2, V3, R1 |

### Example: The Llama Family

Here is how `_llama.py` declares the Llama 3.2 family:

```python
# registry/_llama.py (lines 38-60)
class LlamaModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B"  # → "Llama-3.2-1B"
        return super().construct_model_name(
            base_name, version, size, quant_type, instruct_tag, key
        )

LlamaMeta_3_2_Instruct = ModelMeta(
    org = "meta-llama",
    base_name = "Llama",
    instruct_tags = ["Instruct"],
    model_version = "3.2",
    model_sizes = ["1", "3"],
    model_info_cls = LlamaModelInfo,
    quant_types = [QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)
```

This single `ModelMeta` generates 8 registry entries: 2 sizes × 4 quantization types.

---

## 8.3 The Registration Flow

Registration follows a three-step process:

```
1. Family module defines ModelMeta instances
       │
2. register_*_models() calls _register_models(ModelMeta)
       │
3. _register_models() iterates all combinations:
       for each size:
           for each instruct_tag:
               for each quant_type:
                   register_model() → MODEL_REGISTRY[key] = ModelInfo
```

The `_register_models()` function (lines 150-191 of `registry.py`) handles the combinatorial expansion:

```python
def _register_models(model_meta, include_original_model=False):
    for size in model_meta.model_sizes:
        for instruct_tag in model_meta.instruct_tags:
            for quant_type in _quant_types:
                register_model(
                    org="unsloth",  # All quantized models under unsloth org
                    base_name=model_meta.base_name,
                    version=model_meta.model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=quant_type,
                )
```

Note that all registered models use `org="unsloth"` — even though the original model may come from `meta-llama` or `Qwen`. This is because Unsloth hosts its own pre-quantized copies on the Hugging Face Hub.

---

## 8.4 Name Construction

The name construction follows a consistent pattern:

```
{base_name}-{version}-{size}B[-{instruct_tag}][-{quant_tag}]

Examples:
  Llama-3.2-1B                           (base, no quant)
  Llama-3.2-1B-Instruct                  (instruct, no quant)
  Llama-3.2-1B-Instruct-bnb-4bit         (instruct + BNB)
  Llama-3.2-1B-Instruct-unsloth-bnb-4bit (instruct + dynamic)
  Llama-3.2-1B-Instruct-GGUF             (instruct + GGUF)
```

The `model_path` property prepends the organization:

```
unsloth/Llama-3.2-1B-Instruct-bnb-4bit
```

### Per-Size Quantization Types

Some models have different quantization options per size. The Llama 3.2 Vision family demonstrates this with a dict-based `quant_types`:

```python
LlamaMeta_3_2_Vision = ModelMeta(
    model_sizes = ["11", "90"],
    quant_types = {
        "11": [QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
        "90": [QuantType.NONE],  # 90B only in full precision
    },
)
```

---

## 8.5 Registry Verification

Each family module includes a `__main__` block that verifies all registered models actually exist on Hugging Face:

```python
if __name__ == "__main__":
    register_llama_models(include_original_model=True)
    for model_id, model_info in MODEL_REGISTRY.items():
        info = _check_model_info(model_id)
        print(f"{'✓' if info else '✗'} {model_id}")
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Registry core (ModelInfo, ModelMeta, register) | `unsloth/registry/registry.py` |
| Llama family | `unsloth/registry/_llama.py` |
| Qwen family | `unsloth/registry/_qwen.py` |
| Gemma family | `unsloth/registry/_gemma.py` |
| Mistral family | `unsloth/registry/_mistral.py` |
| DeepSeek family | `unsloth/registry/_deepseek.py` |
| Phi family | `unsloth/registry/_phi.py` |
| Registry docs | `unsloth/registry/REGISTRY.md` |
