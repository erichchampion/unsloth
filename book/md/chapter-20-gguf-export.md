# Chapter 20: GGUF Export and Quantization

> *"From GPU memory to a single file you can run anywhere."*

---

## Introduction

GGUF (GGML Universal Format) is the standard format for running language models with llama.cpp — the most popular local inference engine. It enables running models on CPUs, Apple Silicon, and GPUs without Python or PyTorch. Unsloth integrates with llama.cpp to convert trained models into GGUF format with over 20 quantization methods, from lossless bf16 to aggressive 2-bit quantization. This chapter explains the conversion pipeline, the quantization zoo, and how Ollama integration works.

### What You'll Learn

- The full GGUF quantization method catalog
- The four-step conversion pipeline: save → install → convert → quantize
- llama.cpp installation and building
- SentencePiece tokenizer fixes for GGUF compatibility
- Ollama template integration for `ollama run` workflows

### Prerequisites

- The saving infrastructure from Chapter 19
- The chat templates from Chapter 11
- Basic understanding of model quantization concepts

---

## 20.1 GGUF Quantization Methods

Unsloth supports 20+ GGUF quantization methods. Here are the most commonly used:

| Method | Bits/Weight | Quality | Speed | Size (7B) | Recommendation |
|--------|------------|---------|-------|-----------|----------------|
| `bf16` | 16 | 100% | Slowest | ~14 GB | Maximum quality |
| `f16` | 16 | 100% | Slowest | ~14 GB | Maximum quality |
| `q8_0` | 8 | ~99.5% | Fast | ~7 GB | High quality |
| `q6_k` | 6 | ~99% | Faster | ~5.5 GB | High quality, smaller |
| `q5_k_m` | 5 | ~98% | Fast | ~4.5 GB | **Recommended** |
| `q4_k_m` | 4 | ~97% | Fast | ~4 GB | **Recommended** |
| `q3_k_m` | 3 | ~95% | Fast | ~3 GB | Aggressive |
| `q2_k` | 2 | ~90% | Fastest | ~2.5 GB | Very aggressive |

### Meta-Methods

Three convenience aliases select sensible defaults:

```python
"not_quantized"  # → f16 (fast conversion, large output)
"fast_quantized"  # → q8_0 (fast conversion, medium output)
"quantized"       # → q4_k_m (slow conversion, small output)
```

### Mixed Quantization

The `_k_m` variants (e.g., `q4_k_m`, `q5_k_m`) use **mixed precision** — attention and feed-forward layers use higher-bit quantization (Q6_K) for half their tensors while other tensors use the base quantization level:

```
q4_k_m: Q6_K for half of attention.wv + feed_forward.w2, else Q4_K
q5_k_m: Q6_K for half of attention.wv + feed_forward.w2, else Q5_K
```

This strategy preserves quality where it matters most (attention computation) while aggressively compressing less critical layers.

---

## 20.2 The Conversion Pipeline

GGUF export follows a four-step pipeline:

```
Step 1: Save model as merged 16-bit
  └─ unsloth_save_model(save_method="merged_16bit")
       Dequantize 4-bit → 16-bit, merge LoRA, save safetensors

Step 2: Install/find llama.cpp
  └─ install_llama_cpp() or use_local_gguf(path)
       Builds: llama-quantize, llama-cli, llama-server
       Checks: cURL support for remote model loading
       Flag: -DLLAMA_CURL=ON/OFF

Step 3: Convert HF → GGUF
  └─ convert_to_gguf() / _download_convert_hf_to_gguf()
       Converts safetensors to GGUF binary format
       Maps HF tensor names to GGUF tensor names
       Embeds tokenizer vocabulary

Step 4: Quantize GGUF
  └─ quantize_gguf(method="q4_k_m")
       Runs llama-quantize on the GGUF file
       Produces final quantized GGUF output
```

### Build Targets

Unsloth only builds the specific llama.cpp targets it needs (not the full project):

```python
LLAMA_CPP_TARGETS = [
    "llama-quantize",   # Quantization tool
    "llama-cli",        # Command-line inference
    "llama-server",     # HTTP inference server
]
```

---

## 20.3 SentencePiece Tokenizer Fixes

When converting models with SentencePiece tokenizers, the vocabulary stored in `tokenizer.model` may not include tokens added via `added_tokens.json`. The `fix_sentencepiece_gguf()` function (from `tokenizer_utils.py`) extends the SentencePiece vocabulary:

```python
def fix_sentencepiece_gguf(saved_location):
    # 1. Load tokenizer.model (protobuf format)
    # 2. Load added_tokens.json
    # 3. Verify token IDs are contiguous
    # 4. Extend tokenizer.model pieces with added_tokens
    # 5. Set type=USER_DEFINED, score=-1000.0
    # 6. Write updated tokenizer.model
```

Without this fix, the GGUF model would have a truncated vocabulary, causing generation errors when the model tries to emit tokens beyond the SentencePiece vocabulary.

---

## 20.4 Ollama Integration

Converted GGUF models automatically get Ollama-compatible Modelfile templates, enabling the `ollama run <model>` workflow:

```
Conversion output:
  my_model/
    ├─ model-q4_k_m.gguf       # Quantized weights
    ├─ Modelfile                # Ollama configuration
    └─ README.md                # Model card
```

The Modelfile includes:
- The GGUF file path
- The correct chat template for the model family
- System prompt (if applicable)
- Stop tokens

The template mapping comes from `MODEL_TO_OLLAMA_TEMPLATE_MAPPER` (Chapter 11), which maps over 100 model families to their Ollama Go-template equivalents.

---

## 20.5 Usage Example

```python
# Export to GGUF with q4_k_m quantization
model.save_pretrained_gguf(
    "my_model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)

# Export to multiple quantization levels
for method in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.save_pretrained_gguf(
        f"my_model_{method}",
        tokenizer,
        quantization_method = method,
    )

# Push GGUF directly to Hugging Face
model.save_pretrained_gguf(
    "username/my_model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",
    push_to_hub = True,
    token = "hf_...",
)
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| save_to_gguf() | `unsloth/save.py` |
| ALLOWED_QUANTS dictionary | `unsloth/save.py` (lines 114-141) |
| llama.cpp build/install | `unsloth_zoo/llama_cpp.py` |
| SentencePiece GGUF fix | `unsloth/tokenizer_utils.py` → `fix_sentencepiece_gguf()` |
| Ollama templates | `unsloth/ollama_template_mappers.py` |
