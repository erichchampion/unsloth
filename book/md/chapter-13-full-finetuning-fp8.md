# Chapter 13: Full Fine-Tuning and FP8 Training

---

## Introduction

Beyond LoRA, Unsloth supports full fine-tuning (updating all weights) and FP8 training for maximum efficiency on modern GPUs. These modes route through the `FastModel` code path.

### What You'll Learn

- When to use full fine-tuning vs. LoRA
- FP8 training modes: per-tensor vs. block quantization
- TorchAO integration for FP8
- The `FastModel` fallback path

---

## Notes & Key Points

### 13.1 Full Fine-Tuning

- Enabled with `full_finetuning=True` in `from_pretrained()`
- Routes to `FastModel.from_pretrained()` instead of `FastLanguageModel`
- Uses `torch.compile` for optimization instead of hand-written kernels
- All model parameters are trainable (higher VRAM usage)

### 13.2 FP8 Training

- `load_in_fp8=True` — enables 8-bit floating point quantization
- `load_in_fp8='block'` — block-level FP8 quantization
- `_get_fp8_mode_and_check_settings()` validates FP8 compatibility
- `_offline_quantize_to_fp8()` quantizes the model weights
- Uses TorchAO's FP8 quantization backend

### 13.3 QAT (Quantization-Aware Training)

- `qat_scheme` parameter — enables quantization-aware training
- `_prepare_model_for_qat()` configures fake quantization layers

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastModel fallback | `unsloth/models/loader.py` |
| FP8 utilities | `unsloth/models/loader_utils.py` |
| FP8 kernels | `unsloth/kernels/fp8.py` |
| TorchAO conversion | `unsloth/models/_utils.py` → `_convert_torchao_model` |
