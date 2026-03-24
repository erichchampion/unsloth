# Chapter 16: Vision and Multimodal Fine-Tuning

---

## Introduction

Unsloth supports fine-tuning vision-language models (VLMs) including Gemma 3, Qwen3.5, Llama 3.2 Vision, and Mistral Ministral. This chapter covers the multimodal training pipeline.

### What You'll Learn

- How VLMs are loaded and patched
- Vision data collation with `UnslothVisionDataCollator`
- Image/audio preprocessing with `process_vision_info()`
- Architecture-specific vision handling

---

## Notes & Key Points

### 16.1 Vision Model Loading

- `is_multimodal` flag in the model registry marks vision models
- `vision.py` (64K) — Vision-specific model patches and processing
- Uses HF `ProcessorMixin` for combined image+text tokenization
- Auto-detects VLMs via `ForConditionalGeneration` architecture suffix

### 16.2 Vision Data Handling

- `UnslothVisionDataCollator` — from `unsloth_zoo/vision_utils.py`
- `process_vision_info()` — preprocesses images for model input
- Handles: images (JPEG, PNG), PDFs (via rendering), documents

### 16.3 Packing Limitations

- VLMs disable sample packing and padding-free automatically
- Custom data collators are required for vision inputs
- The trainer auto-detects VLMs and adjusts settings

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Vision patches | `unsloth/models/vision.py` |
| Vision data collator | `unsloth_zoo/vision_utils.py` (external) |
| VLM detection | `unsloth/trainer.py` (lines 340-356) |
