# Chapter 16: Vision and Multimodal Fine-Tuning

> *"Teaching models to see as well as read."*

---

## Introduction

Vision-Language Models (VLMs) combine a language model with a vision encoder, enabling the model to understand images alongside text. Unsloth supports fine-tuning several VLM families — Gemma 3, Qwen 2.5 VL, Llama 3.2 Vision, and Pixtral — with the same LoRA/QLoRA optimizations available for text-only models. The vision pipeline requires specialized data handling, image preprocessing, and model patching that this chapter explores in detail.

### What You'll Learn

- How VLMs are detected and loaded
- The `vision.py` module and its patching strategy
- `UnslothVisionDataCollator` for image+text batching
- Image preprocessing with `process_vision_info()`
- Limitations: why packing and padding-free are disabled for VLMs

### Prerequisites

- The model loading pipeline from Chapter 7
- The trainer infrastructure from Chapter 14
- Basic understanding of vision encoders (ViT, SigLIP)

---

## 16.1 VLM Detection

VLMs are identified through two mechanisms during model loading:

```python
# trainer.py (lines 340-356)
# Check 1: Architecture name ends with "ForConditionalGeneration"
architectures = getattr(model_config, "architectures", [])
is_vlm = any(x.endswith("ForConditionalGeneration") for x in architectures)

# Check 2: Model config has a vision_config attribute
is_vlm = is_vlm or hasattr(model_config, "vision_config")
```

In the registry, VLMs are marked with `is_multimodal=True`:

```python
LlamaMeta_3_2_Vision = ModelMeta(
    base_name = "Llama",
    model_version = "3.2",
    model_sizes = ["11", "90"],
    model_info_cls = LlamaVisionModelInfo,
    is_multimodal = True,
    quant_types = {
        "11": [QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
        "90": [QuantType.NONE],
    },
)
```

---

## 16.2 The Vision Module

The `models/vision.py` file (64K) handles all vision-specific model modifications:

```
vision.py responsibilities:
  ├─ Vision encoder patching (optimize ViT/SigLIP forward pass)
  ├─ Connector patching (vision→language projection)
  ├─ Image preprocessing (resize, normalize, pad)
  ├─ Multi-image handling (multiple images per conversation turn)
  └─ Audio handling (for models that support audio input)
```

### Supported VLM Architectures

| Model | Vision Encoder | Connector | Language Model |
|-------|---------------|-----------|----------------|
| Llama 3.2 Vision | ViT | Cross-attention | Llama 3.2 |
| Gemma 3 | SigLIP | Linear projection | Gemma 3 |
| Qwen 2.5 VL | ViT | Resampler | Qwen 2.5 |
| Pixtral | Custom ViT | MLP | Mistral |

---

## 16.3 Vision Data Handling

### UnslothVisionDataCollator

The `UnslothVisionDataCollator` (from `unsloth_zoo/vision_utils.py`) handles the complexities of batching multimodal data:

```python
from unsloth_zoo.vision_utils import UnslothVisionDataCollator

data_collator = UnslothVisionDataCollator(processor)
```

It handles:
- **Image pixel values** — padding images to the same resolution within a batch
- **Attention masks** — marking which tokens correspond to image vs. text
- **Image position IDs** — indicating where images appear in the token sequence
- **Label masking** — setting labels to -100 for image tokens (don't compute loss on them)

### process_vision_info()

This utility function preprocesses images from various sources:

```python
from unsloth_zoo.vision_utils import process_vision_info

# Handles:
# - Local file paths
# - URLs (downloads automatically)
# - PIL Image objects
# - PDF pages (renders to images)
# - Base64-encoded images
```

---

## 16.4 Training Limitations for VLMs

Several training optimizations are automatically disabled when training VLMs:

| Feature | Status with VLMs | Reason |
|---------|-----------------|--------|
| Sample packing | ❌ Disabled | Variable image sizes make concatenation unreliable |
| Padding-free | ❌ Disabled | Image token padding requires consistent handling |
| Standard data collator | ❌ Replaced | Must use `UnslothVisionDataCollator` |
| `flash_attention_2` | ✅ Enabled | Optimized for mixed modality attention |

The disabling happens in the SFTTrainer wrapper (see Chapter 14, Section 14.3):

```python
blocked = (
    isinstance(processing_class, ProcessorMixin)
    or is_vlm
    or ...
)
if blocked:
    config.packing = False
    config.padding_free = False
```

---

## 16.5 VLM Fine-Tuning Example

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)

# Use the vision data collator
from unsloth_zoo.vision_utils import UnslothVisionDataCollator
data_collator = UnslothVisionDataCollator(tokenizer)

trainer = SFTTrainer(
    model = model,
    train_dataset = vision_dataset,
    data_collator = data_collator,
    args = SFTConfig(output_dir="./vision_output"),
)
trainer.train()
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Vision model patches | `unsloth/models/vision.py` |
| Vision data collator | `unsloth_zoo/vision_utils.py` |
| VLM detection | `unsloth/trainer.py` (lines 340-356) |
| VLM registry entries | `unsloth/registry/_llama.py` (Vision models) |
