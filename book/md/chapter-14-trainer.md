# Chapter 14: The Trainer — UnslothTrainer, Packing, and Padding-Free

---

## Introduction

`UnslothTrainer` extends TRL's `SFTTrainer` with automatic packing, padding-free batching, and embedding-specific learning rates. The trainer module also patches all TRL trainers for backwards compatibility.

### What You'll Learn

- `UnslothTrainer` and `UnslothTrainingArguments`
- Sample packing: what it is and when it activates
- Padding-free batching: auto-detection and configuration
- TRL backwards compatibility patching

---

## Notes & Key Points

### 14.1 UnslothTrainer

```python
class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        # Supports separate learning rate for embeddings
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
```

- Extends `SFTTrainer` with `embedding_learning_rate` support
- `UnslothTrainingArguments` adds `embedding_learning_rate` to `SFTConfig`

### 14.2 Sample Packing

- Concatenates multiple training samples into a single sequence
- Avoids wasting compute on padding tokens
- **"2x faster and uses less VRAM"**
- Controlled by `packing=True` in training config
- Auto-enabled in SFT when not blocked by data collators or VLMs

### 14.3 Padding-Free Batching

- Alternative to packing: variable-length sequences without padding
- Auto-enabled when `padding_free=None` (default) and compatible model
- Blocklist: `gemma2` (softcapping issues), `gpt_oss` (FlexAttention issues)
- Disabled for VLMs and custom data collators
- Uses `configure_padding_free()` from `unsloth.utils`

### 14.4 TRL Patching

- `_patch_trl_trainer()` wraps all TRL trainer classes for backwards compatibility
- Handles the `tokenizer` → `processing_class` rename in newer TRL
- Handles `TrainingArguments` → `*Config` migration in TRL ≥ 0.13
- Patches SFTTrainer's `__init__` to auto-enable packing/padding-free

### 14.5 Gradient Accumulation Fix

- For older transformers (≤ 4.45.2), provides custom `unsloth_train()` 
- Fixes gradient accumulation bug that existed in older versions

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Trainer class | `unsloth/trainer.py` |
| Packing utilities | `unsloth/utils/packing.py` |
| Padding-free config | `unsloth/utils/__init__.py` |
| Training utils | `unsloth_zoo/training_utils.py` (external) |
