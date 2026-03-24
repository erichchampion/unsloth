# Chapter 14: The Trainer — UnslothTrainer, Packing, and Padding-Free

> *"The trainer is where configuration becomes computation."*

---

## Introduction

Unsloth's training infrastructure extends Hugging Face's TRL (Transformer Reinforcement Learning) library with two critical optimizations — sample packing and padding-free batching — plus a set of backwards-compatibility patches that keep existing code working across TRL version upgrades. The main classes, `UnslothTrainer` and `UnslothTrainingArguments`, are thin wrappers that add embedding-specific learning rates and auto-configure training for maximum efficiency.

This chapter traces the code through `unsloth/trainer.py` (484 lines) to explain how training is configured, optimized, and executed.

### What You'll Learn

- `UnslothTrainer` and `UnslothTrainingArguments` class structure
- Sample packing: what it is, when it activates, and why it's 2× faster
- Padding-free batching: the alternative to packing
- TRL backwards-compatibility patching
- The gradient accumulation fix for older transformers

### Prerequisites

- The SFTTrainer from TRL
- Understanding of training loops and gradient accumulation
- The model loading pipeline from Chapter 7

---

## 14.1 UnslothTrainingArguments

`UnslothTrainingArguments` extends TRL's `SFTConfig` (or `TrainingArguments` for older TRL) with a single addition — `embedding_learning_rate`:

```python
# trainer.py (lines 133-136)
class UnslothTrainingArguments(TrainingArguments):
    def __init__(self, embedding_learning_rate: float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_learning_rate = embedding_learning_rate
```

This allows embedding layers (input embeddings and language model head) to use a different learning rate than the rest of the network, which is important when training these layers alongside LoRA adapters.

---

## 14.2 UnslothTrainer

`UnslothTrainer` extends `SFTTrainer` with a custom optimizer creation method:

```python
# trainer.py (lines 182-198)
class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()
        # Create optimizer with separate parameter groups
        self.optimizer = _create_unsloth_optimizer(
            self.model, optimizer_cls, optimizer_kwargs, embedding_learning_rate
        )
        return self.optimizer
```

The `_create_unsloth_optimizer()` function (lines 139-179) creates two parameter groups:

| Group | Parameters | Learning Rate |
|-------|-----------|---------------|
| `non_embeddings` | All LoRA adapters, norms | Standard LR (e.g., 2e-4) |
| `embeddings` | `embed_tokens`, `lm_head` | `embedding_learning_rate` (e.g., 5e-5) |

Detection is based on the parameter name: any parameter ending in `modules_to_save.default.weight` is classified as an embedding parameter.

---

## 14.3 Sample Packing

Sample packing is Unsloth's signature training optimization. Instead of padding every training sample to the same length — wasting compute on `[PAD]` tokens — packing concatenates multiple samples into a single sequence:

```
Without packing (batch_size=3, max_length=8):
  [Sample A] [PAD] [PAD] [PAD]   ← 50% wasted
  [Sample B] [PAD] [PAD] [PAD] [PAD]   ← 63% wasted
  [Sample C] [PAD] [PAD]   ← 25% wasted

With packing (batch_size=1, max_length=8):
  [Sample A] [Sample B] [Sample C]   ← 0% wasted
  Attention mask ensures samples don't attend to each other
```

The result: **2× faster training and less VRAM usage**.

### When Packing Activates

Packing is controlled by `packing=True` in the training config. The `_patch_sft_trainer_auto_packing()` function (lines 319-452) wraps `SFTTrainer.__init__` to:

1. Check if the model type is compatible (blocklist: `gemma2`, `gpt_oss`)
2. Check if the input has a custom data collator (incompatible with packing)
3. Check if the model is a VLM (incompatible with packing)
4. If compatible, call `configure_sample_packing()` and `enable_sample_packing()`

### Packing Blocklist

Some architectures are incompatible with packing:

```python
# trainer.py (lines 57-60)
PADDING_FREE_BLOCKLIST = {
    "gemma2",   # slow_attention_softcapping has torch.compile issues
    "gpt_oss",  # FlexAttention doesn't handle padding_free correctly
}
```

---

## 14.4 Padding-Free Batching

Padding-free batching is an alternative to packing. Instead of concatenating samples, it passes variable-length sequences to the model without any padding, using attention masks to handle the variable lengths:

- **Auto-enabled** when `padding_free=None` (the default) and the model is compatible
- **Disabled** for VLMs, models in the blocklist, and when a custom data collator is used
- Can be **explicitly disabled** via the `UNSLOTH_DISABLE_AUTO_PADDING_FREE` environment variable

The auto-detection logic:

```python
# trainer.py (lines 69-76)
def _should_auto_padding_free(config):
    if config is None or _AUTO_PADDING_FREE_ENV_DISABLED:
        return False
    if getattr(config, "packing", False):  # Packing takes priority
        return False
    return getattr(config, "padding_free", None) is None
```

---

## 14.5 TRL Backwards Compatibility

TRL's API has changed significantly between versions. Unsloth patches all TRL trainer classes for backwards compatibility:

### Version Migration Issues

| TRL Change | Unsloth Fix |
|-----------|-------------|
| `tokenizer` → `processing_class` | Auto-rename in kwargs |
| `TrainingArguments` → `*Config` | Auto-convert args to Config objects |
| Thin wrapper `__init__(*args, **kwargs)` | MRO walking to find real parameters |

The `_patch_trl_trainer()` function (lines 455-483) discovers all TRL trainer/config pairs dynamically:

```python
def _patch_trl_trainer():
    trl_trainers = set(x[:-len("Trainer")] for x in dir(trl.trainer) if x.endswith("Trainer"))
    trl_configs = set(x[:-len("Config")] for x in dir(trl.trainer) if x.endswith("Config"))
    # Patch only classes that have both a Trainer and a Config
    for x in (trl_trainers & trl_configs):
        trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)
```

---

## 14.6 Gradient Accumulation Fix

For older transformers (≤ 4.45.2), there was a bug in gradient accumulation that caused incorrect loss scaling. Unsloth provides a custom `unsloth_train()` function that fixes this:

```python
# trainer.py (lines 105-124)
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)  # Standard training
else:
    def unsloth_train(trainer, *args, **kwargs):
        return _unsloth_train(trainer)  # Custom fixed training loop
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| UnslothTrainer | `unsloth/trainer.py` |
| Sample packing utilities | `unsloth/utils/packing.py` |
| Padding-free config | `unsloth/utils/__init__.py` |
| Training utils (gradient fix) | `unsloth_zoo/training_utils.py` |
| TRL patching | `unsloth/trainer.py` → `_patch_trl_trainer()` |
