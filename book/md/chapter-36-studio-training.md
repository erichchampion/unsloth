# Chapter 36: Studio Training — Configuration and Observability

---

## Introduction

Studio provides a graphical interface for configuring and monitoring training runs, with live loss curves, GPU utilization graphs, and training parameter editing.

### What You'll Learn

- Training configuration via the Studio UI
- Real-time training metrics and observability
- GPU usage monitoring
- Integration with the core `UnslothTrainer`

---

## Notes & Key Points

### 36.1 Training Configuration

- GUI for setting: model, dataset, learning rate, epochs, LoRA rank, etc.
- Maps to `UnslothTrainingArguments` and `SFTConfig` internally
- Supports SFT, GRPO, and other TRL trainers

### 36.2 Observability

- Live training loss graphs
- GPU memory usage tracking
- Step/epoch progress indicators
- Customizable metric graphs

### 36.3 Platform Support

- NVIDIA: full training support
- AMD: training via Core CLI, Studio support coming
- macOS: chat and data recipes only; MLX training coming soon

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Studio backend | `studio/backend/` |
| Training CLI | `unsloth_cli/commands/train.py` |
| Trainer | `unsloth/trainer.py` |
