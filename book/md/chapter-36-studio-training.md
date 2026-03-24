# Chapter 36: Studio Training — Configuration and Observability

> *"Watch your model learn in real time."*

---

## Introduction

Training a language model involves dozens of hyperparameters — learning rate, batch size, LoRA rank, gradient accumulation steps, warmup ratio, and more. Studio's training interface provides a graphical configuration panel that maps directly to `UnslothTrainingArguments` (Chapter 14), plus real-time observability with live loss curves, GPU utilization graphs, and training speed metrics. This chapter covers the training workflow from configuration through monitoring to completion.

### What You'll Learn

- The training configuration UI and its backend mapping
- Real-time metrics: loss curves, learning rate schedule, GPU stats
- Training lifecycle: start, pause, resume, stop
- Integration with SFT, GRPO, and other TRL trainers
- Platform-specific training capabilities

### Prerequisites

- The trainer from Chapter 14
- LoRA/QLoRA from Chapter 12
- The studio architecture from Chapter 34

---

## 36.1 Training Configuration

The Studio UI presents training parameters as a form, organized into sections:

### Model Settings

| Parameter | Maps To | Default |
|-----------|---------|---------|
| Base model | `model_name` | — |
| LoRA rank | `r` | 16 |
| LoRA alpha | `lora_alpha` | 16 |
| Target modules | `target_modules` | Auto-detected |
| Load in 4-bit | `load_in_4bit` | True |

### Training Hyperparameters

| Parameter | Maps To | Default |
|-----------|---------|---------|
| Learning rate | `learning_rate` | 2e-4 |
| Batch size | `per_device_train_batch_size` | 2 |
| Gradient accumulation | `gradient_accumulation_steps` | 4 |
| Max steps | `max_steps` | 60 |
| Warmup steps | `warmup_steps` | 5 |
| Weight decay | `weight_decay` | 0.01 |
| Max sequence length | `max_seq_length` | 2048 |

### Dataset Settings

| Parameter | Description |
|-----------|-------------|
| Dataset source | HF Hub name or uploaded file |
| Text column | Column containing training text |
| Template | Chat template to apply |
| Packing | Enable sample packing (Chapter 14) |

All parameters are validated before training starts, and sensible defaults are pre-filled based on the selected model size.

---

## 36.2 Training Lifecycle

```
Configure → Validate → Start → Monitor → Complete/Stop
    │                    │                      │
    └── Edit params      ├── Pause/Resume       ├── Save checkpoint
                         └── View metrics        └── Export model
```

### Backend Orchestration

The training route (`routes/training.py`) delegates to `core/training/`:

```python
# Simplified training start flow:
@router.post("/api/training/start")
async def start_training(config: TrainingConfig):
    # 1. Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(config.model_name, ...)

    # 2. Apply LoRA
    model = FastLanguageModel.get_peft_model(model, r=config.lora_rank, ...)

    # 3. Load dataset
    dataset = load_dataset(config.dataset_name, ...)

    # 4. Create trainer
    trainer = UnslothTrainer(model=model, args=training_args, ...)

    # 5. Start training in background thread
    training_task = asyncio.create_task(trainer.train())
    return {"status": "started", "task_id": training_task.id}
```

---

## 36.3 Real-Time Observability

Studio streams training metrics to the frontend via WebSocket:

| Metric | Update Frequency | Visualization |
|--------|-----------------|---------------|
| Training loss | Every step | Line chart |
| Learning rate | Every step | Line chart (overlay) |
| GPU memory usage | Every 5 seconds | Area chart |
| GPU utilization | Every 5 seconds | Gauge |
| Tokens/second | Every step | Counter |
| ETA | Every step | Countdown timer |
| Current step / total | Every step | Progress bar |

### Loss Smoothing

The loss chart displays both raw step-level loss (light, semi-transparent) and an exponentially smoothed moving average (bold), making trends visible even with noisy per-step values.

---

## 36.4 Trainer Integration

Studio supports multiple TRL trainers through the same UI:

| Trainer | Use Case | Config Key |
|---------|----------|-----------|
| `SFTTrainer` | Supervised fine-tuning | `trainer_type="sft"` |
| `GRPOTrainer` | RL with reward functions | `trainer_type="grpo"` |
| `DPOTrainer` | Direct Preference Optimization | `trainer_type="dpo"` |

Each trainer type reveals relevant configuration options (e.g., GRPO shows the reward function editor, DPO shows reference model settings).

---

## 36.5 Platform Support

| Platform | Training | GPU Monitoring | Live Metrics |
|----------|----------|---------------|--------------|
| NVIDIA Linux | ✅ Full | ✅ nvidia-smi | ✅ Full |
| AMD Linux | ✅ Core CLI | ⚠️ rocm-smi | ✅ Full |
| macOS (Apple Silicon) | ⚠️ MLX (coming) | ✅ Metal stats | ✅ Full |
| Google Colab | ✅ Full | ✅ nvidia-smi | ✅ via tunnel |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Training routes | `studio/backend/routes/training.py` |
| Training core | `studio/backend/core/training/` |
| UnslothTrainer | `unsloth/trainer.py` |
| Training arguments | `unsloth/trainer.py` → `UnslothTrainingArguments` |
