# Chapter 5: The CLI Entry Point — Typer Commands and Configuration

> *"One command to train, one to infer, one to export — and one to rule them all."*

---

## Introduction

The `unsloth` command-line tool is the bridge between the user's terminal and the Core library's Python API. Built with Typer — a type-hint-driven framework layered on top of Click — it exposes six commands that cover the complete model lifecycle: training, inference, export, checkpoint management, and Studio launch.

This chapter dissects the CLI's architecture, showing how commands are registered, how configuration flows from YAML files through Pydantic models to the training backend, and how the decorator-based option system automatically generates CLI flags from config schemas.

### What You'll Learn

- How the CLI is registered as a console script via `pyproject.toml`
- The command structure and what each command does
- The Pydantic configuration system (`Config`, `TrainingConfig`, `LoraConfig`, etc.)
- How `options.py` auto-generates CLI flags from Pydantic models
- The train command's end-to-end flow from CLI invocation to training loop

### Prerequisites

- Basic familiarity with CLI tools and argument parsing
- Understanding of Pydantic model validation
- The Core library concepts from Chapter 1

---

## 5.1 Entry Point Registration

The CLI is registered as a console script in `pyproject.toml`:

```toml
[project.scripts]
unsloth = "unsloth_cli:app"
```

When you type `unsloth` in your terminal, Python executes `unsloth_cli/__init__.py`, which imports all command modules and registers them with the Typer app:

```python
# unsloth_cli/__init__.py (complete file — just 23 lines)
import typer

from unsloth_cli.commands.train import train
from unsloth_cli.commands.inference import inference
from unsloth_cli.commands.export import export, list_checkpoints
from unsloth_cli.commands.ui import ui
from unsloth_cli.commands.studio import studio_app

app = typer.Typer(
    help = "Command-line interface for Unsloth training, inference, and export.",
    context_settings = {"help_option_names": ["-h", "--help"]},
)

app.command()(train)
app.command()(inference)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.command()(ui)
app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")
```

Notice the two registration patterns: `app.command()` for flat commands and `app.add_typer()` for the `studio` subcommand group. This means `unsloth studio` is itself a Typer app with its own sub-commands (like `unsloth studio setup`).

---

## 5.2 Available Commands

| Command | Module | Purpose |
|---------|--------|---------|
| `unsloth train` | `commands/train.py` (145 lines) | Fine-tune a model with LoRA or full fine-tuning |
| `unsloth inference` | `commands/inference.py` (67 lines) | Run inference on a loaded model |
| `unsloth export` | `commands/export.py` (133 lines) | Export to merged-16bit, merged-4bit, GGUF, or LoRA |
| `unsloth list-checkpoints` | `commands/export.py` | Scan and list training checkpoints |
| `unsloth ui` | `commands/ui.py` (96 lines) | Launch a simple terminal UI |
| `unsloth studio` | `commands/studio.py` (200 lines) | Studio sub-commands: setup, launch, configure |

---

## 5.3 The Configuration System

The CLI's configuration is built on a hierarchy of Pydantic models defined in `config.py`. This gives you type validation, default values, and automatic serialization for free:

```python
# unsloth_cli/config.py — the configuration hierarchy
class Config(BaseModel):
    model: Optional[str] = None
    data: DataConfig         # dataset, local_dataset, format_type
    training: TrainingConfig # max_seq_length, learning_rate, batch_size, etc.
    lora: LoraConfig         # lora_r, lora_alpha, target_modules, etc.
    logging: LoggingConfig   # wandb, tensorboard settings
```

### DataConfig

Controls where training data comes from:

```python
class DataConfig(BaseModel):
    dataset: Optional[str] = None          # HF dataset name
    local_dataset: Optional[List[str]] = None  # Local file paths
    format_type: Literal["auto", "alpaca", "chatml", "sharegpt"] = "auto"
```

### TrainingConfig

All hyperparameters for the training loop:

```python
class TrainingConfig(BaseModel):
    training_type: Literal["lora", "full"] = "lora"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    weight_decay: float = 0.01
    random_seed: int = 3407
    packing: bool = False
    gradient_checkpointing: Literal["unsloth", "true", "none"] = "unsloth"
```

### LoraConfig

LoRA adapter parameters:

```python
class LoraConfig(BaseModel):
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
```

Configuration can come from a YAML file, CLI flags, or both — CLI flags override file values:

```bash
# From YAML file only
unsloth train --config training.yaml

# CLI flags override config values
unsloth train --config training.yaml --learning-rate 1e-5 --num-epochs 5

# All CLI flags, no config file
unsloth train --model unsloth/Llama-3.2-1B-Instruct --dataset alpaca_gpt4
```

The `load_config()` function handles both YAML and JSON:

```python
# config.py (lines 132-149)
def load_config(path: Optional[Path]) -> Config:
    if not path:
        return Config()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    return Config(**data)
```

---

## 5.4 Auto-Generated CLI Options: `options.py`

The most interesting piece of the CLI architecture is `options.py`, which contains a decorator that **automatically generates Typer CLI flags from a Pydantic model's fields**. This eliminates the tedious boilerplate of declaring every option twice (once in the config, once in the CLI).

```python
# options.py — the key decorator
@add_options_from_config(Config)
def train(config: Optional[Path] = ..., config_overrides: dict = None):
    ...
```

The `add_options_from_config` decorator:

1. **Introspects the Pydantic model** — iterates all fields, including nested sub-models (`DataConfig`, `TrainingConfig`, `LoraConfig`, `LoggingConfig`)
2. **Generates CLI flags** — converts `learning_rate` → `--learning-rate`, `load_in_4bit` → `--load-in-4bit/--no-load-in-4bit`
3. **Builds a new function signature** — injects the generated parameters into the function's signature so Typer sees them
4. **Collects overrides at runtime** — when the function is called, the wrapper extracts all provided CLI values into a `config_overrides` dict

This means adding a new config field to `TrainingConfig` automatically creates a corresponding CLI flag — no extra code needed.

---

## 5.5 The Train Command: End-to-End Flow

The `train` command (`commands/train.py`) is the most complex CLI command. Here is its execution flow:

```
unsloth train --model ... --dataset ... [--config training.yaml]
    │
    ├─ 1. load_config(config_file)          Load YAML/JSON or defaults
    ├─ 2. cfg.apply_overrides(**cli_flags)   CLI flags override config values
    ├─ 3. Validate tokens (HF, WandB)       Check env vars and CLI args
    ├─ 4. Check --dry-run                   If set, dump config and exit
    ├─ 5. Validate model + dataset exist    Early error if missing
    ├─ 6. Detect LoRA adapter               If checkpoint has adapter_config.json
    │
    ├─ 7. UnslothTrainer()                  Instantiate trainer backend
    ├─ 8. trainer.load_model(...)           Load model via FastLanguageModel
    ├─ 9. trainer.prepare_model_for_training(...)  Apply LoRA / gradient checkpointing
    ├─ 10. trainer.load_and_format_dataset(...)    Load + format dataset
    ├─ 11. trainer.start_training(...)      Launch training in background thread
    │
    └─ 12. Poll until complete              Sleep loop with Ctrl+C handling
```

The trainer runs in a background thread, allowing the CLI to handle `KeyboardInterrupt` gracefully and stop training cleanly.

---

## 5.6 The Export Command

The `export` command supports four output formats:

```bash
unsloth export ./outputs/checkpoint-100 ./exported --format gguf --quantization q4_k_m
```

| Format | Flag | Description |
|--------|------|-------------|
| `merged-16bit` | `--format merged-16bit` | Full-precision merged weights |
| `merged-4bit` | `--format merged-4bit` | 4-bit quantized merged weights |
| `gguf` | `--format gguf` | GGUF file for llama.cpp / Ollama |
| `lora` | `--format lora` | LoRA adapter only |

GGUF quantization methods: `q4_k_m`, `q5_k_m`, `q8_0`, `f16`.

All export formats support `--push-to-hub` to upload directly to Hugging Face Hub.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| CLI app definition | `unsloth_cli/__init__.py` |
| Train command | `unsloth_cli/commands/train.py` |
| Inference command | `unsloth_cli/commands/inference.py` |
| Export command | `unsloth_cli/commands/export.py` |
| Studio commands | `unsloth_cli/commands/studio.py` |
| UI command | `unsloth_cli/commands/ui.py` |
| Configuration models | `unsloth_cli/config.py` |
| Auto option generation | `unsloth_cli/options.py` |
| Console script entry | `pyproject.toml` → `[project.scripts]` |
