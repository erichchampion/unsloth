# Chapter 5: The CLI Entry Point — Typer Commands and Configuration

---

## Introduction

The `unsloth` command-line tool is the primary way to interact with the project. Built with Typer, it exposes commands for training, inference, export, and launching Studio.

### What You'll Learn

- How the CLI is registered via `pyproject.toml` console scripts
- The command structure: `train`, `inference`, `export`, `list-checkpoints`, `ui`, `studio`
- Training configuration via `config.py`
- Shared CLI options in `options.py`

---

## Notes & Key Points

### 5.1 Entry Point Registration

```toml
# pyproject.toml
[project.scripts]
unsloth = "unsloth_cli:app"
```

- The Typer `app` is defined in `unsloth_cli/__init__.py`
- Commands are registered via `app.command()` decorators
- The `studio` subcommand uses `app.add_typer()` for a nested command group

### 5.2 Available Commands

| Command | Module | Purpose |
|---------|--------|---------|
| `unsloth train` | `commands/train.py` | Fine-tune a model from CLI |
| `unsloth inference` | `commands/inference.py` | Run inference on a model |
| `unsloth export` | `commands/export.py` | Export model to GGUF or other formats |
| `unsloth list-checkpoints` | `commands/export.py` | List available training checkpoints |
| `unsloth ui` | `commands/ui.py` | Launch simple UI |
| `unsloth studio` | `commands/studio.py` | Studio subcommands (setup, launch) |

### 5.3 Configuration

- `config.py` — Defines training configuration as a Pydantic/dataclass model
- `options.py` — Shared Typer `Option` definitions (model name, output dir, etc.)

### 5.4 CLI → Core Library Bridge

- CLI commands import directly from `unsloth` core: `FastLanguageModel`, `unsloth_save_model`, etc.
- The `train` command constructs arguments and passes them to `UnslothTrainer`
- The `inference` command loads a model via `FastLanguageModel.from_pretrained()` and runs generation

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| CLI app | `unsloth_cli/__init__.py` |
| Train command | `unsloth_cli/commands/train.py` |
| Inference command | `unsloth_cli/commands/inference.py` |
| Export command | `unsloth_cli/commands/export.py` |
| Configuration | `unsloth_cli/config.py` |
| Options | `unsloth_cli/options.py` |
