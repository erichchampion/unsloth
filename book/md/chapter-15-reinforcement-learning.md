# Chapter 15: Reinforcement Learning — GRPO and RL Workflows

---

## Introduction

Unsloth provides the most VRAM-efficient RL library for LLMs, with up to 80% less VRAM for GRPO. This chapter covers the RL infrastructure including reward modeling, policy optimization, and sandboxed code execution.

### What You'll Learn

- GRPO (Group Relative Policy Optimization) workflow
- RL-specific model patches in `rl.py` and `rl_replacements.py`
- Sandboxed code execution for reward evaluation
- Integration with `unsloth_zoo` RL environments

---

## Notes & Key Points

### 15.1 RL in Unsloth

- Uses TRL's `GRPOTrainer` (patched by Unsloth for compatibility)
- `unsloth/models/rl.py` (82K) — Core RL model modifications
- `unsloth/models/rl_replacements.py` (69K) — Replacement functions for RL ops
- 80% less VRAM through memory-efficient rollout strategies

### 15.2 RL Environments

```python
from unsloth_zoo.rl_environments import (
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
    Benchmarker,
    is_port_open,
    launch_openenv,
)
```

- Sandboxed execution for code-generated rewards
- Time-limited execution to prevent infinite loops
- Module locking for safe evaluation

### 15.3 FP8 Reinforcement Learning

- Combine FP8 quantization with RL training
- Significant memory savings allow RL on consumer GPUs
- Documented in blog posts and notebooks

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RL model patches | `unsloth/models/rl.py` |
| RL replacements | `unsloth/models/rl_replacements.py` |
| RL environments | `unsloth_zoo/rl_environments.py` (external) |
| DPO support | `unsloth/models/dpo.py` |
