# Chapter 15: Reinforcement Learning — GRPO and RL Workflows

> *"Reward is the only signal that matters."*

---

## Introduction

Reinforcement learning from human feedback (RLHF) and its variants have become essential for aligning language models with human preferences. Unsloth provides the most VRAM-efficient RL training infrastructure for LLMs, with up to 80% less memory usage for GRPO (Group Relative Policy Optimization) compared to standard implementations. This chapter covers the RL infrastructure — from reward modeling and policy optimization to sandboxed code execution for automated reward evaluation.

### What You'll Learn

- The GRPO algorithm and how it differs from PPO
- RL-specific model patches in `rl.py` and `rl_replacements.py`
- Sandboxed code execution for reward evaluation
- DPO (Direct Preference Optimization) support
- Memory-efficient RL through Unsloth optimizations

### Prerequisites

- The LoRA/QLoRA concepts from Chapter 12
- The trainer infrastructure from Chapter 14
- Basic understanding of RL concepts (reward, policy, optimization)

---

## 15.1 RL Training in Unsloth

Unsloth's RL training builds on TRL's trainer classes, patching them with Unsloth's memory optimizations. The key files are substantial:

| File | Size | Purpose |
|------|------|---------|
| `models/rl.py` | 82K | Core RL model modifications and patching |
| `models/rl_replacements.py` | 69K | Replacement functions for RL operations |
| `models/dpo.py` | DPO | Direct Preference Optimization support |

The RL patches follow the same pattern as the training patches in Chapter 9 — replace standard PyTorch operations with fused Triton kernels, but specifically optimized for the unique computational patterns of RL training (rollouts, advantage computation, reference model comparisons).

---

## 15.2 GRPO: Group Relative Policy Optimization

GRPO is the primary RL algorithm used with Unsloth. It simplifies PPO by eliminating the need for a separate value (critic) model:

```
PPO:
  Policy model   (7B params - trainable)
  Value model    (7B params - trainable)
  Reference model (7B params - frozen)
  Total: ~21B parameters in memory

GRPO:
  Policy model    (7B params - trainable)
  Reference model (7B params - frozen, can be offloaded)
  Total: ~14B parameters in memory (33% less)
```

### How GRPO Works

1. **Generate rollouts** — the policy model generates multiple completions for each prompt
2. **Score responses** — a reward function (model or rule-based) scores each completion
3. **Compute group advantages** — normalize rewards within each prompt's group of completions
4. **Update policy** — use the advantages to update the policy model via gradient descent

### The GRPO Algorithm

```
Algorithm: Group Relative Policy Optimization

Input: Policy π_θ, Reference policy π_ref, Prompt dataset D, 
       Group size G, KL coefficient β

For each training step:
  1. Sample prompt p from D
  2. Generate G completions: {o₁, o₂, ..., o_G} ~ π_θ(· | p)
  3. Compute rewards: rᵢ = R(oᵢ, p) for each completion
  
  4. Compute group advantages (the "group relative" part):
     mean_r = mean(r₁, r₂, ..., r_G)
     std_r  = std(r₁, r₂, ..., r_G)
     Aᵢ = (rᵢ - mean_r) / (std_r + ε)    ← normalize within group
  
  5. Compute policy gradient loss:
     L = -Σᵢ Aᵢ · log π_θ(oᵢ|p)           ← REINFORCE with advantages
       + β · KL(π_θ || π_ref)               ← KL penalty for stability

  6. Update θ via gradient descent on L
```

The key insight is step 4: by normalizing rewards within each group, GRPO eliminates the need for an absolute value baseline (which PPO gets from its critic network). This is why GRPO doesn't need a value model — the group mean *is* the baseline.

### Reward Functions

Unsloth supports multiple types of reward functions in GRPO:

```python
# Rule-based reward (no model needed):
def accuracy_reward(completions, prompt):
    """Check if the answer is mathematically correct."""
    scores = []
    for completion in completions:
        answer = extract_answer(completion)
        scores.append(1.0 if answer == expected else 0.0)
    return scores

# Format reward (check output structure):
def format_reward(completions, prompt):
    """Reward proper XML/JSON formatting."""
    scores = []
    for completion in completions:
        has_tags = "<answer>" in completion and "</answer>" in completion
        scores.append(1.0 if has_tags else 0.0)
    return scores

# Usage: multiple rewards are combined
trainer = GRPOTrainer(
    reward_funcs=[accuracy_reward, format_reward],  # Both applied
    ...
)
```

### Memory Optimization in Unsloth

Unsloth's GRPO implementation achieves 80% less VRAM through:
- **Chunked rollout generation** — generate in batches rather than all at once
- **Reference model offloading** — move the frozen reference model to CPU during gradient computation
- **Gradient checkpointing** — recompute activations during the backward pass
- **FP8 + LoRA** — combining quantized weights with parameter-efficient training

---

## 15.3 RL Environments and Sandboxed Execution

For code-generation tasks, RL rewards can come from actually executing the generated code. Unsloth provides sandboxed execution environments:

```python
from unsloth_zoo.rl_environments import (
    check_python_modules,         # Verify allowed modules
    create_locked_down_function,  # Create sandboxed execution context
    execute_with_time_limit,      # Run code with timeout
    Benchmarker,                  # Benchmark reward computation
    is_port_open,                 # Check if sandbox server is available
    launch_openenv,               # Launch sandbox environment
)
```

### Safety Features

| Feature | Purpose |
|---------|---------|
| Module locking | Only allow safe Python modules (no `os`, `sys`, `subprocess`) |
| Time limits | Kill execution after timeout to prevent infinite loops |
| Memory limits | Prevent memory exhaustion |
| Port isolation | Run sandboxed code in isolated network space |

---

## 15.4 DPO: Direct Preference Optimization

DPO is an alternative to GRPO that skips the explicit reward model. Instead, it directly optimizes the policy using pairs of preferred/dispreferred responses:

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model = model,
    args = DPOConfig(output_dir="./dpo_output"),
    train_dataset = preference_dataset,
    processing_class = tokenizer,
)
```

Unsloth's `models/dpo.py` patches the DPO forward pass for memory efficiency, applying the same Triton kernel optimizations used in SFT training.

---

## 15.5 RL Training Examples

```python
# GRPO with Unsloth
from trl import GRPOTrainer, GRPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct", load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    args = GRPOConfig(
        output_dir = "./grpo_output",
        per_device_train_batch_size = 1,
        num_generations = 4,  # Generate 4 completions per prompt
    ),
    train_dataset = prompts_dataset,
    reward_funcs = [accuracy_reward, format_reward],
)
trainer.train()
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RL model patches | `unsloth/models/rl.py` |
| RL replacements | `unsloth/models/rl_replacements.py` |
| RL environments | `unsloth_zoo/rl_environments.py` |
| DPO support | `unsloth/models/dpo.py` |
| TRL integration | `unsloth/trainer.py` → `_patch_trl_trainer()` |
