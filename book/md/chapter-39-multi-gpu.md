# Chapter 39: Multi-GPU Training

> *"Scale beyond a single card."*

---

## Introduction

While Unsloth is primarily optimized for single-GPU training (where its Triton kernels deliver the most impact), many production workloads require distributing training across multiple GPUs. Unsloth provides experimental multi-GPU support through integration with Hugging Face Accelerate and PyTorch's distributed training primitives. This chapter covers the current state of multi-GPU training, device map preparation, and the specific challenges of distributing quantized models.

### What You'll Learn

- Current multi-GPU capabilities and their experimental status
- Device map preparation for distributed training
- How quantized (4-bit) models are placed across GPUs
- Per-device RoPE caching to avoid cross-GPU data transfers
- Integration with Accelerate and torchrun

### Prerequisites

- The FastLlamaModel from Chapter 30
- QLoRA quantization from Chapter 12
- Basic distributed training concepts (data parallel, model parallel)

---

## 39.1 Current State

Multi-GPU training in Unsloth is functional but experimental:

```python
# Multi-GPU training launch:
# Method 1: accelerate
accelerate launch --num_processes 2 train.py

# Method 2: torchrun
torchrun --nproc_per_node=2 train.py
```

### Optimization Level

| Feature | Single GPU | Multi-GPU |
|---------|-----------|-----------|
| Triton kernels | ✅ Full | ✅ Full (per rank) |
| Fused cross-entropy | ✅ | ✅ |
| Fast LoRA | ✅ | ✅ |
| Gradient accumulation | ✅ | ✅ (synced) |
| Flash Attention | ✅ | ✅ |
| torch.compile | ✅ | ⚠️ Experimental |

Each GPU rank loads its own copy of the Triton kernels, so the kernel-level optimizations apply per-GPU just as they do on a single GPU. The distribution overhead comes from gradient synchronization.

---

## 39.2 Device Map Preparation

Distributing a quantized model across GPUs requires careful device placement:

```python
from unsloth.models.loader_utils import prepare_device_map

# Determine distributed configuration
distributed_device_map, is_distributed = prepare_device_map()

if is_distributed:
    # Each rank gets assigned specific GPU(s)
    device_map = distributed_device_map
else:
    device_map = "auto"  # Single GPU
```

### Why Standard device_map Fails

Standard Hugging Face `device_map="auto"` can fail with quantized models because:
1. **BitsAndBytes quantization state** — 4-bit weights have associated `quant_state` tensors that must be on the same device
2. **Accelerate relocation** — Accelerate's device relocation can corrupt quantization metadata
3. **Layer splitting** — Placing half a layer on GPU 0 and half on GPU 1 breaks quantization

`prepare_device_map()` solves these by keeping entire layers (including their `quant_state`) on single devices.

---

## 39.3 Data Parallel vs. Model Parallel

Unsloth's multi-GPU support primarily uses **data parallelism**:

```
Data Parallel (current):
  GPU 0: full model copy + batch slice 0    ─→ gradients ─┐
  GPU 1: full model copy + batch slice 1    ─→ gradients ─┤
  GPU 2: full model copy + batch slice 2    ─→ gradients ─┤
  GPU 3: full model copy + batch slice 3    ─→ gradients ─┘
                                                           ↓
                                               AllReduce (sync gradients)
                                                           ↓
                                               Update weights (all GPUs)
```

For QLoRA (4-bit), each GPU only needs ~4GB for the frozen base model plus ~100MB for LoRA adapters, making data parallel feasible even on consumer multi-GPU setups.

### Model Parallel (Future)

Model parallelism (splitting layers across GPUs) is not yet supported with Unsloth's optimizations, but is on the roadmap for very large models that don't fit on a single GPU even in 4-bit.

---

## 39.4 Multi-GPU RoPE Caching

RoPE cos/sin values must be cached per device to avoid expensive cross-GPU transfers:

```python
# Per-device caching (from FastLlamaModel)
multi_gpu_cos_cached = {}  # {device_id: cos_tensor}
multi_gpu_sin_cached = {}  # {device_id: sin_tensor}

def get_rope_embeddings(position_ids, device):
    if device not in multi_gpu_cos_cached:
        # Compute and cache on first use per GPU
        multi_gpu_cos_cached[device] = compute_cos(position_ids).to(device)
        multi_gpu_sin_cached[device] = compute_sin(position_ids).to(device)
    return multi_gpu_cos_cached[device], multi_gpu_sin_cached[device]
```

Without this caching, every forward pass would trigger a GPU-to-GPU transfer of the RoPE embeddings, adding significant latency.

---

## 39.5 GPU Count Detection

```python
# From loader_utils.py
DEVICE_COUNT = torch.cuda.device_count()

if DEVICE_COUNT > 1:
    logger.info(f"Unsloth: Detected {DEVICE_COUNT} GPUs — using distributed mode")
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs are visible |
| `LOCAL_RANK` | Current process GPU index |
| `WORLD_SIZE` | Total number of processes |
| `MASTER_ADDR` | Coordination server address |
| `MASTER_PORT` | Coordination server port |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Device map preparation | `unsloth/models/loader_utils.py` → `prepare_device_map()` |
| Multi-GPU RoPE caching | `unsloth/models/llama.py` |
| GPU count detection | `unsloth/models/loader_utils.py` → `DEVICE_COUNT` |
| Distributed training | HF Accelerate (external) |
