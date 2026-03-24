# Chapter 39: Multi-GPU Training

---

## Introduction

Unsloth provides experimental multi-GPU support with "major improvements coming soon." This chapter covers the current state of multi-GPU training.

### What You'll Learn

- Current multi-GPU capabilities and limitations
- Device map preparation for distributed training
- How quantized models are placed across GPUs
- Integration with HF Accelerate

---

## Notes & Key Points

### 39.1 Current State

- Multi-GPU is available but marked as experimental
- Uses HF `accelerate` for distributed coordination
- `torchrun` support for multi-process training
- `DEVICE_COUNT` tracks available GPUs

### 39.2 Device Map Preparation

```python
from unsloth.models.loader_utils import prepare_device_map

# In multi-GPU mode, each rank gets its own device
distributed_device_map, is_dist = prepare_device_map()
if is_dist:
    device_map = distributed_device_map
```

- Quantized models require per-rank device placement
- Avoids Accelerate device relocation errors with quantized weights

### 39.3 Multi-GPU RoPE Caching

- `multi_gpu_cos_cached` / `multi_gpu_sin_cached` — per-device caches
- Each GPU maintains its own cos/sin cache to avoid cross-device transfers

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Device map | `unsloth/models/loader_utils.py` → `prepare_device_map()` |
| Multi-GPU caching | `unsloth/models/llama.py` |
