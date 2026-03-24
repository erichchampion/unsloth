# Chapter 6: Device Detection and GPU Setup

---

## Introduction

Unsloth must detect the GPU type (NVIDIA CUDA, AMD ROCm/HIP, Intel XPU, or CPU) at import time to configure the correct backends, enable/disable features, and select appropriate kernels.

### What You'll Learn

- How `DEVICE_TYPE` is determined at startup
- CUDA capability checks and bfloat16 support detection
- Triton and bitsandbytes linking
- AMD/Intel-specific configuration paths

---

## Notes & Key Points

### 6.1 The Device Detection Chain

```python
# unsloth/__init__.py imports from unsloth_zoo
from unsloth_zoo.device_type import (
    is_hip, get_device_type, DEVICE_TYPE,
    DEVICE_TYPE_TORCH, DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)
```

- `DEVICE_TYPE` is one of: `"cuda"`, `"hip"`, `"xpu"`
- `DEVICE_COUNT` — number of GPUs available
- `ALLOW_PREQUANTIZED_MODELS` — whether pre-quantized BnB models are compatible

### 6.2 CUDA Setup

- Checks `torch.cuda.get_device_capability()` for compute capability
- `SUPPORTS_BFLOAT16 = major_version >= 8` (Ampere and above)
- Patches `torch.cuda.is_bf16_supported()` for compatibility
- Attempts to link CUDA via `ldconfig` if bitsandbytes fails

### 6.3 Triton + bitsandbytes Linking

- Triton version ≥ 3.0.0 uses `triton.backends.nvidia.driver.libcuda_dirs`
- Older versions use `triton.common.build.libcuda_dirs`
- If linking fails, Unsloth tries `ldconfig /usr/lib64-nvidia` or scans `/usr/local/cuda-*`

### 6.4 AMD and Intel

- AMD ROCm: configures `amdgpu_asic_id_table_path` early
- AMD: may disable bitsandbytes 4-bit if not stable
- Intel XPU: imports bitsandbytes, checks Triton support

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Device detection | `unsloth/device_type.py` (delegates to `unsloth_zoo`) |
| CUDA/bfloat16 setup | `unsloth/__init__.py` (lines 198-310) |
| AMD GPU config | `unsloth/import_fixes.py` → `configure_amdgpu_asic_id_table_path()` |
