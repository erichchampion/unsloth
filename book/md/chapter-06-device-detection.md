# Chapter 6: Device Detection and GPU Setup

> *"Before any kernel launches, Unsloth must know what hardware it's running on."*

---

## Introduction

Unsloth must detect the GPU type and capabilities at import time — before any model is loaded or any kernel is compiled. This detection determines which features are available (bfloat16, Flash Attention, 4-bit quantization), which code paths are taken, and whether certain workarounds are needed for specific hardware.

The detection happens in two locations: `unsloth/device_type.py` (which delegates to `unsloth_zoo`) handles the basic platform identification, while `unsloth/__init__.py` (lines 198–310) handles CUDA-specific setup including bfloat16 capability, Triton linking, and bitsandbytes integration.

### What You'll Learn

- How `DEVICE_TYPE` is determined at startup
- The CUDA capability check and bfloat16 support detection
- Triton and bitsandbytes linking — including the `ldconfig` fallback chain
- AMD ROCm quirks: warp sizes, pre-quantized model compatibility
- Intel XPU detection
- How capability flags propagate to model loading decisions

### Prerequisites

- Understanding of GPU compute capabilities (NVIDIA SM versions)
- Basic familiarity with CUDA, ROCm, and XPU concepts
- The `__init__.py` import sequence from Chapter 1

---

## 6.1 The Device Detection Chain

Device detection starts in `unsloth/device_type.py`, which exports seven constants used throughout the codebase:

```python
# unsloth/device_type.py — the exports
__all__ = [
    "is_hip",
    "get_device_type",
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "ALLOW_PREQUANTIZED_MODELS",
    "ALLOW_BITSANDBYTES",
]
```

The core logic is in `get_device_type()`, a cached function that probes PyTorch's accelerator support:

```python
# device_type.py (lines 36-59)
@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Fallback: check torch.accelerator (PyTorch 2.4+)
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError(
                "Unsloth cannot find any torch accelerator? You need a GPU."
            )
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
    )
```

The detection order matters. CUDA is checked first because ROCm's HIP runtime also exposes `torch.cuda` as a compatibility layer. The `is_hip()` function distinguishes the two by checking `torch.version.hip`:

```python
@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))
```

### The DEVICE_TYPE_TORCH Alias

A subtle but important detail: for AMD GPUs, `DEVICE_TYPE` is `"hip"`, but `DEVICE_TYPE_TORCH` is set to `"cuda"`. This is because ROCm's PyTorch uses the CUDA API surface — functions like `torch.cuda.is_available()`, `torch.autocast("cuda")`, and `torch.cuda.device_count()` all work on AMD hardware through HIP translation.

```python
# device_type.py (lines 62-66)
DEVICE_TYPE: str = get_device_type()
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip":
    DEVICE_TYPE_TORCH = "cuda"  # HIP uses CUDA API surface
```

---

## 6.2 CUDA Capability and bfloat16 Detection

Once the device type is known, `__init__.py` checks the GPU's compute capability to determine bfloat16 support:

```python
# unsloth/__init__.py (lines 199-216)
if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = major_version >= 8
```

Compute capability 8.0 corresponds to NVIDIA's Ampere architecture (A100, RTX 3090). Older GPUs (Turing, Volta) use float16 instead. This flag affects:

- The default training dtype
- Whether certain Triton kernels can use bfloat16 accumulators
- Flash Attention availability (requires Ampere+)

The code also patches `torch.cuda.is_bf16_supported()` to handle a compatibility issue between PyTorch versions. PyTorch 2.4+ added an `including_emulation` parameter to this function. Unsloth standardizes the interface so the rest of the codebase can call it without worrying about the PyTorch version:

```python
# __init__.py (lines 203-215)
old_is_bf16_supported = torch.cuda.is_bf16_supported
if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):
    def is_bf16_supported(including_emulation=False):
        return old_is_bf16_supported(including_emulation)
else:
    def is_bf16_supported():
        return SUPPORTS_BFLOAT16
torch.cuda.is_bf16_supported = is_bf16_supported
```

---

## 6.3 Triton and bitsandbytes Linking

After capability detection, `__init__.py` ensures that Triton and bitsandbytes are properly linked to the CUDA runtime. This is where things get interesting — and sometimes messy.

```
import triton
    │
    ├─ Triton >= 3.0.0: from triton.backends.nvidia.driver import libcuda_dirs
    └─ Triton < 3.0.0:  from triton.common.build import libcuda_dirs
    │
import bitsandbytes
    │
    ├─ Test: bnb.functional.lib.cdequantize_blockwise_fp32
    └─ Test: libcuda_dirs()
    │
    ├─ Success → Continue
    └─ Failure → ldconfig fallback chain:
        ├─ Try: ldconfig /usr/lib64-nvidia
        ├─ Try: Scan /usr/local/cuda-* for latest version
        ├─ ldconfig /usr/local/cuda-XX.X
        ├─ Reload bitsandbytes and triton
        └─ Retry tests → Warn if still failing
```

The `ldconfig` fallback is particularly important for cloud environments (RunPod, Lambda Labs, Kaggle) where CUDA libraries may be installed in non-standard locations. The script scans `/usr/local/` for directories matching `cuda-*`, sorts by version number, and links the latest one:

```python
# __init__.py (lines 256-274)
possible_cudas = subprocess.check_output(["ls", "-al", "/usr/local"])
    .decode("utf-8").split("\n")
find_cuda = re.compile(r"[\s](cuda\-[\d\.]{2,})$")
possible_cudas = [find_cuda.search(x) for x in possible_cudas]
possible_cudas = [x.group(1) for x in possible_cudas if x is not None]

latest_cuda = np.argsort(
    [float(find_number.search(x).group(1)) for x in possible_cudas]
)[::-1][0]
os.system(f"ldconfig /usr/local/{latest_cuda}")
```

---

## 6.4 AMD ROCm: Warp Sizes and Quantization Compatibility

AMD GPU support has a subtle complication: **warp sizes differ between GPU families**, and this affects 4-bit quantization compatibility.

| Device Family | Warp Size | Pre-quantized Models? |
|---------------|-----------|----------------------|
| NVIDIA (all) | 32 | ✅ Always |
| AMD Radeon (Navi / RDNA) | 32 | ✅ bitsandbytes ≥ 0.49.0 |
| AMD Instinct (MI / CDNA) | 64 | ✅ bitsandbytes ≥ 0.49.2 |

Unsloth's pre-quantized models on Hugging Face Hub use a block size of 64. On AMD GPUs with a warp size of 64 (Instinct MI series), this only works with bitsandbytes 0.49.2 or later. The `device_type.py` file contains careful version-gating logic to handle this:

```python
# device_type.py (lines 96-132)
ALLOW_PREQUANTIZED_MODELS: bool = True
ALLOW_BITSANDBYTES: bool = True

if DEVICE_TYPE == "hip":
    # Check bitsandbytes availability
    try:
        import bitsandbytes
    except:
        ALLOW_PREQUANTIZED_MODELS = False
        ALLOW_BITSANDBYTES = False

    if ALLOW_BITSANDBYTES:
        if Version(bitsandbytes.__version__) >= Version("0.49.2"):
            pass  # Full support
        elif Version(bitsandbytes.__version__) >= Version("0.49.0"):
            # Check if this is a 64-warp GPU (Instinct)
            from bitsandbytes.cextension import ROCM_WARP_SIZE_64
            ALLOW_PREQUANTIZED_MODELS = not ROCM_WARP_SIZE_64
        else:
            # Inspect source code for hardcoded blocksize behavior
            from bitsandbytes.nn.modules import Params4bit
            if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in \
                inspect.getsource(Params4bit):
                ALLOW_PREQUANTIZED_MODELS = False
```

These flags are checked later in `loader.py` when resolving model names — if `ALLOW_PREQUANTIZED_MODELS` is `False`, the loader strips the `-bnb-4bit` suffix and downloads full-precision weights instead.

---

## 6.5 Intel XPU Detection

Intel GPU support follows a simpler path:

```python
elif DEVICE_TYPE == "xpu":
    SUPPORTS_BFLOAT16 = torch.xpu.is_bf16_supported()
    import bitsandbytes as bnb
    # TODO: check triton for intel installed properly.
```

Intel XPU support uses `torch.xpu` APIs and checks bfloat16 support directly. Triton verification is noted as a TODO.

---

## 6.6 How Capability Flags Propagate

The constants set during device detection propagate throughout the codebase:

```
DEVICE_TYPE ──────→ loader.py: controls which Fast*Model classes are available
                  → __init__.py: controls Triton/bnb linking path
                  → kernels/: controls which kernels can be compiled

SUPPORTS_BFLOAT16 → loader.py: selects default dtype (bf16 vs fp16)
                  → trainer.py: configures mixed precision training

ALLOW_BITSANDBYTES → loader.py: enables/disables 4-bit model loading
                   → prints warning if disabled

ALLOW_PREQUANTIZED_MODELS → loader.py: strips -bnb-4bit suffix if False
                          → downloads full-precision weights instead

DEVICE_COUNT ─────→ loader_utils.py: multi-GPU device map preparation
                  → RoPE cache initialization (per-device cos/sin caches)
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Device type detection | `unsloth/device_type.py` |
| Canonical implementation | `unsloth_zoo/device_type.py` |
| CUDA/bfloat16 setup | `unsloth/__init__.py` (lines 198-216) |
| Triton + bnb linking | `unsloth/__init__.py` (lines 226-310) |
| AMD GPU ID table config | `unsloth/import_fixes.py` → `configure_amdgpu_asic_id_table_path()` |
| Capability flag usage | `unsloth/models/loader.py` (lines 362-368, 399-411) |
