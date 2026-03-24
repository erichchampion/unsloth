# Chapter 24: Fast LoRA Kernels

> *"Three kernel launches become one."*

---

## Introduction

When using QLoRA, every linear layer performs three separate operations: dequantize the 4-bit base weights to float16/32, multiply by the input, then add the LoRA correction term `s · B(Ax)`. Each of these is a separate CUDA kernel launch with its own memory read/write cycle. Unsloth's fast LoRA kernel (`fast_lora.py`, 730 lines) fuses all three into a single Triton kernel, cutting memory traffic by roughly 3× and eliminating kernel launch overhead.

### What You'll Learn

- The standard LoRA forward pass and its inefficiencies
- How the fused kernel combines dequant + matmul + LoRA
- The `get_lora_parameters_bias()` utility function
- Forward and backward pass fusion
- Performance characteristics and when fusion helps most

### Prerequisites

- LoRA and QLoRA from Chapter 12
- The kernel architecture from Chapter 22
- Basic Triton programming concepts

---

## 24.1 Standard LoRA Forward Pass

In a standard LoRA implementation, the forward pass for a single linear layer requires multiple steps:

```python
# Standard LoRA forward (3 separate kernel launches)

# Launch 1: Dequantize 4-bit weights to float16
W_16bit = dequantize_nf4(W_4bit, quant_state)    # Read W_4bit, write W_16bit

# Launch 2: Matrix multiply
y = W_16bit @ x                                    # Read W_16bit + x, write y

# Launch 3: LoRA correction
lora_out = (scaling * B) @ (A @ x)                 # Read A, B, x, write lora_out
y = y + lora_out                                    # Read y + lora_out, write y
```

Each launch reads from and writes to global GPU memory, creating three complete memory round-trips.

---

## 24.2 Fused LoRA Kernel

Unsloth's kernel performs all operations in a single pass:

```python
# Fused kernel (1 kernel launch)
y = fast_lora_forward(W_4bit, quant_state, A, B, scaling, x)
# Internally:
#   1. Load chunk of W_4bit, dequantize in-register
#   2. Load corresponding chunk of x
#   3. Compute partial matmul, accumulate
#   4. Load A, B chunks, compute LoRA correction
#   5. Add correction to result
#   6. Store final output — only one write to global memory
```

### Memory Traffic Comparison

For a layer with `d=4096, k=4096, r=16`:

| Operation | Standard | Fused |
|-----------|----------|-------|
| Read W_4bit | 8 MB | 8 MB |
| Write W_16bit | 32 MB | **0** (stays in registers) |
| Read W_16bit | 32 MB | **0** |
| Read/write LoRA (A, B) | 0.5 MB | 0.5 MB |
| **Total memory traffic** | **72.5 MB** | **8.5 MB** |

The fused kernel reduces memory traffic by **~8.5×** for this layer.

---

## 24.3 get_lora_parameters_bias()

This utility function (in `utils.py`) extracts all the components needed for the fused kernel:

```python
def get_lora_parameters_bias(layer):
    """Extract W, quant_state, A, B, scaling, bias from a LoRA layer."""
    # Returns:
    #   W          - base weight (4-bit packed)
    #   quant_state - bitsandbytes quantization metadata
    #   A          - LoRA down-projection matrix [r, in_features]
    #   B          - LoRA up-projection matrix [out_features, r]
    #   s          - LoRA scaling factor (alpha / r)
    #   bias       - optional bias term
```

This function handles the different layer types:
- `Bnb_Linear4bit` — bitsandbytes 4-bit layer
- `Peft_Linear4bit` — PEFT-wrapped 4-bit LoRA layer
- `Peft_Linear` — PEFT-wrapped full-precision LoRA layer

---

## 24.4 Backward Pass

The backward pass is also fused. Standard backpropagation through LoRA requires:

1. Gradient w.r.t. output → gradient w.r.t. base matmul
2. Gradient w.r.t. LoRA B matrix
3. Gradient w.r.t. LoRA A matrix
4. Gradient w.r.t. input x

The fused backward kernel computes all four gradients in a single pass, reusing the dequantized weights and intermediate values.

---

## 24.5 When Fusion Helps Most

| Scenario | Speedup from Fusion |
|----------|-------------------|
| QLoRA (4-bit base + LoRA) | **High** — dequantization dominates |
| LoRA (FP16 base + LoRA) | **Medium** — no dequant, but matmul fusion helps |
| Full fine-tuning (no LoRA) | **N/A** — no LoRA correction to fuse |
| Large rank (r=64+) | **High** — LoRA correction is more expensive |
| Small rank (r=8) | **Medium** — LoRA cost is small relative to base matmul |

## 24.6 Quantization Format Handling

The fused LoRA kernel must handle bitsandbytes' NF4 encoding format, which packs two 4-bit values into each byte:

```
NF4 encoding: each byte holds 2 weights
  Byte layout: [w1_high:4][w0_low:4]

Dequantization (inside the kernel):
  1. Extract 4-bit index from packed byte
  2. Look up in NF4 quantization table (16 entries)
  3. Multiply by per-block scale factor
  4. Result: float16 or float32 weight value
```

### The NF4 Quantization Table

NF4 uses a fixed set of 16 values optimized for the normal distribution of neural network weights:

```
Index:  0       1       2       3       4       5       6       7
Value: -1.000  -0.692  -0.523  -0.395  -0.286  -0.195  -0.113  -0.035

Index:  8       9       10      11      12      13      14      15
Value:  0.035   0.113   0.195   0.286   0.395   0.523   0.692   1.000
```

These 16 values are the quantiles of a standard normal distribution — spaced more densely near zero where most weights cluster. This is why NF4 outperforms uniform 4-bit quantization for neural network weights.

### Block-wise Dequantization

Weights are quantized in blocks of 64 values, each sharing a single FP16 scale factor:

```python
# Dequantization (simplified):
block_size = 64
for block_idx in range(num_blocks):
    scale = scales[block_idx]                    # FP16 scale factor
    for i in range(block_size):
        nf4_index = packed_weights[block_idx, i]  # 4-bit index (0-15)
        weight_fp16 = NF4_TABLE[nf4_index] * scale  # Dequantized value
```

The kernel also handles the double quantization variant (`bnb_4bit_use_double_quant=True`), where the block scale factors themselves are quantized to 8-bit, saving an additional ~0.4 bits per weight.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Fast LoRA kernel | `unsloth/kernels/fast_lora.py` |
| LoRA parameter extraction | `unsloth/kernels/utils.py` → `get_lora_parameters_bias()` |
| Dequantization | `unsloth/kernels/utils.py` → `fast_dequantize()` |
