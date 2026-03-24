# Chapter 19: Saving Models — LoRA, Merged 16-bit, and 4-bit

> *"A model that can't be saved is a model that was never trained."*

---

## Introduction

After training completes, the model must be persisted to disk for later use — whether for resumed training, inference, GGUF export, or deployment. Unsloth's `unsloth_save_model()` function (in `save.py`, 3,246 lines) is the central saving entry point. It supports three save methods, handles everything from LoRA adapter-only exports to full weight merging with dequantization, and manages the complex memory choreography needed when merging 4-bit weights back to 16-bit on memory-limited systems.

### What You'll Learn

- The three save methods: `"lora"`, `"merged_16bit"`, `"merged_4bit"`
- How `_merge_lora()` dequantizes and merges weights
- The VRAM → RAM → disk memory spill strategy
- Sharding and `safe_serialization` trade-offs
- Tokenizer saving and padding side management

### Prerequisites

- The LoRA concepts from Chapter 12
- Understanding of model weight formats (safetensors, state_dict)
- The model loading pipeline from Chapter 7

---

## 19.1 Save Methods

| Method | What It Saves | Size (7B) | Use Case |
|--------|--------------|-----------|----------|
| `"lora"` | Adapter weights only | ~100 MB | Default. Fast. Resume training later. |
| `"merged_16bit"` | Full model in FP16 | ~14 GB | Required for GGUF export. Production inference. |
| `"merged_4bit"` | Full model in 4-bit | ~3.5 GB | Inference-only deployment. |

### Usage

```python
# Save LoRA adapters only (fastest)
model.save_pretrained("my_lora_adapter")

# Save merged 16-bit model (for GGUF or deployment)
model.save_pretrained_merged("my_model_16bit", save_method="merged_16bit")

# Save merged 4-bit model (caution: irreversible quantization)
model.save_pretrained_merged("my_model_4bit", save_method="merged_4bit_forced")
```

Note the `"merged_4bit"` method deliberately raises an error unless you use `"merged_4bit_forced"`. This is a safety measure — merging to 4-bit loses accuracy permanently and should only be a final step.

---

## 19.2 LoRA Merging — _merge_lora()

The `_merge_lora()` function (lines 199-228) is the core algorithm that combines LoRA adapters with base weights:

```python
def _merge_lora(layer, name):
    W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)

    # Step 1: Dequantize base weights from 4-bit to float32
    if quant_state is not None:
        W = fast_dequantize(W, quant_state)

    # Step 2: Transpose to [out_features, in_features]
    W = W.to(torch.float32).t()

    # Step 3: Merge LoRA: W += s * (A^T @ B^T)
    if A is not None:
        W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha=s)

    # Step 4: Check for numerical issues
    maximum_element = torch.max(W.min().abs(), W.max())
    if not torch.isfinite(maximum_element).item():
        raise ValueError(f"Merge failed. {name} has infinity elements.")

    # Step 5: Cast back to target dtype and transpose
    return W.t().to(dtype), bias
```

The `fast_dequantize()` kernel efficiently converts bitsandbytes 4-bit NF4 weights back to float32 for the merge computation.

---

## 19.3 Memory Spill Strategy

Merging a 7B model from 4-bit to 16-bit quadruples its memory footprint. On a 24 GB GPU, there may not be enough room to hold both versions. Unsloth handles this with a three-tier spill strategy:

```
Tier 1: GPU VRAM
  ├─ Check: torch.cuda.memory_allocated() + W.nbytes < max_vram
  └─ Action: state_dict[name] = W  (fastest)

Tier 2: System RAM
  ├─ Check: max_ram - W.nbytes > 0
  └─ Action: state_dict[name] = W.to("cpu")
  └─ Note: Currently disabled due to memory leak issues

Tier 3: Disk
  ├─ Action: torch.save(W, filename)
  │          then torch.load(filename, mmap=True)
  └─ Result: Memory-mapped tensor (slowest but always works)
```

The function calculates available memory accounting for sharding overhead:

```python
max_ram = psutil.virtual_memory().available
max_ram -= sharded_ram_usage  # Reserve space for safetensors shard
max_ram = int(max(0, max_ram) * maximum_memory_usage)  # 90% safety margin
```

---

## 19.4 Sharding and Serialization

### Safe Serialization

By default, Unsloth uses safetensors format (`safe_serialization=True`). However, on systems with ≤2 CPUs (common in Colab), safetensors is 10× slower than PyTorch's pickle format:

```python
if safe_serialization and (n_cpus <= 2):
    logger.warning("Using safe_serialization is 10x slower with 2 CPUs.")
    safe_serialization = False
    save_function = fast_save_pickle
```

### Shard Size

The `max_shard_size` parameter (default: `"5GB"`) controls how large each saved file can be. Larger shards mean fewer files but more memory needed during saving.

---

## 19.5 Layer-by-Layer Processing

The merged_16bit path iterates through every layer in the model:

```python
for j, layer in enumerate(ProgressBar(internal_model.model.layers)):
    for item in LLAMA_WEIGHTS:  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        W, bias = _merge_lora(eval(f"layer.{item}"), name)
        state_dict[name] = W  # or spill to RAM/disk

    for item in LLAMA_LAYERNORMS:  # input_layernorm, post_attention_layernorm, etc.
        state_dict[name] = eval(f"layer.{item}.weight.data")
```

Special handling for tied weights — if `embed_tokens` and `lm_head` share the same data pointer, only one copy is saved.

---

## 19.6 Tokenizer Saving

The tokenizer is saved alongside the model with one important adjustment:

```python
# Set padding side for inference
old_padding_side = tokenizer.padding_side
tokenizer.padding_side = "left"     # Left padding for batched generation
tokenizer.save_pretrained(...)
tokenizer.padding_side = old_padding_side  # Revert for continued training
```

This ensures that saved tokenizers default to left-padding, which is required for correct batched inference.

---

## 19.7 Colab/Kaggle Disk Space

On Colab and Kaggle, disk space is limited (~20GB). Unsloth automatically frees the Hugging Face model cache before saving:

```python
if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
    _free_cached_model(internal_model)  # Frees 4-16GB
```

This deletes the downloaded model weights from the HF cache, relying on the in-memory model for the save operation.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| unsloth_save_model() | `unsloth/save.py` |
| _merge_lora() | `unsloth/save.py` (lines 199-228) |
| fast_dequantize() | `unsloth/kernels/__init__.py` |
| Disk space management | `unsloth/save.py` → `_free_cached_model()` |
