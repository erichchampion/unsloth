# Inside Unsloth: Running and Training Local AI Models

**A Code-Level Guide to Features, Workflows, and Internals**

---

## Part I: Orientation

1. ✅ [Chapter 1: What is Unsloth?](chapter-01-what-is-unsloth.md)
2. ✅ [Chapter 2: Repository Tour — From Root to Source Tree](chapter-02-repository-tour.md)
3. ✅ [Chapter 3: Technology Stack and Key Dependencies](chapter-03-technology-stack.md)

---

## Part II: Installation and Setup

4. ✅ [Chapter 4: Installing Unsloth — Studio, Core, and Docker](chapter-04-installation.md)
5. ✅ [Chapter 5: The CLI Entry Point — Typer Commands and Configuration](chapter-05-cli-entry-point.md)
6. ✅ [Chapter 6: Device Detection and GPU Setup](chapter-06-device-detection.md)

---

## Part III: Running Models (Inference)

7. ✅ [Chapter 7: Loading a Model — FastLanguageModel.from_pretrained](chapter-07-loading-a-model.md)
8. ✅ [Chapter 8: The Model Registry — Mapping Names to Weights](chapter-08-model-registry.md)
9. ✅ [Chapter 9: Model Dispatch — Architecture-Specific Fast Paths](chapter-09-model-dispatch.md)
10. ✅ [Chapter 10: Inference with vLLM — fast_inference Mode](chapter-10-fast-inference.md)
11. ✅ [Chapter 11: Chat Templates and Tokenizer Utilities](chapter-11-chat-templates.md)

---

## Part IV: Training Models

12. ✅ [Chapter 12: LoRA and QLoRA — Parameter-Efficient Fine-Tuning](chapter-12-lora-qlora.md)
13. ✅ [Chapter 13: Full Fine-Tuning and FP8 Training](chapter-13-full-finetuning-fp8.md)
14. ✅ [Chapter 14: The Trainer — UnslothTrainer, Packing, and Padding-Free](chapter-14-trainer.md)
15. ✅ [Chapter 15: Reinforcement Learning — GRPO and RL Workflows](chapter-15-reinforcement-learning.md)
16. ✅ [Chapter 16: Vision and Multimodal Fine-Tuning](chapter-16-vision-multimodal.md)
17. ✅ [Chapter 17: Embedding and Sentence-Transformer Training](chapter-17-embedding-training.md)
18. ✅ [Chapter 18: Data Preparation — Raw Text, Synthetic Data, and Data Recipes](chapter-18-data-preparation.md)

---

## Part V: Saving, Exporting, and Deploying

19. ✅ [Chapter 19: Saving Models — LoRA, Merged 16-bit, and 4-bit](chapter-19-saving-models.md)
20. ✅ [Chapter 20: GGUF Export and Quantization](chapter-20-gguf-export.md)
21. ✅ [Chapter 21: Pushing to Hugging Face Hub](chapter-21-pushing-to-hub.md)

---

## Part VI: Custom Triton Kernels — The Engine Room

22. ✅ [Chapter 22: Kernel Architecture Overview](chapter-22-kernel-architecture.md)
23. ✅ [Chapter 23: Cross-Entropy Loss Kernel](chapter-23-cross-entropy-kernel.md)
24. ✅ [Chapter 24: Fast LoRA Kernels](chapter-24-fast-lora-kernels.md)
25. ✅ [Chapter 25: RoPE Embedding Kernel](chapter-25-rope-kernel.md)
26. ✅ [Chapter 26: SwiGLU, GeGLU, and Activation Kernels](chapter-26-activation-kernels.md)
27. ✅ [Chapter 27: LayerNorm and RMSNorm Kernels](chapter-27-layernorm-kernels.md)
28. ✅ [Chapter 28: FlexAttention and FP8 Kernels](chapter-28-flex-attention-fp8.md)
29. ✅ [Chapter 29: MoE Grouped GEMM Kernels](chapter-29-moe-kernels.md)

---

## Part VII: Model Architectures — The Fast* Classes

30. ✅ [Chapter 30: FastLlamaModel — The Reference Implementation](chapter-30-fast-llama.md)
31. ✅ [Chapter 31: Gemma, Gemma 2, and Gemma 3 Support](chapter-31-gemma-models.md)
32. ✅ [Chapter 32: Qwen 2, Qwen 3, and Qwen 3 MoE](chapter-32-qwen-models.md)
33. ✅ [Chapter 33: Mistral, Cohere, Granite, and Falcon H1](chapter-33-other-architectures.md)

---

## Part VIII: Unsloth Studio — The Web Interface

34. ✅ [Chapter 34: Studio Architecture — Backend and Frontend](chapter-34-studio-architecture.md)
35. ✅ [Chapter 35: Studio Chat — Inference, Tool Calling, and Code Execution](chapter-35-studio-chat.md)
36. ✅ [Chapter 36: Studio Training — Configuration and Observability](chapter-36-studio-training.md)
37. ✅ [Chapter 37: Data Recipes — Visual Dataset Creation](chapter-37-data-recipes.md)

---

## Part IX: Advanced Topics

38. ✅ [Chapter 38: Import-Time Monkey Patching — How Unsloth Optimizes Libraries](chapter-38-monkey-patching.md)
39. ✅ [Chapter 39: Multi-GPU Training](chapter-39-multi-gpu.md)
40. ✅ [Chapter 40: Attention Dispatch and Memory Optimization](chapter-40-attention-dispatch.md)

---

*Inside Unsloth: Running and Training Local AI Models | Complete Edition*
