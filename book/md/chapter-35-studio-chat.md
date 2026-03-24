# Chapter 35: Studio Chat — Inference, Tool Calling, and Code Execution

---

## Introduction

Studio's chat interface provides a complete conversational AI experience with model inference, tool calling with self-healing, and sandboxed code execution.

### What You'll Learn

- Chat session management
- Tool calling with auto-healing retry
- Code execution in sandbox environments
- Multi-format file uploads (images, audio, PDFs, DOCX)

---

## Notes & Key Points

### 35.1 Chat Interface Features

- Conversational UI with streaming responses
- Auto-tune inference parameters (temperature, top-p, etc.)
- Custom chat template selection
- Upload images, audio, PDFs, code files, and DOCX for context

### 35.2 Tool Calling

- Self-healing tool calling: if a tool call fails, the model retries with corrected JSON
- Web search integration
- Structured output validation

### 35.3 Code Execution

- LLMs can generate and test code in sandboxed environments
- Similar to Claude artifacts or ChatGPT code interpreter
- Uses locked-down execution from `unsloth_zoo.rl_environments`

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Chat routes | `studio/backend/routes/` |
| Inference core | `studio/backend/core/` |
| Code execution | `unsloth_zoo/rl_environments.py` |
