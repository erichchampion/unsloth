# Chapter 35: Studio Chat — Inference, Tool Calling, and Code Execution

> *"Run your model like a product, not a notebook."*

---

## Introduction

Studio's chat interface transforms a fine-tuned model into a conversational AI assistant with capabilities beyond basic text generation. It supports streaming responses, multi-format file uploads (images, PDFs, audio, code), self-healing tool calling, and sandboxed code execution. This chapter explores how the chat system is implemented across the backend inference core and the frontend UI.

### What You'll Learn

- Chat session management and message history
- Streaming inference with token-by-token delivery
- Tool calling with automatic retry and self-healing
- Sandboxed code execution (Python, JavaScript)
- Multi-format file uploads and processing

### Prerequisites

- The studio architecture from Chapter 34
- Chat templates from Chapter 11
- Fast inference from Chapter 10

---

## 35.1 Chat Session Management

Each chat session maintains its own conversation history:

```python
# Simplified session model
class ChatSession:
    id: str                          # Unique session identifier
    model_name: str                  # Which model is loaded
    messages: list[ChatMessage]      # Full conversation history
    parameters: InferenceParams      # Temperature, top_p, max_tokens, etc.
    template: str                    # Chat template name
```

### Inference Parameters

Studio provides auto-tuning of inference parameters, but users can manually adjust:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Temperature | 0.7 | 0.0–2.0 | Randomness of sampling |
| Top-P | 0.9 | 0.0–1.0 | Nucleus sampling threshold |
| Top-K | 50 | 1–100 | Top-K sampling |
| Max tokens | 2048 | 1–32768 | Maximum response length |
| Repetition penalty | 1.1 | 1.0–2.0 | Penalize repeated tokens |

---

## 35.2 Streaming Inference

Chat responses stream token-by-token using Server-Sent Events (SSE):

```
Client                          Server
  │                               │
  ├── POST /api/inference/chat ──→│
  │   {messages, params}          │
  │                               ├── Load model (if not loaded)
  │                               ├── Apply chat template
  │                               ├── Start generation
  │←── SSE: {"token": "The"}  ────┤
  │←── SSE: {"token": " answer"} ─┤
  │←── SSE: {"token": " is"}  ────┤
  │←── SSE: {"token": " 42"}  ────┤
  │←── SSE: {"done": true}  ──────┤
  │                               │
```

The backend uses Python's `async` generators with FastAPI's `StreamingResponse` for efficient token delivery.

---

## 35.3 Tool Calling with Self-Healing

Studio supports function calling — the model can invoke predefined tools (web search, calculations, data lookups) and receive their results:

```
User:     "What's the weather in Tokyo?"
Model:    <tool_call>{"name": "web_search", "args": {"query": "weather tokyo"}}</tool_call>
System:   [executes web search, returns results]
Model:    "The current temperature in Tokyo is 22°C with partly cloudy skies."
```

### Self-Healing Retry

If the model generates malformed tool call JSON, Studio automatically retries:

```
Attempt 1: {"name": "search", "args": {"query": "weather"    ← Missing closing brace
            → Parse error detected
Attempt 2: Model receives error feedback, regenerates
            {"name": "search", "args": {"query": "weather"}}  ← Valid JSON
            → Tool execution proceeds
```

This retry loop typically succeeds within 1–2 attempts, making tool calling robust even with smaller models.

---

## 35.4 Code Execution

Studio includes a sandboxed code execution environment, similar to ChatGPT's Code Interpreter:

```python
# The model can generate and execute code
response = """
Let me calculate that for you:
```python
import math
result = math.factorial(20)
print(f"20! = {result}")
```
"""
# Studio detects the code block, executes in sandbox, returns output
# Output: "20! = 2432902008176640000"
```

The sandbox uses `unsloth_zoo.rl_environments` — the same locked-down execution environment used for GRPO reward computation (Chapter 15). This ensures:
- **No filesystem access** beyond a temporary directory
- **No network access**
- **Timeout enforcement** (default: 30 seconds)
- **Memory limits**

---

## 35.5 File Uploads

Studio supports multi-format file uploads for context:

| Format | Processing | Use Case |
|--------|-----------|----------|
| Images (PNG, JPG) | Vision model input | VLM conversations |
| PDF | Text extraction (OCR if needed) | Document Q&A |
| DOCX | Text extraction | Document Q&A |
| Audio | Transcription | Audio context |
| Code files | Syntax-highlighted display | Code review |
| CSV/JSON | Structured preview | Data analysis |

Uploaded files are processed server-side and their content is injected into the conversation context.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Inference routes | `studio/backend/routes/inference.py` |
| Inference core | `studio/backend/core/inference/` |
| Code execution | `unsloth_zoo/rl_environments.py` |
| Chat templates | `unsloth/chat_templates.py` |
