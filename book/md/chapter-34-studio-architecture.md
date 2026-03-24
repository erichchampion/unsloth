# Chapter 34: Studio Architecture — Backend and Frontend

> *"Point, click, train."*

---

## Introduction

Unsloth Studio is a full-featured web application for running and training language models without writing code. It provides a browser-based interface for model inference, training configuration, dataset creation, and model export. The architecture follows a standard modern pattern: a FastAPI backend serves both the REST/WebSocket API and the bundled React frontend. This chapter examines the overall architecture, directory layout, and key design decisions.

### What You'll Learn

- Backend architecture: FastAPI routes, core modules, state management
- Frontend: React SPA with Vite build tooling
- API structure: REST endpoints and real-time WebSocket communication
- Plugin system for extensibility
- Licensing differences from the core library

### Prerequisites

- The CLI entry point from Chapter 5
- Basic understanding of web application architecture (REST, WebSocket, SPA)

---

## 34.1 Backend Architecture

The backend is built on FastAPI, organized into clear layers:

```
studio/backend/
├── main.py                    FastAPI app initialization, middleware, CORS
├── run.py                     Server runner (uvicorn launcher)
├── _platform_compat.py        OS-specific compatibility (macOS/Linux/Windows)
├── colab.py                   Google Colab tunnel support
│
├── routes/                    API endpoint definitions
│   ├── auth.py               Authentication/token management
│   ├── models.py             Model listing, loading, unloading
│   ├── inference.py          Chat, completion, streaming
│   ├── training.py           Training configuration, start/stop, metrics
│   ├── datasets.py           Dataset upload, listing, preview
│   └── export.py             GGUF export, Hub push
│
├── core/                      Business logic (no HTTP concerns)
│   ├── inference/            Model loading, generation, tool calling
│   ├── training/             Training orchestration, metrics collection
│   ├── data_recipe/          Visual data pipeline (Chapter 37)
│   └── export/               GGUF conversion, Hub upload
│
├── auth/                      Authentication and session management
├── plugins/                   Plugin system for extensions
├── loggers/                   Logging configuration
├── utils/                     Shared utilities
└── tests/                     Backend test suite
```

### Request Flow

```
Browser → HTTP Request → FastAPI Router → Route Handler
                                            ↓
                                       Core Module → Unsloth Library
                                            ↓
                                       Response → JSON/SSE/WebSocket → Browser
```

---

## 34.2 Frontend Architecture

The frontend is a React Single Page Application built with Vite:

```
studio/frontend/
├── src/
│   ├── app/                   App root, routing, layout
│   ├── components/            Reusable UI components
│   ├── features/              Feature-specific modules (chat, training, data)
│   ├── stores/                State management (Zustand or similar)
│   ├── hooks/                 Custom React hooks
│   ├── types/                 TypeScript type definitions
│   ├── config/                Configuration constants
│   ├── lib/                   Third-party library wrappers
│   ├── utils/                 Utility functions
│   └── assets/                Static assets
├── public/
│   ├── fonts/                 Custom Hellix font
│   └── Sloth emojis/         Branded emoji/icon assets
└── package.json               Dependencies and build scripts
```

### Real-Time Communication

Training metrics and chat responses use server-sent events (SSE) or WebSocket connections for real-time streaming:

```
Chat streaming:    POST /api/inference/chat → SSE stream of tokens
Training metrics:  WebSocket /ws/training  → live loss, LR, GPU stats
```

---

## 34.3 Plugin System

The `plugins/` directory supports extending Studio's functionality:

```
studio/backend/plugins/
└── data-designer-unstructured-seed/   # Example: advanced data generation
```

Plugins can add new data processing nodes, custom export formats, or additional inference backends.

---

## 34.4 Platform Compatibility

`_platform_compat.py` handles OS-specific differences:

| Platform | Chat | Training | Data Recipes |
|----------|------|----------|-------------|
| NVIDIA Linux | ✅ Full | ✅ Full | ✅ Full |
| AMD Linux | ✅ Full | ✅ Core CLI | ⚠️ Partial |
| macOS (Apple Silicon) | ✅ Full | ⚠️ MLX coming | ✅ Full |
| Google Colab | ✅ via tunnel | ✅ Full | ✅ Full |

For Colab, `colab.py` sets up a tunnel (e.g., ngrok) so Studio can be accessed from an external browser.

---

## 34.5 Licensing

Studio uses a different license from the core library:

| Component | License |
|-----------|---------|
| `unsloth/` (core library) | Apache 2.0 |
| `studio/` (web interface) | AGPL-3.0 |
| `unsloth_zoo/` (extensions) | Apache 2.0 |

The AGPL license requires that modifications to Studio must be shared if deployed as a network service.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastAPI app | `studio/backend/main.py` |
| Server runner | `studio/backend/run.py` |
| API routes | `studio/backend/routes/` |
| Business logic | `studio/backend/core/` |
| Frontend app | `studio/frontend/src/` |
| Platform compat | `studio/backend/_platform_compat.py` |
