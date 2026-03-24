# Chapter 34: Studio Architecture — Backend and Frontend

---

## Introduction

Unsloth Studio is a web UI for running and training models. It uses a FastAPI backend and a React frontend, communicating over HTTP and WebSocket APIs.

### What You'll Learn

- Backend architecture: FastAPI routes, state management, plugins
- Frontend: React SPA with real-time updates
- Backend directory structure
- Authentication and session management

---

## Notes & Key Points

### 34.1 Backend Structure

```
studio/backend/
├── main.py              # FastAPI app initialization
├── run.py               # Server runner
├── routes/              # API endpoint definitions
├── core/                # Business logic (data recipes, inference)
├── models/              # Pydantic data models
├── state/               # Application state management
├── auth/                # Authentication
├── plugins/             # Plugin system
├── loggers/             # Logging configuration
├── utils/               # Shared utilities
└── tests/               # Backend tests
```

### 34.2 Frontend

- React SPA at `studio/frontend/`
- Includes Node.js build tooling (package.json, vite config)
- Bundled assets served by the backend
- Real-time training metrics via WebSocket or SSE

### 34.3 Licensing

- Studio components are under AGPL-3.0 (not Apache 2.0)
- This includes all files under `studio/`
- Core library remains Apache 2.0

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Backend app | `studio/backend/main.py` |
| Server runner | `studio/backend/run.py` |
| API routes | `studio/backend/routes/` |
| Frontend app | `studio/frontend/` |
| Platform compat | `studio/backend/_platform_compat.py` |
