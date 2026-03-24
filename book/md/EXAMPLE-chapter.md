# Chapter 1: What is OpenCode?

> *"The open source AI coding agent."*

---

## Introduction

OpenCode (https://github.com/anomalyco/opencode) is a terminal-first AI coding agent that gives large language models the ability to read, write, search, and execute code on your behalf. Unlike IDE-embedded copilots that suggest inline completions, OpenCode operates as a full agentic loop: it receives a natural-language prompt, plans its approach, executes multi-step tool calls, and iterates until the task is done -- or it asks you a question.

The project is MIT-licensed and hosted at https://github.com/anomalyco/opencode.

### What You'll Learn

- What separates an AI coding *agent* from an AI coding *assistant*
- OpenCode's design philosophy: open source, provider-agnostic, terminal-native
- The client-server architecture that decouples the UI from the engine
- How the project positions itself in the landscape (contrast with Claude Code, Cursor, Copilot)

### Prerequisites

- Comfortable reading TypeScript
- Basic familiarity with LLM concepts (tokens, context windows, tool calling)
- Experience with terminal-based developer tooling

---

## Notes & Key Points

### 1.1 Agent vs. Assistant

- **Assistants** respond to a single prompt; **agents** loop: prompt --> plan --> act --> observe --> continue
- OpenCode implements a multi-step agentic loop in `session/prompt.ts` -- the `loop()` function runs until the model emits a finish reason that is not `tool-calls`
- Key insight: the model's finish reason drives continuation logic

### 1.2 Design Philosophy

- **Provider-agnostic**: supports 20+ AI providers through the Vercel AI SDK
- **Open source**: MIT licensed, full source visibility
- **Terminal-native**: built by Neovim users; the TUI is a first-class citizen
- **Client-server**: the agent engine runs as an HTTP server (Hono), any client (TUI, web, mobile) can drive it via the OpenCode SDK
- **LSP-aware**: out-of-the-box Language Server Protocol support for diagnostics

### 1.3 Getting Started

**As a user** -- install and run with a single command:

```bash
npx opencode@latest
```

This downloads the latest release, detects your project directory, and launches the TUI. You'll need an API key for at least one provider (e.g., `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment).

**As a developer** -- clone and run from source:

> **Important:** This book was written against a specific snapshot of the OpenCode codebase. To ensure every file path, function name, and code excerpt matches what you see on screen, check out the exact commit used throughout the book:

```bash
git clone https://github.com/anomalyco/opencode.git
cd opencode
git checkout 129fe1e35   # the commit this book was written against
bun install              # requires Bun >= 1.3.10
bun run dev              # launches the TUI against the current directory
```

The `bun run dev` command is shorthand for:

```bash
bun run --cwd packages/opencode --conditions=browser src/index.ts
```

On first launch, OpenCode:
1. Detects the project directory (looks for `.git`, `package.json`, etc.)
2. Creates a local SQLite database at `~/.opencode/data.db` for session storage
3. Loads configuration from `opencode.json` (project-level) and `~/.config/opencode/config.json` (global)
4. Starts the Hono HTTP server on a random available port
5. Launches the TUI client connected to that server

### 1.4 Architecture at 10,000 Feet

```
+-----------------------------------------------------------+
|                      CLI / TUI / Web                      |
|                     (opencode-ai/sdk)                     |
+------------------------+----------------------------------+
                         | HTTP / in-process fetch
+------------------------v----------------------------------+
|                    Hono HTTP Server                       |
|          (server/server.ts -- routes, OpenAPI)            |
+-----------------------------------------------------------+
| Session Layer   | Agent Layer   | Provider Layer          |
| (session/*.ts)  | (agent.ts)    | (provider/provider.ts)  |
+-----------------+---------------+-------------------------+
|             Tool Registry (tool/registry.ts)              |
|   bash - read - write - edit - grep - glob - webfetch     |
+-----------------------------------------------------------+
|           Permission - Bus - Snapshot - MCP - LSP         |
+-----------------------------------------------------------+
```

### 1.5 How OpenCode Differs from Claude Code

Directly from the README:
- 100% open source
- Not coupled to any single provider
- Out-of-the-box LSP support
- TUI focus (Ink-based terminal rendering)
- Client-server architecture enabling remote operation

### 1.6 The User Experience vs. What Happens Inside

To understand why the codebase is structured the way it is, it helps to contrast what the *user* experiences with what the *engine* does behind the scenes.

**What the user sees** (terminal output):

```
$ opencode
> Build a React component that renders a sortable table

~ Reading project structure...
~ Reading src/components/Table.tsx
~ Writing src/components/SortableTable.tsx
~ Running: npm run typecheck
~ Edit src/components/SortableTable.tsx (fixing type error)
~ Running: npm run typecheck
[x] Done -- created src/components/SortableTable.tsx
```

The user types one sentence and watches a stream of status updates scroll past. It looks almost trivially simple -- but behind those seven lines, the entire system activates:

**What the engine does** (abbreviated trace):

1. **CLI** (`src/index.ts`) -- parses the command, invokes the `tui` or `run` handler
2. **Bootstrap** (`cli/bootstrap.ts`) -- discovers the project root, detects git, opens the SQLite database, starts the HTTP server
3. **Session** (`session/index.ts`) -- creates a session record with a unique ID, persists it to SQLite
4. **Prompt** (`session/prompt.ts`) -- stores the user message, resolves attached files, enters the agentic loop
5. **System prompt** (`session/system.ts`) -- assembles the model's instructions: base rules, project context, tool list, skill docs
6. **Agent resolution** (`agent/agent.ts`) -- loads the `build` agent with its model, permissions, and tool set
7. **LLM bridge** (`session/llm.ts`) -- calls `streamText()` from the AI SDK with all the assembled options
8. **Stream processing** (`session/processor.ts`) -- consumes the token stream, detects tool calls, emits bus events
9. **Tool execution** (`tool/*.ts`) -- runs each tool (read, write, bash) through the permission system
10. **Loop iteration** -- the `loop()` function detects `finishReason === "tool-calls"` and calls the LLM again
11. **Verification** -- the model calls `bash` to run the type checker, reads the output, makes corrections
12. **Compaction** (if needed, `session/compaction.ts`) -- summarizes earlier messages when the context window fills
13. **Completion** -- the model returns `finishReason === "stop"` and the loop exits

Steps 7-11 repeat multiple times. What the user experiences as "a few status lines" is often 3-8 full round trips to the LLM, each with hundreds of tool calls in between.

### 1.7 A Prompt's Journey Through Every Layer

The following diagram traces a single user prompt from entry to exit, showing every major subsystem it touches. This is the roadmap for the rest of the book -- each numbered step corresponds to one or more chapters.

```
User Input
    |
    v
+--------------------+  Ch 4
| CLI Entry (yargs)  |------------------------------------------+
+--------+-----------+                                          |
         v                                                      |
+--------------------+  Ch 5                                    |
| Bootstrap          |  - project discovery                     |
|                    |  - database init                         |
|                    |  - server start                          |
+--------+-----------+                                          |
         v                                                      |
+--------------------+  Ch 6, 11                                |
| Config + Agent     |  - load opencode.json                    |
| Resolution         |  - resolve build agen                    |
|                    |  - merge permissions                     |
+--------+-----------+                                          |
         v                                                      |
+--------------------+  Ch 14                                   |
| Prompt Ingestion   |  - parse user text                       |
|                    |  - resolve @file references              |
|                    |  - store user message                    |
+--------+-----------+                                          |
         v                                                      |
+------------------------------------------------+              |
|              AGENTIC LOOP (Ch 18)              |              |
|                                                |              |
|  +------------------------------------------+  |              |
|  | System Prompt Assembly (Ch 15)           |  |              |
|  |  - base rules + tools + environment      |  |              |
|  +--------------------+---------------------+  |              |
|                       v                        |              |
|  +------------------------------------------+  |              |
|  | LLM Call (Ch 8, 9, 16)                   |  |              |
|  |  - provider --> model --> streamText()   |  |              |
|  +--------------------+---------------------+  |              |
|                       v                        |              |
|  +------------------------------------------+  |              |
|  | Stream Processing (Ch 17)                |  |              |
|  |  - tokens --> parts --> bus events       |  |              |
|  +--------------------+---------------------+  |              |
|                       v                        |              |
|  +------------------------------------------+  |              |
|  | Tool Execution (Ch 19-24)                |  |              |
|  |  - permission check (Ch 25)              |  |              |
|  |  - run tool --> return result            |  |              |
|  |  - snapshot file changes (Ch 27)         |  |              |
|  +--------------------+---------------------+  |              |
|                       v                        |              |
|  finishReason === "tool-calls"? --Yes--> LOOP  |              |
|                       | No                     |              |
|                       v                        |              |
|  Compaction needed? (Ch 28) --Yes--> LOOP      |              |
|                       | No                     |              |
|                       v                        |              |
|                    EXIT LOOP                   |              |
+------------------------------------------------+              |
         |                                                      |
         v                                                      |
+--------------------+  Ch 12, 13                               |
| Session Persist    |  - store assistant message               |
|                    |  - update session title                  |
+--------+-----------+                                          |
         v                                                      |
+--------------------+  Ch 26, 31-33                            |
| Bus --> Client     |  - SSE events --> TUI/web/SDK            |
+--------------------+                                          |
```

Every box in this diagram is a chapter in this book. By the time you reach the final chapter you'll have traced a prompt from the user's keystroke all the way to the model's response and back.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Entry point | `packages/opencode/src/index.ts` |
| Server | `packages/opencode/src/server/server.ts` |
| Agentic loop | `packages/opencode/src/session/prompt.ts` |
| Agent definitions | `packages/opencode/src/agent/agent.ts` |
| Provider layer | `packages/opencode/src/provider/provider.ts` |

