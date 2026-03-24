# Inside OpenCode: How an AI Coding Agent Works

**A Code-Level Walkthrough of Architecture, Execution, and Design**

---

You've used AI coding tools. But do you know how they actually work?

*Inside OpenCode* is the first book to crack open a production AI coding agent and walk you through every layer — from the CLI entry point to the agentic loop, from tool execution to stream processing, from permission systems to context compaction.

OpenCode is the open-source AI coding agent built by the team at Anomaly. It connects to 20+ LLM providers, executes multi-step tool chains autonomously, and runs entirely in your terminal. This book traces every line of its TypeScript source, showing you exactly how a natural-language prompt becomes working code.

## What You'll Learn

**The Agentic Loop** — How a `while(true)` loop drives the entire system. The model calls tools, reads results, and decides whether to continue or stop. You'll see the exact control flow that turns a single prompt into 15–30 autonomous steps.

**Tool Architecture** — How 22 built-in tools (file read/write/edit, bash execution, web fetch, code search, sub-agent spawning) are registered, permission-checked, and executed. How MCP extends the tool set at runtime.

**Provider Abstraction** — One interface, twenty backends. How a model ID string like `anthropic/claude-sonnet-4-20250514` is resolved through fuzzy matching, config overrides, and the Vercel AI SDK into a callable language model.

**Stream Processing** — How raw LLM token streams are parsed into structured parts, how tool calls are detected mid-stream, and how events flow through the bus to drive real-time UI updates.

**The Full Stack** — Bootstrap and project discovery. Configuration merging. Session persistence in SQLite. Snapshot-based git change tracking. Context compaction when conversations grow too long. Permission rules that balance autonomy with safety.

## Who This Book Is For

- **Engineers building AI-powered tools** who want a complete, working reference architecture
- **Developers curious about what happens inside** tools like Claude Code, Cursor, and GitHub Copilot
- **Contributors to OpenCode** who need a deep understanding of the codebase
- **Technical leaders evaluating AI agents** who want to understand the engineering behind the capabilities

## What Makes This Book Different

This isn't a tutorial or a conceptual overview. Every chapter points to specific source files, traces concrete execution paths, and references the project's test suite as executable documentation. Diagrams show data flow through real functions. Code excerpts are annotated with the design decisions behind them.

The book covers 33 chapters across eight parts — from orientation and startup through providers, sessions, the prompt-to-response loop, tools, supporting infrastructure, and client interfaces. Each chapter includes a source file map and test references so you can follow along in the code.

**33 chapters · 8 parts · 100% open source**

*By Erich Champion · ISBN 979-8-9937022-3-0*

## Keywords

1. AI coding agent architecture source code
2. LLM agentic loop tool calling internals
3. open source AI assistant TypeScript
4. building AI developer tools from scratch
5. how AI coding assistants work under the hood
6. multi-step autonomous code generation
7. AI agent permission systems and safety
