# Inside OpenCode: How an AI Coding Agent Works

**A Code-Level Walkthrough of Architecture, Execution, and Design**

---

## Part I: Orientation

1. [Chapter 1: What is OpenCode?](chapter-01-what-is-opencode.md)
2. [Chapter 2: Repository Tour -- From Monorepo Root to Source Tree](chapter-02-repository-tour.md)
3. [Chapter 3: Technology Stack and Key Dependencies](chapter-03-technology-stack.md)

---

## Part II: Startup and Initialization

4. [Chapter 4: CLI Entry Point -- Parsing Commands with Yargs](chapter-04-cli-entry-point.md)
5. [Chapter 5: Bootstrap -- Project Discovery, Database, and Instance Lifecycle](chapter-05-bootstrap.md)
6. [Chapter 6: Configuration -- Loading, Merging, and Validating Settings](chapter-06-configuration.md)
7. [Chapter 7: The Server -- Building an HTTP API with Hono](chapter-07-server.md)

---

## Part III: Providers and Models

8. [Chapter 8: The Provider Abstraction -- One Interface, Twenty Backends](chapter-08-provider-abstraction.md)
9. [Chapter 9: Loading a Model -- From Config String to Language Model](chapter-09-loading-a-model.md)
10. [Chapter 10: Authentication and Credentials](chapter-10-authentication.md)

---

## Part IV: Agents and Sessions

11. [Chapter 11: Agents -- Roles, Permissions, and Personalities](chapter-11-agents.md)
12. [Chapter 12: Sessions -- Creating, Persisting, and Resuming Conversations](chapter-12-sessions.md)
13. [Chapter 13: Messages -- The Data Model Behind Every Turn](chapter-13-messages.md)

---

## Part V: The Prompt-to-Response Loop

14. [Chapter 14: Receiving a Prompt -- From User Input to the Agentic Loop](chapter-14-receiving-a-prompt.md)
15. [Chapter 15: The System Prompt -- Identity, Context, and Instructions](chapter-15-system-prompt.md)
16. [Chapter 16: Calling the LLM -- The streamText Bridge](chapter-16-calling-the-llm.md)
17. [Chapter 17: Stream Processing -- From LLM Events to Persistent State](chapter-17-stream-processing.md)
18. [Chapter 18: The Agentic Loop -- Multi-Step Execution and Continuation](chapter-18-agentic-loop.md)

---

## Part VI: Tools -- Giving the Agent Hands

19. [Chapter 19: Tool Architecture -- Definition, Registry, and Execution](chapter-19-tool-architecture.md)
20. [Chapter 20: File System Tools -- Read, Write, Edit, and Glob](chapter-20-filesystem-tools.md)
21. [Chapter 21: Code Intelligence Tools -- Grep, Code Search, and File Discovery](chapter-21-code-intelligence.md)
22. [Chapter 22: The Bash Tool -- Command Execution and Permission Parsing](chapter-22-bash-tool.md)
23. [Chapter 23: Web Tools -- Fetch and Search](chapter-23-web-tools.md)
24. [Chapter 24: The Task Tool -- Spawning Sub-Agents](chapter-24-task-tool.md)

---

## Part VII: Supporting Infrastructure

25. [Chapter 25: The Permission System -- Rules, Requests, and Approvals](chapter-25-permissions.md)
26. [Chapter 26: The Event Bus -- Typed Events and Subscribers](chapter-26-event-bus.md)
27. [Chapter 27: Snapshots and Reverting -- Git-Based Change Tracking](chapter-27-snapshots.md)
28. [Chapter 28: Context Compaction -- Summarizing Long Conversations](chapter-28-compaction.md)
29. [Chapter 29: MCP -- Connecting External Tool Servers](chapter-29-mcp.md)
30. [Chapter 30: Plugins -- Extending OpenCode at Runtime](chapter-30-plugins.md)

---

## Part VIII: Clients and Interfaces

31. [Chapter 31: The TUI -- Terminal User Interface](chapter-31-tui.md)
32. [Chapter 32: The CLI -- Commands and Non-Interactive Mode](chapter-32-cli.md)
33. [Chapter 33: The JavaScript SDK -- Programmatic Access](chapter-33-sdk.md)

---

> **Note on Test References:** Chapters 5-33 include a **Test References** section listing the relevant test files from `packages/opencode/test/`. These tests serve as executable documentation -- each one demonstrates a specific behavior discussed in the chapter. Run them with `bun test <path>` from the `packages/opencode` directory.

---

*Inside OpenCode: How an AI Coding Agent Works | Complete Edition*
