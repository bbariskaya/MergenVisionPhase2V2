# prompt-memory OpenCode Integration Design

**Date:** 2026-07-18  
**Scope:** Bind Claude-style prompt-memory hooks into OpenCode, add slash skills (`/remember`, `/recall`, `/get_full_context`, `/cross-repo-researcher`), and make memory retrieval/writing nearly automatic so agents never lose context.

## 1. Goals

- Never rely on reading the OpenCode SQLite DB for automatic capture. Only explicit user messages and agent decisions are stored.
- On every user message: capture the message, retrieve relevant memory, and inject a compact digest into the system prompt so the agent sees context without having to call a tool.
- On compaction / idle / session start: persist snapshots and restore them into the system prompt.
- Keep the TypeScript plugin as a thin shim; all business logic lives in Python (`prompt-memory-mcp` store and a dedicated OpenCode backend CLI).
- Provide OpenCode slash skills that mirror Claude’s `/remember`, `/recall`, `/get_full_context`, plus a new `/cross-repo-researcher` that dispatches a subagent to another repo and stores findings.

## 2. Constraints

- OpenCode plugins must have a TypeScript/JavaScript entry point.
- `chat.message`, `experimental.chat.system.transform`, `experimental.session.compacting`, and `event` hooks are available.
- `tool.execute.before` can mutate tool arguments but cannot inject model-visible text, so code-discovery reminders are delivered via `system.transform` instead.
- MCP server `prompt-memory-mcp` is already registered in `~/.config/opencode/opencode.json`.

## 3. Components

```
┌─────────────────────────────────────────────────────────────────────┐
│ OpenCode                                                            │
│  ┌─────────────────┐   ┌────────────────────┐   ┌───────────────┐  │
│  │ TS plugin shim  │   │ Slash skills       │   │ MCP server    │  │
│  │ ~/.opencode/... │   │ .opencode/skills/  │   │ prompt-memory │  │
│  └────────┬────────┘   └────────────────────┘   └───────┬───────┘  │
│           │                                              │          │
│           │ subprocess (JSON I/O)                        │ MCP      │
│           ▼                                              │ protocol │
│  ┌────────────────────────────────────────┐              │          │
│  │ Python backend                         │              │          │
│  │ ~/.opencode/plugins/prompt-memory/     │              │          │
│  │ prompt_memory_opencode.py              │              │          │
│  └────────┬───────────────────────────────┘              │          │
│           │ imports from /home/user/Workspace/myMcp/src  │          │
│           ▼                                              │          │
│  ┌────────────────────────────────────────┐              │          │
│  │ prompt_memory_mcp store (SQLite)       │◀─────────────┘          │
│  │ ~/.cache/prompt-memory-mcp/store.db    │                         │
│  └────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. TS Plugin Shim (`~/.opencode/plugins/prompt-memory/index.ts`)

Implements these hooks and forwards to the Python backend. Every subprocess call is fire-and-forget with try/catch; failures are logged and never block OpenCode.

| Hook | Purpose |
|------|---------|
| `experimental.chat.system.transform` | Inject session reminder + code-discovery reminder + recent memory digest. |
| `chat.message` | Extract user text from `output.parts`, filter noise, call `store_message`. Then call `retrieve_for_message` and cache digest for next system transform. |
| `experimental.session.compacting` | Call `compact_session`; push returned compactions summary into `output.context`. |
| `event: session.created` | Call `ensure_session`. |
| `event: session.status` (idle) | Call `snapshot_session` to capture outcome of completed work. |
| `tool` | Expose plugin tools: `recall`, `remember`, `compact_recap`, `get_full_context`, `opencode_memory_status`. |

### Subprocess contract

```ts
const result = await $`python3 ${PYTHON_BACKEND} ${command} ${JSON.stringify(args)}`.quiet().nothrow();
return JSON.parse(result.text() || "{}") as Record<string, unknown>;
```

Commands are idempotent where possible (`ensure_session`, `store_message` with deterministic UUIDs, `compact_session` with unique snapshot UUIDs).

## 5. Python Backend Commands (`prompt_memory_opencode.py`)

New commands in addition to existing `recall`, `remember`, `compact_recap`, `session_reminder`, `code_discovery_gate`:

- `ensure_session({session_id, project_name, root})` → create `Session` and `Project` nodes if missing.
- `store_message({session_id, message_id, role, text, timestamp, cwd, git_branch})` → store only user messages as `Message` nodes under the session; UUID = `message:opencode:{session_id}:{message_id}`.
- `compact_session({session_id, max_messages=30})` → read recent user messages, create `Decision` node `opencode-compaction-{session_id}-{timestamp}`; return summary.
- `snapshot_session({session_id})` → lightweight Decision with last N user messages, fired on idle.
- `retrieve_for_message({session_id, query, limit=5})` → hybrid search over `Memory`, `Decision`, `Message` using the user message text; return digest text.
- `get_full_context({session_id, project_name, limit=10})` → replicate `~/.claude/skills/get_full_context`: return a compact Turkish summary of project, active tasks, codebase position, and ask-back sentence.
- `opencode_memory_status({session_id})` → return latest snapshot and memory count.

All commands use `Store` from `/home/user/Workspace/myMcp/src/prompt_memory_mcp`.

## 6. Proactive Memory Loop

1. User sends a message.
2. `chat.message` hook stores it.
3. Same hook calls `retrieve_for_message(query=user_text, session_id)`.
4. The returned digest is cached in the plugin instance for the current session.
5. Next `system.transform` injects the cached digest (plus reminders) into the system prompt.
6. Therefore the agent sees relevant prior context without calling `recall` first.
7. On compaction/idle the whole recent window is persisted as a `Decision`, which is also ranked high in later retrieval.

This replaces the disabled bridge’s “echo” problem by:
- Only indexing user messages and decisions/memories, not assistant/tool outputs.
- Injecting a short digest (default 500 chars, max 3 memories) instead of large raw dumps.
- Busting the cache on session id change.

## 7. Slash Skills (`.opencode/skills/`)

Create these OpenCode skill files in the workspace (they are auto-discovered from `.opencode/skills/` and `.claude/skills/`):

- `prompt-memory/SKILL.md` — generic reminder of available recall/remember tools.
- `remember/SKILL.md` — trigger `/remember`, scan recent context, ask user, then call `remember` plugin tool.
- `recall/SKILL.md` — trigger `/recall`, call `retrieve_memory`-like bulk retrieval via the `recall` plugin tool.
- `get_full_context/SKILL.md` — trigger `/get_full_context`, call the `get_full_context` plugin tool.
- `cross-repo-researcher/SKILL.md` — trigger `/cross-repo-researcher`.

## 8. `/cross-repo-researcher` Skill

Mirrors the Claude skill but uses OpenCode tooling.

**Trigger:** `/cross-repo-researcher [repo_path] [project_name]`  
**Defaults:** `/home/user/MergenVisionDemo`, `MergenVisionDemo`

**Steps:**

1. Call `remember`/`store_memory` to create the parent `Project: MergenVision` node and external repo `Memory: <ProjectName>` node if missing.
2. Use `task` tool with `subagent_type: "explore"` to dispatch an ultra-thorough agent into the target repo.
3. The subagent is allowed only:
   - `Read`
   - `codebase-memory-mcp` graph tools (target repo must be indexed first)
   - `prompt-memory-mcp` `store_memory`, `search_memory`, `relate_nodes`
4. The subagent writes a markdown knowledge pack to `.opencode/cross-repo-intelligence/<project>-insights.md` and creates topic Memory nodes under the external repo parent in prompt-memory.
5. The main agent reads the markdown and confirms the created memory nodes, then returns a one-sentence Turkish summary.

## 9. Configuration Updates

`~/.config/opencode/opencode.json` already references the plugin and MCP correctly. Only the plugin source files and skills will change; no config schema change is required.

For completeness, ensure these env vars stay on the MCP entry:

```json
"PMM_AUTO_INDEX": "false",
"PMM_INDEX_OPENCODE_DB": "false",
"PMM_INDEX_CLAUDE_LOGS": "false"
```

All indexing is explicit (slash command or hook-driven snapshot), not DB polling.

## 10. Error Handling & Observability

- TS plugin: every hook wrapped in `try/catch`; failures logged to `~/.cache/prompt-memory-mcp/opencode-bridge.log`.
- Python backend: every command returns `{"ok": true, ...}` or `{"ok": false, "error": ...}`; never throws.
- If the Python backend fails to open the store, the plugin logs and disables automatic capture; explicit tools still return friendly errors.
- MCP server health is independent; plugin failures do not kill MCP.

## 11. Testing Plan

1. **Python backend unit tests** under `~/.opencode/plugins/prompt-memory/tests/` using a temporary SQLite store:
   - `store_message` deduplication by UUID.
   - `compact_session` creates a Decision.
   - `retrieve_for_message` returns relevant Memory.
   - `get_full_context` returns a non-empty summary.
   - `ensure_session` creates Project/Session nodes.
2. **Plugin build smoke**:
   - `cd ~/.opencode/plugins/prompt-memory && npm run build` (if available) or `npx tsc --noEmit`.
3. **Integration smoke**:
   - Restart OpenCode, verify `/get_full_context` returns context.
   - Send a user message, verify a `Message` node is stored.
   - Compact, verify a `Decision` snapshot is stored.

## 12. Future Extensions (out of scope for first PR)

- Semantic embeddings for OpenCode messages (currently FTS-only for the proactive digest).
- Automatic `cross-repo-researcher` trigger when user mentions an external repo name.
- Cross-session summarization (today we keep messages per session).
