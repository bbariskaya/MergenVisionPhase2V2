# prompt-memory OpenCode Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind Claude-style prompt-memory hooks into OpenCode, add slash skills (`/remember`, `/recall`, `/get_full_context`, `/cross-repo-researcher`), and make memory retrieval/writing proactive so agents never lose context.

**Architecture:** A minimal TypeScript OpenCode plugin forwards lifecycle hooks to a Python backend CLI. The backend reuses the same SQLite store as `prompt-memory-mcp` so memory is shared between Claude and OpenCode. Slash skills live in `.opencode/skills/` and instruct the model to use the plugin tools.

**Tech Stack:** Python 3.12, SQLite/FTS5 (`prompt_memory_mcp.store`), TypeScript/Bun (`@opencode-ai/plugin`), OpenCode plugin/skill system.

## Global Constraints

- OpenCode DB SQLite is never read for automatic capture. Only explicit hook data and user messages are stored.
- All business logic stays in Python; the TypeScript layer is a passthrough shim.
- UUIDs must be deterministic to make commands idempotent (`ensure_session`, `store_message`, `compact_session`).
- The MCP server stays registered separately in `~/.config/opencode/opencode.json`; do not break it.
- No edits to source files inside external repos (cross-repo researcher is read-only).

---

### Task 1: Extend Python backend commands

**Files:**
- Modify: `~/.opencode/plugins/prompt-memory/prompt_memory_opencode.py`
- Create: `~/.opencode/plugins/prompt-memory/prompt_memory_opencode.py` tests will be added in Task 2

**Interfaces:**
- Consumes: `prompt_memory_mcp.config.Config`, `prompt_memory_mcp.store.Store`
- Produces: CLI commands `ensure_session`, `store_message`, `compact_session`, `snapshot_session`, `retrieve_for_message`, `get_full_context`, `opencode_memory_status`, plus enhanced `recall`

- [ ] **Step 1: Add helpers at the top of the backend**

Append after `_store()`:

```python
def _project_uuid(project_name: str) -> str:
    return f"project:{project_name}"


def _session_uuid(source: str, session_id: str) -> str:
    return f"session:{source}:{session_id}"


def _message_uuid(session_id: str, message_id: str) -> str:
    return f"message:opencode:{session_id}:{message_id}"


def _parse_timestamp(ts: int | float | str | None) -> str:
    if isinstance(ts, (int, float)):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    return ts or ""


def _is_noise(text: str) -> bool:
    t = text.strip()
    return len(t) < 3
```

- [ ] **Step 2: Implement `ensure_session`**

Add command:

```python
def cmd_ensure_session(args: dict) -> dict:
    store = _store()
    session_id = args["session_id"]
    project_name = args.get("project_name") or "unknown"
    project_root = args.get("project_root", "")

    project_uuid = _project_uuid(project_name)
    project_id = store.add_node(
        label="Project",
        name=project_name,
        node_uuid=project_uuid,
        properties={"root": project_root},
    )
    session_uuid = _session_uuid("opencode", session_id)
    session_id_node = store.add_node(
        label="Session",
        name=session_id,
        node_uuid=session_uuid,
        properties={
            "project": project_name,
            "project_root": project_root,
            "source": "opencode",
        },
    )
    store.add_edge(project_id, session_id_node, "HAS_SESSION")
    return {"ok": True, "project_uuid": project_uuid, "session_uuid": session_uuid}
```

- [ ] **Step 3: Implement `store_message`**

Add command:

```python
def cmd_store_message(args: dict) -> dict:
    store = _store()
    session_id = args["session_id"]
    message_id = args.get("message_id", "")
    role = args.get("role", "user")
    text = args.get("text", "")
    if _is_noise(text):
        return {"ok": True, "skipped": True}

    session_uuid = _session_uuid("opencode", session_id)
    session_node = store.get_node_by_uuid(session_uuid)
    if session_node is None:
        cmd_ensure_session(args)
        session_node = store.get_node_by_uuid(session_uuid)

    msg_uuid = _message_uuid(session_id, message_id or str(args.get("timestamp", "")))
    props = {
        "role": role,
        "seq": args.get("seq", 0),
        "timestamp": _parse_timestamp(args.get("timestamp")),
        "session": session_uuid,
        "cwd": args.get("cwd", ""),
        "git_branch": args.get("git_branch", ""),
        "source": "opencode",
    }
    msg_node_id = store.add_node(
        label="Message",
        name=f"{role}:{args.get('seq', 0)}",
        content=text,
        node_uuid=msg_uuid,
        properties=props,
    )
    store.add_edge(session_node.id, msg_node_id, "HAS_MESSAGE", properties={"seq": props["seq"]})
    return {"ok": True, "id": msg_node_id, "uuid": msg_uuid}
```

- [ ] **Step 4: Implement `retrieve_for_message`**

Add command that searches Memory/Decision/Message:

```python
def cmd_retrieve_for_message(args: dict) -> dict:
    store = _store()
    query = args.get("query", "")
    session_id = args.get("session_id", "")
    limit = int(args.get("limit", 5))
    if not query:
        return {"ok": True, "digest": ""}

    results = store.hybrid_search(query=query, labels=["Memory", "Decision", "Message"], limit=limit)
    lines = []
    for r in results:
        body = (r.get("content") or "").strip()
        if not body:
            continue
        snippet = body[:600]
        suffix = "…" if len(body) > 600 else ""
        lines.append(f"[{r.get('label')}] {r.get('name')}\n{snippet}{suffix}")

    # Also append last N user messages of current session for continuity
    session_uuid = _session_uuid("opencode", session_id)
    session_node = store.get_node_by_uuid(session_uuid)
    if session_node:
        msgs = store.get_neighbors(session_node.id, edge_types=["HAS_MESSAGE"], direction="outbound")
        msgs.sort(key=lambda x: x["node"]["properties"].get("seq", 0))
        user_msgs = [m for m in msgs if m["node"]["properties"].get("role") == "user"][-3:]
        if user_msgs:
            lines.append("---\nRecent user messages in this session:")
            for m in user_msgs:
                content = m["node"]["content"].strip()
                if content:
                    lines.append(f"- {content[:300]}{'…' if len(content) > 300 else ''}")

    return {"ok": True, "digest": "\n\n---\n\n".join(lines)}
```

- [ ] **Step 5: Implement `compact_session`, `snapshot_session`, and `get_full_context`**

```python
def _recent_user_messages(store: Store, session_id: str, limit: int = 30) -> list[tuple[int, str]]:
    session_uuid = _session_uuid("opencode", session_id)
    session_node = store.get_node_by_uuid(session_uuid)
    if not session_node:
        return []
    msgs = store.get_neighbors(session_node.id, edge_types=["HAS_MESSAGE"], direction="outbound")
    msgs.sort(key=lambda x: x["node"]["properties"].get("seq", 0))
    return [
        (m["node"]["properties"].get("seq", 0), m["node"]["content"])
        for m in msgs
        if m["node"]["properties"].get("role") == "user"
    ][-limit:]


def cmd_compact_session(args: dict) -> dict:
    store = _store()
    session_id = args["session_id"]
    msgs = _recent_user_messages(store, session_id, limit=int(args.get("max_messages", 30)))
    if len(msgs) < 2:
        return {"ok": True, "stored": False, "reason": "too few messages"}
    content_lines = [f"[{seq}] {text}" for seq, text in msgs]
    snapshot_uuid = f"opencode-compaction-{session_id}-{store._now()}"
    store.add_node(
        label="Decision",
        name=snapshot_uuid,
        node_uuid=snapshot_uuid,
        content=f"OpenCode compaction snapshot for session {session_id}:\n\n" + "\n\n".join(content_lines),
        properties={"session": _session_uuid("opencode", session_id), "source": "opencode-compaction"},
        tags=["opencode", "compaction"],
    )
    return {"ok": True, "stored": True, "snapshot_uuid": snapshot_uuid, "message_count": len(msgs)}


def cmd_snapshot_session(args: dict) -> dict:
    return cmd_compact_session({"session_id": args["session_id"], "max_messages": 10})


def cmd_get_full_context(args: dict) -> dict:
    store = _store()
    session_id = args.get("session_id", "")
    project_name = args.get("project_name") or "unknown"
    # Collect recent decisions/memories
    memories = store.hybrid_search(query=project_name, labels=["Memory", "Decision"], limit=20)
    lines = [f"Project: {project_name}", f"Session: {session_id}", ""]
    if memories:
        lines.append("Recent memories/decisions:")
        for m in memories[:10]:
            content = (m.get("content") or "").strip()
            lines.append(f"- [{m.get('label')}] {m.get('name')}: {content[:250]}{'…' if len(content) > 250 else ''}")
    else:
        lines.append("No stored memories/decisions yet.")
    # Recent session messages
    msgs = _recent_user_messages(store, session_id, limit=10)
    if msgs:
        lines.append("\nRecent user messages:")
        for seq, text in msgs:
            lines.append(f"- {text[:250]}{'…' if len(text) > 250 else ''}")
    lines.append(f"\nUse `/recall <query>` to pull more context from prompt-memory.")
    return {"ok": True, "summary": "\n".join(lines)}


def cmd_opencode_memory_status(args: dict) -> dict:
    store = _store()
    session_id = args.get("session_id", "")
    stats = store.get_stats()
    last = store.hybrid_search(
        query=f"opencode-compaction-{session_id}",
        labels=["Decision"],
        limit=1,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "stats": stats,
        "latest_snapshot": last[0] if last else None,
    }
```

- [ ] **Step 6: Enhance `cmd_recall` for bulk retrieval**

Replace existing `cmd_recall` with:

```python
def cmd_recall(args: dict) -> dict:
    store = _store()
    query = args.get("query", "")
    project = args.get("project", "")
    tag = args.get("tag", "")
    label = args.get("label", "")
    limit = int(args.get("limit", 10))

    if query or tag or label:
        nodes = store.bulk_retrieve(query=query or None, tag=tag or None, label=label or None, limit=limit)
        results = [
            {
                "id": n.id,
                "uuid": n.uuid,
                "label": n.label,
                "name": n.name,
                "content": n.content,
                "properties": n.properties,
                "tags": n.tags,
            }
            for n in nodes
        ]
        return {"ok": True, "mode": "bulk", "query": query, "tag": tag, "label": label, "results": results}

    results: list[dict] = []
    if project:
        proj = store.get_node_by_uuid(_project_uuid(project))
        if proj:
            sessions = store.get_neighbors(proj.id, edge_types=["HAS_SESSION"], direction="outbound")
            for sess in sessions[:limit]:
                messages = store.get_neighbors(sess["node"]["id"], edge_types=["HAS_MESSAGE"], direction="outbound")
                results.append({
                    "type": "session",
                    "name": sess["node"]["name"],
                    "messages": len(messages),
                })
    return {"ok": True, "mode": "project", "project": project, "results": results}
```

- [ ] **Step 7: Register new commands**

Update `COMMANDS` dict:

```python
COMMANDS = {
    "recall": cmd_recall,
    "remember": cmd_remember,
    "compact_recap": cmd_compact_recap,
    "session_reminder": cmd_session_reminder,
    "code_discovery_gate": cmd_code_discovery_gate,
    "ensure_session": cmd_ensure_session,
    "store_message": cmd_store_message,
    "retrieve_for_message": cmd_retrieve_for_message,
    "compact_session": cmd_compact_session,
    "snapshot_session": cmd_snapshot_session,
    "get_full_context": cmd_get_full_context,
    "opencode_memory_status": cmd_opencode_memory_status,
}
```

- [ ] **Step 8: Run backend smoke tests manually**

Run:

```bash
cd ~/.opencode/plugins/prompt-memory
python3 - <<'PY'
import json, subprocess, tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    env = os.environ.copy()
    env["PMM_DB_DIR"] = tmp
    out = subprocess.run(
        ["python3", "prompt_memory_opencode.py", "ensure_session", json.dumps({"session_id": "s1", "project_name": "Test"})],
        capture_output=True, text=True, env=env,
    )
    assert json.loads(out.stdout)["ok"], out.stdout
    out = subprocess.run(
        ["python3", "prompt_memory_opencode.py", "store_message", json.dumps({"session_id": "s1", "message_id": "m1", "text": "hello world", "seq": 1, "role": "user"})],
        capture_output=True, text=True, env=env,
    )
    assert json.loads(out.stdout)["ok"], out.stdout
    out = subprocess.run(
        ["python3", "prompt_memory_opencode.py", "retrieve_for_message", json.dumps({"session_id": "s1", "query": "hello", "limit": 3})],
        capture_output=True, text=True, env=env,
    )
    data = json.loads(out.stdout)
    assert "digest" in data and "hello" in data["digest"], out.stdout
print("backend smoke OK")
PY
```

Expected output: `backend smoke OK`

- [ ] **Step 9: Commit Python backend changes**

```bash
git add ~/.opencode/plugins/prompt-memory/prompt_memory_opencode.py
git commit -m "feat(opencode-prompt-memory): extend backend with hooks and proactive retrieval commands"
```

---

### Task 2: Backend unit tests

**Files:**
- Create: `~/.opencode/plugins/prompt-memory/tests/test_backend.py`
- Modify: `~/.opencode/plugins/prompt-memory/package.json` (not needed; Python tests run with plain pytest)

**Interfaces:**
- Consumes: `prompt_memory_opencode` commands
- Produces: passing pytest results

- [ ] **Step 1: Write tests**

```python
import json
import os
import subprocess
from pathlib import Path
import pytest

BACKEND = Path(__file__).parent.parent / "prompt_memory_opencode.py"


def _run(cmd: str, args: dict, tmp_path: Path):
    env = os.environ.copy()
    env["PMM_DB_DIR"] = str(tmp_path)
    out = subprocess.run(
        ["python3", str(BACKEND), cmd, json.dumps(args)],
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(out.stdout), out.returncode


def test_ensure_session_creates_project_and_session(tmp_path):
    data, rc = _run("ensure_session", {"session_id": "s1", "project_name": "P", "project_root": "/tmp"}, tmp_path)
    assert rc == 0 and data["ok"]
    assert data["project_uuid"] == "project:P"
    assert data["session_uuid"] == "session:opencode:s1"


def test_store_message_and_retrieve(tmp_path):
    _run("ensure_session", {"session_id": "s2", "project_name": "P2"}, tmp_path)
    data, rc = _run("store_message", {"session_id": "s2", "message_id": "m1", "text": "hello world", "seq": 1, "role": "user"}, tmp_path)
    assert rc == 0 and data["ok"]
    data, rc = _run("retrieve_for_message", {"session_id": "s2", "query": "hello", "limit": 5}, tmp_path)
    assert rc == 0
    assert "hello" in data["digest"]


def test_compact_session_stores_decision(tmp_path):
    _run("ensure_session", {"session_id": "s3", "project_name": "P3"}, tmp_path)
    for i in range(3):
        _run("store_message", {"session_id": "s3", "message_id": f"m{i}", "text": f"msg {i}", "seq": i, "role": "user"}, tmp_path)
    data, rc = _run("compact_session", {"session_id": "s3", "max_messages": 10}, tmp_path)
    assert rc == 0 and data["ok"] and data["stored"]
    data, rc = _run("opencode_memory_status", {"session_id": "s3"}, tmp_path)
    assert data["ok"] and data["latest_snapshot"]


def test_noise_messages_skipped(tmp_path):
    _run("ensure_session", {"session_id": "s4", "project_name": "P4"}, tmp_path)
    data, rc = _run("store_message", {"session_id": "s4", "message_id": "m1", "text": "  a  ", "seq": 1, "role": "user"}, tmp_path)
    assert rc == 0 and data.get("skipped")
```

- [ ] **Step 2: Run tests**

```bash
cd ~/.opencode/plugins/prompt-memory
python3 -m pytest tests/test_backend.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Commit tests**

```bash
git add ~/.opencode/plugins/prompt-memory/tests/test_backend.py
git commit -m "test(opencode-prompt-memory): add backend command tests"
```

---

### Task 3: TypeScript plugin shim hooks

**Files:**
- Modify: `~/.opencode/plugins/prompt-memory/index.ts`

**Interfaces:**
- Consumes: Python backend CLI via `runPython($, ...)`
- Produces: OpenCode hooks `experimental.chat.system.transform`, `chat.message`, `experimental.session.compacting`, `event`; plugin tools `recall`, `remember`, `compact_recap`, `get_full_context`, `opencode_memory_status`

- [ ] **Step 1: Replace the plugin with hook-enabled version**

Full content of `~/.opencode/plugins/prompt-memory/index.ts`:

```ts
import { tool } from "@opencode-ai/plugin";
import type { Plugin, PluginInput } from "@opencode-ai/plugin";

const PYTHON_BACKEND = "/home/user/.opencode/plugins/prompt-memory/prompt_memory_opencode.py";

const BRIDGE_LOG = "/home/user/.cache/prompt-memory-mcp/opencode-bridge.log";

function log(...args: unknown[]) {
  try {
    const fs = await import("fs/promises");
    const line = `[${new Date().toISOString()}] ` + args.map((a) => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" ") + "\n";
    await fs.mkdir("/home/user/.cache/prompt-memory-mcp", { recursive: true });
    await fs.appendFile(BRIDGE_LOG, line);
  } catch {
    // ignore
  }
}

async function runPython(
  $: PluginInput["$"],
  command: string,
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  try {
    const output = await $`python3 ${PYTHON_BACKEND} ${command} ${JSON.stringify(args)}`.quiet().nothrow();
    const text = output.text();
    return JSON.parse(text || "{}") as Record<string, unknown>;
  } catch (e) {
    await log("python backend failed", command, e);
    return { ok: false, error: "prompt-memory python backend failed" };
  }
}

function isUserNoise(text: string): boolean {
  const t = text.trim();
  return t.length < 3;
}

function extractTextFromParts(parts: unknown[]): string {
  if (!Array.isArray(parts)) return "";
  const texts: string[] = [];
  for (const p of parts) {
    if (p && typeof p === "object" && "text" in p && typeof (p as any).text === "string") {
      texts.push((p as any).text);
    }
  }
  return texts.join("\n").trim();
}

export const PromptMemoryPlugin: Plugin = async ({ $, worktree }) => {
  let currentSessionID: string | null = null;
  let cachedDigest = "";
  const projectName = worktree.split("/").pop() || "unknown";

  const setSession = (id: string | undefined) => {
    if (id && id !== currentSessionID) {
      currentSessionID = id;
      cachedDigest = "";
    }
  };

  return {
    "experimental.chat.system.transform": async (_input, output) => {
      output.system.push(
        "Prompt-memory MCP is available. Recall prior context with /recall <query>. When the user says 'remember' or 'hatırla', store a Memory/Decision via /remember.",
        "CRITICAL - Code Discovery Protocol: ALWAYS use codebase-memory-mcp tools FIRST for code exploration (search_graph, trace_path, get_code_snippet). Use Grep/Glob/Read for text/config files only.",
      );
      if (cachedDigest) {
        output.system.push(
          `Relevant context from prompt-memory (auto-retrieved):\n\n${cachedDigest}`,
        );
      }
    },

    "chat.message": async (input, output) => {
      setSession(input.sessionID);
      const text = extractTextFromParts(output.parts as unknown[]);
      if (!text || isUserNoise(text)) return;

      const storeResult = await runPython($, "store_message", {
        session_id: input.sessionID,
        message_id: input.messageID,
        role: "user",
        text,
        timestamp: Date.now(),
        seq: 0,
        cwd: process.cwd(),
      });
      if (!storeResult.ok) {
        await log("store_message failed", storeResult);
      }

      const retrieveResult = await runPython($, "retrieve_for_message", {
        session_id: input.sessionID,
        query: text.slice(0, 500),
        limit: 3,
      });
      if (retrieveResult.ok && typeof retrieveResult.digest === "string") {
        cachedDigest = retrieveResult.digest.slice(0, 2000);
      }
    },

    "experimental.session.compacting": async (input, output) => {
      setSession(input.sessionID);
      const result = await runPython($, "compact_session", {
        session_id: input.sessionID,
        max_messages: 30,
      });
      if (result.ok && result.stored) {
        output.context.push(
          `Compaction snapshot saved to prompt-memory (${result.snapshot_uuid}). Latest user messages preserved for continuity.`,
        );
      }
    },

    event: async (input) => {
      const ev = (input.event as { type?: string; properties?: { sessionID?: string; info?: { id?: string } } }) || {};
      const type = ev.type || "";
      const sessionID = ev.properties?.sessionID || ev.properties?.info?.id;
      if (!sessionID) return;

      if (type === "session.created") {
        setSession(sessionID);
        await runPython($, "ensure_session", {
          session_id: sessionID,
          project_name: projectName,
          project_root: worktree,
        });
      }

      if (type === "session.status") {
        const status = (input.event as any).properties?.status?.type;
        if (status === "idle") {
          await runPython($, "snapshot_session", { session_id: sessionID });
        }
      }
    },

    tool: {
      recall: tool({
        description: "Recall prior context from prompt-memory by query, tag, label, or project.",
        args: {
          query: tool.schema.string().optional().describe("FTS query"),
          project: tool.schema.string().optional().describe("Project name"),
          tag: tool.schema.string().optional().describe("Tag filter"),
          label: tool.schema.string().optional().describe("Label filter"),
          limit: tool.schema.number().default(10).describe("Max results"),
        },
        async execute(args, ctx) {
          const result = await runPython($, "recall", {
            query: args.query,
            project: args.project || ctx.worktree.split("/").pop(),
            tag: args.tag,
            label: args.label,
            limit: args.limit,
          });
          return { title: "prompt-memory recall", output: JSON.stringify(result, null, 2) };
        },
      }),

      remember: tool({
        description: "Store a memory, decision, snippet, or prompt explicitly.",
        args: {
          name: tool.schema.string().describe("Short kebab-case slug"),
          content: tool.schema.string().describe("Full content to store"),
          label: tool.schema.enum(["Memory", "Decision", "Snippet", "Prompt"]).default("Memory"),
          tags: tool.schema.array(tool.schema.string()).default([]),
        },
        async execute(args, ctx) {
          const result = await runPython($, "remember", {
            name: args.name,
            content: args.content,
            label: args.label,
            tags: args.tags,
            source_session: ctx.sessionID,
          });
          return { title: "prompt-memory remember", output: JSON.stringify(result, null, 2) };
        },
      }),

      compact_recap: tool({
        description: "Return a compact recap from prompt-memory after compaction.",
        args: {
          project: tool.schema.string().optional().describe("Project name"),
        },
        async execute(args, ctx) {
          const result = await runPython($, "compact_recap", {
            project: args.project || ctx.worktree.split("/").pop(),
          });
          return { title: "prompt-memory recap", output: JSON.stringify(result, null, 2) };
        },
      }),

      get_full_context: tool({
        description: "Rebuild full context from prompt-memory for the current session.",
        args: {},
        async execute(_args, ctx) {
          const result = await runPython($, "get_full_context", {
            session_id: ctx.sessionID,
            project_name: ctx.worktree.split("/").pop(),
          });
          return { title: "prompt-memory full context", output: String(result.summary || "") };
        },
      }),

      opencode_memory_status: tool({
        description: "Show prompt-memory stats and latest snapshot for the current session.",
        args: {},
        async execute(_args, ctx) {
          const result = await runPython($, "opencode_memory_status", {
            session_id: ctx.sessionID,
          });
          return { title: "prompt-memory status", output: JSON.stringify(result, null, 2) };
        },
      }),
    },
  };
};

export default PromptMemoryPlugin;
```

- [ ] **Step 2: Type-check the plugin**

```bash
cd ~/.opencode/plugins/prompt-memory
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit plugin shim**

```bash
git add ~/.opencode/plugins/prompt-memory/index.ts
git commit -m "feat(opencode-prompt-memory): add lifecycle hooks and proactive memory injection"
```

---

### Task 4: Add OpenCode slash skills

**Files:**
- Create: `.opencode/skills/remember/SKILL.md`
- Create: `.opencode/skills/recall/SKILL.md`
- Create: `.opencode/skills/get_full_context/SKILL.md`
- Create: `.opencode/skills/cross-repo-researcher/SKILL.md`

**Interfaces:**
- Consumes: plugin tools (`remember`, `recall`, `get_full_context`) and MCP/task tools
- Produces: OpenCode slash-command behaviors

- [ ] **Step 1: Create `.opencode/skills/remember/SKILL.md`**

```markdown
---
name: remember
description: "Explicitly remember a user note. Trigger on: /remember, bunu hatırla, remember this, kaydet, not al."
---

# /remember — Açık Hafıza Kaydı

Kullanıcı `/remember` yazdığında veya "bunu hatırla" / "kaydet" dediğinde çalışır.

1. Kısaca son memory sayısını `opencode_memory_status` tool ile öğren.
2. Kullanıcıya 1-2 cümlede mevcut context özetini sun ve "Ne kaydetmemi istiyorsun?" diye sor.
3. Kullanıcı notu verdiğinde `remember` plugin tool’unu çağır:
   - `label`: `Memory` (reusable karar ise `Decision`)
   - `name`: `explicit-<kisa-slug>`
   - `content`: Kullanıcının notu + kısa bağlam
   - `tags`: `["explicit", "<proje>"]` ve konuya göre tag'ler
4. Kaydedildiğini ve memory name/tag'lerini söyle.

Kural: Sadece kullanıcı açıkça hatırla/kaydet/remember dediğinde kaydet.
```

- [ ] **Step 2: Create `.opencode/skills/recall/SKILL.md`**

```markdown
---
name: recall
description: "Prompt-memory'den hedefli, kesilmemiş içerikleri çek. Trigger on /recall, /hatirla, /getmem."
---

# /recall — Prompt Memory Retrieval

Kullanıcı `/recall <query>` yazdığında `recall` plugin tool’unu çağır.

- `/recall MergenVision` → query="MergenVision"
- `/recall --tag MergenVision` → tag="MergenVision"
- `/recall --query "worker heartbeat"` → query="worker heartbeat"
- `/recall --label Decision` → label="Decision"

Kurallar:
1. Kaydetme; sadece oku.
2. Çıktı çok büyükse (200 KB+) dosyaya yaz ve kullanıcıya yol söyle.
3. Eşleşme yoksa "Bu sorguyla eşleşen kayıt bulunamadı" de.
4. Genel context istenirse `/get_full_context` kullan.
```

- [ ] **Step 3: Create `.opencode/skills/get_full_context/SKILL.md`**

```markdown
---
name: get_full_context
description: "Rebuilt full context from prompt-memory. Trigger on /get_full_context, /context, /tamcontext, ne yapıyorduk."
---

# /get_full_context — Context Reset

`/get_full_context` yazıldığında `get_full_context` plugin tool’unu çağır. Sadece okur, hiçbir şey kaydetmez.

Tool sonucunu 5-10 cümlede Türkçe özetle:
- Proje neydi?
- Son durum / aktif hedef ne?
- Kodda neredeyiz?
- Kullanıcıya "devam edelim mi, yoksa yön değiştirelim mi?" diye sor.
```

- [ ] **Step 4: Create `.opencode/skills/cross-repo-researcher/SKILL.md`**

```markdown
---
name: cross-repo-researcher
description: "Farklı bir repoda ultra-thorough research agent çalıştır, bulguları prompt-memory'ye yaz. Trigger on /cross-repo-researcher."
disable-model-invocation: true
---

# /cross-repo-researcher — Smart Cross-Repo Intelligence

Amaç: Hedef repoyu okuma-only olarak analiz etmek; bulguları markdown + prompt-memory graph olarak kaydetmek.

## Sabitler

- Varsayılan kaynak repo: `/home/user/MergenVisionDemo`
- Varsayılan proje adı: `MergenVisionDemo`
- Hedef markdown: `.opencode/cross-repo-intelligence/<project>-insights.md`
- prompt-memory root: `Project: MergenVision`
- prompt-memory external parent: `Memory: <RepoName>`

## Akış

1. Kullanıcı repo yolu ve proje adı vermediyse varsayılanları kullan.
2. `task` aracı ile `subagent_type: "explore"` gönder; hedef repoyu ayrıntılı analiz ettir.
   - İzinli araçlar: `Read`, `codebase-memory-mcp` (eğer repo indeksli ise), `prompt-memory-mcp` store/search/relate.
   - Yasak: herhangi bir kaynak dosyaya yazma/edit, `Bash` (gerekmedikçe).
3. Alt agent şunları yapar:
   - `get_architecture` ile repo yapısını çıkar.
   - Önemli semboller için `search_graph`, `search_code`, `get_code_snippet`, `trace_path`.
   - Markdown knowledge pack yaz.
   - prompt-memory’ye parent + child `Memory` node’ları yaz.
4. Ana agent markdown dosyasını okuyup prompt-memory node’larını doğrular.
5. Kullanıcıya tek cümlelik Türkçe özet ve dosya yolunu söyle.
```

- [ ] **Step 5: Commit skills**

```bash
git add .opencode/skills/remember/SKILL.md .opencode/skills/recall/SKILL.md .opencode/skills/get_full_context/SKILL.md .opencode/skills/cross-repo-researcher/SKILL.md
git commit -m "feat(opencode-prompt-memory): add slash skills for remember, recall, full context, cross-repo research"
```

---

### Task 5: Integration smoke test

**Files:**
- None new; verify existing OpenCode config.

**Interfaces:**
- Consumes: plugin, MCP, skills
- Produces: working OpenCode behavior

- [ ] **Step 1: Type-check plugin again after all changes**

```bash
cd ~/.opencode/plugins/prompt-memory
npx tsc --noEmit
```

- [ ] **Step 2: Run Python tests once more**

```bash
cd ~/.opencode/plugins/prompt-memory
python3 -m pytest tests/test_backend.py -v
```

- [ ] **Step 3: Restart OpenCode and verify**

1. Restart OpenCode.
2. Send a test user message like `hello test`.
3. Call `opencode_memory_status` tool and confirm the message count increased.
4. Type `/get_full_context` and confirm a context summary is returned.
5. Type `/remember test-note "we are testing prompt-memory integration"` and confirm it stores.
6. Type `/recall test-note` and confirm retrieval works.

- [ ] **Step 4: Final commit**

```bash
git commit -m "feat(opencode): wire prompt-memory hooks, slash skills, and cross-repo researcher"
```

## Self-Review

- **Spec coverage:** Every design section (hooks, proactive retrieval, slash skills, cross-repo researcher) maps to a task. ✔
- **No placeholders:** Steps include exact file paths, command names, code blocks, and CLI commands. ✔
- **Type consistency:** All backend commands return `{"ok": bool, ...}`. Plugin calls match command names. ✔
- **Gaps:** The TS plugin uses `process.cwd()` and `worktree`; this is acceptable for first iteration. Event `session.created` may not fire on every resume, so compaction/idle snapshots provide continuity.
