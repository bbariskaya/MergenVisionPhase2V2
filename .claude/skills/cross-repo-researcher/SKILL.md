---
name: cross-repo-researcher
description: "MergenVisionDemo reposunu codebase-memory-mcp ile ULTRA-THOROUGH modda arastirir, kritik pattern'leri cikarir ve sonuclari hem Phase2v2'ye knowledge pack dosyasi hem de prompt-memory graph'ine yazar. Kullanim: /cross-repo-researcher"
disable-model-invocation: true
allowed-tools: Agent Bash(mkdir *) Read mcp__prompt-memory-mcp__store_memory mcp__prompt-memory-mcp__search_memories mcp__prompt-memory-mcp__relate_nodes
---

# /cross-repo-researcher — Smart Cross-Repo Intelligence (Ultra-Thorough Mode)

Amaç: Hedef repoyu codebase-memory-mcp kullanarak **kapsamlı, çapraz-doğrulanmış ve detaylı** bir şekilde analiz etmek. Bulgular hem markdown knowledge pack olarak hem de prompt-memory graph'inde küçük, tag'li node'lara dönüştürülerek kaydedilir. Model qwen'dir; ultra-thorough davranış tamamen prompt engineering, zorunlu exploration adımları ve self-verification talimatları ile sağlanır.

## Sabitler

- Kaynak proje (okuma-only): `MergenVisionDemo`
- Kaynak repo kökü: `/home/user/MergenVisionDemo`
- Hedef markdown dosyası: `/home/user/Workspace/MergenVisionPhase2v2/.claude/cross-repo-intelligence/mergenvision-demo-insights.md`
- prompt-memory root parent: `MergenVision` Project node (cross-repo intelligence root)
- prompt-memory external repo parent: `MergenVisionDemo` Memory node
- prompt-memory topic child'ları: `MergenVisionDemo: overview`, `MergenVisionDemo: bulk-enrollment`, `MergenVisionDemo: gpu-runtime`, `MergenVisionDemo: persistence`, `MergenVisionDemo: identity-model`, `MergenVisionDemo: recommendations`

## Kurallar (kesin)

1. **Asla hedef repo source file'larını değiştir.** Sadece `Read` ve codebase-memory-mcp tool'ları ile oku.
2. **Yazma izinleri sadece şunlar:** Phase2v2'deki hedef markdown dosyası ve prompt-memory graph'i. Başka hiçbir dosyaya `Write` veya `Edit` yapma.
3. **Agent kullan.** `Agent` tool'u ile görevi başlat. Agent tamamlandığında kendiliğinden kapanacak.
4. **Model qwen'dir; Anthropic modelleri (opus/sonnet/fable/haiku) kullanılmaz.** Ultra-thorough davranış prompt engineering, zorunlu exploration adımları ve self-verification ile sağlanır.
5. **Agent'a sadece izin verilen tool'ları söyle:**
   - `Read` (sadece okuma)
   - `mcp__codebase-memory-mcp__search_graph`
   - `mcp__codebase-memory-mcp__search_code`
   - `mcp__codebase-memory-mcp__trace_path`
   - `mcp__codebase-memory-mcp__get_code_snippet`
   - `mcp__codebase-memory-mcp__query_graph`
   - `mcp__codebase-memory-mcp__get_architecture`
   - `mcp__prompt-memory-mcp__search_memories`
   - `mcp__prompt-memory-mcp__store_memory`
   - `mcp__prompt-memory-mcp__relate_nodes`
   - `Write` (sadece hedef markdown dosyasına)
6. **Agent asla şunları kullanmasın:** `Edit`, `Bash`, `NotebookEdit`, herhangi bir source file'a `Write`, codebase-memory ADR write.

## Agent görevi (Ultra-Thorough)

Agent'a şu prompt'u ver. Bu prompt zorunlu exploration protokolü, çift doğrulama ve self-verification içerir:

```
You are a cross-repo research agent running in ULTRA-THOROUGH mode. Your job is to analyze the target repository exhaustively using codebase-memory-mcp and persist the findings in TWO places:

1. A structured markdown knowledge pack at the target markdown path below.
2. The prompt-memory graph as small, tagged, linked Memory nodes under the MergenVision cross-repo root.

ULTRA-THOROUGH RULES:
- Think extensively before each conclusion. Do not rush.
- Every factual claim about files, symbols, numbers, or behavior must be verified by at least TWO independent codebase-memory searches (e.g., search_graph + search_code, or search_code + trace_path).
- If two searches disagree, perform a third search or read the source snippet to resolve.
- If a topic or symbol is not found, explicitly state "NOT FOUND" and list what you searched for. Do not hallucinate.
- Use get_architecture once at the start to understand the repo structure.
- For every major symbol you cite, use trace_path to confirm its callers and callees.
- Before writing the final output, re-run your key searches one more time to confirm file paths and symbol names.

Repository-specific constants (fill these in before running):
- Source project name for codebase-memory-mcp: "MergenVisionDemo"
- Source repo root (for context only, never modify): /home/user/MergenVisionDemo
- Target markdown file: /home/user/Workspace/MergenVisionPhase2v2/.claude/cross-repo-intelligence/mergenvision-demo-insights.md
- External repo parent node name for prompt-memory: "MergenVisionDemo"
- External repo tag prefix: "mergenvision-demo"
- Child node prefix: "MergenVisionDemo: <topic>"

Hard constraints:
- NEVER modify any source file in the source repo.
- You may ONLY write to the target markdown file in the Phase2v2 repo.
- Use Read only for source references if needed.
- Prefer codebase-memory-mcp tools for exploration.
- For prompt-memory, use search_memories, store_memory, and relate_nodes only.

MANDATORY EXPLORATION PROTOCOL for each topic:
1. search_graph — natural-language query for the topic (e.g., "bulk enrollment producer consumer queue").
2. search_code — grep-style search for the most important symbols you discovered.
3. get_code_snippet — read the critical function/class bodies to confirm behavior.
4. trace_path — follow CALLS edges inbound and outbound for the critical symbols.
5. query_graph (optional) — if you need complexity metrics or entry-point lists.

Investigate these topics:
1. Bulk enrollment architecture: BulkEnrollmentService, _extract_batch_faces, _persist_batch, producer/consumer flow, batch sizes, executors.
2. GPU hot-path: GpuFacePipeline.extract_batch / extract_bytes / embed, batch sizes, DeviceTensor/BufferArena, how images stay on GPU.
3. Native runtime integration: how the C++/CUDA runtime is built, loaded, and called from Python (pybind11, CMake, Dockerfile).
4. Persistence pattern: MinIO upload, PostgreSQL bulk upserts (pg_insert on_conflict), Qdrant upsert_batch.
5. Identity/person model: FaceIdentity, Person, PersonPhoto, FaceSample, deterministic IDs.
6. Worker/API orchestration: bulk_orchestrator, gpu_worker, bulk_jobs routes, ProcessRecord lifecycle.
7. Anything else Phase2v2 could reuse for bulk enrollment or a future video pipeline.

Markdown output format:
# <RepoName> Cross-Repo Knowledge Pack

## Executive Summary
3-5 sentences summarizing the repo and the most important takeaways for Phase2v2.

## Bulk Enrollment
- Key files and symbols (with verified paths)
- Algorithm / flow
- Batch sizes and GPU usage
- What Phase2v2 can copy

## GPU / Native Runtime
- Key files and symbols
- Batch inference details
- Python ↔ native bridge

## Persistence & Storage
- MinIO, Postgres, Qdrant patterns
- Bulk upsert examples (verified code snippets)

## Identity Model
- Person/FaceIdentity/FaceSample design and deterministic IDs

## Worker / API Orchestration
- How jobs are dispatched, tracked, and resumed/cancelled

## Actionable Recommendations for Phase2v2
Numbered list of concrete next steps. Each recommendation must cite the file/symbol it is based on.

SELF-VERIFICATION CHECKLIST (perform before writing):
- [ ] Every file path in the markdown also appears in a codebase-memory search result.
- [ ] Every symbol name is confirmed by search_code or get_code_snippet.
- [ ] Every number (batch size, concurrency limit, vector dimension) is backed by source.
- [ ] Prompt-memory node names and tags match the constants above exactly.

Prompt-memory graph output instructions:
- First, search_memories for a Project node named "MergenVision". If not found, create it with label Project, name "MergenVision", content "Cross-repo intelligence root for the MergenVision family. Holds linked memories for Phase2v2 (CURRENT repository) and external reference repositories. NEVER assume external patterns exist in Phase2v2 unless explicitly ported.", tags ["mergenvision", "cross-repo", "parent"].
- Create (or reuse if found) a Memory node named "MergenVisionDemo" with content "SEPARATE EXTERNAL REFERENCE REPOSITORY: /home/user/MergenVisionDemo. Parent node for all intelligence gathered from MergenVisionDemo. NEVER assume these patterns exist in Phase2v2 unless explicitly ported." and tags ["mergenvision-demo", "external-repo", "cross-repo", "parent"].
- Relate the "MergenVisionDemo" node to the "MergenVision" root with edge type RELATED_TO and properties {"role": "external-repo"}.
- Create exactly these child Memory nodes under "MergenVisionDemo":
  * "MergenVisionDemo: overview" — tags ["mergenvision-demo", "external-repo", "cross-repo", "overview"]
  * "MergenVisionDemo: bulk-enrollment" — tags ["mergenvision-demo", "external-repo", "cross-repo", "bulk-enrollment"]
  * "MergenVisionDemo: gpu-runtime" — tags ["mergenvision-demo", "external-repo", "cross-repo", "gpu-runtime"]
  * "MergenVisionDemo: persistence" — tags ["mergenvision-demo", "external-repo", "cross-repo", "persistence"]
  * "MergenVisionDemo: identity-model" — tags ["mergenvision-demo", "external-repo", "cross-repo", "identity-model"]
  * "MergenVisionDemo: recommendations" — tags ["mergenvision-demo", "external-repo", "cross-repo", "recommendations"]
- Each child node content must start with "EXTERNAL REPO: MergenVisionDemo — " and be concise (only key files/symbols, numbers, and concrete takeaways; agents must not have to read long text).
- Relate each child to the "MergenVisionDemo" parent with edge type RELATED_TO and properties {"role": "<topic>"}.

After writing the markdown file, creating all prompt-memory nodes, and completing the self-verification checklist, return a one-sentence summary in Turkish. Then terminate.
```

## Çalışma akışı

1. Hedef dizinin var olduğundan emin ol: `.claude/cross-repo-intelligence/`
2. Agent'ı **qwen modeliyle** ve yukarıdaki ultra-thorough prompt ile çalıştır. Model değiştirilemez; derinlik tamamen promptun zorunlu exploration protokolü, çift doğrulama ve self-verification checklist'i ile sağlanır.
3. Agent çalışırken ara verme; her topic için search_graph → search_code → get_code_snippet → trace_path döngüsünün tamamlandığından emin ol.
4. Agent tamamlandığında:
   - Hedef markdown dosyasını `Read` ile doğrula.
   - prompt-memory'de `MergenVision` root node'unu ve altındaki external-repo parent + child node'ları doğrula (`mcp__prompt-memory__get_memory` ile uuid'den).
   - Her child node'un `EXTERNAL REPO: <RepoName> —` prefix'i ve doğru tag'leri taşıdığını kontrol et.
5. Kullanıcıya kısa özet, dosya yolunu, prompt-memory root uuid'sini ve oluşturulan child node sayısını söyle.
