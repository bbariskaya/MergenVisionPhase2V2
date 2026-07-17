# Reference Decision Log — Phase 1 Sprint 01

## Decision Record Template

| Field | Value |
|---|---|
| Decision ID | — |
| Local feature/symbol | — |
| Reference URL | — |
| Repository commit/tag | — |
| Access date | 2026-07-16 |
| Repository license | — |
| Inspected upstream files/symbols | — |
| Behavior adopted | — |
| Behavior explicitly rejected | — |
| Local modifications | — |
| Failing test/reproducer | — |
| Parity/runtime acceptance command | — |
| Known limitation | — |

---

## DEC-001 — SQLAlchemy 2.0 DeclarativeBase

| Field | Value |
|---|---|
| Decision ID | DEC-001 |
| Local feature/symbol | `backend/app/infrastructure/persistence/sqlalchemy/base.py` |
| Reference URL | <https://docs.sqlalchemy.org/en/20/orm/mapping_api.html#sqlalchemy.orm.DeclarativeBase> |
| Repository commit/tag | SQLAlchemy 2.0.31 |
| Access date | 2026-07-16 |
| Repository license | MIT |
| Inspected upstream files/symbols | `DeclarativeBase` class |
| Behavior adopted | Use `class Base(DeclarativeBase): pass` instead of legacy `declarative_base()`. |
| Behavior explicitly rejected | Legacy `declarative_base()` factory. |
| Local modifications | None. |
| Failing test/reproducer | `tests/integration/persistence/test_migrations.py::test_upgrade_head_creates_tables` |
| Parity/runtime acceptance command | `make phase1-sprint-01-postgres` |
| Known limitation | None. |

---

## DEC-002 — UUIDv7 via `uuid7` PyPI Package

| Field | Value |
|---|---|
| Decision ID | DEC-002 |
| Local feature/symbol | `backend/app/infrastructure/uuid7.py` |
| Reference URL | <https://pypi.org/project/uuid7/0.1.0/> |
| Repository commit/tag | uuid7 0.1.0 |
| Access date | 2026-07-16 |
| Repository license | MIT (per PyPI) |
| Inspected upstream files/symbols | `uuid_extensions.uuid7()` |
| Behavior adopted | Import `uuid7` from the `uuid_extensions` module and cast to `uuid.UUID`. |
| Behavior explicitly rejected | UUIDv4 and custom UUIDv7 implementation. |
| Local modifications | `generate_uuid7()` wraps `uuid.UUID(str(uuid7()))` for type safety. |
| Failing test/reproducer | `tests/unit/infrastructure/test_uuid7.py` |
| Parity/runtime acceptance command | `cd backend && python -m pytest tests/unit/infrastructure/test_uuid7.py -v` |
| Known limitation | `uuid7()` return type is untyped; wrapped to satisfy mypy strict mode. |

---

## DEC-003 — Async MinIO SDK Calls via `asyncio.to_thread`

| Field | Value |
|---|---|
| Decision ID | DEC-003 |
| Local feature/symbol | `backend/app/infrastructure/storage/minio_adapter.py` |
| Reference URL | <https://docs.min.io/aistor/developers/sdk/python/api/> |
| Repository commit/tag | minio 7.2.7 |
| Access date | 2026-07-16 |
| Repository license | Apache-2.0 (per PyPI) |
| Inspected upstream files/symbols | `Minio.put_object`, `Minio.stat_object`, `Minio.remove_object`, `Minio.bucket_exists`, `Minio.make_bucket` |
| Behavior adopted | Run synchronous MinIO SDK methods in `asyncio.to_thread` to avoid blocking the event loop. |
| Behavior explicitly rejected | Using an async MinIO SDK (none is official) or calling sync SDK directly from coroutines. |
| Local modifications | Adapter exposes `upload`, `stat`, `delete` and `_ensure_bucket` as async methods. |
| Failing test/reproducer | `tests/integration/storage/test_minio_adapter.py::test_upload_and_stat_valid_webp` |
| Parity/runtime acceptance command | `make phase1-sprint-01-minio` |
| Known limitation | None. |

---

## DEC-004 — Qdrant `query_points` Non-Deprecated Search API

| Field | Value |
|---|---|
| Decision ID | DEC-004 |
| Local feature/symbol | `backend/app/infrastructure/vectors/qdrant_adapter.py::query` |
| Reference URL | <https://qdrant.tech/documentation/concepts/search/> |
| Repository commit/tag | qdrant-client 1.18.2 |
| Access date | 2026-07-16 |
| Repository license | Apache-2.0 (per PyPI) |
| Inspected upstream files/symbols | `AsyncQdrantClient.query_points`, `Filter`, `FieldCondition`, `MatchValue`, `PointStruct` |
| Behavior adopted | Use `query_points()` with a `must=[FieldCondition(key="active", match=MatchValue(value=True))]` filter. |
| Behavior explicitly rejected | Deprecated `search()` method. |
| Local modifications | Payload carries `sample_id`, `face_id`, `active`, and `model_version` to align with `AGENTS.md` §20. |
| Failing test/reproducer | `tests/integration/vectors/test_qdrant_adapter.py::test_active_filter_excludes_inactive` |
| Parity/runtime acceptance command | `make phase1-sprint-01-qdrant` |
| Known limitation | `model_version` is a static sprint marker until `inference_profile` is introduced. |

---

## DEC-005 — Deterministic 512-D Unit Embedding Fixtures

| Field | Value |
|---|---|
| Decision ID | DEC-005 |
| Local feature/symbol | `backend/tests/fixtures/embedding_fixtures.py` |
| Reference URL | `requirements/ProjectRequirements.md` §3 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | `vector_a` = `[1.0, 0.0, ...]` and `vector_b` = `[0.0, 1.0, ...]`, both exactly 512 values and unit normalized. |
| Behavior explicitly rejected | Random embeddings, non-unit vectors. |
| Local modifications | `cosine_similarity()` helper for assertions. |
| Failing test/reproducer | `tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_orthogonal_vector_b_returns_new_anonymous_different_face` |
| Parity/runtime acceptance command | `make phase1-sprint-01-lifecycle` |
| Known limitation | Fixtures are test-only; not a production threshold calibration. |

---

## DEC-006 — Test-Only `0.95` Match Threshold

| Field | Value |
|---|---|
| Decision ID | DEC-006 |
| Local feature/symbol | `MATCH_THRESHOLD = 0.95` in lifecycle tests |
| Reference URL | `requirements/ProjectRequirements.md` §3 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Use `0.95` as a clear test threshold for deterministic unit vectors. |
| Behavior explicitly rejected | Claiming the threshold is production-calibrated. |
| Local modifications | Threshold is passed into `resolve_or_create` to keep it explicit. |
| Failing test/reproducer | `tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_vector_a_repeated_returns_anonymous_same_face` |
| Parity/runtime acceptance command | `make phase1-sprint-01-lifecycle` |
| Known limitation | Must be replaced with a calibrated threshold before production recognition claims. |

---

## DEC-007 — `NullPool` for Integration-Test Async Engine

| Field | Value |
|---|---|
| Decision ID | DEC-007 |
| Local feature/symbol | `backend/app/infrastructure/persistence/sqlalchemy/session.py` |
| Reference URL | <https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html> |
| Repository commit/tag | SQLAlchemy 2.0.31 |
| Access date | 2026-07-16 |
| Repository license | MIT |
| Inspected upstream files/symbols | `create_async_engine`, `NullPool` |
| Behavior adopted | Use `poolclass=NullPool` so each test session creates and closes its own asyncpg connection. |
| Behavior explicitly rejected | Default connection pool, which caused `cannot perform operation: another operation is in progress` under pytest-asyncio session-scoped tests. |
| Local modifications | `NullPool` imported from `sqlalchemy.pool`. |
| Failing test/reproducer | `tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_vector_a_repeated_returns_anonymous_same_face` |
| Parity/runtime acceptance command | `make phase1-sprint-01-lifecycle` |
| Known limitation | Higher per-request connection overhead; acceptable for tests and early foundation. |

---

## DEC-008 — pytest-asyncio Session Loop with Synchronous Per-Test Cleanup

| Field | Value |
|---|---|
| Decision ID | DEC-008 |
| Local feature/symbol | `backend/tests/integration/lifecycle/conftest.py` |
| Reference URL | <https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/run_session_tests_in_same_loop.html> |
| Repository commit/tag | pytest-asyncio 0.23.8 |
| Access date | 2026-07-16 |
| Repository license | Apache-2.0 (per PyPI) |
| Inspected upstream files/symbols | `pytest.mark.asyncio(scope="session")`, `pytest_asyncio` plugin internals |
| Behavior adopted | Module-level `pytestmark = pytest.mark.asyncio(scope="session")`; autouse cleanup runs in an isolated `asyncio.run()` loop before each test. |
| Behavior explicitly rejected | Function-scoped async cleanup fixture, which forced function-scoped loops and broke asyncpg connection reuse. |
| Local modifications | `_clean_lifecycle_stores()` is a sync fixture that calls `asyncio.run(_clean_stores_async())`. |
| Failing test/reproducer | `tests/integration/lifecycle/test_identity_storage_lifecycle.py` second and later tests |
| Parity/runtime acceptance command | `make phase1-sprint-01-lifecycle` |
| Known limitation | Cleanup runs in a separate event loop; adds a small per-test overhead. |

---

## DEC-009 — Unified `resolve_or_create` Public API

| Field | Value |
|---|---|
| Decision ID | DEC-009 |
| Local feature/symbol | `backend/app/application/services/identity_storage_lifecycle_service.py::resolve_or_create` |
| Reference URL | `PLAN.md` §4.4 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Single public `resolve_or_create(crop_bytes, embedding, bbox, match_threshold)` decides whether a face is new or existing. |
| Behavior explicitly rejected | Separate `store_new_identity()` and `recognize_existing()` public methods requiring the caller to decide. |
| Local modifications | Internal private helpers `_accept_candidate` and `_create_new_identity`. |
| Failing test/reproducer | `tests/integration/lifecycle/test_identity_storage_lifecycle.py` |
| Parity/runtime acceptance command | `make phase1-sprint-01-lifecycle` |
| Known limitation | Does not solve concurrent same-unknown races; deferred. |

---

## DEC-010 — Docker Compose Environment Variables for Local Secrets

| Field | Value |
|---|---|
| Decision ID | DEC-010 |
| Local feature/symbol | `docker-compose.yml`, `backend/.env`, `backend/.env.example` |
| Reference URL | `AGENTS.md` §30 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | PostgreSQL and MinIO credentials are loaded from `backend/.env` via Docker Compose `env_file`; no hardcoded secrets in `docker-compose.yml`. |
| Behavior explicitly rejected | Hardcoded database and MinIO passwords in `docker-compose.yml`. |
| Local modifications | `backend/.env` created with local-dev placeholders (Git-ignored); `backend/.env.example` documents the required variables. |
| Failing test/reproducer | `make phase1-sprint-01-acceptance` (smoke test for compose env loading) |
| Parity/runtime acceptance command | `make phase1-sprint-01-acceptance` |
| Known limitation | `backend/.env` contains local-dev-only values and must not be committed. |

---

## DEC-011 — Dedicated Test Namespace and Fail-Closed Resource Guard

| Field | Value |
|---|---|
| Decision ID | DEC-011 |
| Local feature/symbol | `docker-compose.test.yml`, `backend/tests/support/resource_guard.py`, `backend/.env.test` |
| Reference URL | `AGENTS.md` §30 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Run all acceptance tests against an isolated `mergenvision-s01-test` Compose project and validate the environment before cleanup with `assert_safe_test_environment()`. |
| Behavior explicitly rejected | Reusing development/production services or relying on "best-effort" cleanup without guardrails. |
| Local modifications | Added `docker-compose.test.yml`, `backend/.env.test`, `backend/tests/support/resource_guard.py`, and per-test autouse cleanup guards. |
| Failing test/reproducer | `tests/unit/support/test_resource_guard.py` |
| Parity/runtime acceptance command | `make phase1-sprint-01-acceptance` |
| Known limitation | Guard values are hardcoded for Sprint 01; future sprints will need namespace expansion. |

---

## DEC-012 — Unit of Work Factory Pattern

| Field | Value |
|---|---|
| Decision ID | DEC-012 |
| Local feature/symbol | `backend/app/application/ports/unit_of_work.py::UnitOfWorkFactory`, `backend/app/infrastructure/persistence/sqlalchemy/unit_of_work.py` |
| Reference URL | `AGENTS.md` §16 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Inject a `UnitOfWorkFactory` callable into services; each workflow creates its own `SqlAlchemyUnitOfWork` instance. |
| Behavior explicitly rejected | Sharing a single mutable Unit of Work instance across concurrent operations. |
| Local modifications | Refactored `IdentityStorageLifecycleService.__init__` to accept `unit_of_work_factory` instead of a unit-of-work instance. |
| Failing test/reproducer | Concurrent integration tests under pytest-asyncio session scope. |
| Parity/runtime acceptance command | `make phase1-sprint-01-lifecycle` |
| Known limitation | None. |

---

## DEC-013 — Optimistic Locking for Identity Mutation

| Field | Value |
|---|---|
| Decision ID | DEC-013 |
| Local feature/symbol | `backend/app/infrastructure/persistence/sqlalchemy/repositories/face_identity.py::update_with_expected_version` |
| Reference URL | `AGENTS.md` §19 |
| Repository commit/tag | SQLAlchemy 2.0.31 |
| Access date | 2026-07-16 |
| Repository license | MIT |
| Inspected upstream files/symbols | `update(...).where(...).values(...).returning(...)` |
| Behavior adopted | Perform conditional `UPDATE ... WHERE version = expected_version RETURNING version`; raise `ConcurrentUpdateError` when no row is updated. |
| Behavior explicitly rejected | Blind `UPDATE` without version check or pessimistic row locking. |
| Local modifications | Added `update_with_expected_version` to `FaceIdentityRepository` and wired it into enrollment/deactivation. |
| Failing test/reproducer | Concurrent enrollment/deactivation race scenarios. |
| Parity/runtime acceptance command | `make phase1-sprint-01-acceptance` |
| Known limitation | Caller must capture `expected_version` before mutating the entity in memory. |

---

## DEC-014 — Cross-Store Synchronous Compensation

| Field | Value |
|---|---|
| Decision ID | DEC-014 |
| Local feature/symbol | `backend/app/application/services/identity_storage_lifecycle_service.py` |
| Reference URL | `AGENTS.md` §23 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | If MinIO upload or Qdrant upsert fails during new-identity creation, mark the sample/identity failed and best-effort delete partial external state. |
| Behavior explicitly rejected | Leaving active identities with zero active samples after an external-store failure. |
| Local modifications | Rewrote `resolve_or_create` and `add_sample` with explicit try/except/compensation blocks and `_persist_resolution_failure`. |
| Failing test/reproducer | `tests/integration/lifecycle/test_failure_paths.py` |
| Parity/runtime acceptance command | `make phase1-sprint-01-failure` |
| Known limitation | Compensation is synchronous and best-effort; orphaned objects/vectors may remain if cleanup fails. |

---

## DEC-015 — Injected `IdGenerator` Port for UUIDv7

| Field | Value |
|---|---|
| Decision ID | DEC-015 |
| Local feature/symbol | `backend/app/application/ports/id_generator.py`, `backend/app/infrastructure/uuid7.py::Uuid7Generator` |
| Reference URL | `AGENTS.md` §14 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Define an `IdGenerator` port with `new_uuid7()` and inject a `Uuid7Generator` adapter into services and tests. |
| Behavior explicitly rejected | Calling `uuid_extensions.uuid7()` directly from application/domain code. |
| Local modifications | Added port and adapter; updated service constructor and tests to accept `id_generator`. |
| Failing test/reproducer | `tests/unit/test_domain_dependency_boundary.py` |
| Parity/runtime acceptance command | `make phase1-sprint-01-acceptance` |
| Known limitation | None. |

---

## DEC-016 — MinIO Upload Idempotency and SHA-256 Metadata

| Field | Value |
|---|---|
| Decision ID | DEC-016 |
| Local feature/symbol | `backend/app/infrastructure/storage/minio_adapter.py` |
| Reference URL | MinIO Python SDK docs |
| Repository commit/tag | minio 7.2.20 |
| Access date | 2026-07-16 |
| Repository license | Apache-2.0 |
| Inspected upstream files/symbols | `Minio.put_object`, `Minio.stat_object`, `x-amz-meta-*` |
| Behavior adopted | Compute SHA-256, store as metadata, verify on stat, and return `ObjectStat(bucket, key, size, sha256)`. Reject same-key/different-content uploads. |
| Behavior explicitly rejected | Uploading blindly without integrity check or silently overwriting existing objects. |
| Local modifications | Extended `ObjectStat`, added SHA computation, and added conflict detection. |
| Failing test/reproducer | `tests/integration/storage/test_minio_adapter.py` |
| Parity/runtime acceptance command | `make phase1-sprint-01-minio` |
| Known limitation | None. |

---

## DEC-017 — Qdrant Payload Validation and Indexes

| Field | Value |
|---|---|
| Decision ID | DEC-017 |
| Local feature/symbol | `backend/app/infrastructure/vectors/qdrant_adapter.py` |
| Reference URL | Qdrant documentation |
| Repository commit/tag | qdrant-client 1.18.0 |
| Access date | 2026-07-16 |
| Repository license | Apache-2.0 |
| Inspected upstream files/symbols | `AsyncQdrantClient.create_payload_index`, `PayloadSchemaType` |
| Behavior adopted | Validate payload `sample_id` matches point id and create payload indexes for `face_id`, `active`, and `model_version`. |
| Behavior explicitly rejected | Trusting unvalidated Qdrant payloads or running searches without payload indexes. |
| Local modifications | Added validation loop in `query()` and explicit payload index creation after collection creation. |
| Failing test/reproducer | `tests/integration/vectors/test_qdrant_adapter.py` |
| Parity/runtime acceptance command | `make phase1-sprint-01-qdrant` |
| Known limitation | `model_version` is a static sprint marker until `inference_profile` is introduced. |

---

| Field | Value |
|---|---|
| Decision ID | DEC-010 |
| Local feature/symbol | `docker-compose.yml`, `backend/.env`, `backend/.env.example` |
| Reference URL | `AGENTS.md` §30 |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | PostgreSQL and MinIO credentials are loaded from `backend/.env` via Docker Compose `env_file`; no hardcoded secrets in `docker-compose.yml`. |
| Behavior explicitly rejected | Hardcoded database and MinIO passwords in `docker-compose.yml`. |
| Local modifications | `backend/.env` created with local-dev placeholders (Git-ignored); `backend/.env.example` documents the required variables. |
| Failing test/reproducer | `make phase1-sprint-01-acceptance` (smoke test for compose env loading) |
| Parity/runtime acceptance command | `make phase1-sprint-01-acceptance` |
| Known limitation | `backend/.env` contains local-dev-only values and must not be committed. |

---

## DEC-018 — Public Media/Biometric Exposure Remediation

| Field | Value |
|---|---|
| Decision ID | DEC-018 |
| Local feature/symbol | `research/friends_characters/`, `prompt2.txt`, `.gitignore`, `tests/unit/test_gitignore.py` |
| Reference URL | `AGENTS.md` §30 (Security and privacy baseline) |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Removed `research/friends_characters/**` and `prompt2.txt` from the current worktree; expanded `.gitignore` to block models, videos, galleries, crops, embeddings, reports, and debug media; added a fail-closed test that fails if forbidden patterns become tracked again. |
| Behavior explicitly rejected | Rewriting Git history to erase the committed cast images and biometric-derived `gallery_centroids.json`; deleting `test_videos/Friends.mp4` or user-provided gallery images. |
| Local modifications | Worktree deletion only; history retained for senior review. |
| Failing test/reproducer | `tests/unit/test_gitignore.py::test_friends_characters_directory_removed_from_worktree` |
| Parity/runtime acceptance command | `make video-reference-unit` |
| Known limitation | Old commit history still contains the removed files; a separate history-cleanup operation is required before the repository can be made public. |

---

## DEC-019 — Isolated Video Reference Lab Exception

| Field | Value |
|---|---|
| Decision ID | DEC-019 |
| Local feature/symbol | `docs/implementation/CURRENT_SPRINT.md` |
| Reference URL | `AGENTS.md` §4 (Mandatory implementation order) |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | N/A |
| Behavior adopted | Documented an explicit user-approved isolated video-reference correctness spike that does not begin product Sprint 02, does not alter the mandatory implementation order, and cannot claim production video/GPU completion. |
| Behavior explicitly rejected | Treating this correction as authorization to start product Sprint 02 or to declare production video recognition complete. |
| Local modifications | Added isolated-research-exception paragraph to `CURRENT_SPRINT.md` without changing Sprint 01 status. |
| Failing test/reproducer | Manual review of `CURRENT_SPRINT.md` |
| Parity/runtime acceptance command | `git diff docs/implementation/CURRENT_SPRINT.md` |
| Known limitation | The exception is valid only for this isolated laboratory correction task. |

---

## DEC-020 — Evaluation Compares Observations Through Tracklet → Canonical Mapping

| Field | Value |
|---|---|
| Decision ID | DEC-020 |
| Local feature/symbol | `research/video_reference_lab/src/mergenvision_video_lab/evaluation.py::evaluate_identity` |
| Reference URL | Sprint 002 spec (observation / raw tracklet / canonical track separation) |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `_build_observation_to_canonical_map` helper |
| Behavior adopted | Ground-truth anchors label `observation_id`s; clusters contain `raw_tracklet_id`s; evaluate by mapping each observation to its tracklet and then to its canonical cluster. |
| Behavior explicitly rejected | Treating observation IDs as cluster members. |
| Local modifications | New `_build_observation_to_canonical_map`; `evaluate_identity` accepts `assignments` and `canonical_map`. |
| Failing test/reproducer | Old `tests/unit/test_evaluation.py::test_evaluate_identity_pairwise` (validated broken semantics). |
| Parity/runtime acceptance command | `make video-reference-unit` |
| Known limitation | Pairwise metrics require ground truth; `Friends.mp4` has none. |

---

## DEC-021 — Gallery `known` Requires `decision_reason == "gallery_match"`

| Field | Value |
|---|---|
| Decision ID | DEC-021 |
| Local feature/symbol | `research/video_reference_lab/src/mergenvision_video_lab/evaluation.py::evaluate_gallery` |
| Reference URL | Sprint 002 spec (display_label safety rule) |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `CanonicalTrack.decision_reason`, `CanonicalTrack.display_label` |
| Behavior adopted | A track is counted as `known` only when `decision_reason == "gallery_match"` and `display_label is not None`. |
| Behavior explicitly rejected | Counting a track as known just because `gallery_top1_label` exists. |
| Local modifications | `if track.decision_reason == "gallery_match" and track.display_label is not None`. |
| Failing test/reproducer | Old `tests/unit/test_evaluation.py::test_evaluate_gallery_known_unknown`. |
| Parity/runtime acceptance command | `make video-reference-unit` |
| Known limitation | Does not change the underlying gallery thresholds. |

---

## DEC-022 — Per-Tracker Raw Tracklet ID Allocation

| Field | Value |
|---|---|
| Decision ID | DEC-022 |
| Local feature/symbol | `research/video_reference_lab/src/mergenvision_video_lab/tracking/byte_tracker.py::ByteTrackIoUTracker` |
| Reference URL | FoundationVision ByteTrack tracklet ID lifecycle |
| Repository commit/tag | Local adaptation |
| Access date | 2026-07-16 |
| Repository license | MIT (ByteTrack upstream) |
| Inspected upstream files/symbols | `Tracklet.__init__`, `ByteTrackIoUTracker.__init__` |
| Behavior adopted | Each tracker instance owns its own `_next_tracklet_id` counter; new tracklets receive IDs via `_allocate_tracklet_id`. |
| Behavior explicitly rejected | Global class-level `_id_counter` reset in every tracker constructor, which made multi-instance behavior non-deterministic. |
| Local modifications | `Tracklet.__init__` accepts optional `_id_allocator`; tracker passes its allocator. |
| Failing test/reproducer | `tests/unit/test_chunk_invariance.py::test_chunk_invariance_assignments_and_tracklets` after removing manual counter reset. |
| Parity/runtime acceptance command | `make video-reference-unit` |
| Known limitation | Class-level fallback counter is kept for direct `Tracklet()` construction in tests. |

---

## DEC-023 — Benchmark Reports Real `max_active_tracks_estimate`

| Field | Value |
|---|---|
| Decision ID | DEC-023 |
| Local feature/symbol | `research/video_reference_lab/src/mergenvision_video_lab/benchmark.py::benchmark_replay` |
| Reference URL | N/A |
| Repository commit/tag | N/A |
| Access date | 2026-07-16 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `ByteTrackIoUTracker.active_tracklet_ids` |
| Behavior adopted | `benchmark_replay` queries `tracker.active_tracklet_ids()` each frame and records the maximum. |
| Behavior explicitly rejected | Returning `0` for `max_active_tracks_estimate`. |
| Local modifications | `one_run` returns `max_active`; measured runs keep the maximum across runs. |
| Failing test/reproducer | Friends benchmark JSON previously showed `max_active_tracks_estimate: 0`. |
| Parity/runtime acceptance command | Full `run-friends --max-frames 300`. |
| Known limitation | Counts tracked tracklets only; lost tracklets are not included in the active estimate. |

---

## DEC-024 — Native Execution Slot Lifecycle

| Field | Value |
|---|---|
| Decision ID | DEC-024 |
| Local feature/symbol | `backend/native/image_runtime/src/pipeline.h`, `backend/native/image_runtime/src/pipeline.cpp`, `ExecutionSlot` |
| Reference URL | `AGENTS.md` §13, §18 |
| Repository commit/tag | N/A |
| Access date | 2026-07-17 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `ExecutionSlot::available()`, `ExecutionSlot::acquire()`, `ExecutionSlot::release()` |
| Behavior adopted | Add explicit `ExecutionSlot::State` enum (`uninitialized`, `initialized`, `unavailable`, `in_use`) so a slot whose engine/decoder/stream fails to initialize is marked `unavailable` and is never acquired by `ImageRuntime`. |
| Behavior explicitly rejected | Continuing to use a plain `bool in_use_` that silently reports `available() == true` after constructor failure. |
| Local modifications | Added `state_` member; constructor sets `unavailable` on every early-return path and `initialized` only on full success. |
| Failing test/reproducer | `backend/tests/native/test_image_runtime_safety.py::test_image_runtime_constructor_rejects_broken_slots` |
| Parity/runtime acceptance command | `make phase2-step0-native` inside the pinned TensorRT container. |
| Known limitation | C++ source compiles only when CUDA/TensorRT headers are available; host `.venv` skips native tests. |

---

## DEC-025 — Model Profile JSON Parsing in Native Runtime

| Field | Value |
|---|---|
| Decision ID | DEC-025 |
| Local feature/symbol | `backend/native/image_runtime/src/model_profile.cpp`, `backend/native/image_runtime/src/model_profile.h`, `backend/app/infrastructure/model_profile.py` |
| Reference URL | `AGENTS.md` §21, §33; canonical profile JSON |
| Repository commit/tag | N/A |
| Access date | 2026-07-17 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `ModelProfile::from_py_dict`, detector/recognizer/alignment sections |
| Behavior adopted | Parse `alignment.crop_size` as `[h, w]`, validate square, store as `alignment_crop_h/w`. Validate detector/recognizer dynamic_profile.max shapes against the parsed input shapes and declared batch maxima. |
| Behavior explicitly rejected | Treating `crop_size` as scalar int; ignoring dynamic profile contract. |
| Local modifications | Added `get_list` helper and `validate_4d_shape`; replaced scalar parse with list parse; added dynamic profile checks. |
| Failing test/reproducer | `backend/tests/native/test_image_runtime_surface.py::test_image_runtime_parses_alignment_crop_size_as_list` |
| Parity/runtime acceptance command | `make phase2-step0-native` inside the pinned TensorRT container. |
| Known limitation | Host `.venv` cannot import `image_runtime`; real parse validation needs container build. |

---

## DEC-026 — Native `ImageRuntime` Python Contract Takes a Parsed Profile Dict

| Field | Value |
|---|---|
| Decision ID | DEC-026 |
| Local feature/symbol | `backend/native/image_runtime/src/bindings.cpp`, `backend/app/infrastructure/runtime/native_image_recognition_adapter.py`, `backend/tests/native/test_image_runtime_*.py` |
| Reference URL | `AGENTS.md` §21 |
| Repository commit/tag | N/A |
| Access date | 2026-07-17 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `ImageRuntime` pybind constructor, `NativeImageRecognitionAdapter.__init__` |
| Behavior adopted | C++ `ImageRuntime` keeps `const py::dict&` constructor; Python adapter loads the canonical `ModelProfile`, validates it, and passes `model_dump(by_alias=True)` dict to the native module. Native tests use the same dict. |
| Behavior explicitly rejected | Changing the C++ constructor to load a filesystem path, which would duplicate JSON validation and bypass the Pydantic contract. |
| Local modifications | `NativeImageRecognitionAdapter._init_runtime` now calls `ModelProfile.load(...)`; native tests call `_load_profile()` helper. |
| Failing test/reproducer | `backend/tests/native/test_image_runtime_safety.py` (path-string variant was broken) |
| Parity/runtime acceptance command | `make phase2-step0-native` inside the pinned TensorRT container. |
| Known limitation | None once module is built. |

---

## DEC-027 — TensorRT Engine Build Manifest Script and Digest-Pinned Container

| Field | Value |
|---|---|
| Decision ID | DEC-027 |
| Local feature/symbol | `backend/scripts/build_engines.py`, `backend/Dockerfile.gpu` |
| Reference URL | `AGENTS.md` §12, §33 |
| Repository commit/tag | N/A |
| Access date | 2026-07-17 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `trtexec --onnx`, `trtexec --minShapes/--optShapes/--maxShapes` |
| Behavior adopted | Script builds both engines from ONNX using `trtexec` with the exact dynamic profiles from the JSON, verifies ONNX SHA256, computes engine SHA256, and updates the `engine_manifest` container digest, TensorRT/CUDA versions, GPU compute capability/uuid, and build timestamp. |
| Behavior explicitly rejected | Shell-only builder with untracked commands and manual SHA updates; mutable TensorRT base image tag. |
| Local modifications | Added `backend/scripts/build_engines.py`; `Dockerfile.gpu` already pins TensorRT image digest. |
| Failing test/reproducer | `make phase2-step0-native` when `engine_manifest` is missing or stale. |
| Parity/runtime acceptance command | Run `backend/scripts/build_engines.py` inside the pinned TensorRT container and verify manifest engine SHA256. |
| Known limitation | Script cannot be executed on a non-GPU host; it requires `nvidia-smi`, `nvcc`, and `trtexec` from the container. |

---

## DEC-028 — Configurable Native CUDA Architectures

| Field | Value |
|---|---|
| Decision ID | DEC-028 |
| Local feature/symbol | `backend/native/image_runtime/CMakeLists.txt` |
| Reference URL | CMake `CMAKE_CUDA_ARCHITECTURES` documentation |
| Repository commit/tag | N/A |
| Access date | 2026-07-17 |
| Repository license | Proprietary |
| Inspected upstream files/symbols | `cmake_policy`, `CUDA_ARCHITECTURES` setting |
| Behavior adopted | Default architecture list `75;80;86;89` is kept but can be overridden via the `CMAKE_CUDA_ARCHITECTURES` environment variable to allow local build tuning. |
| Behavior explicitly rejected | Permanently hardcoding `75` only, which would produce broken code for newer GPUs. |
| Local modifications | `CMakeLists.txt` reads `CMAKE_CUDA_ARCHITECTURES` from env, else applies the default list. |
| Failing test/reproducer | Native build on a compute capability 8.6 or 8.9 GPU. |
| Parity/runtime acceptance command | `make phase2-step0-native` inside the pinned TensorRT container. |
| Known limitation | User may need to set a narrower architecture list for faster container rebuilds. |
