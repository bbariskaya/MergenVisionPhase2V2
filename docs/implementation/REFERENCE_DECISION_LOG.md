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
| Local modifications | Payload restricted to `face_id` and `active`. |
| Failing test/reproducer | `tests/integration/vectors/test_qdrant_adapter.py::test_active_filter_excludes_inactive` |
| Parity/runtime acceptance command | `make phase1-sprint-01-qdrant` |
| Known limitation | Search result is always cross-checked against PostgreSQL active state. |

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
