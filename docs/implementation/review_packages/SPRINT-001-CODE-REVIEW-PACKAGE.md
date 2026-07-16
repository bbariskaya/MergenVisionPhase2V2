# Phase 1 Sprint 01 — Code Review Package

**Correction base commit:** `675549670ab65daec9ffedae3e62ddb4f4478dc3`<br>
**Status:** `CORRECTION_IN_PROGRESS` — senior review ready<br>
**Date:** 2026-07-16

This package documents the forensic correction pass performed on Phase 1 Sprint 01 (Minimal Identity Storage Foundation) in response to senior feedback. The previous "PASS" claim has been withdrawn.

## 1. Sprint identity

| Field | Value |
|---|---|
| Sprint | Phase 1 Sprint 01 — Minimal Identity Storage Foundation |
| Reviewed base commit | `675549670ab65daec9ffedae3e62ddb4f4478dc3` |
| Current HEAD (uncommitted) | `675549670ab65daec9ffedae3e62ddb4f4478dc3` |
| Branch | `main` |
| Origin | `https://github.com/bbariskaya/MergenVisionPhase2V2.git` |

`git status --short` shows only the expected correction-pass changes; no unrelated user changes were overwritten.

## 2. Executive verdict

**Verdict:** `PASS — READY_FOR_SENIOR_REVIEW`

Scope is Phase 1 Sprint 01 storage-foundation correction only. This is **not** a claim that full Phase 1 is complete, and Sprint 02 has **not** been started.

## 3. Original finding closure matrix

| ID | Finding | Severity | Fix location | Status |
|---|---|---|---|---|
| F-001 | Unsafe cross-store test cleanup | P0 | `docker-compose.test.yml`, `backend/tests/support/resource_guard.py`, `backend/tests/integration/conftest.py` | CLOSED |
| F-002 | Acceptance missing repository/UUID tests | P1 | `Makefile::phase1-sprint-01-full-test` runs `tests` tree | CLOSED |
| F-003 | Shared mutable UoW instance | P1 | `backend/app/application/ports/unit_of_work.py::UnitOfWorkFactory`, `backend/tests/integration/lifecycle/test_concurrent_uow_isolation.py` | CLOSED |
| F-004 | Stale candidate / empty `max()` crash | P1 | `backend/app/application/services/identity_storage_lifecycle_service.py::resolve_or_create`, `backend/tests/integration/lifecycle/test_candidate_validation.py` | CLOSED |
| F-005 | Negative/non-finite score semantics | P1 | `_to_match_confidence` clamps to `[0,1]`; threshold compare uses raw finite score, `backend/tests/integration/lifecycle/test_candidate_validation.py::test_negative_cosine_yields_new_anonymous_with_zero_confidence` | CLOSED |
| F-006 | Vector query stuck-process failure | P1 | `resolve_or_create` fails process on `VectorStore.query` exception, `backend/tests/integration/lifecycle/test_candidate_validation.py::test_vector_query_failure_fails_process_and_creates_nothing` | CLOSED |
| F-007 | New identity cross-store compensation | P1 | `IdentityStorageLifecycleService::_create_new_identity`, `_persist_resolution_failure`, `_delete_object_best_effort`, `backend/tests/integration/lifecycle/test_failure_paths.py` | CLOSED |
| F-008 | `add_sample()` failure handling | P1 | `IdentityStorageLifecycleService::add_sample`, `backend/tests/integration/lifecycle/test_failure_paths.py` | CLOSED |
| F-009 | Pending sample incorrectly active | P1 | `backend/app/domain/entities/face_sample.py`, migration `0002_sprint01_correctness.py` | CLOSED |
| F-010 | Fake optimistic locking | P1 | `backend/app/infrastructure/persistence/sqlalchemy/repositories/face_identity.py::update_with_expected_version`, `backend/tests/integration/lifecycle/test_optimistic_locking.py` | CLOSED |
| F-011 | Hardcoded MinIO bucket | P1 | `backend/app/infrastructure/storage/minio_adapter.py::ObjectStat.bucket`, `backend/app/application/services/identity_storage_lifecycle_service.py` | CLOSED |
| F-012 | Missing migration invariants/indexes | P1 | `backend/app/infrastructure/persistence/alembic/versions/0002_sprint01_correctness.py`, `backend/tests/integration/persistence/test_migrations.py` | CLOSED |
| F-013 | UUIDv4 ORM defaults | P1 | Removed from all SQLAlchemy models | CLOSED |
| F-014 | Incomplete restart persistence proof | P1 | `backend/scripts/restart_persistence_probe.py`, `Makefile::phase1-sprint-01-restart`, `backend/tests/integration/lifecycle/test_restart_persistence.py` | CLOSED |
| F-015 | Missing lock/bootstrap/Docker/CI | P1 | `backend/requirements.lock`, `backend/Dockerfile`, `.dockerignore`, `Makefile::bootstrap`, `.github/workflows/phase1-sprint-01.yml` | CLOSED |
| F-016 | Documentation false claims | P2 | `docs/implementation/IMPLEMENTATION_DETAILS.md`, `docs/implementation/REFERENCE_DECISION_LOG.md`, `README.md` | CLOSED |
| F-017 | Duplicated `AGENTS.md` content / empty `Plan` file | P2 | `AGENTS.md` duplicate removed, root `Plan` deleted | CLOSED |
| F-018 | Portable MCP config | P2 | `.mcp.json` command changed to `codebase-memory-mcp` | CLOSED |
| F-019 | Fresh-clone env/Compose failure | P2 | `backend/.env.example`, `backend/.env.test.example`, `Makefile::bootstrap`, `README.md` | CLOSED |
| F-020 | Qdrant malformed/stale candidate handling | P2 | `backend/app/infrastructure/vectors/qdrant_adapter.py::query`, payload validation, `backend/tests/integration/lifecycle/test_candidate_validation.py` | CLOSED |

## 4. Schema evidence

Migration `0001_identity_storage_foundation.py` is unchanged. Migration `0002_sprint01_correctness.py` adds the required CHECK constraints, indexes, and repairs the `face_sample.is_active` default to `false`.

ORM models mirror the migration head:

- No `default=uuid.uuid4` on business keys.
- `FaceSample.is_active` default `false`.
- Lifecycle constraints on `face_identity`, `process_record`, `face_sample`, `recognition_result`.

`tests/integration/persistence/test_migrations.py` verifies both `upgrade head` from a fresh DB and `upgrade 0001 -> head`, plus invalid-insert rejection.

## 5. Runtime / data flow summary

- `resolve_or_create`: top-`candidate_limit` Qdrant search, per-candidate PostgreSQL validation, stale/malformed/non-finite skip, clamped confidence, new-identity compensation on external-store failure.
- `add_sample`: pending → upload → vector index → active; failure marks sample failed and best-effort cleans partial state.
- `enroll_identity`: optimistic-lock `UPDATE ... WHERE version = expected_version`.
- `deactivate_identity`: optimistic-lock identity inactive, samples inactive, Qdrant best-effort deactivation; **MinIO objects are not deleted** in Sprint 01.

## 6. Test safety

- Dedicated Compose project: `mergenvision-s01-test`
- PostgreSQL test DB: `mergenvision_s01_test` on `127.0.0.1:55432`
- MinIO test bucket: `mergenvision-s01-test-face-samples` on `127.0.0.1:59000`
- Qdrant test collection: `mergenvision_s01_test_face_samples_v1` on `127.0.0.1:56333`
- `assert_safe_test_environment()` runs before any cleanup.
- `make phase1-sprint-01-acceptance` never runs `docker compose down -v` or deletes volumes.
- No main/dev resource cleanup is performed.

## 7. Raw validation

All commands were executed from `/home/user/Workspace/MergenVisionPhase2v2`.

| Command | Result |
|---|---|
| `make phase1-sprint-01-preflight` | 11 guard unit tests passed; Compose config valid |
| `make phase1-sprint-01-static` | `ruff check .` passed; `mypy .` passed (80 files) |
| `make phase1-sprint-01-format-check` | 80 files already formatted |
| `make phase1-sprint-01-unit` | 44 passed |
| `make phase1-sprint-01-full-test` | **77 passed** (44 unit + 33 integration) |
| `make phase1-sprint-01-restart` | Probe seed/restart/verify passed across PG, MinIO, Qdrant |
| `make phase1-sprint-01-image-build` | Image built; container import printed `backend-import-ok` |
| `git diff --check` | Clean |

Full acceptance was executed **twice** successfully on 2026-07-16.

## 8. Restart proof

`backend/scripts/restart_persistence_probe.py` seeds an anonymous identity, restarts `postgres-test`, `minio-test`, and `qdrant-test`, then verifies:

- PostgreSQL identity/sample/process/result persisted and consistent.
- MinIO object stat size matches fixture.
- Qdrant point remains searchable and returns the correct `sample_id`/`face_id`.

## 9. Security / privacy

- Object key format: `faces/{face_id}/{sample_id}/aligned.webp` — no name/metadata.
- Qdrant payload contains only `sample_id`, `face_id`, `active`, `model_version`.
- Raw exceptions are not persisted or exposed; sanitized `error_code` values are used.
- No real secrets committed; `backend/.env` is Git-ignored.

## 10. Real vs synthetic

- Embeddings are deterministic 512-D unit vectors (`vector_a`, `vector_b`).
- They prove storage, lifecycle, and candidate-selection correctness.
- They do **not** prove production recognition accuracy; no real detector/recognizer/GPU/API/UI/video path is implemented.

## 11. Known limitations

- `poolclass=NullPool` is used for integration-test stability.
- If a PostgreSQL transaction fails after MinIO upload or Qdrant upsert succeeds, an orphan object or vector point may remain; no persistent outbox/saga/reconciliation worker exists yet.
- Sprint 01 deactivation does **not** delete MinIO objects.
- The `0.95` threshold is a test fixture, not a production-calibrated recognition threshold.
- No FastAPI endpoints, UI, real detector/recognizer, video pipeline, or GPU path.

## 12. MCP / tool accountability

- `codebase-memory-mcp`: used for architecture and caller/callee discovery.
- `mcp__plugin_context7_context7__resolve-library-id` / `query-docs`: used for SQLAlchemy, Alembic, MinIO, Qdrant official docs.
- DeepWiki/Exa/Postman/Playwright: not used.
- `21st`: FORBIDDEN_NOT_USED.

## 13. Changed files

```
AGENTS.md
Makefile
README.md
backend/README.md
backend/.env.example
backend/.env.test
backend/.env.test.example
backend/Dockerfile
backend/app/application/ports/id_generator.py
backend/app/application/ports/repositories.py
backend/app/application/ports/unit_of_work.py
backend/app/application/services/identity_storage_lifecycle_service.py
backend/app/domain/entities/face_identity.py
backend/app/domain/entities/face_sample.py
backend/app/domain/value_objects.py
backend/app/infrastructure/config.py
backend/app/infrastructure/persistence/alembic/versions/0002_sprint01_correctness.py
backend/app/infrastructure/persistence/sqlalchemy/models/*.py
backend/app/infrastructure/persistence/sqlalchemy/repositories/face_identity.py
backend/app/infrastructure/persistence/sqlalchemy/repositories/face_sample.py
backend/app/infrastructure/storage/minio_adapter.py
backend/app/infrastructure/uuid7.py
backend/app/infrastructure/vectors/qdrant_adapter.py
backend/pyproject.toml
backend/requirements.lock
backend/scripts/restart_persistence_probe.py
backend/tests/conftest.py
backend/tests/integration/conftest.py
backend/tests/integration/lifecycle/conftest.py
backend/tests/integration/lifecycle/test_candidate_validation.py
backend/tests/integration/lifecycle/test_concurrent_uow_isolation.py
backend/tests/integration/lifecycle/test_failure_paths.py
backend/tests/integration/lifecycle/test_identity_storage_lifecycle.py
backend/tests/integration/lifecycle/test_inactive_rejection.py
backend/tests/integration/lifecycle/test_multiple_samples.py
backend/tests/integration/lifecycle/test_optimistic_locking.py
backend/tests/integration/lifecycle/test_restart_persistence.py
backend/tests/integration/persistence/test_migrations.py
backend/tests/integration/persistence/test_repositories.py
backend/tests/integration/storage/conftest.py
backend/tests/integration/storage/test_minio_adapter.py
backend/tests/integration/vectors/conftest.py
backend/tests/integration/vectors/test_qdrant_adapter.py
backend/tests/support/resource_guard.py
backend/tests/unit/domain/test_face_identity.py
backend/tests/unit/domain/test_face_sample.py
backend/tests/unit/domain/test_process_record.py
backend/tests/unit/infrastructure/test_uuid7.py
backend/tests/unit/support/test_resource_guard.py
backend/tests/unit/test_domain_dependency_boundary.py
docker-compose.test.yml
.github/workflows/phase1-sprint-01.yml
.mcp.json
```

## 14. Git / destructive operations

- No `git add`, `commit`, `push`, `merge`, `reset`, `checkout --discard`, or history rewrite performed.
- No `docker compose down -v`, `docker volume rm`, `docker system prune`, or unrelated container restart performed.

## 15. Senior-review readiness

All mandatory local gates passed twice. The correction is **READY_FOR_SENIOR_REVIEW**.

Phase 1 Sprint 01 storage-foundation correction senior review\'e hazırdır. Bu, Phase 1\'in tamamlandığı anlamına gelmez. Sonraki sprint başlatılmamıştır.
