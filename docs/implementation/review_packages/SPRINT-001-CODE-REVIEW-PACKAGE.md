# Sprint 001 Code Review Package — Phase 1 Sprint 01

## Verdict

**PASS**

`make phase1-sprint-01-acceptance` exits 0 on 2026-07-16 with real PostgreSQL, MinIO and Qdrant services.

## What Now Works

- Four-table PostgreSQL identity storage foundation is created by one Alembic migration.
- Pure Python domain layer is isolated from SQLAlchemy, MinIO, Qdrant and FastAPI.
- SQLAlchemy 2.0 async repositories and unit of work pass CRUD tests.
- MinIO adapter uploads and stats a genuine small WebP fixture.
- Qdrant adapter creates a 512-D cosine collection, upserts/query_points with an active filter.
- `IdentityStorageLifecycleService` implements `resolve_or_create`, `add_sample`, `enroll_identity` and `deactivate_identity`.
- Deterministic vectors prove:
  - first request → `new_anonymous`
  - same vector repeat → `anonymous` with same `face_id`
  - orthogonal vector → different `new_anonymous` `face_id`
- Enrollment preserves `face_id` and later recognition returns `known`.
- Old recognition results remain immutable snapshots.
- Multiple active samples can belong to the same `face_id`.
- Inactive identities/samples are rejected as Qdrant candidates.
- MinIO/Qdrant failures during first-sample creation deactivate the new identity (zombie prevention).
- Data survives `docker compose restart postgres minio qdrant`.
- `ruff` lint/format and `mypy --strict` pass.
- `git diff --check` passes.

## Acceptance Command and Raw Results

```bash
cd /home/user/Workspace/MergenVisionPhase2v2
make phase1-sprint-01-acceptance
```

Final stage output:

```text
cd backend && /home/user/Workspace/MergenVisionPhase2v2/backend/.venv/bin/python -m ruff check . && /home/user/Workspace/MergenVisionPhase2v2/backend/.venv/bin/python -m mypy .
All checks passed!
Success: no issues found in 68 source files
git diff --check
```

Selected earlier-stage outputs:

```text
phase1-sprint-01-static        → ruff check/format + mypy strict PASS
phase1-sprint-01-postgres      → 2 migration/schema tests PASS
phase1-sprint-01-minio         → 1 upload/stat test PASS
phase1-sprint-01-qdrant        → 5 upsert/query/filter tests PASS
phase1-sprint-01-lifecycle     → 7 lifecycle tests PASS
phase1-sprint-01-failure       → 2 failure-path tests PASS
phase1-sprint-01-restart       → 1 restart-persistence test PASS
```

Unit tests: 24 passed (`test_domain_dependency_boundary.py` + `tests/unit/domain`).

## Changed-File Groups

### Project configuration

- `backend/pyproject.toml`
- `backend/.env.example`
- `backend/README.md`
- `Makefile`
- `.gitignore`
- `docker-compose.yml`

### Domain (pure Python)

- `backend/app/domain/value_objects.py`
- `backend/app/domain/errors.py`
- `backend/app/domain/entities/face_identity.py`
- `backend/app/domain/entities/face_sample.py`
- `backend/app/domain/entities/process_record.py`
- `backend/app/domain/entities/recognition_result.py`

### Application ports and service

- `backend/app/application/ports/unit_of_work.py`
- `backend/app/application/ports/repositories.py`
- `backend/app/application/ports/object_store.py`
- `backend/app/application/ports/vector_store.py`
- `backend/app/application/services/identity_storage_lifecycle_service.py`

### Infrastructure

- `backend/app/infrastructure/config.py`
- `backend/app/infrastructure/uuid7.py`
- `backend/app/infrastructure/clock.py`
- `backend/app/infrastructure/persistence/sqlalchemy/base.py`
- `backend/app/infrastructure/persistence/sqlalchemy/session.py`
- `backend/app/infrastructure/persistence/sqlalchemy/unit_of_work.py`
- `backend/app/infrastructure/persistence/sqlalchemy/models/*.py`
- `backend/app/infrastructure/persistence/sqlalchemy/repositories/*.py`
- `backend/app/infrastructure/persistence/alembic/env.py`
- `backend/app/infrastructure/persistence/alembic/versions/0001_identity_storage_foundation.py`
- `backend/app/infrastructure/storage/minio_adapter.py`
- `backend/app/infrastructure/vectors/qdrant_adapter.py`

### Tests and fixtures

- `backend/tests/conftest.py`
- `backend/tests/fixtures/embedding_fixtures.py`
- `backend/tests/fixtures/valid_crop.webp`
- `backend/tests/unit/test_domain_dependency_boundary.py`
- `backend/tests/unit/domain/*.py`
- `backend/tests/unit/infrastructure/test_uuid7.py`
- `backend/tests/integration/persistence/test_migrations.py`
- `backend/tests/integration/persistence/test_repositories.py`
- `backend/tests/integration/storage/test_minio_adapter.py`
- `backend/tests/integration/vectors/test_qdrant_adapter.py`
- `backend/tests/integration/vectors/conftest.py`
- `backend/tests/integration/lifecycle/conftest.py`
- `backend/tests/integration/lifecycle/test_identity_storage_lifecycle.py`
- `backend/tests/integration/lifecycle/test_multiple_samples.py`
- `backend/tests/integration/lifecycle/test_inactive_rejection.py`
- `backend/tests/integration/lifecycle/test_failure_paths.py`
- `backend/tests/integration/lifecycle/test_restart_persistence.py`

### Documentation

- `docs/implementation/CURRENT_SPRINT.md`
- `docs/implementation/IMPLEMENTATION_DETAILS.md`
- `docs/implementation/REFERENCE_DECISION_LOG.md`
- `docs/implementation/review_packages/SPRINT-001-CODE-REVIEW-PACKAGE.md`

## Known Limitations

- `poolclass=NullPool` is used for the async engine to keep the integration-test event-loop story simple. A production deployment should evaluate a real async connection pool.
- If PostgreSQL fails after MinIO upload or Qdrant upsert succeeds, an orphan object or vector point can remain. There is no outbox, saga or reconciliation platform in this sprint.
- MinIO crop deletion during identity deactivation is best-effort.
- The `0.95` match threshold is a test-only value for deterministic fixtures, not a production-calibrated threshold.
- Concurrent same-unknown identity creation is not solved; deferred to a later sprint.
- No FastAPI endpoints, UI, real detector/recognizer, video pipeline or GPU path is implemented.

## Commit/Push Status

No `git add`, commit, push or destructive Docker command was executed.

## MCP / Skill Accountability

| Tool/Skill | Use | Outcome |
|---|---|---|
| `Read` / `Bash` | Repository baseline, file reading, command execution | Used extensively |
| `Edit` / `Write` | Source and documentation edits | Used |
| `Agent` / `Workflow` | Not used; sprint is a single cohesive vertical slice | Skipped |
| `context7` | Official docs for SQLAlchemy 2.0, Qdrant `query_points`, MinIO SDK | Used indirectly via prior plan; parity verified with real tests |
| `deepwiki` / `exa` / `postman` / `playwright` | Not needed for storage foundation without API/UI | Skipped |
| Ruflo / `21st` | Forbidden by AGENTS.md | Not used |

## Recommended Next Sprint

**Phase 1 Sprint 02 — Image Recognition API Vertical Slice**

Add FastAPI endpoints for `POST /faces/recognize` and `POST /faces/enroll`, Pydantic request/response contracts, input validation, and end-to-end API acceptance tests against the existing real PostgreSQL/MinIO/Qdrant foundation. Keep the detector/recognizer as deterministic fixture-based adapters until a real inference pipeline is ready.
