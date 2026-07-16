# MergenVision — Phase 1 Sprint 01

**Minimal Identity Storage Foundation**

This repository contains the backend foundation for MergenVision: a layered, async Python service that stores face identities, samples, process records and recognition results across PostgreSQL, MinIO and Qdrant.

> **Status:** `CORRECTION_IN_PROGRESS`. Sprint 01 is being hardened for senior review. The previous "PASS" claim has been withdrawn. See [`docs/implementation/CURRENT_SPRINT.md`](docs/implementation/CURRENT_SPRINT.md).

## What is implemented

- **Domain layer** (`backend/app/domain/`) — pure Python entities/value objects with explicit lifecycle state machines.
- **Application layer** (`backend/app/application/`) — ports (repositories, object store, vector store, unit of work, id generator) and the `IdentityStorageLifecycleService`.
- **Infrastructure layer** (`backend/app/infrastructure/`):
  - SQLAlchemy 2.0 async PostgreSQL persistence with Alembic migrations.
  - MinIO object-store adapter (async wrapper around the official sync SDK).
  - Qdrant vector-store adapter (512-D cosine collection, payload indexes).
- **UUIDv7** business identifiers via the `uuid7` PyPI package.
- **Optimistic locking** on identity enrollment and deactivation.
- **Cross-store synchronous compensation** when MinIO/Qdrant writes fail during identity creation.
- **Fail-closed test resource guard** that refuses to run cleanup against non-test resources.

## Quick start

Requirements:

- Docker & Docker Compose v2
- Python 3.12
- GNU Make

```bash
# 1. Clone the repository
git clone <repo-url>
cd MergenVisionPhase2v2

# 2. Create local environment files, venv, install deps and pull test images
make bootstrap

# 3. Run the full Sprint 01 acceptance suite
make phase1-sprint-01-acceptance
```

`make bootstrap` copies `backend/.env.test.example` → `backend/.env.test` and `backend/.env.example` → `backend/.env` only if those files do not already exist.

## Test isolation

Acceptance tests use a dedicated Compose namespace, `mergenvision-s01-test`, with isolated host ports and named volumes:

| Service | Host port | Container port |
|---|---|---|
| postgres-test | `127.0.0.1:55432` | `5432` |
| minio-test API | `127.0.0.1:59000` | `9000` |
| minio-test console | `127.0.0.1:59001` | `9001` |
| qdrant-test HTTP | `127.0.0.1:56333` | `6333` |
| qdrant-test gRPC | `127.0.0.1:56334` | `6334` |

The [`tests/support/resource_guard.py`](backend/tests/support/resource_guard.py) module requires exact environment values (test bucket/collection names, localhost endpoints) before any cleanup runs. The Makefile sources `backend/.env.test` for every test target.

## Useful targets

```bash
make phase1-sprint-01-up            # start test services and run migrations
make phase1-sprint-01-down          # stop test services (no volume deletion)
make phase1-sprint-01-static        # ruff + mypy
make phase1-sprint-01-unit          # unit tests only
make phase1-sprint-01-integration   # integration tests only
make phase1-sprint-01-acceptance    # full acceptance (static + unit + integration + git diff --check)
```

## Safety rules

- `make phase1-sprint-01-acceptance` never runs `docker compose down -v` or deletes volumes.
- `backend/.env.test` contains only dummy test credentials and is safe to commit; `backend/.env` is ignored and holds local development secrets.
- No destructive cleanup of main/dev resources is performed.

## Documentation

- Sprint scope and status: [`docs/implementation/CURRENT_SPRINT.md`](docs/implementation/CURRENT_SPRINT.md)
- Implementation details: [`docs/implementation/IMPLEMENTATION_DETAILS.md`](docs/implementation/IMPLEMENTATION_DETAILS.md)
- Decision log: [`docs/implementation/REFERENCE_DECISION_LOG.md`](docs/implementation/REFERENCE_DECISION_LOG.md)
- Code review package: [`docs/implementation/review_packages/SPRINT-001-CODE-REVIEW-PACKAGE.md`](docs/implementation/review_packages/SPRINT-001-CODE-REVIEW-PACKAGE.md)

## License

Proprietary — see `pyproject.toml`.
