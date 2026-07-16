# Phase 1 Sprint 01 — Implementation Details

## 1. Scope

This document records the concrete implementation choices for the Minimal Identity Storage Foundation sprint. It covers the four-table PostgreSQL schema, the pure Python domain layer, SQLAlchemy 2.0 async infrastructure adapters, MinIO/Qdrant adapters, the lifecycle service and the integration-test strategy.

## 2. PostgreSQL Schema

One Alembic migration creates the tables in this order:

1. `face_identity`
2. `process_record`
3. `face_sample`
4. `recognition_result`

### 2.1 `face_identity`

| Column | Type | Notes |
|---|---|---|
| `face_id` | UUID | Primary key. Generated as UUIDv7 in application code. |
| `status` | VARCHAR(16) | `anonymous` or `known`. |
| `is_active` | BOOLEAN | `true` for active identities; `false` after deactivation. |
| `display_name` | VARCHAR(255) | Non-null only after enrollment. |
| `identity_metadata` | JSONB | `{}` default. |
| `version` | INTEGER | Optimistic-lock counter. |
| `created_at` / `updated_at` / `deleted_at` | TIMESTAMPTZ | `deleted_at` set on deactivation. |

Indexes:

- `face_identity_status_is_active_idx`
- `face_identity_created_at_idx`

### 2.2 `process_record`

| Column | Type | Notes |
|---|---|---|
| `process_id` | UUID | Primary key, UUIDv7. |
| `process_type` | VARCHAR(32) | `image_recognize`, `face_enroll`, `face_delete`. |
| `status` | VARCHAR(16) | `processing`, `completed`, `failed`. |
| `face_count` | INTEGER | Number of faces processed, if completed. |
| `error_code` | VARCHAR(64) | Failure code, if failed. |
| `details` | JSONB | `{}` default. |
| `created_at` / `completed_at` | TIMESTAMPTZ | Completion timestamp, when applicable. |

Indexes:

- `process_record_status_created_at_idx`
- `process_record_process_type_created_at_idx`

### 2.3 `face_sample`

| Column | Type | Notes |
|---|---|---|
| `sample_id` | UUID | Primary key, UUIDv7. |
| `face_id` | UUID | Foreign key to `face_identity` with `ON DELETE RESTRICT`. |
| `state` | VARCHAR(16) | `pending`, `active`, `failed`, `inactive`. |
| `bucket` | VARCHAR(64) | MinIO bucket once active. |
| `object_key` | VARCHAR(512) | MinIO key once active. |
| `failure_code` | VARCHAR(64) | Set when state becomes `failed`. |
| `created_at` / `activated_at` / `deactivated_at` | TIMESTAMPTZ | Lifecycle timestamps. |

Indexes:

- `face_sample_face_id_sample_state_idx`
- `face_sample_bucket_key_unique_idx` (unique)

### 2.4 `recognition_result`

Immutable snapshot of a recognition decision.

| Column | Type | Notes |
|---|---|---|
| `result_id` | UUID | Primary key, UUIDv7. |
| `process_id` | UUID | Foreign key to `process_record`. |
| `face_id` | UUID | Foreign key to `face_identity`. |
| `sample_id` | UUID | Foreign key to `face_sample`, nullable. |
| `status` | VARCHAR(16) | `known`, `anonymous`, `new_anonymous`. |
| `bounding_box` | JSONB | Bounding box at processing time. |
| `match_confidence` | NUMERIC(4,3) | Cosine score or highest rejected score. |
| `result_metadata` | JSONB | `{}` default. |
| `created_at` | TIMESTAMPTZ | Auto-generated. |

Index:

- `recognition_result_process_id_result_index_idx`

## 3. Domain Layer

The domain layer is pure Python and does not import SQLAlchemy, MinIO, Qdrant, asyncpg or FastAPI. It lives under `backend/app/domain/`.

Key state machines:

- `FaceIdentity`: `anonymous` ↔ `known` via `promote_to_known()`; deactivation via `deactivate()`.
- `FaceSample`: `pending` → `active`/`failed`/`inactive`.
- `ProcessRecord`: `processing` → `completed`/`failed`.
- `RecognitionResult`: immutable snapshot; no update method.

## 4. Infrastructure Adapters

### 4.1 SQLAlchemy 2.0

- `DeclarativeBase` subclass as the ORM base.
- `create_async_engine(..., poolclass=NullPool)` is used for the integration-test session to avoid asyncpg greenlet/connection-reuse issues across pytest-asyncio session-scoped tests.
- `SqlAlchemyUnitOfWork` creates a fresh `AsyncSession` on each `async with` entry, begins a transaction explicitly, and closes the session on exit.
- A `UnitOfWorkFactory` callable is injected into services so every concurrent workflow receives its own unit-of-work instance instead of a shared mutable object.

### 4.2 UUIDv7

The `uuid7` PyPI package is used. It exposes `uuid_extensions.uuid7()`. The wrapper casts the result to `uuid.UUID(str(uuid7()))` for type safety.

### 4.3 MinIO

- Development bucket: `mergenvision-face-samples`; dedicated test bucket: `mergenvision-s01-test-face-samples`.
- Object key: `faces/{faceId}/{sampleId}/aligned.webp`
- The official MinIO Python SDK is synchronous; blocking calls are isolated via `asyncio.to_thread`.
- `upload()` computes a SHA-256 checksum, stores it as object metadata, and validates the upload by calling `stat()` immediately after.
- `upload()` is idempotent for identical key/content: uploading the same bytes to the same key returns the existing object stat. Uploading different bytes to an existing key raises a conflict.

### 4.4 Qdrant

- Development collection: `face_samples_v1`; dedicated test collection: `mergenvision_s01_test_face_samples_v1`.
- Vector: 512-D cosine
- Point ID: `sample_id` as string
- Payload: `{"sample_id": "<uuid>", "face_id": "<uuid>", "active": true, "model_version": "<settings.model_version>"}`
- Search uses the non-deprecated `query_points()` API with an `active=True` filter.
- Query results are validated: the payload `sample_id` must match the point id and both IDs must be valid UUIDs.
- Payload indexes are created for `face_id`, `active`, and `model_version`.
- PostgreSQL always validates that the candidate identity and sample are active before trusting a Qdrant result.

### 4.5 Docker Compose environment

- Development services are defined in `docker-compose.yml` and load `backend/.env`.
- Acceptance tests use `docker-compose.test.yml` with the isolated `mergenvision-s01-test` project, named test-only volumes, and `backend/.env.test`.
- `backend/.env.example` and `backend/.env.test.example` contain documented placeholders.
- Real `.env` files are ignored by Git; `backend/.env.test` is tracked because it contains only dummy test credentials.

## 5. IdentityStorageLifecycleService

The service is constructed with a `UnitOfWorkFactory`, `ObjectStore`, `VectorStore`, `IdGenerator` and an optional `candidate_limit`.

### 5.1 `resolve_or_create`

1. Validate the crop, embedding, bounding box and threshold.
2. Create a `process_record` with status `processing`.
3. Query Qdrant for the top `candidate_limit` candidates.
4. Skip non-finite scores and candidates below `match_threshold`.
5. For each remaining candidate, verify in PostgreSQL that the identity and sample are active and that the sample belongs to the identity.
6. If a candidate is accepted:
   - Persist an `anonymous` or `known` result.
   - Complete the process.
   - Return the existing `face_id`.
7. Otherwise create a new anonymous identity/sample, upload the crop, index the vector, persist a `new_anonymous` result, and complete the process.

`match_confidence` for an unmatched result is the highest rejected candidate score, or `0.0` if no candidate exists.

### 5.2 `add_sample`

Requires an active identity, creates a new pending sample, uploads the crop, indexes the vector, then marks the sample active using the bucket and key returned by `ObjectStat`.

### 5.3 `enroll_identity`

Creates a `face_enroll` process, requires an active anonymous identity, promotes it to `known` with display name and metadata, and commits the change using `update_with_expected_version` (optimistic locking). Old recognition results remain unchanged.

### 5.4 `deactivate_identity`

Creates a `face_delete` process, marks the identity inactive with `deleted_at` using optimistic locking, marks all active samples inactive, completes the process, then best-effort disables the corresponding Qdrant points. MinIO crop deletion is best-effort and documented as a limitation.

### 5.5 Cross-store compensation

If MinIO upload or Qdrant upsert fails during the first sample creation of a new identity:

- The sample is marked `failed`.
- The process is marked `failed`.
- The newly created identity is deactivated and `deleted_at` is set.
- The partial MinIO object or Qdrant point is removed best-effort.
- No recognition result is created.

This prevents an active identity with zero active samples.

## 6. Testing Strategy

- TDD execution order was followed for domain entities, migration, repositories, MinIO, Qdrant and lifecycle flows.
- Deterministic 512-D unit vectors are used: `vector_a` = `[1,0,0,...]`, `vector_b` = `[0,1,0,...]`.
- Test-only match threshold: `0.95`.
- Integration tests run against the dedicated `mergenvision-s01-test` Docker Compose namespace on non-conflicting host ports:
  - PostgreSQL `127.0.0.1:55432:5432`
  - MinIO API `127.0.0.1:59000:9000`, console `127.0.0.1:59001:9001`
  - Qdrant HTTP `127.0.0.1:56333:6333`, gRPC `127.0.0.1:56334:6334`
- `tests/support/resource_guard.py` implements a fail-closed guard that rejects any test run whose environment variables point outside the test namespace.
- The lifecycle test module uses a synchronous autouse cleanup fixture that runs store cleanup in an isolated `asyncio.run()` loop, while the async tests run under a pytest-asyncio session-scoped event loop. This avoids asyncpg greenlet/connection-reuse failures.
- The full acceptance command is `make phase1-sprint-01-acceptance`.

## 7. Known Limitations

- `poolclass=NullPool` is used for the async engine to keep the integration tests stable. A real deployment may prefer a connection pool such as `AsyncAdaptedQueuePool` with proper reset behavior.
- If PostgreSQL transaction fails after MinIO upload or Qdrant upsert succeeds, an orphan object or vector point may remain. There is no outbox, saga or reconciliation worker in this sprint.
- Sprint 01 does not delete MinIO objects during deactivation; biometric crop retention/deletion policy is deferred.
- The `0.95` threshold is a test fixture value, not a production-calibrated recognition threshold.
- No FastAPI endpoints, UI, real detector/recognizer, video pipeline, or GPU path is implemented.

## 8. Acceptance Command

```bash
make phase1-sprint-01-acceptance
```

All targets passed twice on 2026-07-16 (77 tests total: 44 unit + 33 integration, ruff, mypy, format check, Docker build/import, restart probe, git diff --check).
