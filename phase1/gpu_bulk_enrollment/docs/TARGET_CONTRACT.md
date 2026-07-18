# Phase 2 Target Contract

This document captures the read-only contracts the Phase 1 bulk enrollment tool must satisfy so that enrolled identities are visible to the existing Phase 2 image and video pipelines.

## PostgreSQL

### `person`

| Column | Type | Notes |
|--------|------|-------|
| `person_id` | UUID PK | Phase 1 generates deterministic UUIDv7 or namespace-based UUID; schema expects UUID. |
| `display_name` | VARCHAR(255) NOT NULL | From manifest subject display name. |
| `person_metadata` | JSONB NOT NULL default `{}` | May hold `source_dataset`, `external_subject_key`, `source_namespace`. |
| `is_active` | BOOLEAN NOT NULL default true | Soft delete uses `deleted_at` + `is_active=false`. |
| `version` | INTEGER NOT NULL default 1 | Constraint `version >= 1`. |
| `created_at/updated_at/deleted_at` | TIMESTAMPTZ | `deleted_at` null unless soft-deleted. |

Constraint: `(is_active=true AND deleted_at IS NULL) OR (is_active=false AND deleted_at IS NOT NULL)`.

### `face_identity`

| Column | Type | Notes |
|--------|------|-------|
| `face_identity_id` | UUID PK | Deterministic UUID per subject. |
| `person_id` | UUID nullable | Set when status is `known`. |
| `display_name` | VARCHAR(255) nullable | Required when `status='known'`. |
| `status` | VARCHAR(20) nullable | `pending`, `known`, `redirect`, `deleted`. |
| `is_active` | BOOLEAN NOT NULL | `false` when `redirect` or `deleted`. |
| `redirect_to_face_id` | UUID nullable | Set when `status='redirect'`. |

Check constraints:
- `known` requires `person_id` and `display_name`.
- `redirect` implies `is_active=false`.
- Active/deleted consistency.

Phase 1 will create `pending` rows during reservation and promote to `known` on successful persistence.

### `face_sample`

| Column | Type | Notes |
|--------|------|-------|
| `sample_id` | UUID PK | Deterministic UUID; used as Qdrant point ID. |
| `face_identity_id` | UUID NOT NULL | Canonical face identity. |
| `person_id` | UUID nullable | Person reference; must match `face_identity.person_id` when known. |
| `bucket` | VARCHAR nullable | Required when active. |
| `object_key` | VARCHAR nullable | Required when active; technical UUID key. |
| `model_version` | VARCHAR NOT NULL | Same as Phase 2 `settings.model_version`. |
| `preprocess_version` | VARCHAR NOT NULL | From `model_profile.json`. |
| `embedding_model` | VARCHAR NOT NULL | Same as `model_version` (GlintR100). |
| `detector_model` | VARCHAR NOT NULL | Detector model tag (RetinaFace). |
| `bbox` | JSONB | `{x, y, width, height}` in original image coordinates. |
| `landmarks` | JSONB | 5 landmarks `[{x,y}, ...]` in original image coordinates. |
| `quality_score` | FLOAT | Detector confidence or quality primitive. |
| `status` | VARCHAR(20) NOT NULL | `pending`, `active`, `failed`, `inactive`. |
| `activated_at` | TIMESTAMPTZ | Required when active. |

Check constraints:
- Active requires `bucket`, `object_key`, `activated_at` not null.
- Status enum limited.

Phase 1 will write `pending` rows, then upload to MinIO, batch upsert Qdrant, then update to `active`.

## MinIO

- Bucket: `settings.minio_bucket_name`.
- Object key format: `enrollments/{person_id}/{sample_id}`.
- Content type: `image/webp` for aligned crop.
- User metadata: `sha256` = SHA-256 of object bytes.
- Idempotency: if same key exists with same size/sha256, reuse existing stat.

## Qdrant

- Collection: `settings.qdrant_collection_name`.
- Vector: 512-D float32, distance = `COSINE`.
- Point ID: `str(sample_id)` (UUID string).
- Payload:
  - `sample_id`: str(sample_id)
  - `face_id`: str(face_identity_id)
  - `active`: true
  - `model_version`: settings.model_version

## Model / Preprocess Contract

- `model_version`: from Phase 2 `.env` / settings.
- `preprocess_version`: from `model_profile.json` field.
- Detector input: 640×640 RGB float32 NCHW.
- Recognizer input: 112×112 RGB float32 NCHW.
- Embedding: 512-D, L2-normalized on GPU.
- Landmark template (ArcFace canonical):
  - (38.2946, 51.6963)
  - (73.5318, 51.5014)
  - (56.0252, 71.7366)
  - (41.5493, 92.3655)
  - (70.7299, 92.2041)

## Identity Query Filter Used by Phase 2 Video Pipeline

Qdrant query filter used by `QdrantVectorStore.query`:
- `active == true`
- `model_version == settings.model_version`

Therefore Phase 1 must write `active=true` and the exact `model_version` string.
