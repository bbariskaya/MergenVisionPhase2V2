# Phase 2 Target Contract

This document captures the read-only contracts the Phase 1 isolated GPU bulk enrollment tool must satisfy so that enrolled identities are visible to the existing Phase 2 image and video pipelines.

Source-of-truth order:

1. This document.
2. Current HEAD `backend/app/infrastructure/persistence/sqlalchemy/models/{person,face_identity,face_sample}.py`.
3. Current HEAD `backend/app/infrastructure/storage/minio_adapter.py`.
4. Current HEAD `backend/app/infrastructure/vectors/qdrant_adapter.py`.
5. MergenVisionDemo commit `5bf4b4c57542b26058e8d068186faee06c0fc29c` compute/queue/ID patterns.

## PostgreSQL

### `person`

| Column | Type | Notes |
|--------|------|-------|
| `person_id` | UUID PK | Deterministic UUIDv5 over HMAC(secret, namespace:key). |
| `display_name` | VARCHAR(255) NOT NULL | From manifest subject display name. User rename must be preserved; never blindly overwritten on conflict. |
| `person_metadata` | JSONB NOT NULL default `{}` | May hold `source_namespace`, `external_subject_key` fingerprint. Raw source key must not leak. |
| `is_active` | BOOLEAN NOT NULL default true | Soft delete uses `deleted_at` + `is_active=false`. |
| `version` | INTEGER NOT NULL default 1 | Constraint `version >= 1`. Do not increment during idempotent bulk re-runs. |
| `created_at/updated_at/deleted_at` | TIMESTAMPTZ | Managed by ORM defaults; do not inject synthetic timestamps. |

Constraint: `(is_active=true AND deleted_at IS NULL) OR (is_active=false AND deleted_at IS NOT NULL)`.

### `face_identity`

| Column | Type | Notes |
|--------|------|-------|
| `face_id` | UUID PK | Deterministic UUIDv5 over HMAC(secret, namespace:key). Same person/model-independent identity. |
| `status` | VARCHAR(16) NOT NULL | `anonymous` or `known`. Phase 1 bulk writes `known`. |
| `is_active` | BOOLEAN NOT NULL default true | `false` when redirected or deleted. Do not reactivate a redirected/inactive identity. |
| `display_name` | VARCHAR(255) nullable | Required when `status='known'`. |
| `identity_metadata` | JSONB NOT NULL default `{}` | May hold `source_namespace`, `external_subject_key` fingerprint. |
| `person_id` | UUID nullable | Set when `status='known'`; FK to `person.person_id`. |
| `redirect_to_face_id` | UUID nullable | Set when this identity redirects to a canonical face. Phase 1 bulk must not enroll into a redirected identity. |
| `version` | INTEGER NOT NULL default 1 | Constraint `version >= 1`. |
| `created_at/updated_at/deleted_at` | TIMESTAMPTZ | ORM defaults. |

Check constraints:
- `status IN ('anonymous', 'known')`.
- `known` requires `person_id IS NOT NULL AND display_name IS NOT NULL AND btrim(display_name) != ''`.
- `redirect_to_face_id IS NOT NULL` implies `is_active=false`.
- Active/deleted consistency.

Phase 1 bulk creates one canonical `known` `FaceIdentity` per subject, linked to the deterministic `Person`.

### `face_sample`

| Column | Type | Notes |
|--------|------|-------|
| `sample_id` | UUID PK | Deterministic UUIDv5 over `face_id:image_sha256:model_version:preprocess_version`. Used as Qdrant point ID. |
| `face_id` | UUID NOT NULL | FK to `face_identity.face_id`. |
| `state` | VARCHAR(16) NOT NULL | `pending`, `active`, `failed`, `inactive`. Correct column name is `state`, not `status`. |
| `bucket` | VARCHAR(255) nullable | Required when `state='active'`. |
| `object_key` | VARCHAR(1024) nullable | Required when `state='active'`; technical key `faces/{face_id}/{sample_id}/aligned.jpg`. |
| `failure_code` | VARCHAR(64) nullable | Required when `state='failed'`. |
| `is_active` | BOOLEAN NOT NULL default false | `true` only for `state='active'`. |
| `created_at` | TIMESTAMPTZ | ORM default. |
| `activated_at` | TIMESTAMPTZ nullable | Required when `state='active'`. |
| `deactivated_at` | TIMESTAMPTZ nullable | Set when `state='inactive'`. |

Check constraints:
- `state IN ('pending', 'active', 'failed', 'inactive')`.
- `pending` implies `is_active=false`, `bucket/object_key NULL`, `activated_at/deactivated_at NULL`, `failure_code NULL`.
- `active` implies `is_active=true`, `bucket/object_key NOT NULL`, `activated_at NOT NULL`, `failure_code NULL`.
- `failed` implies `is_active=false`, `failure_code NOT NULL`.
- `inactive` implies `is_active=false`, `bucket/object_key NOT NULL`, `activated_at/deactivated_at NOT NULL`.

Phase 1 writes `state='pending'`, then uploads to MinIO, upserts Qdrant, then updates to `state='active'` with `activated_at`.

## MinIO

- Bucket: configured via `MV_MINIO_BUCKET_NAME`; no dangerous default.
- Object key format: `faces/{face_id}/{sample_id}/aligned.jpg`.
- Content type: `image/jpeg`.
- User metadata: `sha256` = SHA-256 of object bytes.
- Idempotency: if same key exists, verify size and SHA match; reject conflict.
- Only accepted aligned 112×112 face crops are written. Raw dataset JPEG must never be stored under this key.

## Qdrant

- Collection: configured via `MV_QDRANT_COLLECTION_NAME`; default collection name is `face_samples_retinaface_r50_glintr100_v1`.
- Vector: 512-D float32, distance = `COSINE`.
- Point ID: `str(sample_id)` (UUID string).
- Payload:
  - `sample_id`: str(sample_id)
  - `face_id`: str(face_id)
  - `active`: true
  - `model_version`: exact model version string
- Required payload indexes:
  - `face_id`: KEYWORD
  - `active`: BOOL
  - `model_version`: KEYWORD
- Collection must be validated on startup. Network/auth/contract errors must fail-closed, not silently create a collection.

## Model / Preprocess Contract

- `model_version`: `retinaface_r50_glintr100_v1` unless overridden by env.
- `preprocess_version`: from `config/model_profile.json` field.
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

`QdrantVectorStore.query` filters by:

- `active == true`
- `model_version == settings.model_version`

Therefore Phase 1 must write `active=true` and the exact `model_version` string.

## Deterministic ID Contract

Required env:

```text
MV_PHASE1_BULK_ID_HMAC_KEY=<secret>
```

Derivation:

```text
identity_key = source_namespace + ":" + normalized_external_subject_key
identity_hmac = HMAC-SHA256(MV_PHASE1_BULK_ID_HMAC_KEY, identity_key)
person_id = UUIDv5(PERSON_NAMESPACE, identity_hmac)
face_id = UUIDv5(FACE_NAMESPACE, identity_hmac)
sample_id = UUIDv5(
    SAMPLE_NAMESPACE,
    face_id + ":" + source_image_sha256 + ":" + model_version + ":" + preprocess_version
)
```

- `person_id` and `face_id` are stable across model/preprocess changes.
- `sample_id` changes when image bytes, model version, or preprocess version change.
- Resume uses a journal fingerprint of the HMAC key; mismatch raises `BLOCKED_ID_NAMESPACE_MISMATCH`.
- Raw `external_subject_key`, folder path, or display name must not appear in logs, object keys, Qdrant payload, or benchmark reports.
