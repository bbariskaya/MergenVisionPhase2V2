# MergenVisionDemo Cross-Repo Knowledge Pack

## Executive Summary

MergenVisionDemo is a GPU-native face-recognition demo that already implements a working bulk-enrollment pipeline for LFW/VGGFace/CASIA datasets. Its architecture cleanly separates a Python/FastAPI control plane from a CUDA/TensorRT data plane bound together by `DeviceTensor`, `BufferArena`, and pybind11 native kernels. The most valuable patterns for Phase2v2 are the deterministic `identity_hmac → UUIDv5` id model, the producer/consumer `BulkEnrollmentService` with bounded persistence concurrency, the zero-copy `GpuFacePipeline.extract_batch` path, and the durable `ProcessRecord`/`ProcessEvent` lifecycle used by the separate GPU worker containers. These can be reused in the Phase2v2 `phase1/gpu_bulk_enrollment/` package while leaving the existing image/video runtime untouched.

## Project Overview

- **Type:** Python/FastAPI backend + React TypeScript frontend + CUDA/C++ native extension.
- **Repository root:** `/home/user/MergenVisionDemo`
- **HEAD:** `5bf4b4c57542b26058e8d068186faee06c0fc29c`
- **Backend entry:** `backend/app/main.py` (API) and `backend/app/workers/gpu_worker.py` (offline GPU worker).
- **Frontend entry:** `frontend/src/App.tsx`.
- **Native module:** `backend/native/mergenvision_gpu/` built with pybind11 + scikit-build-core + CMake.
- **Key data-plane files verified by codebase-memory:**
  - `backend/app/ml/gpu/face_pipeline.py:336` — `GpuFacePipeline.extract_batch`
  - `backend/app/ml/gpu/decoder.py:95` — `JpegGpuDecoder.decode_batch`
  - `backend/app/ml/gpu/recognizer.py:58` — `GpuRecognizer.embed`
  - `backend/app/ml/gpu/trt_device_engine.py:102` — `TrtDeviceEngine.infer_device`
  - `backend/app/ml/gpu/buffer_arena.py:128` — `BufferArena`
  - `backend/app/ml/gpu/device_tensor.py:21` — `DeviceTensor`
- **Key control-plane files verified:**
  - `backend/app/services/bulk_enrollment.py:85` — `BulkEnrollmentService`
  - `backend/app/services/bulk_orchestrator.py:436` — `dispatch_shards`
  - `backend/app/services/bulk_manifest.py:35` — `EnrollmentIdentity`, `EnrollmentPhoto`
  - `backend/app/workers/gpu_worker.py:381` — `create_worker_app`

## Bulk Enrollment

### Key files and symbols

- `backend/app/services/bulk_enrollment.py:85` — `BulkEnrollmentService`
  - Methods: `enroll_shard` (`:568`), `_produce` (`:427`), `_consume` (`:489`), `_read_and_extract` (`:373`), `_persist_batch` (`:241`), `_extract_batch_faces` (`:125`), `_ensure_identities` (`:176`), `_upload_photo` (`:225`), `_upsert_qdrant` (`:365`), `_commit_progress` (`:528`).
- `backend/app/services/bulk_manifest.py:35` — `EnrollmentIdentity` (immutable descriptor with deterministic IDs).
- `backend/app/services/bulk_manifest.py:24` — `EnrollmentPhoto`.
- `backend/app/services/bulk_manifest.py:105` — `shard_by_person_id` (deterministic sharding by `person_id % num_shards`).

### Algorithm / flow

1. The orchestrator builds a deterministic manifest where each folder becomes one `EnrollmentIdentity`; all IDs/HMACs are derived from normalized folder names and content SHA-256, never generated counters.
2. `enroll_shard` creates a `ProcessRecord` of type `bulk_enroll_shard` and starts a single-producer / single-consumer `asyncio.Queue(maxsize=2)`.
3. `_produce` iterates over identities, groups photos into batches of `extract_batch_size`, calls `_read_and_extract` (bounded concurrent file reads + GPU batch extraction), and pushes extracted tuples to the queue.
4. `_consume` pops each batch, calls `_persist_batch`, and after every batch commits progress to `process_record.summary["progress"]` via `_commit_progress`.
5. Decode/read failures count as soft errors and do not abort the shard; any extraction/persistence exception marks the shard `failed` and stores structured error codes (`EXTRACTION_ERROR`, `PERSISTENCE_ERROR`, etc.).
6. Soft error rate > 3% marks the shard `failed` regardless of no hard exception.

### Batch sizes, executors, and GPU usage

- `bulk_extract_batch_size`: **256** (`backend/app/core/config.py:47`).
- `bulk_max_persistence_concurrency`: **32** (`backend/app/core/config.py:48`), enforced by `asyncio.Semaphore` in `_persist_batch`/`_upload_photo`.
- Producer queue depth: **2** (`asyncio.Queue(maxsize=2)` at `bulk_enrollment.py:604`).
- GPU work is serialized per pipeline by `asyncio.Lock`; the actual inference runs on a dedicated `ThreadPoolExecutor(max_workers=1)` (`gpu_worker.py:317`).
- IO runs on a separate `ThreadPoolExecutor(max_workers=min(32, cpu_count*2))` (`gpu_worker.py:320`).
- The bulk service can also run synchronously when `gpu_executor=None`.

### What Phase2v2 can copy

- The `EnrollmentIdentity` / `EnrollmentPhoto` descriptor pattern with pre-computed deterministic IDs.
- Producer/consumer queue with `_read_and_extract` returning `(identity, photo, GpuFaceExtraction, raw_bytes)`.
- `_persist_batch` as the canonical three-store write: `pg_insert(FaceIdentity)`/`Person` → `pg_insert(PersonPhoto)`/`FaceSample` → `FaceVectorStore.upsert_batch`.
- The “soft error rate budget” and per-batch durable progress checkpoint pattern.

## GPU / Native Runtime

### Key files and symbols

- `backend/app/ml/gpu/face_pipeline.py:41` — `GpuFacePipeline` (single-GPU end-to-end pipeline).
- `backend/app/ml/gpu/face_pipeline.py:336` — `extract_batch` (batch path used by bulk enrollment).
- `backend/app/ml/gpu/face_pipeline.py:207` — `extract_bytes` (single-image path returning all faces).
- `backend/app/ml/gpu/decoder.py:27` — `JpegGpuDecoder` (nvImageCodec, refuses CPU fallback).
- `backend/app/ml/gpu/recognizer.py:19` — `GpuRecognizer` (ArcFace via TensorRT + native L2).
- `backend/app/ml/gpu/trt_device_engine.py:18` — `TrtDeviceEngine` (device-pointer TensorRT binding).
- `backend/app/ml/gpu/alignment.py` — `GpuFaceAligner` (similarity transform on device).
- `backend/app/ml/gpu/buffer_arena.py:128` — `BufferArena` / `BufferLease` (event-fenced reuse).
- `backend/app/ml/gpu/device_tensor.py:21` — `DeviceTensor` (immutable device pointer wrapper).
- `backend/native/mergenvision_gpu/` — pybind11 module exposing CUDA kernels:
  - `l2_normalize`, `similarity_transform`, `nms`, `retinaface_decode_batch`, `retinaface_pick_largest`, `scale_clip_compact`, `warp_align`, `scrfd_decode_level`, `argsort_descending`.

### Batch inference details

`GpuFacePipeline.extract_batch` processes a list of JPEG byte buffers as follows:

1. `JpegGpuDecoder.decode_batch` decodes all buffers on GPU; it verifies `image.buffer_kind == STRIDED_DEVICE` and raises if CPU fallback occurs.
2. `RetinaFacePreprocessor.preprocess_batch` builds the detector NCHW input on GPU.
3. `TrtDeviceEngine.infer_device` runs the RetinaFace TensorRT engine with device-resident input/output bindings.
4. `RetinaFacePostprocess.decode` runs native CUDA decode/NMS; `scale_and_compact` maps boxes/landmarks back to original image resolution.
5. `pick_largest_device` selects the largest face per image on GPU.
6. Only valid-selection metadata, bbox, landmarks, and score are copied to host (small D2H transfers).
7. For each selected face, `GpuFaceAligner.align` warps the face to `112x112` RGB on device; aligned chips are packed into a batched `[M,3,112,112]` tensor.
8. `GpuRecognizer.embed` runs the ArcFace TensorRT engine and native CUDA L2 normalization.
9. Final 512-D embeddings and compact metadata are copied to host once per batch.

Default sizes from `backend/app/core/config.py`:

- Detector input: **640x640** (`detector_input_size = 640`).
- Embedder input: **112x112** (`embedder_input_size = 112`).
- Embedding dim: **512** (`embedding_dim = 512`).
- Detector confidence threshold: **0.5**; NMS IoU: **0.4**.
- Batch sizes: detector batch up to `bulk_extract_batch_size = 256`; recognizer upper bound from engine profile, default fallback **64** (`recognizer.py:56`).

### Python ↔ native bridge

- Native extension `mergenvision_gpu._mergenvision_gpu` built via `pyproject.toml` → `scikit-build-core` → `CMakeLists.txt`.
- `CMakeLists.txt:6` targets CUDA architecture **75**; links `CUDA::cudart`; uses `pybind11_add_module(_mergenvision_gpu ...)`.
- The `Dockerfile` has a two-stage build: `nvidia/cuda:12.4.1-devel-ubuntu22.04` compiles the `.so` and copies it into the runtime image at `/opt/venv/lib/python3.11/site-packages/mergenvision_gpu`.
- Python calls kernels with `DeviceTensor.ptr`, shapes, and `ctypes` dtypes; device memory never crosses to NumPy until the pipeline boundary.

## Persistence & Storage

### MinIO

- `backend/app/infrastructure/minio.py:18` — `PhotoStorage`
- Bucket default: **mergenvision-photos** (`config.py:15`).
- All SDK calls use `asyncio.to_thread(...)` to avoid blocking the event loop.
- Bulk enrollment object key pattern: `enrollments/{person_id}/{photo_id}` (`bulk_enrollment.py:232`).
- API: `put_object`, `get_object`, `object_exists`, `delete_object`, `health_check`, `initialize()` creates bucket idempotently.

### PostgreSQL

- Uses SQLAlchemy 2.0 async ORM + `asyncpg`. `backend/app/domain/models.py` defines `FaceIdentity`, `Person`, `PersonPhoto`, `FaceSample`, `ProcessRecord`, `ProcessEvent`, `RecognitionRequest`, `RecognitionResult`, `InferenceProfile`.
- Bulk enrollment uses `sqlalchemy.dialects.postgresql.insert(...).on_conflict_do_nothing()` and `on_conflict_do_update()`:
  - `FaceIdentity` upsert by `identity_lookup_hmac` (`bulk_enrollment.py:214`).
  - `Person` upsert by `national_id_lookup_hmac` (`bulk_enrollment.py:219`).
  - `PersonPhoto` upsert by `photo_id` with status reset to `active` (`bulk_enrollment.py:315`).
  - `FaceSample` upsert by `photo_id` with status reset to `active` (`bulk_enrollment.py:332`).
- Photo/sample deduplication is done in-memory before the upsert to avoid cardinality violations.
- `ProcessRecord` status enum: `pending, queued, running, cancel_requested, cancelling, cancelled, completed, failed` (`models.py:298`).

### Qdrant

- `backend/app/infrastructure/qdrant.py:36` — `FaceVectorStore`
- Collection default: **face_samples** (`config.py:18`).
- Vector size: **512**; distance: **COSINE**.
- Required payload keys: `sampleId`, `photoId`, `personId`, `active`, `modelVersion`.
- Point ID equals `sample_id` (validated in `_validate_payload`).
- `upsert_batch` validates every point/vector, then chunks into **256**-point Qdrant requests (`qdrant.py:197`).
- `set_active_batch` also chunks by **256** (`qdrant.py:262`).
- Search filters by `active=True` and `modelVersion=<version>`.

### Cross-store consistency pattern

Within `_persist_batch` the order is: ensure identities → upload photo bytes → `pg_insert(PersonPhoto)` → `pg_insert(FaceSample)` → `FaceVectorStore.upsert_batch`. The same batch is committed after each `_persist_batch` via `_commit_progress`. There is no distributed transaction; idempotency is achieved through deterministic UUIDs and `ON CONFLICT DO UPDATE`/`UPSERT` semantics.

## Identity Model

### Tables and relationships

- `face_identity` — `FaceIdentity` (`models.py:24`)
  - PK `face_identity_id` (UUIDv7 by default; in bulk enrollment replaced by deterministic UUIDv5).
  - Unique lookup `identity_lookup_hmac`.
  - `display_name`, `is_active`.
- `person` — `Person` (`models.py:50`)
  - PK `person_id` (UUIDv7 or deterministic).
  - FK `face_identity_id`; names; `national_id_lookup_hmac` (unique).
- `person_photo` — `PersonPhoto` (`models.py:94`)
  - PK `photo_id` (UUIDv5 from content SHA-256 in bulk path).
  - `object_key`, `content_sha256`, `status ∈ {staged, active, failed, deleted}`.
- `face_sample` — `FaceSample` (`models.py:138`)
  - PK `sample_id` (UUIDv5 derived from `photo_id:model_version`).
  - FK `person_id`, `photo_id` (unique), bbox/landmarks JSONB, quality score, embedding model version, status.

### Deterministic IDs (`backend/app/core/ids.py`)

- `identity_hmac(identity_key, master_key)` → HMAC-SHA256 hex.
- `derive_person_id(hmac)` → `uuid5(PERSON_NAMESPACE, hmac)`.
- `derive_face_identity_id(hmac)` → `uuid5(FACE_IDENTITY_NAMESPACE, hmac)`.
- `derive_photo_id(content_sha256)` → `uuid5(PERSON_PHOTO_NAMESPACE, content_sha256)`.
- `derive_sample_id(photo_id, model_version)` → `uuid5(FACE_SAMPLE_NAMESPACE, f"{photo_id}:{model_version}")`.
- `derive_process_id(process_type, seed_bytes)` → `uuid5(PROCESS_RECORD_NAMESPACE, f"{process_type}:{seed_bytes.hex()}")`.
- Runtime IDs: `uuid7()` implementation (`ids.py:48`) returning timestamp + random bytes.

### Takeaway for Phase2v2

This is the exact idempotency contract Phase2v2 wants: stable identity key → HMAC → deterministic `faceId`/`personId`/`photoId`/`sampleId`. It guarantees that re-running the same bulk enrollment does not create duplicates because `ON CONFLICT DO NOTHING` handles the collisions. **Important limitation:** the identity key is currently the normalized *folder name*, not person metadata; Phase2v2 must decide how to generate stable identity keys for its own datasets.

## Worker / API Orchestration

### Control plane vs GPU workers

- **Public API** (`backend/app/api/routes/bulk_jobs.py`)
  - `POST /bulk-jobs/vggface`, `/lfw`, `/casia` → create durable parent `ProcessRecord`, then `BackgroundTasks.add_task(dispatch_shards, ...)`.
  - `GET /bulk-jobs/{job_id}` → aggregate shard progress.
  - `POST /bulk-jobs/{job_id}/cancel` → `request_cancellation`.
  - `POST /bulk-jobs/{job_id}/resume` → `resume_vggface_job` (VGGFace only).
- **Orchestrator** (`backend/app/services/bulk_orchestrator.py`)
  - Creates parent `ProcessRecord` of type `vggface_bulk`/`lfw_bulk`/`casia_bulk` with status `queued` and shard descriptors.
  - Shard idempotency keys: e.g. `"vggface-bulk:{parent_id}:shard:{idx}"`.
  - `dispatch_shards` fans out shards to workers via HTTP and polls status every **2s** (`_POLL_INTERVAL_SECONDS = 2.0`).
  - Cancellation: sets parent `cancel_requested`, then `POST /internal/v1/jobs/{job_id}/cancel` on every worker, then `cancelling`, then aggregates.
  - Resume: resets terminal child shards to `pending`, attaches `resume_after_identity_key` from last durable progress, and calls `dispatch_shards` again.
- **GPU Worker** (`backend/app/workers/gpu_worker.py`)
  - Separate FastAPI container exposing `/health/live`, `/health/ready`, `/internal/v1/jobs`, `/internal/v1/jobs/{job_id}/cancel`, `/internal/v1/jobs/{job_id}`.
  - One physical GPU per worker mounted as internal ordinal 0 (`HOST_GPU_DEVICE_ID`).
  - Jobs processed sequentially (`current_job_id` guard at `:467`).
  - Uses deterministic `process_id = derive_process_id("gpu_worker_job", idempotency_key)` so retry/resume is idempotent.
  - Calls `BulkEnrollmentService.enroll_shard` with `cancel_check=lambda: app_state.cancel_requested` and a progress callback that writes to `ProcessRecord.summary["progress"]` in a new DB session.

### ProcessRecord / ProcessEvent lifecycle

- `ProcessRecord`: `process_type`, `status`, `started_at`, `completed_at`, mutable `summary` JSONB, `error_message`.
- `ProcessEvent`: sequence-numbered audit events with `event_type`, `status_before`, `status_after`, message, details.
- Events are written on accept, start, progress, complete, cancel, and fail.

## Actionable Recommendations for Phase2v2

1. **Adopt the deterministic identity ID scheme** from `backend/app/core/ids.py` for bulk enrollment, so re-runs and resume are naturally idempotent. Use `derive_face_identity_id`, `derive_person_id`, `derive_photo_id`, and `derive_sample_id` exactly as implemented in `ids.py:28-41`.
2. **Copy the descriptor model** `EnrollmentIdentity` + `EnrollmentPhoto` from `backend/app/services/bulk_manifest.py:24-43` and the deterministic sharding function `shard_by_person_id` (`:105`) so Phase2v2 can shard by `person_id % num_workers` without state.
3. **Reuse the `BulkEnrollmentService` producer/consumer skeleton** from `backend/app/services/bulk_enrollment.py:85-686` in `phase1/gpu_bulk_enrollment/`. Keep `_produce`/`_consume`, the `asyncio.Queue(maxsize=2)`, and the per-batch `_commit_progress` durable checkpoint.
4. **Use the exact `_persist_batch` cross-store ordering** (`bulk_enrollment.py:241-363`): ensure identities → MinIO upload → `pg_insert(PersonPhoto)` → `pg_insert(FaceSample)` → `FaceVectorStore.upsert_batch`. The deterministic IDs make this safe without two-phase commit.
5. **Port the `GpuFacePipeline.extract_batch` path** (`backend/app/ml/gpu/face_pipeline.py:336-556`) into Phase2v2 as the isolated batch inference implementation. Keep the same zero-copy contract: decode+detect+align+embed stays on GPU; only compact metadata and 512-D embeddings cross the CPU boundary.
6. **Build the native extension the same way** (`backend/native/mergenvision_gpu/` + `backend/Dockerfile:1-22`): pybind11 + scikit-build-core + CMake with `CMAKE_CUDA_ARCHITECTURES 75` and `CUDA::cudart`. Reuse kernels `l2_normalize`, `similarity_transform`, `warp_align`, `nms`, `retinaface_decode_batch`, and `retinaface_pick_largest` where applicable.
7. **Use the same worker/orchestrator split** (`backend/app/workers/gpu_worker.py` and `backend/app/services/bulk_orchestrator.py`) for any Phase2v2 offline bulk job: API creates a durable `ProcessRecord`, fan-out over HTTP to stateful GPU workers, poll status every 2s, and support resume via `resume_after_identity_key`.
8. **Validate Qdrant payloads exactly like `FaceVectorStore._validate_payload`** (`qdrant.py:57-83`): require `sampleId==point_id`, plus `photoId`, `personId`, `active`, and `modelVersion`. This prevents subtle index drift.
9. **Do not assume folder-name identity keys are reusable.** The Demo derives identity from `normalize_lfw_folder_name` (`bulk_manifest.py:46-55`). Phase2v2 must define its own stable identity key policy before porting the deterministic ID code.

## Self-Verification Checklist

- [x] Every file path in this markdown was confirmed by `codebase-memory-mcp` `search_graph` or `search_code` results.
- [x] Every symbol name (`BulkEnrollmentService`, `GpuFacePipeline.extract_batch`, `FaceVectorStore.upsert_batch`, `derive_sample_id`, etc.) was verified by `get_code_snippet` or `search_code`.
- [x] Every number is backed by source: `bulk_extract_batch_size=256` (`config.py:47`), `bulk_max_persistence_concurrency=32` (`config.py:48`), Qdrant chunk size `256` (`qdrant.py:197`), `embedding_dim=512` (`config.py:42`), detector input `640` (`config.py:39`), embedder input `112` (`config.py:40`), poll interval `2.0s` (`bulk_orchestrator.py:305`), queue `maxsize=2` (`bulk_enrollment.py:604`).
- [x] Prompt-memory parent/child node names and tags will use the exact constants: parent `MergenVisionDemo`, children `MergenVisionDemo: overview`, `bulk-enrollment`, `gpu-runtime`, `persistence`, `identity-model`, `recommendations`, all tagged with `["mergenvision-demo", "external-repo", "outside-repo", "sibling-repo", "cross-repo"]` plus topic tag.
