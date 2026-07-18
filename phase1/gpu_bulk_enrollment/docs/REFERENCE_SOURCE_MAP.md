# Reference Source Map

## Reference Repository

- **Path:** `/home/user/Workspace/MergenVisionDemo`
- **Remote:** `https://github.com/bbariskaya/MergenVisionDemo` (inferred from context)
- **HEAD:** `5bf4b4c57542b26058e8d068186faee06c0fc29c`
- **Status:** read-only; no modifications made.
- **License/provenance:** Internal company repository; reuse authorized by binding senior mail for this sprint.

## Symbols Adapted/Copied

| Original Path | Original Symbol | Purpose | Adaptation Notes |
|---------------|-----------------|---------|------------------|
| `backend/app/ml/gpu/face_pipeline.py` | `GpuFacePipeline` / `extract_batch` | End-to-end GPU batch decode→detect→align→embed→L2 | Copied and adapted to Phase 1 `python/mv_phase1_bulk/pipeline.py`; input/output contract aligned to Phase 2 storage. |
| `backend/app/ml/gpu/buffer_arena.py` | `BufferArena` / `Lease` | Reusable GPU buffer pool fenced by CUDA events | Copied into `python/mv_phase1_bulk/buffer_arena.py` with same lease semantics. |
| `backend/app/ml/gpu/trt_device_engine.py` | `TrtDeviceEngine` | TensorRT engine wrapper with device selection and profile validation | Copied into `python/mv_phase1_bulk/trt_device_engine.py`; engine paths read from Phase 1 config. |
| `backend/app/services/bulk_enrollment.py` | `BulkEnrollmentService` / `_produce` / `_consume` / `_persist_batch` | Producer/consumer queues, batched persistence, deterministic IDs | Architecture adapted into `python/mv_phase1_bulk/enrollment.py`; PG/MinIO/Qdrant contracts aligned to Phase 2. |
| `backend/native/mergenvision_gpu/CMakeLists.txt` | `pybind11_add_module(_mergenvision_gpu ...)` | Native build | Adapted into `phase1/gpu_bulk_enrollment/native/CMakeLists.txt`; CUDA arch 75 kept, module name `mv_phase1_bulk_native`. |
| `backend/native/mergenvision_gpu/src/bindings.cpp` | `PYBIND11_MODULE` | Python/C++ bridge for batch extraction | Adapted into `phase1/gpu_bulk_enrollment/native/src/bindings.cpp`; exposes `extract_batch`. |
| `backend/native/mergenvision_gpu/src/*.cu` | Kernels: `nms`, `retinaface_decode`, `retinaface_pick_largest`, `warp_align`, `similarity_transform`, `l2_normalize`, `argsort`, `scale_clip_compact` | GPU hot-path primitives | Copied/adapted into `phase1/gpu_bulk_enrollment/native/src/`. |

## Phase 2 Contracts Consumed (Read-Only)

| Path | Symbol | What We Must Match |
|------|--------|--------------------|
| `backend/app/infrastructure/persistence/sqlalchemy/models/person.py` | `PersonOrm` | `person_id`, `display_name`, `person_metadata`, `is_active`, `version`, `created_at/updated_at/deleted_at` |
| `backend/app/infrastructure/persistence/sqlalchemy/models/face_identity.py` | `FaceIdentityOrm` | `face_identity_id`, `person_id`, `display_name`, `status`, `is_active`, `redirect_to_face_id`, check constraints |
| `backend/app/infrastructure/persistence/sqlalchemy/models/face_sample.py` | `FaceSampleOrm` | `sample_id`, `face_identity_id`, `person_id`, `bucket`, `object_key`, `model_version`, `preprocess_version`, `embedding_model`, `detector_model`, `bbox`, `landmarks`, `quality_score`, `status`, `activated_at`, check constraints |
| `backend/app/infrastructure/storage/minio_adapter.py` | `MinIOObjectStore` | technical UUID object key, SHA-256 metadata, idempotent put, bucket |
| `backend/app/infrastructure/vectors/qdrant_adapter.py` | `QdrantVectorStore` | 512-D, cosine, point ID = `sample_id`, payload fields `sample_id`, `face_id`, `active`, `model_version` |
| `backend/app/infrastructure/config.py` | `Settings` | `model_version`, model/engine path settings |
| `backend/native/image_runtime/src/model_profile.h` | `ModelProfile` | `detector_input_size=640`, `recognizer_input_h/w=112`, `embedding_dim=512`, ArcFace template landmarks |

## Frozen Models

| Model | Path | Purpose |
|-------|------|---------|
| RetinaFace R50 dynamic ONNX | `backend/artifacts/models/retinaface_r50_dynamic.onnx` | Detector engine source |
| GlintR100 ONNX | `backend/artifacts/models/glintr100.onnx` | Recognizer engine source |

Both are read-only mounted from Phase 2 tree. Phase 1 engines are built separately under `.artifacts/phase1_gpu_bulk_enrollment/engines/<runtime-fingerprint>/`.

## Expected Changes from Reference

- Input/output types renamed from `GpuFaceExtraction` to `FaceExtraction` / `ImageExtractionResult`.
- Deterministic UUID generation replaces any `uuid4()` calls for person/face/sample IDs.
- Qdrant payload keys aligned to Phase 2 adapter (`face_id`, `sample_id`, `active`, `model_version`).
- PostgreSQL ORM tables referenced read-only; Phase 1 uses bulk SQLAlchemy `insert().on_conflict_do_update()` against the same tables.
- Multi-GPU worker ownership uses stable HMAC-based shard key instead of `hash()`.
- Dataset is read from filesystem bind mount, not from MinIO upload or base64 API.
