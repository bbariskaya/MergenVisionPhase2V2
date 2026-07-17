# Current Sprint: Phase 1 Sprint 02 — Native GPU Image Identity Vertical Slice

## Objective

Implement the real NVIDIA GPU image identity vertical slice:

```text
JPEG
-> nvJPEG decode
-> CUDA preprocessing
-> TensorRT RetinaFace R50
-> CUDA RetinaFace decode/NMS/landmarks
-> CUDA five-point face alignment
-> TensorRT GlintR100
-> CUDA L2-normalized 512-D embedding
-> Python application service
-> PostgreSQL / MinIO / Qdrant
-> FastAPI response
```

## In Scope

- `backend/native/image_runtime/` pybind11 native package:
  - `ImageRuntime` singleton created in FastAPI lifespan.
  - `infer_jpeg(encoded_bytes)` returning compact observations.
  - nvJPEG decode, CUDA preprocessing, TensorRT inference, CUDA postprocess, CUDA alignment, WebP crop encoding.
  - Bounded execution-slot pool and backpressure signals.
- Model profile schema/example for `retinaface_r50_glintr100_v1`.
- TensorRT engine build with profiles:
  - RetinaFace: min=1×3×640×640, opt=8×3×640×640, max=64×3×640×640
  - GlintR100: min=1×3×112×112, opt=8×3×112×112, max=64×3×112×112
- Backend integration:
  - `NativeImageInferenceAdapter` with bounded concurrency.
  - FastAPI lifespan-managed `ImageRuntime` instance.
  - FastAPI routers at `/api/v1`: recognize, enroll, detail, delete, history, process; plus health endpoints.
  - Public response contract (camelCase, no client threshold, no sampleId exposure).
  - `IdentityStorageLifecycleService` refactor for process-scoped `resolve_or_create_for_process`.
  - New Qdrant collection `face_samples_retinaface_r50_glintr100_v1`.
- TDD tests for no-face, single-face lifecycle, multi-face, overload, history, process endpoints.

## Out of Scope

- Video upload/job/tracking.
- GStreamer/DeepStream/NVDEC.
- SCRFD or other models.
- React UI.
- Blur/pose/occlusion calibration.
- Accuracy benchmark / 600 FPS claims.
- National ID / Oracle / 10M-person.
- New PostgreSQL tables (reuse Sprint 01 tables).
- Mutation/deletion of old `face_samples_v1` collection.

## Binding Decisions

- Container: `nvcr.io/nvidia/tensorrt:26.03-py3` digest `sha256:ade1b30517b3d66b911a3cd7faf0146484ab8956098abe66b96b944fa36f4861`.
- Models:
  - `backend/artifacts/models/retinaface_r50_dynamic.onnx` SHA-256 `fd8a87a6f2837d425604e0f88efca91e661947dbc3707f54da53ec27a8dc9a8a`
  - `backend/artifacts/models/glintr100.onnx` SHA-256 `4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf`
- Engines (FP16, TensorRT 10.16):
  - `retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1016.engine`
  - `glintr100.bs1.opt8.max64.fp16.trt1016.engine`
- MinIO crop key keeps Sprint 01 contract: `faces/{faceId}/{sampleId}/aligned.webp`, Content-Type `image/webp`.
- Test env: only `backend/.env.gpu-test.example` tracked; `backend/.env.gpu-test` is generated locally and gitignored.

## Status

`COMPLETED`

See review package: `docs/implementation/review_packages/SPRINT-002-CODE-REVIEW-PACKAGE.md`.

