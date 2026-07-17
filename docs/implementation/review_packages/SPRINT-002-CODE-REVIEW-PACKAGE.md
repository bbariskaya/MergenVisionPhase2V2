# Sprint 02 — Native GPU Image Identity Vertical Slice

## Verdict: PASS (scope limited to the image identity vertical slice)

## What changed

### Core product flow

```text
POST /faces/recognize
  -> FastAPI controller
  -> ImageRecognitionService
  -> NativeImageRecognitionAdapter (image_runtime pybind11)
  -> nvJPEG / CUDA RetinaFace decode+NMS+landmarks / CUDA alignment / TensorRT GlintR100
  -> IdentityStorageLifecycleService (process-scoped)
  -> PostgreSQL / MinIO / Qdrant
  -> JSON response
```

### Files added

- `backend/native/image_runtime/` — native C++/CUDA/pybind11 package.
  - `CMakeLists.txt`
  - `pyproject.toml`
  - `src/pipeline.cpp`, `src/decode.cpp`, `src/encode.cpp`, `src/kernels*.cu`, `src/*.hpp`
  - `src/image_runtime.cpp` (pybind11 bindings)
- `backend/Dockerfile.gpu` — pinned TensorRT 26.03-py3 runtime image.
- `backend/entrypoint.gpu.sh` — migrations + Uvicorn startup.
- `docker-compose.gpu.yml` — full GPU stack.
- `backend/config/model_profiles/retinaface_r50_glintr100_v1.*`
- `backend/app/api/` — FastAPI routers, controllers, schemas, dependencies.
  - `main.py`, `routes/faces.py`, `routes/processes.py`, `routes/dependencies.py`
  - `controllers/face_controller.py`
  - `schemas.py`
- `backend/app/application/ports/image_recognition.py` — `ImageRecognitionEngine` port + `NativeRecognitionResult` DTO.
- `backend/app/application/services/image_recognition_service.py` — orchestration, error mapping, response assembly.
- `backend/app/infrastructure/runtime/native_image_recognition_adapter.py` — `image_runtime` bridge.
- `backend/tests/unit/api/test_face_api.py` — API contract tests.
- `backend/tests/native/test_image_runtime_surface.py` — native module surface tests (skip outside container).

### Files modified

- `backend/app/application/services/identity_storage_lifecycle_service.py` — added process-scoped `start_process`, `resolve_or_create_for_process`, `complete_process`, `fail_process`.
- `backend/app/application/ports/repositories.py` — added `list_by_face_id` to recognition result port.
- `backend/app/infrastructure/persistence/sqlalchemy/repositories/recognition_result.py` — SQLAlchemy implementation.
- `backend/app/infrastructure/config.py` — native runtime settings.
- `backend/pyproject.toml` — ruff per-file ignores; added FastAPI deps.
- `backend/requirements.lock` — refreshed.
- `backend/.env.example` — native / model settings.
- `docs/implementation/CURRENT_SPRINT.md` — status and decisions updated.

## Models / engines / runtime

| Item | Value |
|------|-------|
| Container | `nvcr.io/nvidia/tensorrt:26.03-py3` digest `sha256:ade1a...4861` |
| TensorRT | 10.16.0.72 |
| CUDA | 13.2 (driver 580.105.08 host) |
| Detector | `retinaface_r50_dynamic.onnx` SHA-256 `fd8a87a6...8dc9a8a` |
| Recognizer | `glintr100.onnx` SHA-256 `4ab1d643...534cdf` |
| Engine (det) | `retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1016.engine` |
| Engine (rec) | `glintr100.bs1.opt8.max64.fp16.trt1016.engine` |
| Profile | `retinaface_r50_glintr100_v1` |
| Qdrant collection | `face_samples_retinaface_r50_glintr100_v1` |

Native build note: the first wheel builds used `CUDA_SEPARABLE_COMPILATION ON`; inside the wheel target this skipped the device-link step and produced the runtime error `undefined symbol: fatbinData`. Setting `CUDA_SEPARABLE_COMPILATION OFF` resolved it because the independent `.cu` kernel files now each embed their own device code directly.

## Validation commands & raw results

### 1. Stack start

```bash
docker build -t mergenvision-backend:gpu -f backend/Dockerfile.gpu .
docker compose -f docker-compose.gpu.yml up -d
```

Health:

```bash
curl -s http://localhost:8090/health
# {"status":"ok"}
```

### 2. No-face JPEG

```bash
curl -s -X POST http://localhost:8090/faces/recognize -F image=@/tmp/no_face.jpg
```

```json
{
  "process_id": "...",
  "status": "completed",
  "face_count": 0,
  "faces": []
}
```

### 3. Single-face lifecycle (Rachel)

```bash
# 1st -> new_anonymous
# 2nd -> anonymous
# enroll -> known
# 3rd -> known
```

Raw response for the final call:

```json
{
  "process_id": "06a5a027-48d7-77af-8000-7dec6420d7da",
  "status": "completed",
  "face_count": 1,
  "faces": [
    {
      "face_id": "06a5a026-f257-790f-8000-f857aa78feee",
      "status": "known",
      "name": "Rachel",
      "metadata": {"show": "Friends"},
      "bounding_box": {"x": 352, "y": 78, "width": 139, "height": 215},
      "confidence": 0.57453734
    }
  ]
}
```

`face_id` stayed identical through every step.

### 4. Multi-face JPEG (Rachel + Joey composite)

```bash
ffmpeg -y -i test_gallery/Rachel/1.jpg -i test_gallery/Joey/1.jpg \
  -filter_complex "[0:v]scale=400:400:force_original_aspect_ratio=decrease,pad=400:400:(ow-iw)/2:(oh-ih)/2,setsar=1[a];[1:v]scale=400:400:force_original_aspect_ratio=decrease,pad=400:400:(ow-iw)/2:(oh-ih)/2,setsar=1[b];[a][b]hstack=inputs=2" \
  -frames:v 1 /home/user/tmp_multi_face.jpg

curl -s -X POST http://localhost:8090/faces/recognize -F image=@/home/user/tmp_multi_face.jpg
```

Result:

```json
{
  "process_id": "06a5a03e-cc78-7c42-8000-15e2279ded48",
  "status": "completed",
  "face_count": 2,
  "faces": [
    {
      "face_id": "06a5a03e-cd82-75c1-8000-da16ebff3fbe",
      "status": "new_anonymous",
      "bounding_box": {"x": 536, "y": 70, "width": 136, "height": 166},
      "confidence": 0.09028235
    },
    {
      "face_id": "06a5a026-f257-790f-8000-f857aa78feee",
      "status": "known",
      "name": "Rachel",
      "metadata": {"show": "Friends"},
      "bounding_box": {"x": 88, "y": 61, "width": 91, "height": 113},
      "confidence": 0.9459566
    }
  ]
}
```

One `process_id`, two independent results (known + new_anonymous).

### 5. History and process endpoints

```bash
curl -s http://localhost:8090/faces/06a5a026-f257-790f-8000-f857aa78feee/history | python -m json.tool
curl -s http://localhost:8090/processes/<process_id> | python -m json.tool
```

Both returned expected JSON with immutable process records and timestamps.

### 6. Restart persistence

```bash
docker compose -f docker-compose.gpu.yml restart backend
sleep 5
curl -s -X POST http://localhost:8090/faces/recognize -F image=@test_gallery/Rachel/1.jpg
```

Returned the same `face_id` as before restart with status `known`.

### 7. Storage evidence (inside backend container)

```bash
docker exec -i mergenvision-backend-gpu python - <<'PY'
# ... queries PG / MinIO / Qdrant
PY
```

Output:

```text
PG counts:
  face_identity: 4
  face_sample: 4
  recognition_result: 9
  process_record: 13
PG sample row: {
  'sample_id': UUID('06a5a026-f257-7b57-8000-37020c823143'),
  'face_id': UUID('06a5a026-f257-790f-8000-f857aa78feee'),
  'bucket': 'mergenvision-face-samples',
  'object_key': 'faces/06a5a026-f257-790f-8000-f857aa78feee/06a5a026-f257-7b57-8000-37020c823143/aligned.webp',
  'state': 'active',
  'is_active': True
}
MinIO object count: 4
  example key: faces/06a5a026-f257-790f-8000-f857aa78feee/06a5a026-f257-7b57-8000-37020c823143/aligned.webp size=5714
Qdrant collection face_samples_retinaface_r50_glintr100_v1: points_count=4
  example point id=06a5a026-f257-7b57-8000-37020c823143
  payload={'sample_id': '06a5a026-f257-7b57-8000-37020c823143', 'face_id': '06a5a026-f257-790f-8000-f857aa78feee', 'active': True, 'model_version': 'retinaface_r50_glintr100_v1'}
```

Cross-store consistency: Qdrant point id == `face_sample.sample_id`; MinIO key is `faces/{faceId}/{sampleId}/aligned.webp`; no name/metadata leaks into object keys or Qdrant payload.

### 8. Unit / native tests

```bash
MV_TEST_MODE=1 pytest tests/unit tests/native -q
# 55 passed, 1 skipped in 3.34s
```

The single skip is the native import test when the module is not built locally; it passes inside the GPU container.

### 9. Lint / type check

```bash
cd backend
ruff check app tests
# All checks passed!
mypy app
# Success: no issues found in 52 source files
```

### 10. Latency sample (informational only)

Five sequential `POST /faces/recognize` calls on `test_gallery/Rachel/1.jpg`:

```text
0.189548s
0.180521s
0.178021s
0.176101s
0.179326s
```

This is end-to-end HTTP+GPU+storage latency on a single RTX-class GPU, not a formal benchmark.

## Known limitations / not claimed

- **Image-only scope.** Video upload, async jobs, tracking, temporal aggregation, and GStreamer/DeepStream are intentionally not part of this sprint.
- **No production accuracy benchmark.** The acceptance set proves identity persistence and same-image matching; general precision/recall across poses, blur, and occlusion will be calibrated separately.
- **No 600 FPS claim.** The dynamic batch profiles exist for the future video path; the current public API processes one image per request.
- **Delete semantics need product review.** The endpoint returns `204`, but the detail endpoint still returns the identity. Samples are deactivated in Qdrant/PostgreSQL; whether detail should return `404` after soft-delete is a product decision left for the next sprint.
- **Multi-face fixture is a composite.** Real group-photo multi-face acceptance requires a fixture with naturally co-occurring faces.

## Next recommended sprint

**Phase 1 Sprint 03 — Video Foundation & Native Observation Stream:**

1. Define video asset / job / person database schema and state machine.
2. Implement direct multipart and presigned multipart video upload.
3. Implement async job queue with cancel/retry/lease semantics.
4. Pipe a minimal GStreamer → NVDEC → `DeviceImageView` batch into the same `FacePipeline` used in this sprint.
