# MergenVision Phase 2 Build Notes

## Session recovery
- Repo: `/home/user/Workspace/MergenVisionPhase2v2`
- Branch: `main`
- Baseline: `c555c9a4e0982e77e63593b7de4bc4560715f612`
- HEAD: same as baseline (dirty worktree)
- Build mode active, no subagents.

## Completed before M0.7 turn (M0.1–M0.6)
- M0.1 canonical `/api/v1` API contract, requestId, camelCase, safe errors
- M0.2 health/readiness with real dependency probes
- M0.3 image orchestration guarded lifecycle + partial failure persistence
- M0.4 delete/detail/history semantics on real PostgreSQL (`get_active_by_id`)
- M0.5 bounded JPEG validation (streaming read, magic, dimensions/pixels)
- M0.6 Qdrant `model_version` filter + collection contract validation
- Static/type check passes on changed files.

## M0.7 — native safety fixes (code complete; GPU verification pending container)
### Already present in dirty tree
- `bindings.cpp` releases GIL and uses RAII slot guard.
- `retinaface_postproc.cpp` replaced abort with exceptions and reports `MODEL_CONTRACT_ERROR`.
- `pipeline.cpp` checks alignment status before L2 normalization and rejects degenerate transforms.
- `pipeline.cpp` already deterministic-chunks recognizer crops by `max_faces_`.
- `CMakeLists.txt` reads `CMAKE_CUDA_ARCHITECTURES` from env with Turing+ default.
- `Dockerfile.gpu` has TensorRT digest pinned.

### Implemented in this session
1. `ExecutionSlot::State` enum added; constructor failures set `unavailable`, full success sets `initialized`; unavailable slots never acquired.
2. `model_profile.cpp` parses `alignment.crop_size` as `[h, w]` list and validates dynamic profile shapes.
3. `NativeImageRecognitionAdapter` and native tests now pass a validated Python dict into `ImageRuntime`.
4. Added `backend/scripts/build_engines.py` (ONNX SHA verify, `trtexec` engine build with exact dynamic profiles, SHA256 manifest update).
5. Added root `Makefile` `phase2-step0-*` acceptance targets.
6. `test_image_runtime_surface.py` + `test_image_runtime_safety.py` updated/extended for crop_size, broken-slot, dict contract.

### Yet to verify
- Real C++ compile + native tests inside pinned TensorRT container.
- Engine build script runtime inside pinned TensorRT container.

## Files currently in dirty tree relevant to M0.7
- `backend/native/image_runtime/src/bindings.cpp`
- `backend/native/image_runtime/src/pipeline.cpp` / `pipeline.h`
- `backend/native/image_runtime/src/model_profile.cpp` / `model_profile.h`
- `backend/native/image_runtime/src/retinaface_postproc.cpp`
- `backend/native/image_runtime/src/retinaface_engine.cpp`
- `backend/native/image_runtime/src/glintr100_engine.cpp`
- `backend/native/image_runtime/CMakeLists.txt`
- `backend/Dockerfile.gpu`
- `backend/config/model_profiles/retinaface_r50_glintr100_v1.example.json`
- `backend/tests/native/test_image_runtime_surface.py`
- `backend/tests/native/test_image_runtime_safety.py`
- `backend/app/infrastructure/runtime/native_image_recognition_adapter.py`
- `backend/app/infrastructure/model_profile.py`
- `Makefile`

## Environment
- Python for tests: `backend/.venv/bin/python` (Python 3.12 expected)
- Base `python` is 3.14.6 and lacks fastapi; always use `.venv`.
- `image_runtime` is not installed/built in the current host `.venv`; native tests will skip or must be run inside the pinned GPU container.

## Next work
- Try building `image_runtime` inside the pinned `mergenvision-backend:gpu` container
  to catch C++/CMake errors; the host has no CUDA/TensorRT so it cannot compile natively.
- If container build is blocked by host driver/runtime limits, record the blocker
  and proceed to M1 video control-plane migrations (`backend/alembic/versions/0003_video_control_plane.py`).
- Do **not** claim M0.8 `PASS` until native tests run green inside the pinned container.
