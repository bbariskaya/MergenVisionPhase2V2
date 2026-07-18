# NVDIAgstreamer Cross-Repo Intelligence Pack

**Source repo (READ-ONLY):** `/home/user/NVDIAgstreamer`  
**Purpose:** Reference implementation of a GPU-only GStreamer + NVIDIA DeepStream face-detection / recognition / tracking pipeline, plus Python TensorRT engine and offline identity-resolution tooling.

---

## Executive Summary

NVDIAgstreamer is the Phase 1 benchmark/clean-reference predecessor for the MergenVision family. It demonstrates an end-to-end video hotpath that keeps every heavy operation on the GPU until the final `[N, 512]` face embeddings and metadata are transferred to CPU for Qdrant search and persistence. The repo mixes C++/CUDA DeepStream apps, Python TensorRT engine helpers, offline trackers/resolvers, and extensive governance docs. For MergenVisionPhase2v2 it is a rich source of proven patterns, but it is explicitly **benchmark-only** and contains hardcoded paths/InsightFace non-commercial models that must not be copied blindly into production.

---

## Repo Structure

```
/home/user/NVDIAgstreamer/
├── AGENTS.md                                    # mandatory agent governance
├── GSTREAMER_DEEPSTREAM_QDRANT_ARCHITECTURE.md  # pipeline + GPU/CPU rules
├── TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md     # dynamic-batch engine guidance
├── MODEL_VALIDATION_PLAN.md                     # 20-item model gate
├── MODEL_CANDIDATES_MATRIX.md                   # detector/recognizer options
├── MODEL_RESEARCH_REPORT.md
├── projectultrareport.md
├── notes/anchored_summary.md
├── opensourcereferences/references.md           # upstream links + policy
├── requirements/phase1requirements.md
├── requirements/phase2videorequirements.md
└── phase1/                                      # all implementation code
    ├── configs/
    │   ├── pgie_scrfd_10g.yml
    │   ├── pgie_scrfd_500m.yml
    │   └── tracker_NvDCF_perf.yml
    ├── cpp/
    │   ├── Makefile
    │   ├── deepstream_track_app.cpp
    │   ├── deepstream_render_app.cpp
    │   ├── nvdsinfer_custom_scrfd_parser.cpp
    │   ├── arcface_infer.cpp / arcface_infer.hpp
    │   ├── face_align.cu / face_align.hpp
    │   └── embedding_meta.hpp
    ├── scripts/
    │   ├── build_engines.py
    │   ├── tensorrt_runtime.py
    │   ├── benchmark_lfw.py
    │   ├── annotate_friends_video_abc.py
    │   ├── offline_track_best_shot_annotator.py
    │   ├── resolve_deepstream_tracks.py
    │   ├── render_deepstream_tracks.py
    │   ├── prototype_gallery.py
    │   └── enrolment_quality.py
    ├── tests/
    │   ├── test_batch_invariance.py
    │   ├── test_build_engines.py
    │   ├── test_deepstream_offline_tracker.py
    │   ├── test_enrolment_quality.py
    │   ├── test_offline_track_best_shot_annotator.py
    │   ├── test_prototype_gallery.py
    │   └── test_resolve_accuracy_improvements.py
    ├── pyproject.toml
    └── requirements.txt
```

**Languages indexed:** Python (10 files), C++ (4 files), YAML (3 files), TOML (1 file).  
**Entry points:** `phase1/cpp/deepstream_track_app.cpp:main`, `phase1/cpp/deepstream_render_app.cpp:main`.

---

## Key Files and Symbols

### DeepStream C++ hotpath

| File | Key symbol | Responsibility |
|---|---|---|
| `/home/user/NVDIAgstreamer/phase1/cpp/deepstream_track_app.cpp` | `main()` (line 662) | Builds `filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux -> nvinfer -> nvtracker -> nvvideoconvert -> fakesink`, attaches tracker-src and fakesink-sink probes, writes per-frame JSONL. |
| same | `tracker_src_pad_buffer_probe()` (line 379) | Decodes SCRFD raw tensors (`decodeScrfdTensors`), matches 5-point landmarks to tracked objects by IoU, caches them in a keyed map. |
| same | `fakesink_sink_pad_buffer_probe()` (line 436) | Warps faces on GPU, runs ArcFace, L2-normalizes, appends `[N,512]` embeddings to JSONL output. |
| same | `decodeScrfdTensors()` (line 101) | Probe-side re-implementation of SCRFD decode: 9 output tensors, strides 8/16/32, score/bbox/kps decode, NMS. |
| `/home/user/NVDIAgstreamer/phase1/cpp/deepstream_render_app.cpp` | `main()` (line 258) | Renders named bounding boxes from an observations TSV via `nvdsosd` and encodes to MP4. |
| `/home/user/NVDIAgstreamer/phase1/cpp/nvdsinfer_custom_scrfd_parser.cpp` | `NvDsInferParseCustomSCRFD()` (line 235) | DeepStream custom parser shared library; outputs `NvDsInferObjectDetectionInfo` list. |
| `/home/user/NVDIAgstreamer/phase1/cpp/arcface_infer.cpp` | `phase1::ArcFaceInfer::infer()` (line 102) | Loads TensorRT plan via `NvDsInferContext`, accepts preprocessed GPU NCHW input, returns raw 512-D embeddings on GPU. |
| `/home/user/NVDIAgstreamer/phase1/cpp/arcface_infer.hpp` | `phase1::ArcFaceInfer` class (line 19) | Wrapper with `embedding_dims()`, `initialized()`, `max_batch_size_`. |
| `/home/user/NVDIAgstreamer/phase1/cpp/face_align.cu` | `warp_affine_rgba_to_arcface_nchw()` (line 260) | CUDA kernel that bilinearly samples RGBA NVMM frames to ArcFace 112x112 NCHW and normalizes. |
| same | `l2_normalize_rows()` (line 279) | Row-wise L2 normalization kernel. |
| `/home/user/NVDIAgstreamer/phase1/cpp/embedding_meta.hpp` | `attach_face_embedding()` / `find_face_embedding()` | Helper for attaching L2-normalized embeddings as NvDs user meta. |

### Python TensorRT / offline tooling

| File | Key symbol | Responsibility |
|---|---|---|
| `/home/user/NVDIAgstreamer/phase1/scripts/build_engines.py` | `build_engine()` (line 26) | Builds explicit-batch dynamic-shape TensorRT engines from ONNX with FP16 and SHA256 manifest. |
| same | `default_profile()` (line 90) | Returns min/opt/max shapes for detector/recognizer. |
| `/home/user/NVDIAgstreamer/phase1/scripts/tensorrt_runtime.py` | `TRTEngine` class (line 11) | Runs TensorRT plans; caches one `IExecutionContext` per distinct input shape to avoid TensorRT 10 shape-change corruption. |
| `/home/user/NVDIAgstreamer/phase1/scripts/benchmark_lfw.py` | `SCRFDEngine`, `ArcFaceEngine` (lines 101, 270) | Python TensorRT detector/recognizer used for LFW 1:1 benchmark; includes InsightFace-style decode and preprocessing. |
| `/home/user/NVDIAgstreamer/phase1/scripts/resolve_deepstream_tracks.py` | `resolve_tracks()` (line 487) | Reads DeepStream JSONL, picks top-K observations, embeds best faces, matches Qdrant prototype gallery, merges tracks. |
| `/home/user/NVDIAgstreamer/phase1/scripts/prototype_gallery.py` | `build_prototype_gallery()` / `match_prototypes()` (lines 53, 129) | k-means prototype selection and top-K mean cosine matching for identity resolution. |
| `/home/user/NVDIAgstreamer/phase1/scripts/offline_track_best_shot_annotator.py` | `ByteTrackFace` class (line 182) | Pure-Python ByteTrack-style two-stage face tracker with best-shot identity resolution. |
| `/home/user/NVDIAgstreamer/phase1/scripts/enrolment_quality.py` | `EnrolmentFilter` class (line 12) | Reusable quality gates: size, Laplacian blur, frontalness. |
| `/home/user/NVDIAgstreamer/phase1/scripts/annotate_friends_video_abc.py` | `IoUTracker`, `KalmanTracker` (lines 207, 254) | Simpler Python trackers with identity-history smoothing. |

### Configs and docs

- `/home/user/NVDIAgstreamer/phase1/configs/pgie_scrfd_10g.yml` — `nvinfer` config for SCRFD_10G_KPS, batch 16, custom parser, `output-tensor-meta=1`.
- `/home/user/NVDIAgstreamer/phase1/configs/pgie_scrfd_500m.yml` — same for SCRFD_500M_KPS, batch 1.
- `/home/user/NVDIAgstreamer/phase1/configs/tracker_NvDCF_perf.yml` — NvDCF tuning for face tracking.
- `/home/user/NVDIAgstreamer/GSTREAMER_DEEPSTREAM_QDRANT_ARCHITECTURE.md` — pipeline diagram, plugin list, forbidden patterns.
- `/home/user/NVDIAgstreamer/TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md` — why static batch=1 cannot be patched, dynamic profile recipes.
- `/home/user/NVDIAgstreamer/MODEL_VALIDATION_PLAN.md` — 20-step gate with pass/fail thresholds.
- `/home/user/NVDIAgstreamer/MODEL_CANDIDATES_MATRIX.md` — detector/recognizer trade-offs and license notes.

---

## Pipeline / Flow

### DeepStream GPU hotpath (track app)

```text
filesrc -> qtdemux -> h264parse -> nvv4l2decoder (NVDEC)
  -> nvstreammux (batch-size=1, 1920x1080)
  -> nvinfer (SCRFD TensorRT, custom parser)
  -> nvtracker (NvDCF)
  -> nvvideoconvert (RGBA, 1920x1080)
  -> fakesink
```

Two probes bridge detection metadata with recognition:

1. **Tracker source probe** (`tracker_src_pad_buffer_probe`)
   - Reads `NVDSINFER_TENSOR_OUTPUT_META` from frame user meta.
   - Runs `decodeScrfdTensors()` to recover score/bbox/5-point landmarks.
   - Matches raw detections to tracked `NvDsObjectMeta` rects by IoU.
   - Stores landmarks in a global keyed map: `source_id:frame_num:object_id`.

2. **Fakesink sink probe** (`fakesink_sink_pad_buffer_probe`)
   - Maps the RGBA `NvBufSurface` for the batch slot.
   - Looks up cached landmarks, computes a quality score `conf * log1p(area) * frontal_score`.
   - For each face, estimates a similarity transform to the ArcFace 112 template, inverts it, and launches `warp_affine_rgba_to_arcface_nchw`.
   - Runs `ArcFaceInfer::infer()` on the preprocessed batch.
   - L2-normalizes embeddings on GPU with `l2_normalize_rows`.
   - Copies embeddings to host and writes a JSONL line per frame.

### Offline identity resolution

```text
video.mp4 + tracks.jsonl
  -> load_tracks() groups observations by object_id
  -> embed_best_faces() selects top-K quality observations
       (uses JSONL embeddings if present, else SCRFD + ArcFace)
  -> mean-embed track -> match against Qdrant prototype gallery
  -> merge_same_identity_tracks() joins short-gap segments
  -> propagate_unknown_tracks() links unknowns to nearby known tracks
  -> render_deepstream_tracks.py draws final labels
```

### Offline renderer

```text
filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux
  -> nvvideoconvert (RGBA) -> nvdsosd -> nvvideoconvert (NV12)
  -> nvv4l2h264enc -> h264parse -> qtmux -> filesink
```

`osd_sink_pad_buffer_probe` injects `NvDsObjectMeta` rects/text from a TSV observations file before `nvdsosd` draws them.

---

## Reusable Components for Phase2v2

### 1. SCRFD custom parser pattern
- File: `/home/user/NVDIAgstreamer/phase1/cpp/nvdsinfer_custom_scrfd_parser.cpp`
- Use the `NvDsInferParseCustomFunc` signature, load via `custom-lib-path` + `parse-bbox-func-name` in `nvinfer` config.
- Decode logic: group 9 outputs by last-dim K (1/4/10), sort by anchor count descending for strides 8/16/32, clip boxes, validate geometry, NMS.

### 2. GPU face crop + alignment
- Files: `/home/user/NVDIAgstreamer/phase1/cpp/face_align.cu`, `/home/user/NVDIAgstreamer/phase1/cpp/face_align.hpp`
- `estimate_similarity_transform_2x3()` solves 4x4 normal equations for a similarity transform from 5 source landmarks to the ArcFace template.
- `warp_affine_rgba_to_arcface_nchw()` can be reused for RGBA NVMM -> NCHW ArcFace input directly on GPU.
- `l2_normalize_rows()` is a generic CUDA row-normalization kernel.

### 3. ArcFace TensorRT wrapper
- Files: `/home/user/NVDIAgstreamer/phase1/cpp/arcface_infer.hpp`, `/home/user/NVDIAgstreamer/phase1/cpp/arcface_infer.cpp`
- `phase1::ArcFaceInfer` shows how to drive `NvDsInferContext` with preprocessed GPU input, `queueInputBatchPreprocessed` / `dequeueOutputBatch`, and copy output device-to-device.

### 4. Dynamic-batch TensorRT engine build + runtime
- File: `/home/user/NVDIAgstreamer/phase1/scripts/build_engines.py`
- Pattern: explicit batch network, one optimization profile with min/opt/max, FP16, serialized plan + SHA256 manifest.
- File: `/home/user/NVDIAgstreamer/phase1/scripts/tensorrt_runtime.py`
- Pattern: cache one `IExecutionContext` per distinct input shape; this works around TensorRT 10 shape-change issues.

### 5. Offline tracker patterns
- File: `/home/user/NVDIAgstreamer/phase1/scripts/offline_track_best_shot_annotator.py`
- `ByteTrackFace` demonstrates two-stage high/low-score IoU matching with Hungarian assignment and simple motion prediction.
- File: `/home/user/NVDIAgstreamer/phase1/scripts/annotate_friends_video_abc.py`
- `IoUTracker` / `KalmanTracker` plus `Track.smooth_identity()` are useful for temporal identity smoothing.

### 6. Identity resolution / gallery matching
- File: `/home/user/NVDIAgstreamer/phase1/scripts/prototype_gallery.py`
- `build_prototype_gallery()` clusters per-identity embeddings with k-means and returns real samples closest to centroids.
- `match_prototypes()` / `prototype_means()` compute top-K mean cosine similarity and a margin gap — suitable for a Qdrant-backed or in-memory gallery.
- File: `/home/user/NVDIAgstreamer/phase1/scripts/resolve_deepstream_tracks.py`
- Track-level top-K observation selection, mean embedding voting, track merging, and offline unknown propagation are directly relevant to bulk enrollment.

### 7. Enrolment quality gates
- File: `/home/user/NVDIAgstreamer/phase1/scripts/enrolment_quality.py`
- `EnrolmentFilter` measures size, Laplacian blur, and a frontalness heuristic from 5-point landmarks; reusable for selecting gallery samples.

### 8. Offline render pipeline
- File: `/home/user/NVDIAgstreamer/phase1/cpp/deepstream_render_app.cpp`
- Shows how to overlay named bounding boxes with `nvdsosd` and hardware-encode to MP4.

---

## Build / Runtime Notes

- **C++ build:** `/home/user/NVDIAgstreamer/phase1/cpp/Makefile` targets DeepStream 9.0 (`/opt/nvidia/deepstream/deepstream-9.0`) and CUDA 13.0 (`/usr/local/cuda-13.0`). It builds:
  - `libnvdsinfer_custom_parser_scrfd.so` (custom parser)
  - `deepstream_track_app`
  - `deepstream_render_app`
  - Links: `nvdsgst_meta`, `nvds_meta`, `nvbufsurface`, `cudart`, `nvds_infer`.
- **Python deps (declared):** `onnx`, `onnxruntime`, `requests`, `pytest`.
- **Python deps (used by scripts):** `tensorrt`, `cupy`, `numpy`, `opencv-python`, `scipy`, `qdrant-client`, `insightface`.
- **Artifacts expected at runtime:** `phase1/artifacts/models/MODEL_MANIFEST.json`, `phase1/artifacts/engines/*.plan`, and optionally `phase1/artifacts/friends_gallery_per_image.npz` / `friends_prototypes.npz`.
- **FP16:** recommended first engine precision; INT8 only after the validation gate passes.
- **Dynamic batch engine recipes (from TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md):**
  - Detector SCRFD_34G_KPS 640: min 1, opt 8, max 16.
  - Detector SCRFD_10G_KPS 640: min 1, opt 16, max 32.
  - Recognizer ArcFace R100: min 1, opt 32, max 64.

---

## Limitations and Warnings

1. **Phase 1 benchmark only.** AGENTS.md forbids treating `phase1/` code as production backend; it is disposable after Phase 2 selection.
2. **Non-commercial models.** Primary stack uses InsightFace SCRFD/ArcFace models that require non-commercial/research licensing. Commercial use would need RetinaFace-R50 (MIT) + AuraFace-v1 (Apache 2.0) per `MODEL_CANDIDATES_MATRIX.md`.
3. **Hardcoded paths.** Config files and scripts point to `/home/user/Workspace/MergenVisionCleanVersion/phase1/artifacts/...` and similar; these must not be copied into Phase2v2 as-is.
4. **Probe landmark cache is global and mutex-protected.** The `g_landmarks` map in `deepstream_track_app.cpp` is a correctness workaround for user-meta double-free; it is not scalable to many sources.
5. **Per-frame CUDA allocations.** The fakesink probe allocates/free device buffers for every frame; for higher throughput this should be pooled.
6. **ArcFace context is synchronous.** The `infer()` call synchronizes the default CUDA stream; async pipelining is left as future work.
7. **No CMake/Docker in repo.** Build is Makefile-only and assumes a host DeepStream/CUDA installation.
8. **Model validation gate is mandatory.** `MODEL_VALIDATION_PLAN.md` lists 20 checks (license, ONNX shape, ORT vs TRT comparison, batch invariance, LFW, Qdrant, DeepStream compatibility, throughput/memory). No production code should rely on the model stack until the gate passes.
9. **Forbidden patterns (from architecture docs):** CPU JPEG decode per frame, CPU resize/normalize per crop, full-frame CPU transfer before detection, FAISS GPU as source-of-truth gallery, raw embeddings or PII in Qdrant payload, per-face Qdrant query frame-by-frame, static batch=1 detector as final design.

---

*Generated by cross-repo research agent on 2026-07-18. Source repo was analyzed read-only; no files were modified.*
