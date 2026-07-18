--- NVDIAgstreamer-reusable-patterns [Memory] tags: cross-repo, external-repo, nvdiagstreamer, reusable-patterns ---
EXTERNAL REPO: NVDIAgstreamer — Reusable components for MergenVisionPhase2v2. Reusable patterns: (1) NvDsInfer custom SCRFD parser API and 9-tensor decode logic. (2) GPU RGBA -> ArcFace NCHW warp kernel and L2 normalization in CUDA. (3) ArcFaceInfer wrapper around NvDsInferContext with preprocessed GPU input. (4) TensorRT dynamic-batch engine builder + per-shape context cache pattern. (5) ByteTrack-style two-stage face tracker and identity-history smoothing. (6) Top-K quality observation selection, mean-embedding track resolution, prototype-gallery matching with k-means and top-K averaging. (7) EnrolmentFilter quality gates (size, blur/Laplacian variance, frontalness). (8) Offline render pipeline with nvdsosd + nvv4l2h264enc. Adapt these to Phase2v2's GPU video lab / bulk enrollment, but avoid copying hardcoded paths, FAISS GPU, or production backend logic.

--- NVDIAgstreamer-build-runtime [Memory] tags: cross-repo, external-repo, nvdiagstreamer, build-runtime ---
EXTERNAL REPO: NVDIAgstreamer — Build / runtime notes & limitations. Build: phase1/cpp/Makefile compiles libnvdsinfer_custom_parser_scrfd.so, deepstream_track_app, deepstream_render_app against DeepStream 9.0 and CUDA 13.0 includes; requires nvdsinfer, nvdsgst_meta, nvbufsurface, cudart. Python deps in phase1/requirements.txt: onnx, onnxruntime, requests, pytest; scripts also use tensorrt, cupy, cv2, scipy, qdrant-client, insightface. Runtime: expects TensorRT .plan engines and MODEL_MANIFEST.json under phase1/artifacts. Limitations: Phase 1 benchmark-only; custom parser is C++ only; ArcFace inference in fakesink probe allocates per-frame device buffers; probe landmark cache uses global mutex map (not scalable); Qdrant payload limited to identity metadata; InsightFace models are non-commercial; model validation gate must pass before production use.

--- NVDIAgstreamer-tracking [Memory] tags: cross-repo, external-repo, nvdiagstreamer, tracking ---
EXTERNAL REPO: NVDIAgstreamer — Tracking & identity resolution. nvtracker uses NvDCF with config tracker_NvDCF_perf.yml; Python fallback trackers are in offline_track_best_shot_annotator.py (ByteTrackFace two-stage high/low score matching with simple motion prediction) and annotate_friends_video_abc.py (IoUTracker and KalmanTracker with identity-history smoothing). Identity resolution: resolve_deepstream_tracks.py loads DeepStream JSONL, selects top-K observations by quality (confidence * area * frontal), embeds best faces (or reuses JSONL embeddings), matches against a Qdrant prototype gallery using prototype_gallery.py (k-means prototypes, top-K mean cosine, threshold/margin gate), merges same-identity tracks with short gaps, and propagates unknown tracks spatio-temporally. render_deepstream_tracks.py draws final labels and interpolates bboxes across gaps.

--- NVDIAgstreamer-batch [Memory] tags: cross-repo, external-repo, nvdiagstreamer, batch ---
EXTERNAL REPO: NVDIAgstreamer — Batch processing & TensorRT engine strategy. build_engines.py uses TensorRT builder with explicit batch, one optimization profile per model, FP16 by default, SHA256 manifest. default_profile() gives detector min1/opt16/max32 (10g) or min1/opt8/max16 (500m); recognizer min1/opt32/max64. tensorrt_runtime.py caches one IExecutionContext per distinct input shape to avoid TensorRT 10 Cask shape-change corruption. TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md warns that static batch=1 cannot be safely patched to batch=N; safe dynamic batch requires ONNX exported with dynamic_axes. Batch invariance tests in phase1/tests/test_batch_invariance.py verify cosine similarity > 0.9999 across positions and shapes.

--- NVDIAgstreamer-recognition [Memory] tags: cross-repo, external-repo, nvdiagstreamer, recognition ---
EXTERNAL REPO: NVDIAgstreamer — Face recognition / ArcFace & alignment. ArcFace wrapper: phase1/cpp/arcface_infer.hpp/cpp (class phase1::ArcFaceInfer) loads a TensorRT plan via NvDsInferContext, accepts preprocessed NCHW float GPU input, returns 512-D embeddings on GPU. GPU crop/align: phase1/cpp/face_align.cu/hpp computes a 5-point similarity transform to the ArcFace 112x112 template, inverts it, and warp_affine_rgba_to_arcface_nchw bilinearly samples RGBA NVMM frames into NCHW ArcFace input, applying mean=127.5 / std=127.5 normalization; l2_normalize_rows kernel normalizes embeddings. Python equivalent uses insightface.utils.face_align.norm_crop and cv2.dnn.blobFromImage(s). Recognizer candidates: ArcFace R100@Glint360K (accuracy) and ArcFace w600k_r50 (throughput).

--- NVDIAgstreamer-detection [Memory] tags: cross-repo, external-repo, nvdiagstreamer, detection ---
EXTERNAL REPO: NVDIAgstreamer — Face detection / SCRFD parser. Custom DeepStream parser: phase1/cpp/nvdsinfer_custom_scrfd_parser.cpp exports NvDsInferParseCustomSCRFD; expects 9 output tensors (scores/bboxes/kps for strides 8/16/32); groups layers by last-dimension K, sorts by anchor count, decodes anchors, applies geometry validation (isValidFaceGeometry), NMS (0.4) and top-k (300). Probe-level decode in deepstream_track_app.cpp (decodeScrfdTensors) repeats the logic to recover landmarks, then matches them to tracked NvOSD rects by IoU and stores in a keyed global map. Configs: phase1/configs/pgie_scrfd_10g.yml (batch 16), pgie_scrfd_500m.yml (batch 1), tracker_NvDCF_perf.yml. Model candidates: SCRFD_34G_KPS (accuracy) and SCRFD_10G_KPS (throughput).

--- NVDIAgstreamer-pipeline [Memory] tags: cross-repo, external-repo, nvdiagstreamer, pipeline ---
EXTERNAL REPO: NVDIAgstreamer — DeepStream pipeline & flow. Pipeline: filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux -> nvinfer (SCRFD TensorRT) -> nvtracker (NvDCF) -> nvvideoconvert (RGBA) -> fakesink. Two C++ probes implement the hotpath: tracker_src_pad_buffer_probe decodes SCRFD raw tensors and caches 5-point landmarks per object; fakesink_sink_pad_buffer_probe crops/aligns each face on GPU, runs ArcFace via NvDsInferContext, L2-normalizes embeddings, and writes JSONL per frame. Offline renderer deepstream_render_app.cpp overlays named bboxes using nvdsosd and hardware-encodes to MP4. All heavy compute stays on GPU (memory:NVMM) until final [N,512] embeddings + metadata move to CPU.

--- NVDIAgstreamer-overview [Memory] tags: cross-repo, overview, external-repo, nvdiagstreamer ---
EXTERNAL REPO: NVDIAgstreamer — Overview & repo structure. Repo at /home/user/NVDIAgstreamer; Phase 1 benchmark-only code under phase1/. Languages: Python (tests + scripts) and C++/CUDA (DeepStream apps). Entry points: phase1/cpp/deepstream_track_app.cpp, phase1/cpp/deepstream_render_app.cpp. Key scripts in phase1/scripts/: build_engines.py, tensorrt_runtime.py, benchmark_lfw.py, resolve_deepstream_tracks.py, render_deepstream_tracks.py, offline_track_best_shot_annotator.py, enrolment_quality.py, prototype_gallery.py. Governance docs: AGENTS.md, GSTREAMER_DEEPSTREAM_QDRANT_ARCHITECTURE.md, TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md, MODEL_VALIDATION_PLAN.md, MODEL_CANDIDATES_MATRIX.md.

--- NVDIAgstreamer [Memory] tags: cross-repo, parent, external-repo, nvdiagstreamer, gstreamer, nvidia ---
EXTERNAL REPO: /home/user/NVDIAgstreamer. Read-only reference for NVIDIA/GStreamer/DeepStream video pipeline patterns.

--- phase1-engine-build-complete [Decision] tags: - ---
Phase 1 TensorRT engines built inline using tensorrt==10.3.0 (TensorRT 11 lacks BuilderFlag.FP16). Engines moved to phase1/gpu_bulk_enrollment/artifacts/engines/default/ as requested. Detector profile max batch 256 @ 640x640; recognizer max batch 256 @ 112x112. Manifest persisted in config/model_profile.json with engine SHA-256, TensorRT version, precision=fp16, and UTC build timestamp. phase2-untouched-gate passes.

--- phase1-reference-source-map [Memory] tags: phase1, MergenVision, gpu-bulk-enrollment, checkpoint, reference-source-map ---
Phase 1 reference source map and target contract captured.
- Reference repo: /home/user/Workspace/MergenVisionDemo @ 5bf4b4c57542b26058e8d068186faee06c0fc29c
- Adapted symbols: GpuFacePipeline.extract_batch, BufferArena, TrtDeviceEngine, BulkEnrollmentService produce/consume/persist, native CUDA kernels and pybind11 bindings.
- Phase 2 contracts: PersonOrm, FaceIdentityOrm, FaceSampleOrm, MinIOObjectStore, QdrantVectorStore (512-D cosine, point_id=sample_id), model_version/preprocess_version.
- Frozen models: backend/artifacts/models/retinaface_r50_dynamic.onnx and glintr100.onnx (read-only).
- Docs written: REFERENCE_SOURCE_MAP.md, TARGET_CONTRACT.md.
- Next: package scaffold (step 5).

--- phase1-gpu-bulk-baseline [Memory] tags: phase1, MergenVision, gpu-bulk-enrollment, baseline, checkpoint ---
Phase 1 isolated GPU bulk enrollment baseline established.
- Active repo: /home/user/Workspace/MergenVisionPhase2v2 (main @ 0a2d928)
- Writable prefix: phase1/gpu_bulk_enrollment/**
- Runtime artifacts: .artifacts/phase1_gpu_bulk_enrollment/**
- Protected tree manifest: .artifacts/phase1_gpu_bulk_enrollment/baseline/protected_tree_manifest.txt
- phase2-untouched-gate: PASS (pre-existing user dirty files prompt13.txt and .claude/plans/phase1-gpu-bulk-enrollment-plan.md excluded)
- Reference repo: /home/user/MergenVisionDemo (read-only)
- Next: reference source map and current contract analysis (steps 2-4).

--- MergenVisionPhase2: recommendations [Memory] tags: cross-repo, recommendations, external-repo, mergenvision-phase2 ---
EXTERNAL REPO: MergenVisionPhase2 — Aktarılabilir: katmanlı mimari, NativeWorkerPort sözleşmesi, domain modelleri, deterministik reconciliation, GPU hot-path kuralları. Dikkat: tek video/tek GPU, tracker+batch çelişkisi. Kaçınılmalı: FastAPI/DB/storage olduğunu varsaymak; build artefaktlarını kaynak sanmak; anonymous kaydı reconciliation öncesinde yapmak; tracker+batch>1'i göz ardı etmek. Phase2v2 kendi kalıcılık/API katmanlarını tasarlamalı.

--- MergenVisionPhase2: api-workers [Memory] tags: cross-repo, external-repo, mergenvision-phase2, api-workers ---
EXTERNAL REPO: MergenVisionPhase2 — FastAPI implemente edilmemiş; backend/app/api/__init__.py yalnızca placeholder. Hedef uç noktalar: POST /videos/recognize, GET /videos/jobs/{id}, GET /videos/jobs/{id}/result, DELETE /videos/jobs/{id}, GET /faces/{faceId}/appearances. Mevcut orkestrasyon: SubprocessNativeWorkerAdapter her iş için Docker konteyneri başlatır, stdout JSON/özet satırlarını ayrıştırır. Frontend mock API ile çalışır.

--- MergenVisionPhase2: identity-model [Memory] tags: cross-repo, identity-model, external-repo, mergenvision-phase2 ---
EXTERNAL REPO: MergenVisionPhase2 — Domain modeller: RecognitionObservation, TrackletEvidence, CanonicalVideoPerson, ReconciliationConfig (backend/app/domain/video_tracking.py). Durumlar: known/anonymous/new_anonymous. CPU galeri JSON: canonical_face_id, display_name, 512-boyutlu centroid; SHA-256; L2 normalize; top1/top2 marj eşleştirme. ReconcileVideoIdentities: bilinen→anonim→bilinmeyen complete-link kümeleme; cannot-link çatışma kuralları.

--- MergenVisionPhase2: persistence [Memory] tags: cross-repo, persistence, external-repo, mergenvision-phase2 ---
EXTERNAL REPO: MergenVisionPhase2 — Şu anda PostgreSQL/MinIO/Qdrant yok. Çıktılar yerel dosyalar: detections.jsonl, tracks.json, run_manifest.json (output_dir altında). Gereksinimler hedef çoklu depo tanımlar: PostgreSQL (iş/kişilik), MinIO (video/nesne), Qdrant (embedding). İstenen: deterministik ID, idempotent upsert, açık durum, reconciliation öncesi new_anonymous kalıcılaştırılmamalı.

--- MergenVisionPhase2: gpu-runtime [Memory] tags: cross-repo, external-repo, gpu-runtime, mergenvision-phase2 ---
EXTERNAL REPO: MergenVisionPhase2 — Native pipeline: filesrc→qtdemux→h264parse→nvv4l2decoder→nvstreammux→nvdspreprocess→nvdsretinaface→(nvtracker)→nvvideoconvert→mvfacerecognizer. Motorlar: RetinaFace-R50 TensorRT, GlintR100 TensorRT. CUDA çekirdekleri: retinaface_decode, argsort, nms, scale_clip_compact_xy, warp_align, l2_normalize. Tracker: ByteTrack + MultiSourceTracker. Hot-path kuralı: yalnızca yoğunlaştırılmış metadata D2H, tam tensör değil.

--- MergenVisionPhase2: bulk-processing [Memory] tags: cross-repo, external-repo, mergenvision-phase2, bulk-processing ---
EXTERNAL REPO: MergenVisionPhase2 — Toplu kayıt (bulk enrollment) API’si yok; galeri statik JSON. Toplu çıkarım (batch inference) var: WorkerOptions.batch_size/--batch-size ve RetinaFacePostproc::processBatch (backend/native/worker/retinaface_postproc.cpp). nvstreammux batch-size ile çalışır. Tracker (nvtracker) batch_size>1 ile reddeder (MV_ALLOW_TRACKER_BATCH hariç); --mode fast tracker’sız.

--- MergenVisionPhase2: architecture [Memory] tags: cross-repo, external-repo, mergenvision-phase2, architecture ---
EXTERNAL REPO: MergenVisionPhase2 — Hexagonal benzeri katmanlar: api (placeholder), application/services, domain, ports, infrastructure, native. CLI zinciri: RunVideoDetectionService → NativeWorkerPort → SubprocessNativeWorkerAdapter → Docker native worker. Frontend React 18 mock API ile çalışır. Runtime: DeepStream 9.0 konteyneri; derleme ayrı derleme konteyneriyle yapılır.

--- MergenVisionPhase2: overview [Memory] tags: cross-repo, overview, external-repo, mergenvision-phase2 ---
EXTERNAL REPO: MergenVisionPhase2 — Offline video yüz analiz lab'ı. Repo root: /home/user/Workspace/MergenVisionPhase2. Stack: Python kontrol katmanı (FastAPI yer tutucu), React+Vite frontend, GStreamer/DeepStream/CUDA/TensorRT native worker. Ana girişler: backend/app/cli.py ve backend/native/worker/main.cpp. Katmanlı DDD/ports-adapters mimarisi. Henüz üretim kalıcılığı yok.

--- MergenVisionPhase2 [Memory] tags: cross-repo, parent, external-repo, mergenvision-phase2 ---
EXTERNAL REPO: MergenVisionPhase2 — /home/user/Workspace/MergenVisionPhase2. Parent node for all intelligence gathered from MergenVisionPhase2. NEVER assume these patterns exist in Phase2v2 unless explicitly ported.

--- MergenVisionDemo: recommendations [Memory] tags: cross-repo, recommendations, external-repo, mergenvision-demo ---
EXTERNAL REPO: MergenVisionDemo — Top actionable recommendations for Phase2v2: (1) Reuse BulkEnrollmentService producer/consumer architecture: bounded asyncio.Queue, separate GPU/IO executors, per-batch progress commits for resume/cancel. (2) Port deterministic ID scheme from app/core/ids.py for idempotent bulk imports without existence SELECTs. (3) Copy GpuFacePipeline.extract_batch end-to-end: batched nvImageCodec decode -> TensorRT detector -> native CUDA decode/NMS -> GPU alignment -> batched ArcFace -> native L2 normalize, keeping intermediate data as DeviceTensors. (4) Adopt BufferArena + BufferLease with CUDA event fences instead of per-batch allocate/free. (5) Use TrtDeviceEngine.infer_device style DeviceTensor binding to avoid H2D/D2H copies in inference hot path. (6) Mirror persistence triple: PostgreSQL pg_insert upserts, bounded MinIO semaphore, Qdrant 256-point upsert batches with payload validation. (7) Build single pybind11 native package (e.g. mergenvision_gpu) exposing only pointer/stream functions, compiled in CUDA-devel Docker builder stage. (8) Use worker/control-plane split: API orchestrator owns durable ProcessRecords, dispatches compact shard descriptors to GPU workers over HTTP, polls status. (9) For future video pipeline add GPU video decode stage (NVIDIA Video Codec SDK / PyNvVideoCodec) and feed frame batches into same extract_batch path. (10) Reuse Qdrant payload schema (active bool + modelVersion keyword) and set_active_batch soft-delete pattern.

--- MergenVisionDemo: identity-model [Memory] tags: cross-repo, identity-model, external-repo, mergenvision-demo ---
EXTERNAL REPO: MergenVisionDemo — Identity Model. Schema in app/domain/models.py: FaceIdentity (canonical identity row, unique identity_lookup_hmac, links to many Persons), Person (per-dataset person, unique national_id_lookup_hmac, face_identity_id FK, JSONB details, soft-delete deleted_at), PersonPhoto (one row per stored photo, unique object_key in MinIO, status staged/active/failed/deleted, unique (person_id, content_sha256)), FaceSample (one row per extracted face, photo_id FK unique, JSONB bbox/landmarks, detector_model, embedding_model, quality_score, status). Deterministic IDs in app/core/ids.py: identity_hmac(identity_key, master_key) -> HMAC-SHA256; derive_person_id(hmac) and derive_face_identity_id(hmac) are UUIDv5 over fixed namespaces; derive_photo_id(content_sha256) UUIDv5 over photo hash; derive_sample_id(photo_id, model_version) UUIDv5 over photo_id:model_version. Re-imports are idempotent: same folder/photo/model always produces same UUIDs; ON CONFLICT DO NOTHING/UPDATE handles collisions without pre-SELECTs.

--- MergenVisionDemo: persistence [Memory] tags: cross-repo, persistence, external-repo, mergenvision-demo ---
EXTERNAL REPO: MergenVisionDemo — Persistence & Storage patterns. MinIO: app/infrastructure/minio.py PhotoStorage.put_object stores raw bytes with key enrollments/{person_id}/{photo_id} using asyncio.to_thread; lazy bucket creation. PostgreSQL bulk upserts in app/services/bulk_enrollment.py _persist_batch: FaceIdentity/Person use pg_insert(...).on_conflict_do_nothing(index_elements=[...]); PersonPhoto/FaceSample use pg_insert(...).on_conflict_do_update(index_elements=['photo_id'], set_=...). All writes in one batch per chunk; _commit_progress calls await db.commit() after each activation batch. Qdrant: app/infrastructure/qdrant.py FaceVectorStore.upsert_batch validates payload keys (sampleId, photoId, personId, active, modelVersion) and vector shape/finiteness; sends points in 256-point sub-batches with wait=False for bulk; search_active filters by active=True and modelVersion; set_active_batch toggles active payload field for soft deletes. Sample point payload mirrors Postgres sample ID: sampleId == point.id.

--- MergenVisionDemo: gpu-runtime [Memory] tags: cross-repo, external-repo, gpu-runtime, mergenvision-demo ---
EXTERNAL REPO: MergenVisionDemo — GPU / Native Runtime. Key files: app/ml/gpu/face_pipeline.py (GpuFacePipeline, extract_batch, extract_bytes), app/ml/gpu/decoder.py (JpegGpuDecoder.decode_batch), preprocess.py, retinaface_preprocessor.py, retinaface_postprocess.py, scrfd_postprocess.py, alignment.py (GpuFaceAligner.align, similarity_transform, warp_align), recognizer.py (GpuRecognizer.embed, max_batch), trt_device_engine.py (TrtDeviceEngine.infer_device), buffer_arena.py (BufferArena, BufferLease), device_tensor.py (DeviceTensor), l2_norm.py (l2_normalize_device), native/mergenvision_gpu/src/bindings.cpp (_mergenvision_gpu pybind11 module), Dockerfile (CUDA 12.4 multi-stage build). extract_batch hot path: JpegGpuDecoder.decode_batch -> RetinaFacePreprocessor -> TrtDeviceEngine.infer_device -> RetinaFacePostprocess.decode (CUDA NMS) -> scale_and_compact -> pick_largest_device -> GpuFaceAligner.align -> GpuRecognizer.embed (ArcFace chip batch) -> native L2 normalize. Only embeddings/bboxes/landmarks/scores copied to host. BufferArena pools GPU allocations keyed by shape/dtype and fences reuse with CUDA events. Python passes device pointers (uintptr_t) and stream handle to pybind11 functions; error status written into DeviceTensor([1] int32) and read back after cudaStreamSynchronize.

--- MergenVisionDemo: bulk-enrollment [Memory] tags: bulk-enrollment, cross-repo, external-repo, mergenvision-demo ---
EXTERNAL REPO: MergenVisionDemo — Bulk Enrollment architecture. Key files: app/services/bulk_enrollment.py (BulkEnrollmentService, _produce, _consume, _read_and_extract, _extract_batch_faces, _persist_batch, enroll_shard), app/services/bulk_manifest.py (shard_by_person_id), app/services/bulk_orchestrator.py (dispatch_shards, start_vggface_job, resume_vggface_job), app/workers/gpu_worker.py, app/api/routes/bulk_jobs.py. Flow: enroll_shard -> _produce streams identities/photos into asyncio.Queue(maxsize=2) -> _consume -> _persist_batch -> _commit_progress after each activation batch; orchestrator dispatches shard descriptors to GPU workers over HTTP and polls status. Defaults: bulk_extract_batch_size=256, bulk_max_persistence_concurrency=32, bulk_activation_batch_size=2048. GPU inference serialized per pipeline by asyncio.Lock in single-worker ThreadPoolExecutor; file reads on separate io_executor with min(32, cpu_count*2) workers.

--- MergenVisionDemo: overview [Memory] tags: cross-repo, overview, external-repo, mergenvision-demo ---
EXTERNAL REPO: MergenVisionDemo — FastAPI/React face-recognition backend. Long-lived GPU workers own one CUDA device each, keep a warm GpuFacePipeline across batches, and process durable shard jobs from an API orchestrator. Hot path keeps JPEG decode, detection, alignment, recognition, and L2 norm on GPU via DeviceTensors/BufferArena; only embeddings/metadata touch host. PostgreSQL, MinIO, and Qdrant are updated with batched upserts using deterministic HMAC-derived UUIDs for idempotent cancel/resume. Source pack: .claude/cross-repo-intelligence/mergenvision-demo-insights.md

--- MergenVisionDemo [Memory] tags: cross-repo, parent, external-repo, mergenvision-demo ---
SEPARATE EXTERNAL REFERENCE REPOSITORY: /home/user/MergenVisionDemo. This node is the parent for all cross-repo intelligence gathered from MergenVisionDemo. NEVER assume these patterns exist in Phase2v2 unless explicitly ported. Child nodes cover overview, bulk enrollment, GPU/native runtime, persistence, identity model, and recommendations.

--- MergenVision [Project] tags: mergenvision, cross-repo, parent, phase2v2-current, demo-external ---
Cross-repo intelligence root for the MergenVision family. Holds linked memories for Phase2v2 (CURRENT repository, /home/user/Workspace/MergenVisionPhase2v2) and MergenVisionDemo (SEPARATE EXTERNAL reference repository, /home/user/MergenVisionDemo). NEVER assume Demo patterns exist in Phase2v2 unless explicitly ported. Use this node to traverse related memories instead of reading long markdown files.

--- bulk-enrollment-pattern-from-demo [Decision] tags: bulk-enrollment, milestone-b, MergenVisionDemo, gpu-pipeline, pattern ---
MergenVisionDemo reposundan (`/home/user/MergenVisionDemo/backend/app/services/bulk_enrollment.py`) bulk enrollment pattern incelendi.

**Ana prensipler:**
- GPU hot path: `GpuFacePipeline.extract_batch(image_bytes_list, max_batch=...)`. Detector ve recognizer yüksek batch size ile GPU'da ardışık çalışır; CPU'ya sadece son extraction sonuçları çıkar.
- IO sona bırakılır ve sınırlı paralellik (`asyncio.Semaphore`) ile yapılır: MinIO upload, PostgreSQL `ON CONFLICT DO UPDATE` bulk upsert, Qdrant batch upsert.
- Deterministik ID üretimi ile idempotency: `derive_sample_id(photo_id, model_version)`, `identity_hmac`, `person_id`. Re-run duplicate oluşturmaz.
- Producer/consumer queue: producer okuma+extract, consumer persist. Chunk chunk ilerlenir, her chunk commitlenir.
- Model version sözleşmesi: Qdrant payload `active=True`, `modelVersion`, `personId`; sample tablosunda `embedding_model`.
- Hata sınıflandırması: `no_face`, `decode_error`, `persistence_error`, `fatal_error`; soft error rate > %3 fail.

**Demo şema farkları (Phase2v2'ye adaptasyon için dikkat):**
- Demo: `Person` ← `FaceIdentity` ← `PersonPhoto` ← `FaceSample`.
- Phase2v2: `Person` aggregate root → `FaceIdentity` → `FaceSample` (doğrudan sample MinIO key `faces/{face_id}/{sample_id}/aligned.webp`).
- Demo'da `PersonPhoto` ayrı nesne; Phase2v2'de yok.
- Phase2v2'de `FaceIdentity.status` (`known`/`anonymous`/`new_anonymous`) ve `redirect_to_face_id` alias/redirect mekanizması var; demo'da yok.
- Phase2v2'de image enrollment `POST /faces/{face_id}/enroll` ve `POST /faces/{face_id}/assign` ile; demo'da bulk worker doğrudan oluşturuyor.

**Karar:** Phase2v2 bulk enrollment, demo'nun GPU hot path + batch persistence pattern'ini alır ama domain modeli Phase2v2'ye uyarlar:
1. Manifest: `EnrollmentIdentity` (display_name + photos list) olarak gelir; her identity bir `Person` ve bir `FaceIdentity` üretir.
2. GPU extraction sonrası her fotoğraf bir `FaceSample` olur (face_id aynı identity'nin face_id'si).
3. Bulk upsert: `Person`, `FaceIdentity`, `FaceSample`, Qdrant points tek transaction/flow içinde.
4. Idempotency: deterministik `face_id`/`sample_id` veya `ON CONFLICT DO UPDATE` ile sağlanır.
5. Endpoint: `POST /people/batch` veya ayrı `POST /faces/batch-enroll` olarak eklenebilir; önce people directory batch insert ile başlanır, sonra fotoğraflı bulk enrollment genişletilir.

**Why:** Kullanıcı "detector recognizerın batch size i yüksek gpu hot path izliyo zaten hiç gpu dan inmeden enroll ediyo sadece en son da bulk bi şekilde IO ya giriyo batch şekild eyani cpu da çok az duruyo" diyerek demo'daki yolu işaret etti.
**How to apply:** Phase2v2'nin `IdentityStorageLifecycleService` ve `VideoIdentityResolutionService` ile aynı Qdrant collection/MinIO namespace paylaşımını koru; bulk worker'ı da aynı lifecycle üzerinden çalıştır ki Phase1→Phase2 continuity bozulmasın.

--- milestone-a-backend-complete [Decision] tags: bulk-enrollment, milestone-a, backend, tests, ruff, continuity ---
Milestone A backend tamamlandı ve doğrulandı.
- 255 backend testi geçiyor.
- Ruff tüm `app` ve `tests` üzerinde temiz.
- Kalan tek hata `app/worker/video_worker_main.py` I001 import sort idi; `ruff check --fix` ile çözüldü.
- Shared integration cleanup fixture'ları (`_clean_integration_stores`, `lifecycle_service`, `crop_bytes`) `tests/integration/conftest.py`'e taşındı; tüm entegrasyon modülleri (lifecycle, services, video, persistence) artık aynı cleanup'ı paylaşıyor.
- `face_assign` process_type check constraint'e eklendi (migration 0006 + manuel test DB alter).
- Phase 1 → Phase 2 kimlik sürekliliği mevcut: aynı Qdrant koleksiyonu ve yüz kimlik deposu image enroll ile video recognition arasında paylaşılıyor; `current_status`/`current_name` projeksiyonu canonical identity'den çözülüyor. Video testi `test_video_people_projection_reflects_current_identity_after_enrollment` ile doğrulandı.

**Kalan bağlayıcı görevler:**
1. Bulk/batch people insert endpoint'i tasarla ve implemente et (Milestone B öncü görev).
2. Phase 1'de enroll edilen kişinin Phase 2 video'da known/recognized olarak çıktığını üretim senaryosunda da garantile (canonical resolution + redirect alias zincirlerinin canlı videoda doğru çalıştığını e2e doğrula).

**Why:** Kullanıcı "devam et" ve iki yeni bağlayıcı gereksinim belirtti; önce Milestone A'yı stabilize etmek gerekiyordu.
**How to apply:** Bulk endpoint için `POST /people/batch` veya `POST /people` upsert/modal seçeneklerini değerlendir; identity continuity için video_worker_main pipeline'ında `FaceIdentityStore.get_canonical_by_id` / `resolve_identity` kullanımını ve yeni enrollment sonrası Qdrant payload güncellemelerini denetle.

--- phase1-phase2-identity-continuity-and-bulk-enroll [Decision] tags: phase1, phase2, bulk-enrollment, milestone-b, identity-continuity, video-recognition ---
Kullanıcıdan gelen iki yeni bağlayıcı gereksinim:

1. **Bulk / batch people insert** desteği isteniyor. Tek tek kişi/enroll creation yerine toplu people kaydı atabileceğimiz bir akış olmalı.
2. **Phase 1 ↔ Phase 2 kimlik sürekliliği:** Phase 1’de (image enroll) kaydedilen bir kişi, Phase 2 video işlemede anonim olarak görünmemeli; aynı vektör/face identity store’un devam etmesi sayesinde Phase 2 video pipeline’ında da "known" olarak tanınmalı. Yani image enroll ve video recognition aynı Qdrant koleksiyonunu, aynı face_identity/person store’unu paylaşmalı; video sonuçlarında `status_at_processing` snapshot korunsa bile `current_status` known olmalı.

**Why:** Phase 1 ve Phase 2 ayrı milestone’lar gibi görünse de ürün tek bir kimlik directoriesi üzerinde çalışıyor. Kullanıcı image’den kaydettiği kişiyi videoda da known bekliyor.

**How to apply:**
- Bulk enroll için Milestone B’deki manifest tabanlı dataset enrollment’ı planlarken people/face identity oluşturma akışını aynı lifecycle service’e yönlendir; böylece toplu yüklenenler de aynı vector store’a yazılır.
- Video pipeline’ın `VideoIdentityResolutionService`’i mevcut `match_threshold` ile aynı Qdrant koleksiyonuna sorgu atıyor zaten; tek kontrol edilecek nokta: Phase 1’den kalan sample’ların `is_active=true` ve Qdrant payload’larının doğru şekilde saklanmış olması. Migration/backfill sırasında eski Phase 1 known identity’ler için `person_id` backfill’i zaten yapılıyor.
- Frontend’de video people listesi `current_status`/`current_name` projection’ını kullanıyor; bu zaten Phase 1 enroll sonrası known’u yansıtacak.

--- milestone-a-test-failure-root-causes [Memory] tags: milestone-a, testing, postgres, migration, asyncio, pytest, troubleshooting ---
Milestone A backend entegrasyon testleri sırasında karşılaşılan fail zincirinin kök nedenleri ve çözümleri:

1. `.env.test` source edilmeden test çalıştırılmıştı. Çözüm: `. .env.test` sonrası `.venv/bin/python -m pytest` kullanmak.
2. `tests/integration/services/test_person_assign_and_redirect.py` session-scope asyncio kullanıyordu; ortak `_clean_integration_stores` fixture'ı `asyncio.run(...)` ile yeni event loop açıp kapatıyordu. İkinci testte session loop kayboluyordu (`RuntimeError: There is no current event loop`). Çözüm: cleanup fixture'ı `async def` yapılıp pytest-asyncio loop'una bağlandı; services testi function-scope asyncio'ya çekildi.
3. Migration `0006` person domainini eklerken `process_record.process_type` check constraint'ine `'face_assign'` eklenmemişti; `assign_identity_to_person` işlemi `ck_process_record_type` ihlaliyle fail ediyordu. Çözüm: `0006_person_domain_and_identity_redirect.py` içinde constraint drop+create ile `'face_assign'` eklendi.
4. `test_assign_second_face_to_existing_person_redirects` `unit_of_work_factory` fixture'ını parametre olarak almamıştı (`NameError`). Çözüm: parametre eklendi.
5. `tests/integration/services` modülü `lifecycle/conftest.py` cleanup'ını paylaşmıyordu; önceki testlerden kalan Qdrant vektörleri `vector_b` sorgusunu known yapıyordu. Çözüm: `_clean_stores_async`, `_clean_lifecycle_stores`, `crop_bytes` ve `lifecycle_service` fixture'ları `tests/integration/conftest.py`'ye taşındı.

**Why:** Bu hataların ortak nedeni, yeni person/assign/redirect özelliklerinin hem şema migration'ında (`face_assign` enum), hem test altyapısında (session vs function asyncio scope, shared cleanup), hem de test yazımında aynı anda düşünülmemesiydi.

**How to apply:** Yeni bir domain özelliği eklerken (a) yeni `process_type`/`status` enum değerlerini migration'da hemen genişlet, (b) session-scope asyncio kullanan testlerde cleanup fixture'larının `asyncio.run()` kullanmadığından emin ol, (c) yeni integration modüllerinin ortak conftest cleanup'ini paylaştığından emin ol, (d) test parametrelerinde kullanılan fixture'ları explicit olarak tanımla.

--- milestone-a-progress-2026-07-18 [Memory] tags: milestone-a, backend, progress, task-21 ---
Milestone A backend implementasyonunda 2026-07-18 durumu.

## Tamamlananlar
- Person domain aggregate root eklendi (`backend/app/domain/entities/person.py`); `create`, `rename`, `update_metadata`, `deactivate` metodları var.
- FaceIdentity entity'ye `person_id`, `redirect_to_face_id`, `canonical_face_id` eklendi; `promote_to_known` artık person_id zorunlu, `assign_to_person` merge/redirect yapıyor.
- Domain unit testleri güncellendi (`tests/unit/domain/test_face_identity.py`, `test_person.py`).
- Alembic migration 0006 oluşturuldu: person tablosu, face_identity FK'ları, backfill (geçici `_source_face_id` sütunu ile 1:1 known→person eşleştirme), check constraint'ler.
- SQLAlchemy ORM: `PersonOrm` ve güncellenmiş `FaceIdentityOrm`.
- Repository port'larına `PersonRepository` ve `FaceIdentityRepository.get_canonical_by_id`, `list_by_person_id` eklendi.
- SQLAlchemy implementasyon: `SqlAlchemyPersonRepository`, güncellenmiş `SqlAlchemyFaceIdentityRepository`.
- `UnitOfWork` port ve `SqlAlchemyUnitOfWork` `people` repository ile güncellendi.
- `IdentityStorageLifecycleService`: `enroll_identity` yeni Person oluşturuyor; `assign_identity_to_person` source inactive/redirect, sample'ları Qdrant'ta pasifleştiriyor.
- `ImageRecognitionService`: `_canonical_face_id` çözümleme, `get_identity_detail`, `delete_identity`, sample metodları canonical üzerinden; `assign_face_to_person` eklendi.
- `FaceController`: `assign_to_person`, `EnrollResponse.person_id`.
- `PersonManagementService` + `PersonController` oluşturuldu; list/get/create/update/deactivate + faces.
- Routes: `faces.py` `POST /{face_id}/assign`; `people.py` yeni route'lar.
- `main.py` ve `dependencies.py` person controller wiring.
- `VideoResultService` + schemas: `current_status`/`current_name` projection (snapshot immutable kalıyor, UI current değeri görüyor).
- Static: ruff ve mypy temizlendi (112 source, no issues).
- Unit testler: 181 passed.
- Integration: migration tests 9 passed; lifecycle delete/detail/history 1 passed.

## Kalan / Devam Eden
- `tests/integration/video/test_video_processing_and_result_api.py` ve diğer integration testler çalıştırılacak; person/redirect değişikliklerine göre düzeltilecek.
- Yeni API endpoint'leri için integration/API testleri eklenecek (`/people`, `/faces/{id}/assign`).
- Frontend (Task #22) henüz başlanmadı; #21 bitince başlayacak.

## Önemli Kararlar
- Redirect/alias: source face inactive, canonical pointer; eski snapshot'lar korunur; UI canonical ismi gösterir.
- Enrollment iki mod: new person (mevcut `/enroll`) ve existing person (yeni `/assign`).


--- mergenvision-implementation-decisions-2026-07-18 [Decision] tags: MergenVision, milestone-a, decisions, milestone-b, milestone-c, binding ---
Binding implementation decisions for MergenVision Phase2v2 (2026-07-18).

1. Assign-to-existing-person / merge semantics — REDIRECT/ALIAS
   - Source face_id=A becomes inactive and carries canonical_face_id=B pointer.
   - /faces/A API calls continue to work and resolve to canonical identity B.
   - New recognition and current projection use B.
   - Historical recognition/video snapshots remain immutable as A.
   - Video UI shows B's current name.
   - UI route opened with A may navigate to canonical B.
   - Rationale: preserves history and does not break old links.

2. Bulk dataset ingestion — READ-ONLY BIND MOUNT, not MinIO upload
   - Dedicated bulk-enrollment worker container reads dataset from user-supplied read-only bind mount.
   - Not the normal backend container; separate worker.
   - Dataset local path lives only in worker config; must not leak to API response, logs, or object keys.
   - Worker writes accepted aligned crops to MinIO.
   - PostgreSQL stores manifest, provenance, and SHA.
   - Qdrant receives only accepted embeddings.
   - Future: MinIO-source mode may be added for distributed/remote import.
   - Rationale: avoid copying huge dataset twice.

3. GPU video lab — CAPTURE + REPLAY/SWEEP
   - Capture: real GPU hot path inside pinned mergenvision/deepstream-dev:9.0 container produces immutable observation bundle.
   - Replay/sweep: CPU-fast tracker/quality/template/reconciliation experiments over captured bundle.
   - Core architecture: GPU capture once → CPU replay many → finalist GPU rerun.
   - Rationale: avoids reprocessing video for every threshold experiment while still producing real GPU bundles.

These decisions are binding; do not re-ask.

--- current-tasks-mergenvision-2026-07-18 [Memory] tags: MergenVision, tasks, status, pending-decisions ---
MergenVision Phase2v2 current task state (2026-07-18).

Plan file: /home/user/Workspace/MergenVisionPhase2v2/newmission.md
Decision node: 64812 (newmission-plan-post-discovery-2026-07-18)
Repo HEAD: 2cfde196795a2b783e4494d879bbc48fe3361f69 (main)

Active task list:
- Task #20: Deep project discovery before detailed planning — COMPLETED
- Task #21: Implement Milestone A backend — PENDING (blocked on user decisions)
- Task #22: Implement Milestone A frontend — PENDING (blocked by #21)
- Task #23: Implement Milestone B bulk enrollment — PENDING (blocked by #21)
- Task #24: Implement Milestone C GPU video lab — PENDING

Where we left off:
- Deep discovery finished.
- Detailed plan written to newmission.md.
- Recall skill used to dump MergenVision memory nodes to file.
- Waiting for user decisions on 3 open questions before starting implementation:
  1. merge/redirect strategy for assign-to-existing-person
  2. bulk dataset read location (filesystem vs MinIO upload)
  3. GPU lab DeepStream container dependency vs CPU-only replay

Next action: update newmission.md and prompt-memory Decision node with user decisions, then start Task #21 (Milestone A backend).

--- newmission-plan-post-discovery-2026-07-18 [Decision] tags: MergenVision, newmission, milestone-a, deep-discovery-complete, plan-ready ---
MergenVision Phase2v2 mission plan updated after deep discovery.

Plan file: /home/user/Workspace/MergenVisionPhase2v2/newmission.md (written 2026-07-18)
Repo HEAD: 2cfde196795a2b783e4494d879bbc48fe3361f69 (main)

Key findings from discovery:
- No Person domain; face_identity is the only aggregate.
- VideoResultService.list_people returns stale status_at_processing/name_at_processing.
- Frontend client.ts ignores structured body.error envelope.
- useEnrollMutation does not invalidate video-people queries.
- Enrollment only supports anonymous->known promotion; no merge/redirect.
- Bulk enrollment has no backend code.
- research/video_reference_lab is frozen; gpu_video_lab does not exist.

Current step: start Milestone A backend implementation (Alembic 0006 + Person domain + current projection + enroll unification).

Next actions:
1. Implement Milestone A backend (Task #21).
2. Implement Milestone A frontend (Task #22, blocked by #21).
3. Implement Milestone B bulk enrollment (Task #23, blocked by #21).
4. Implement Milestone C GPU video lab (Task #24).

--- newmission-global-identity-bulk-enrollment-gpu-lab [Decision] tags: compaction-retrieve, MergenVision, newmission, global-identity, bulk-enrollment, gpu-lab, decision ---
Yeni görev: MergenVision Phase2v2'de global identity/enrollment unification, bulk dataset enrollment ve izole GPU video lab.

Plan dosyası: /home/user/Workspace/MergenVisionPhase2v2/newmission.md
Repo HEAD: 2cfde196795a2b783e4494d879bbc48fe3361f69 (main, 2026-07-18)

Milestone A — Global Identity & Enrollment Unification:
- Enrollment call graph + reproducer
- Frontend error envelope parser fix
- Snapshot vs current identity read-model
- Query invalidation/refetch
- New/existing Person assignment UI
- Person schema/backfill migration
- Assign/merge audit semantics
- Acceptance: make global-identity-enrollment-acceptance

Milestone B — Bulk Dataset Enrollment:
- Small rights-cleared fixture manifest
- Person/FaceIdentity reserve/upsert + ingestion lifecycle
- GPU batching, idempotent retry, duplicate prevention
- Admission/quality gate
- Gallery search aggregation
- Bulk enrollment API/UI
- Acceptance: make bulk-enrollment-e2e

Milestone C — Isolated GPU Video Lab:
- research/gpu_video_lab/ scaffold
- Capture once bundle
- Replay/sweep tracker/quality/best-shot/reconciliation variants
- Contact sheets/failure gallery
- Sparse/dense evaluation
- Promotion candidate/shadow mode
- Acceptance: make gpu-video-lab-*

Current step: Milestone A — enrollment call graph + reproducer.

--- agents-md-0001 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 1: # MergenVision Engineering Constitution

--- agents-md-0003 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 3: Bu dosya MergenVision repository'sinde çalışan bütün insan ve AI agent'lar için kalıcı çalışma sözleşmesidir. Kullanıcının güncel açık kararı bu dosyadan üstündür. Sprint'e özel hedefler `docs/implementation/CURRENT_SPRINT.md`, kaynak adaptasyon kararları `docs/implementation/REFERENCE_DECISION_LOG.md` içinde tutulur.

--- agents-md-0005 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 5: ## 1. Ürün misyonu

--- agents-md-0007 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 7: Sistem önce görüntülerde, ardından videolarda çoklu yüz tespiti ve kalıcı yüz kimliği üretir. Her detected yüzün immutable bir `faceId` değeri olur. İlk karşılaşma `new_anonymous`, daha sonraki eşleşme `anonymous`, aynı `faceId` isimlendirildikten sonra `known` sonucunu üretir. Video genişletmesi aynı identity karar motorunu kullanır; tracking yalnız temporal sürekliliği, recognition kalıcı kimliği cevaplar.

--- agents-md-0009 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 9: ## 2. Source-of-truth sırası

--- agents-md-0011 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 11: Çelişki halinde sıra şöyledir:

--- agents-md-0013 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 13: 1. Kullanıcının güncel açık kararı.

--- agents-md-0014 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 14: 2. `requirements/ProjectRequirements.md` içindeki image/identity gereksinimleri.

--- agents-md-0015 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 15: 3. `requirements/videorequirements.md` içindeki additive video gereksinimleri.

--- agents-md-0016 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 16: 4. Kullanıcı tarafından onaylanmış architecture/ADR belgeleri.

--- agents-md-0017 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 17: 5. `docs/implementation/CURRENT_SPRINT.md`.

--- agents-md-0018 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 18: 6. Official vendor documentation ve pinned upstream source.

--- agents-md-0019 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 19: 7. Eski repository kodları ve raporları yalnız lessons-learned kaynağıdır.

--- agents-md-0021 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 21: Eski client özetleri; Oracle, 10M kişi, national ID veya başka ek kapsamları kullanıcı yeniden onaylamadıkça ürün requirement'ı sayılmaz.

--- agents-md-0023 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 23: ## 3. Onaylanmış ürün sınırı

--- agents-md-0025 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 25: Backend bağımsız ve API-first çalışır. Internal React UI, kullanıcı tarafından istenen kontrollü bir extension'dır; backend olmadan çalışamaz ve business/ML/storage mantığı içermez. Product output yeniden encode edilmiş annotated MP4 değil, orijinal video + zaman senkronlu overlay metadata'sıdır. Annotated MP4 yalnız debug/acceptance artifact'ı olarak ayrıca istenirse üretilebilir.

--- agents-md-0027 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 27: ## 4. Zorunlu implementation sırası

--- agents-md-0029 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 29: Sıra atlanmaz:

--- agents-md-0031 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 31: 1. Requirement/contract/ERD/state-machine freeze.

--- agents-md-0032 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 32: 2. PostgreSQL + MinIO + Qdrant foundation.

--- agents-md-0033 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 33: 3. Image `new_anonymous -> anonymous` vertical slice.

--- agents-md-0034 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 34: 4. Aynı `faceId` ile enrollment -> `known`, update/delete/history.

--- agents-md-0035 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 35: 5. Video upload, retention, async job, cancel/retry.

--- agents-md-0036 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 36: 6. Native GPU video observation extraction.

--- agents-md-0037 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 37: 7. Python temporal tracking ve identity reconciliation.

--- agents-md-0038 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 38: 8. Video best-shot/new-anonymous persistence.

--- agents-md-0039 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 39: 9. Person-level aggregation, appearances ve overlay API.

--- agents-md-0040 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 40: 10. Internal UI, hardening, performance ve full E2E.

--- agents-md-0042 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 42: Image identity lifecycle gerçek storage üzerinde geçmeden video recognition PASS ilan edilmez.

--- agents-md-0044 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 44: ## 5. Her görevde zorunlu başlangıç

--- agents-md-0046 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 46: Kod veya doküman değiştirmeden önce:

--- agents-md-0048 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 48: 1. Repository root, branch, HEAD ve `git status --short` doğrulanır.

--- agents-md-0049 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 49: 2. Bu dosya tamamen okunur.

--- agents-md-0050 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 50: 3. `CURRENT_SPRINT.md`, ilgili requirements, architecture ve testler okunur.

--- agents-md-0051 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 51: 4. Dirty worktree'deki kullanıcı değişiklikleri belirlenir ve korunur.

--- agents-md-0052 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 52: 5. Multi-file/sprint işinde `codebase-memory-mcp` ile gerçek caller/callee ve test path'leri keşfedilir; filesystem source ile doğrulanır.

--- agents-md-0053 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 53: 6. İlgili official docs/upstream source `opensourcereferences/references.md` üzerinden seçilir.

--- agents-md-0054 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 54: 7. Plan current sprint ile çelişiyorsa implementation başlatılmaz.

--- agents-md-0056 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 56: ## 6. Repository ve değişiklik güvenliği

--- agents-md-0058 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 58: Aktif repository dışında yazma yapılmaz. Eski repository'ler read-only referanstır. Kullanıcı istemeden `git add`, commit, push, merge, history rewrite, tracked file silme, model/dataset indirme, Docker volume silme veya system CUDA/driver değişikliği yapılmaz. Machine-specific absolute runtime path production source'a yazılmaz.

--- agents-md-0060 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 60: ## 7. Katman sınırı

--- agents-md-0062 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 62: - Python control plane: FastAPI, contracts, domain kararları, orchestration, PostgreSQL, MinIO, Qdrant, tracking, reconciliation, history.

--- agents-md-0063 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 63: - Native data plane: GStreamer/DeepStream, NVDEC/NVMM, CUDA preprocess/alignment, TensorRT detector/recognizer, compact observation emission.

--- agents-md-0064 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 64: - UI: yalnız versioned API'leri tüketir.

--- agents-md-0066 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 66: Domain outer layer'a bağımlı olmaz. API doğrudan SQL/Qdrant/MinIO/GPU çağırmaz. Infrastructure business karar sahibi değildir. Tracker kalıcı identity sahibi değildir.

--- agents-md-0068 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 68: ## 8. Production GPU hot path

--- agents-md-0070 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 70: Hedef video yolu:

--- agents-md-0072 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 72: ```text

--- agents-md-0073 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 73: encoded video

--- agents-md-0074 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 74: -> GStreamer graph

--- agents-md-0075 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 75: -> NVIDIA decoder / NVMM

--- agents-md-0076 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 76: -> DeepStream batching

--- agents-md-0077 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 77: -> CUDA/TensorRT RetinaFace

--- agents-md-0078 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 78: -> CUDA five-point alignment

--- agents-md-0079 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 79: -> TensorRT ArcFace/Glint embedding

--- agents-md-0080 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 80: -> GPU L2 normalization

--- agents-md-0081 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 81: -> compact metadata/embedding CPU boundary

--- agents-md-0082 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 82: ```

--- agents-md-0084 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 84: Production hot path'te full-frame OpenCV/PIL/NumPy decode, CPU resize, raw tensor NumPy postprocess, frame başına zorunlu device synchronize ve sessiz CPU inference fallback yasaktır.

--- agents-md-0086 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 86: ## 9. GStreamer ve DeepStream kararı

--- agents-md-0088 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 88: Bu bir “GStreamer mı DeepStream mi?” seçimi değildir: DeepStream, GStreamer tabanlı NVIDIA data plane'dir. Graph GStreamer ile kurulur; `nvv4l2decoder`, `nvstreammux`, `nvdspreprocess`, `nvinfer` veya doğrulanmış custom native elementler gibi DeepStream/NVIDIA bileşenleri kullanılır.

--- agents-md-0090 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 90: İlk product graph'ında render branch, `nvstreamdemux`, OSD, encoder ve filesink zorunlu değildir. Bunlar GPU observation throughput'unu etkilememelidir. Python NVMM surface map etmez.

--- agents-md-0092 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 92: ## 10. Frame batch ve face batch ayrımı

--- agents-md-0094 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 94: - Frame batch: detector throughput'u için ardışık input frame'leri.

--- agents-md-0095 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 95: - Face batch: bir frame batch içinde bulunan değişken sayıdaki yüzlerin recognizer input'u.

--- agents-md-0097 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 97: Bu iki batch aynı şey değildir. Detector `batch=8` iken recognizer batch'i o sekiz frame'deki geçerli yüz sayısı olabilir. Dynamic TensorRT profile min/opt/max, actual batch, partial final batch ve EOS ayrı test edilir. Batch sonucu frame/PTS sırası kaybedilemez.

--- agents-md-0099 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 99: ## 11. CPU boundary ve backpressure

--- agents-md-0101 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 101: CPU'ya yalnız şu kompakt kayıtlar çıkar:

--- agents-md-0103 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 103: - source ID, frame index, PTS/time base;

--- agents-md-0104 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 104: - original-resolution bbox ve landmarks;

--- agents-md-0105 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 105: - detector score ve quality metrikleri;

--- agents-md-0106 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 106: - 512-D embedding veya seçilmiş embedding evidence;

--- agents-md-0107 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 107: - model/preprocess/profile kimliği.

--- agents-md-0109 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 109: Full frame, NV12/RGBA surface, detector input tensor veya bütün raw TRT outputs CPU'ya taşınmaz. Native producer bounded ring buffer/queue kullanır. Queue davranışı açıkça `backpressure`, kontrollü drop veya job failure olarak tanımlanır; sessiz sınırsız RAM büyümesi yasaktır.

--- agents-md-0111 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 111: ## 12. Performance iddiası disiplini

--- agents-md-0113 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 113: `trtexec` engine FPS, detector-only pipeline FPS, GPU observation FPS ve full E2E FPS ayrı metriklerdir. “600 FPS” denebilmesi için hardware, video, codec, batch, sampling, tracker, recognizer, persistence, model SHA ve ölçüm kapsamı yazılmalıdır.

--- agents-md-0115 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 115: 600 FPS yaklaşık 1.67 ms/frame bütçedir. Python tracker bu bütçe içinde varsayımla PASS sayılamaz. Frozen observation replay; p50/p95/p99 latency, queue depth, backlog, drop, RSS ve sustained-duration raporu gerekir. Detector-only sayı full product performansı değildir.

--- agents-md-0117 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 117: ## 13. Python tracker sözleşmesi

--- agents-md-0119 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 119: İlk tercih Python metadata tracker'dır; C++ rewrite yalnız profiling kanıtıyla yapılır. Her source'un tracker state'i tek sıralı consumer tarafından `PTS/frame` sırasıyla mutate edilir. Batch içindeki frame'ler sırayla tracker'a verilir; batch sınırında state resetlenmez. Kaynak/job seviyesinde paralellik olabilir, aynı source tracker state'inde paralel update olamaz.

--- agents-md-0121 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 121: ByteTrack adapte edilirse low-score detection ikinci association aşamasına ulaşmalıdır. NMS/threshold ile bütün low-score adayları önceden silinip sonra “ByteTrack kullanılıyor” denilemez.

--- agents-md-0123 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 123: ## 14. Tracklet, track ve identity ayrımı

--- agents-md-0125 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 125: - `rawTrackletId`: Kesintisiz temporal tracker segmenti.

--- agents-md-0126 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 126: - `trackId`: Bir video içinde reconciliation sonrası canonical kişi grubu.

--- agents-md-0127 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 127: - `faceId`: Bütün image/video request'leri boyunca kalıcı biyometrik identity.

--- agents-md-0128 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 128: - `detectionId`: Tek processed-frame observation.

--- agents-md-0130 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 130: Scene cut, uzun kayıp veya yeniden giriş yeni raw tracklet üretir. Birinci ve son sahnedeki Rachel'ın aynı `faceId` olması tracker değil, embedding evidence + reconciliation sonucudur. Aynı anda görünen iki farklı yüz için cannot-link uygulanır.

--- agents-md-0132 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 132: ## 15. Identity status semantiği

--- agents-md-0134 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 134: `new_anonymous` yalnız identity'nin yaratıldığı process/job sonucudur; persistent identity type değildir.

--- agents-md-0136 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 136: ```text

--- agents-md-0137 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 137: ilk unmatched process -> result=new_anonymous, identity=anonymous

--- agents-md-0138 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 138: sonraki match          -> result=anonymous, identity=anonymous

--- agents-md-0139 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 139: aynı faceId enroll     -> sonraki result=known, identity=known

--- agents-md-0140 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 140: ```

--- agents-md-0142 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 142: `faceId`, sample ID'leri ve geçmiş sonuçlar rename/enroll sırasında değişmez. Immutable `statusAtProcessing/nameAtProcessing` ile mutable `currentStatus/currentName` ayrılır.

--- agents-md-0144 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 144: ## 16. ID ve concurrency kuralları

--- agents-md-0146 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 146: Persistent ID'ler opaque UUIDv7 olur: `faceId`, `sampleId`, `processId`, `videoId`, `jobId`, `trackId`, `trackletId`. Her HTTP çağrısı `requestId`; business operation `processId`; async video execution `jobId` taşır.

--- agents-md-0148 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 148: Retry için `Idempotency-Key` desteklenir. Aynı key duplicate process, face identity, MinIO object veya Qdrant point oluşturamaz. Concurrent same-unknown race'i için ikinci vector search, bounded lock/reconciliation ve merge yolu bulunur.

--- agents-md-0150 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 150: ## 17. PostgreSQL ownership

--- agents-md-0152 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 152: PostgreSQL authoritative business source-of-truth'tür. En az identity/sample/process/result/event/inference-profile ve video asset/job/person/tracklet/appearance/timeline-index/outbox lifecycle'larını taşır. Embedding ve image/video binary PostgreSQL'e yazılmaz.

--- agents-md-0154 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 154: Historical recognition result immutable snapshot'tır. Current face identity ayrı projection'dır. Uzun per-frame timeline tek JSONB row'a veya unbounded API response'a gömülmez.

--- agents-md-0156 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 156: ## 18. Video job state ve cancellation

--- agents-md-0158 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 158: `active` boolean tek source-of-truth olamaz. En az şu state'ler bulunur:

--- agents-md-0160 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 160: ```text

--- agents-md-0161 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 161: pending, processing, cancelling, completed, failed, cancelled

--- agents-md-0162 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 162: ```

--- agents-md-0164 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 164: `cancellation_requested_at`, `lease_owner`, `lease_expires_at`, `heartbeat_at`, `attempt_no` tutulur. İstenirse `is_active = state IN (pending, processing, cancelling)` derived/generated alanı veya partial index olarak eklenebilir.

--- agents-md-0166 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 166: Worker kısa transaction içinde `FOR UPDATE SKIP LOCKED` ile claim eder, state/lease yazar ve lock'u bırakır. GPU işi boyunca DB transaction/row lock tutulmaz. Cancel ancak native process gerçekten durup resource cleanup tamamlandıktan sonra `cancelled` olur.

--- agents-md-0168 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 168: ## 19. MinIO ownership

--- agents-md-0170 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 170: MinIO binary object owner'dır: input images, original videos, selected face crops, timeline/evidence artifacts. Object key'ler yalnız opaque ID ve teknik segment taşır; name/metadata/secrets içermez.

--- agents-md-0172 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 172: Worker yalnız finalize edilmiş, stat/size/checksum doğrulanmış canonical video objesini işler. Browser stream'i worker ve MinIO'ya iki kez tee edilmez. Video, image ve face-sample retention sınıfları ayrıdır. Source video TTL ile silinirken persistent identity sample crop'u kendiliğinden silinmez.

--- agents-md-0174 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 174: ## 20. Qdrant ownership

--- agents-md-0176 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 176: Qdrant derived ve rebuildable embedding index'tir. Point ID tam olarak `face_sample.sample_id`; vector 512-D; payload yalnız `sample_id`, `face_id`, active flag ve model/preprocess version gibi teknik alanlar taşır.

--- agents-md-0178 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 178: Qdrant name/metadata/history sahibi değildir. Search sonucu final karardan önce PostgreSQL identity/sample lifecycle ile doğrulanır. Collection model-versioned olur; model migration dual-read/dual-write veya rebuild planı olmadan yapılmaz.

--- agents-md-0180 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 180: ## 21. Cross-store consistency

--- agents-md-0182 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 182: PostgreSQL, MinIO ve Qdrant tek transaction paylaşmaz. Yeni sample akışı idempotent state machine'dir:

--- agents-md-0184 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 184: ```text

--- agents-md-0185 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 185: PG reserve pending_blob

--- agents-md-0186 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 186: -> deterministic MinIO upload + SHA verify

--- agents-md-0187 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 187: -> PG blob_ready + outbox

--- agents-md-0188 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 188: -> Qdrant idempotent upsert(sampleId)

--- agents-md-0189 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 189: -> PG indexed/active

--- agents-md-0190 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 190: -> result finalize

--- agents-md-0191 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 191: ```

--- agents-md-0193 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 193: Qdrant index tamamlanmadan sample recognition-ready görünmez. Partial failure için retry, compensation, orphan scan ve reconciliation integration testleri zorunludur.

--- agents-md-0195 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 195: ## 22. Image recognition workflow

--- agents-md-0197 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 197: `POST /faces/recognize` bütün yüzleri bağımsız işler. Invalid/corrupt/empty input structured error; no-face başarılı `faceCount=0` sonucudur. Her detection canonical align/embed/search/lifecycle validation'dan geçer.

--- agents-md-0199 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 199: Existing known aynı `faceId/known`; existing unnamed aynı `faceId/anonymous`; no valid match persistent sample tamamlandıktan sonra `new_anonymous` döner. Mixed known/anonymous/new-anonymous tek response'ta desteklenir.

--- agents-md-0201 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 201: ## 23. Enrollment, update ve delete

--- agents-md-0203 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 203: Enrollment iki explicit mode taşır:

--- agents-md-0205 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 205: 1. New identity: image + name + metadata; 0 yüz error, 2+ yüz explicit policy/error.

--- agents-md-0206 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 206: 2. Existing anonymous promotion: `faceId + name + metadata`; aynı faceId korunur.

--- agents-md-0208 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 208: Bir identity çok sayıda sample taşıyabilir. Update optimistic version kullanır. Delete önce identity'yi search dışında bırakır, sonra outbox ile Qdrant/MinIO cleanup yapar. History gereği hard cascade varsayılmaz; tombstone ve privacy policy açıkça tanımlanır. Duplicate identity merge canonical redirect ve audit ile yapılır.

--- agents-md-0210 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 210: ## 24. Video upload ve retention

--- agents-md-0212 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 212: API direct multipart video kabul etmeye devam eder. Internal UI büyük dosyada presigned multipart upload kullanabilir. Upload complete idempotent olur; backend container/codec, boyut, süre, checksum ve readability doğrular; job yalnız bundan sonra queued olur.

--- agents-md-0214 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 214: Browser local `File` için object URL ile anında preview gösterebilir. Refresh sonrası private MinIO object kısa ömürlü signed Range URL veya authorized proxy ile oynatılır. Incomplete multipart explicit abort ve stale cleanup ile temizlenir.

--- agents-md-0216 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 216: ## 25. Sampling, zaman ve bbox contract'ı

--- agents-md-0218 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 218: Sampling `every_n_frames` veya `frames_per_second` olarak request/config üzerinden seçilebilir. İlk correctness fixture'larında every-frame kullanılır. Canonical zaman integer `pts_ns + time_base`; `frame/fps` tek zaman kaynağı değildir.

--- agents-md-0220 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 220: BBox canonical formatı ve inclusivity dondurulur; API original display-space pixel koordinatı döndürür. Rotation, sample/display aspect ratio, letterbox ve downscale reverse mapping test edilir. Interpolated/held overlay actual detection gibi sunulmaz; provenance alanı taşır.

--- agents-md-0222 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 222: ## 26. Video aggregation

--- agents-md-0224 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 224: Frame-level evidence doğrudan final identity değildir. Tracklet boyunca quality-selected, temporally diverse embedding'ler robust/quality-weighted şekilde birleştirilir; top-1, top-2, margin, threshold ve kullanılan evidence kaydedilir.

--- agents-md-0226 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 226: `firstSeen/lastSeen` PTS tabanlıdır. `totalDuration`, appearance interval toplamıdır; aradaki görünmediği süreyi kapsamaz. Person-level result faceId, public trackId, raw tracklet listesi, appearances ve processed-frame detections erişimi taşır.

--- agents-md-0228 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 228: ## 27. Crop ve sample politikası

--- agents-md-0230 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 230: Her frame crop olarak saklanmaz. İlk baseline: canonical video identity başına en fazla 5 candidate ve en fazla 3 aktif başlangıç sample; exact değer config/calibration ile belirlenir.

--- agents-md-0232 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 232: Minimum face size, blur, pose, occlusion, landmark geometry, alignment residual, detector score ve temporal diversity değerlendirilir. L2-normalized ArcFace embedding normu image quality olarak kullanılamaz. Existing known identity'ye video sample otomatik eklemek gallery poisoning riski nedeniyle ayrı güçlü gate gerektirir.

--- agents-md-0234 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 234: ## 28. Internal UI ve overlay extension

--- agents-md-0236 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 236: UI ayrı service/container'dır ve backend API olmadan çalışmaz. Product playback original video üzerinde Canvas/SVG overlay'dir. `requestVideoFrameCallback().metadata.mediaTime`, `ResizeObserver`, DPR, fullscreen, seek, playback-rate, VFR ve `object-fit` offset'leri test edilir.

--- agents-md-0238 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 238: İsim her detection record'una bake edilmez. Immutable timeline `trackId/bbox/PTS`; mutable identity map `faceId/currentName/currentStatus/version` taşır. Rename sonrası eski video yeniden render edilmeden yeni isim gösterir. Bütün timeline tek seferde yüklenmez; zaman chunk'ları prefetch edilir.

--- agents-md-0240 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 240: ## 29. API, process, log ve history

--- agents-md-0242 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 242: Versioned OpenAPI contract zorunludur. Requirement endpointleri korunur: image recognize/enroll/detail/delete/history/process ve video recognize/job/status/result/cancel/appearances. Ek upload/playback/timeline/SSE endpointleri extension olarak açıkça dokümante edilir.

--- agents-md-0244 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 244: Process record, result, identity ve job persistence mandatory business data'dır. Yalnız auxiliary diagnostic logging/metrics best-effort olabilir. Logger failure ana inference'ı bozmaz; result persistence failure başarı gibi dönemez. History pagination ve immutable snapshot/current projection ayrımını destekler.

--- agents-md-0246 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 246: ## 30. Security ve privacy baseline

--- agents-md-0248 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 248: Input image/video ve face crop biyometrik veridir. Bucket'lar private, signed URL kısa ömürlü, service credentials least-privilege olur. Name/metadata object key, Qdrant payload, raw logs ve error response'a sızmaz. Enrollment/update/delete/merge authorization gerektirir.

--- agents-md-0250 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 250: Upload content type'a güvenilmez; gerçek container/codec/decode probe edilir. Size, duration, pixel/decompression ve concurrency limitleri config'ten gelir. Secrets hardcode/default boş olamaz. Qdrant public network'e auth/TLS olmadan açılmaz.

--- agents-md-0252 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 252: ## 31. Test-driven development

--- agents-md-0254 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 254: Production behavior ve bug fix sırası:

--- agents-md-0256 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 256: 1. Failing test veya minimal reproducer.

--- agents-md-0257 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 257: 2. Minimum implementation.

--- agents-md-0258 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 258: 3. Targeted unit test.

--- agents-md-0259 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 259: 4. Integration/contract test.

--- agents-md-0260 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 260: 5. Gerçek PostgreSQL/MinIO/Qdrant veya GPU runtime smoke.

--- agents-md-0261 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 261: 6. Lint/type/build.

--- agents-md-0262 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 262: 7. Diff/scope/review.

--- agents-md-0264 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 264: Mock, build, plugin registration, engine deserialize veya file existence gerçek runtime/correctness kanıtı değildir.

--- agents-md-0266 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 266: ## 32. Debugging, verification ve benchmark

--- agents-md-0268 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 268: Runtime failure'da `systematic-debugging` uygulanır: stuck stage belirlenir, buffer/meta/tensor/frame/PTS ve process lifetime gözlemlenir; rastgele timeout/pool/threshold değiştirilmez. Hung container/process temizlenir ve GPU allocation'ın process lifetime mı leak mi olduğu ayrılır.

--- agents-md-0270 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 270: Completion öncesi `verification-before-completion` uygulanır. Benchmark warmup, tekrar, median/p95, hardware UUID, engine SHA, config ve raw JSON report içerir. CPU tracker replay, GPU observation, storage-disabled E2E ve full E2E ayrı benchmark'lanır.

--- agents-md-0272 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 272: ## 33. Reference-first ve provenance

--- agents-md-0274 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 274: Implementasyon model hafızasından yazılmaz. Önce `opensourcereferences/references.md` içinden official docs ve pinned upstream source seçilir; ilgili gerçek symbol/call path okunur; sonra failing test yazılır.

--- agents-md-0276 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 276: Adapte edilen her source için URL, commit/tag, erişim tarihi, repository/per-file license, adapte edilen symbol, yapılan değişiklik, reddedilen alternatif ve local parity gate `REFERENCE_DECISION_LOG.md` içine yazılır. Paper veya README tek başına production contract değildir. Code license ile model-weight license ayrı doğrulanır.

--- agents-md-0278 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 278: ## 34. MCP ve skill accountability

--- agents-md-0280 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 280: Yeni sprint/multi-file discovery'de `codebase-memory-mcp`; version-sensitive library davranışında `context7`; upstream repository mimarisi/symbol path'inde `deepwiki`; eksik/current primary source aramasında `exa`; API runtime acceptance'ta `postman`; gerçek UI E2E'de `playwright` kullanılır. GitHub plugin/MCP varsa aktif repo ve upstream source doğrulamasında tercih edilir. `21st` ve Ruflo kullanılmaz.

--- agents-md-0282 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 282: Skill sırası göreve göre uygulanır:

--- agents-md-0284 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 284: - `using-superpowers`: workflow governance;

--- agents-md-0285 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 285: - `brainstorming`: yeni architecture/product kararları;

--- agents-md-0286 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 286: - `writing-plans`: multi-file implementation planı;

--- agents-md-0287 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 287: - `executing-plans`: onaylanmış plan;

--- agents-md-0288 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 288: - `test-driven-development`: production behavior;

--- agents-md-0289 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 289: - `systematic-debugging`: failure/root cause;

--- agents-md-0290 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 290: - `verification-before-completion`: bütün completion claim'leri;

--- agents-md-0291 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 291: - `receiving-code-review` / `requesting-code-review`: review lifecycle.

--- agents-md-0293 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 293: Finalde her MCP ve kullanılan skill için gerçekten ne yaptığı veya neden skipped olduğu yazılır. Çağrılmayan araç `used` gösterilmez.

--- agents-md-0295 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 295: ## 35. Sprint, review ve completion sözleşmesi

--- agents-md-0297 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 297: Her sprint cohesive, çalışan bir vertical outcome veya açık teknik gate üretir. Report-only sprint açılmaz. Sprint sonunda `CURRENT_SPRINT.md` ve `IMPLEMENTATION_DETAILS.md` güncellenir; meaningful implementation için `docs/implementation/review_packages/SPRINT-<NNN>-CODE-REVIEW-PACKAGE.md` hazırlanır.

--- agents-md-0299 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 299: Completion verdict yalnız `PASS`, `PARTIAL`, `BLOCKED` veya `NOT_TESTED` olur. Final cevapta çalışan kullanıcı davranışı, exact validation komutları, raw sonuç özeti, changed-source map, known limitations, MCP/skill accountability ve tek önerilen sonraki sprint bulunur. Kanıtsız `production-ready`, `GPU-only`, `600 FPS`, `fully optimized` veya `accuracy verified` denmez.

--- agents-md-0301 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 301: Sistem önce görüntülerde, ardından videolarda çoklu yüz tespiti ve kalıcı yüz kimliği üretir. Her detected yüzün immutable bir `faceId` değeri olur. İlk karşılaşma `new_anonymous`, daha sonraki eşleşme `anonymous`, aynı `faceId` isimlendirildikten sonra `known` sonucunu üretir. Video genişletmesi aynı identity karar motorunu kullanır; tracking yalnız temporal sürekliliği, recognition kalıcı kimliği cevaplar.

--- agents-md-0303 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 303: ## 2. Source-of-truth sırası

--- agents-md-0305 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 305: Çelişki halinde sıra şöyledir:

--- agents-md-0307 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 307: 1. Kullanıcının güncel açık kararı.

--- agents-md-0308 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 308: 2. `requirements/ProjectRequirements.md` içindeki image/identity gereksinimleri.

--- agents-md-0309 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 309: 3. `requirements/videorequirements.md` içindeki additive video gereksinimleri.

--- agents-md-0310 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 310: 4. Kullanıcı tarafından onaylanmış architecture/ADR belgeleri.

--- agents-md-0311 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 311: 5. `docs/implementation/CURRENT_SPRINT.md`.

--- agents-md-0312 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 312: 6. Official vendor documentation ve pinned upstream source.

--- agents-md-0313 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 313: 7. Eski repository kodları ve raporları yalnız lessons-learned kaynağıdır.

--- agents-md-0315 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 315: Eski client özetleri; Oracle, 10M kişi, national ID veya başka ek kapsamları kullanıcı yeniden onaylamadıkça ürün requirement'ı sayılmaz.

--- agents-md-0317 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 317: ## 3. Onaylanmış ürün sınırı

--- agents-md-0319 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 319: Backend bağımsız ve API-first çalışır. Internal React UI, kullanıcı tarafından istenen kontrollü bir extension'dır; backend olmadan çalışamaz ve business/ML/storage mantığı içermez. Product output yeniden encode edilmiş annotated MP4 değil, orijinal video + zaman senkronlu overlay metadata'sıdır. Annotated MP4 yalnız debug/acceptance artifact'ı olarak ayrıca istenirse üretilebilir.

--- agents-md-0321 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 321: ## 4. Zorunlu implementation sırası

--- agents-md-0323 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 323: Sıra atlanmaz:

--- agents-md-0325 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 325: 1. Requirement/contract/ERD/state-machine freeze.

--- agents-md-0326 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 326: 2. PostgreSQL + MinIO + Qdrant foundation.

--- agents-md-0327 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 327: 3. Image `new_anonymous -> anonymous` vertical slice.

--- agents-md-0328 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 328: 4. Aynı `faceId` ile enrollment -> `known`, update/delete/history.

--- agents-md-0329 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 329: 5. Video upload, retention, async job, cancel/retry.

--- agents-md-0330 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 330: 6. Native GPU video observation extraction.

--- agents-md-0331 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 331: 7. Python temporal tracking ve identity reconciliation.

--- agents-md-0332 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 332: 8. Video best-shot/new-anonymous persistence.

--- agents-md-0333 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 333: 9. Person-level aggregation, appearances ve overlay API.

--- agents-md-0334 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 334: 10. Internal UI, hardening, performance ve full E2E.

--- agents-md-0336 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 336: Image identity lifecycle gerçek storage üzerinde geçmeden video recognition PASS ilan edilmez.

--- agents-md-0338 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 338: ## 5. Her görevde zorunlu başlangıç

--- agents-md-0340 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 340: Kod veya doküman değiştirmeden önce:

--- agents-md-0342 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 342: 1. Repository root, branch, HEAD ve `git status --short` doğrulanır.

--- agents-md-0343 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 343: 2. Bu dosya tamamen okunur.

--- agents-md-0344 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 344: 3. `CURRENT_SPRINT.md`, ilgili requirements, architecture ve testler okunur.

--- agents-md-0345 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 345: 4. Dirty worktree'deki kullanıcı değişiklikleri belirlenir ve korunur.

--- agents-md-0346 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 346: 5. Multi-file/sprint işinde `codebase-memory-mcp` ile gerçek caller/callee ve test path'leri keşfedilir; filesystem source ile doğrulanır.

--- agents-md-0347 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 347: 6. İlgili official docs/upstream source `opensourcereferences/references.md` üzerinden seçilir.

--- agents-md-0348 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 348: 7. Plan current sprint ile çelişiyorsa implementation başlatılmaz.

--- agents-md-0350 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 350: ## 6. Repository ve değişiklik güvenliği

--- agents-md-0352 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 352: Aktif repository dışında yazma yapılmaz. Eski repository'ler read-only referanstır. Kullanıcı istemeden `git add`, commit, push, merge, history rewrite, tracked file silme, model/dataset indirme, Docker volume silme veya system CUDA/driver değişikliği yapılmaz. Machine-specific absolute runtime path production source'a yazılmaz.

--- agents-md-0354 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 354: ## 7. Katman sınırı

--- agents-md-0356 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 356: - Python control plane: FastAPI, contracts, domain kararları, orchestration, PostgreSQL, MinIO, Qdrant, tracking, reconciliation, history.

--- agents-md-0357 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 357: - Native data plane: GStreamer/DeepStream, NVDEC/NVMM, CUDA preprocess/alignment, TensorRT detector/recognizer, compact observation emission.

--- agents-md-0358 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 358: - UI: yalnız versioned API'leri tüketir.

--- agents-md-0360 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 360: Domain outer layer'a bağımlı olmaz. API doğrudan SQL/Qdrant/MinIO/GPU çağırmaz. Infrastructure business karar sahibi değildir. Tracker kalıcı identity sahibi değildir.

--- agents-md-0362 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 362: ## 8. Production GPU hot path

--- agents-md-0364 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 364: Hedef video yolu:

--- agents-md-0366 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 366: ```text

--- agents-md-0367 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 367: encoded video

--- agents-md-0368 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 368: -> GStreamer graph

--- agents-md-0369 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 369: -> NVIDIA decoder / NVMM

--- agents-md-0370 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 370: -> DeepStream batching

--- agents-md-0371 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 371: -> CUDA/TensorRT RetinaFace

--- agents-md-0372 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 372: -> CUDA five-point alignment

--- agents-md-0373 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 373: -> TensorRT ArcFace/Glint embedding

--- agents-md-0374 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 374: -> GPU L2 normalization

--- agents-md-0375 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 375: -> compact metadata/embedding CPU boundary

--- agents-md-0376 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 376: ```

--- agents-md-0378 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 378: Production hot path'te full-frame OpenCV/PIL/NumPy decode, CPU resize, raw tensor NumPy postprocess, frame başına zorunlu device synchronize ve sessiz CPU inference fallback yasaktır.

--- agents-md-0380 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 380: ## 9. GStreamer ve DeepStream kararı

--- agents-md-0382 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 382: Bu bir “GStreamer mı DeepStream mi?” seçimi değildir: DeepStream, GStreamer tabanlı NVIDIA data plane'dir. Graph GStreamer ile kurulur; `nvv4l2decoder`, `nvstreammux`, `nvdspreprocess`, `nvinfer` veya doğrulanmış custom native elementler gibi DeepStream/NVIDIA bileşenleri kullanılır.

--- agents-md-0384 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 384: İlk product graph'ında render branch, `nvstreamdemux`, OSD, encoder ve filesink zorunlu değildir. Bunlar GPU observation throughput'unu etkilememelidir. Python NVMM surface map etmez.

--- agents-md-0386 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 386: ## 10. Frame batch ve face batch ayrımı

--- agents-md-0388 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 388: - Frame batch: detector throughput'u için ardışık input frame'leri.

--- agents-md-0389 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 389: - Face batch: bir frame batch içinde bulunan değişken sayıdaki yüzlerin recognizer input'u.

--- agents-md-0391 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 391: Bu iki batch aynı şey değildir. Detector `batch=8` iken recognizer batch'i o sekiz frame'deki geçerli yüz sayısı olabilir. Dynamic TensorRT profile min/opt/max, actual batch, partial final batch ve EOS ayrı test edilir. Batch sonucu frame/PTS sırası kaybedilemez.

--- agents-md-0393 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 393: ## 11. CPU boundary ve backpressure

--- agents-md-0395 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 395: CPU'ya yalnız şu kompakt kayıtlar çıkar:

--- agents-md-0397 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 397: - source ID, frame index, PTS/time base;

--- agents-md-0398 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 398: - original-resolution bbox ve landmarks;

--- agents-md-0399 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 399: - detector score ve quality metrikleri;

--- agents-md-0400 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 400: - 512-D embedding veya seçilmiş embedding evidence;

--- agents-md-0401 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 401: - model/preprocess/profile kimliği.

--- agents-md-0403 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 403: Full frame, NV12/RGBA surface, detector input tensor veya bütün raw TRT outputs CPU'ya taşınmaz. Native producer bounded ring buffer/queue kullanır. Queue davranışı açıkça `backpressure`, kontrollü drop veya job failure olarak tanımlanır; sessiz sınırsız RAM büyümesi yasaktır.

--- agents-md-0405 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 405: ## 12. Performance iddiası disiplini

--- agents-md-0407 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 407: `trtexec` engine FPS, detector-only pipeline FPS, GPU observation FPS ve full E2E FPS ayrı metriklerdir. “600 FPS” denebilmesi için hardware, video, codec, batch, sampling, tracker, recognizer, persistence, model SHA ve ölçüm kapsamı yazılmalıdır.

--- agents-md-0409 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 409: 600 FPS yaklaşık 1.67 ms/frame bütçedir. Python tracker bu bütçe içinde varsayımla PASS sayılamaz. Frozen observation replay; p50/p95/p99 latency, queue depth, backlog, drop, RSS ve sustained-duration raporu gerekir. Detector-only sayı full product performansı değildir.

--- agents-md-0411 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 411: ## 13. Python tracker sözleşmesi

--- agents-md-0413 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 413: İlk tercih Python metadata tracker'dır; C++ rewrite yalnız profiling kanıtıyla yapılır. Her source'un tracker state'i tek sıralı consumer tarafından `PTS/frame` sırasıyla mutate edilir. Batch içindeki frame'ler sırayla tracker'a verilir; batch sınırında state resetlenmez. Kaynak/job seviyesinde paralellik olabilir, aynı source tracker state'inde paralel update olamaz.

--- agents-md-0415 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 415: ByteTrack adapte edilirse low-score detection ikinci association aşamasına ulaşmalıdır. NMS/threshold ile bütün low-score adayları önceden silinip sonra “ByteTrack kullanılıyor” denilemez.

--- agents-md-0417 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 417: ## 14. Tracklet, track ve identity ayrımı

--- agents-md-0419 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 419: - `rawTrackletId`: Kesintisiz temporal tracker segmenti.

--- agents-md-0420 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 420: - `trackId`: Bir video içinde reconciliation sonrası canonical kişi grubu.

--- agents-md-0421 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 421: - `faceId`: Bütün image/video request'leri boyunca kalıcı biyometrik identity.

--- agents-md-0422 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 422: - `detectionId`: Tek processed-frame observation.

--- agents-md-0424 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 424: Scene cut, uzun kayıp veya yeniden giriş yeni raw tracklet üretir. Birinci ve son sahnedeki Rachel'ın aynı `faceId` olması tracker değil, embedding evidence + reconciliation sonucudur. Aynı anda görünen iki farklı yüz için cannot-link uygulanır.

--- agents-md-0426 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 426: ## 15. Identity status semantiği

--- agents-md-0428 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 428: `new_anonymous` yalnız identity'nin yaratıldığı process/job sonucudur; persistent identity type değildir.

--- agents-md-0430 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 430: ```text

--- agents-md-0431 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 431: ilk unmatched process -> result=new_anonymous, identity=anonymous

--- agents-md-0432 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 432: sonraki match          -> result=anonymous, identity=anonymous

--- agents-md-0433 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 433: aynı faceId enroll     -> sonraki result=known, identity=known

--- agents-md-0434 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 434: ```

--- agents-md-0436 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 436: `faceId`, sample ID'leri ve geçmiş sonuçlar rename/enroll sırasında değişmez. Immutable `statusAtProcessing/nameAtProcessing` ile mutable `currentStatus/currentName` ayrılır.

--- agents-md-0438 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 438: ## 16. ID ve concurrency kuralları

--- agents-md-0440 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 440: Persistent ID'ler opaque UUIDv7 olur: `faceId`, `sampleId`, `processId`, `videoId`, `jobId`, `trackId`, `trackletId`. Her HTTP çağrısı `requestId`; business operation `processId`; async video execution `jobId` taşır.

--- agents-md-0442 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 442: Retry için `Idempotency-Key` desteklenir. Aynı key duplicate process, face identity, MinIO object veya Qdrant point oluşturamaz. Concurrent same-unknown race'i için ikinci vector search, bounded lock/reconciliation ve merge yolu bulunur.

--- agents-md-0444 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 444: ## 17. PostgreSQL ownership

--- agents-md-0446 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 446: PostgreSQL authoritative business source-of-truth'tür. En az identity/sample/process/result/event/inference-profile ve video asset/job/person/tracklet/appearance/timeline-index/outbox lifecycle'larını taşır. Embedding ve image/video binary PostgreSQL'e yazılmaz.

--- agents-md-0448 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 448: Historical recognition result immutable snapshot'tır. Current face identity ayrı projection'dır. Uzun per-frame timeline tek JSONB row'a veya unbounded API response'a gömülmez.

--- agents-md-0450 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 450: ## 18. Video job state ve cancellation

--- agents-md-0452 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 452: `active` boolean tek source-of-truth olamaz. En az şu state'ler bulunur:

--- agents-md-0454 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 454: ```text

--- agents-md-0455 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 455: pending, processing, cancelling, completed, failed, cancelled

--- agents-md-0456 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 456: ```

--- agents-md-0458 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 458: `cancellation_requested_at`, `lease_owner`, `lease_expires_at`, `heartbeat_at`, `attempt_no` tutulur. İstenirse `is_active = state IN (pending, processing, cancelling)` derived/generated alanı veya partial index olarak eklenebilir.

--- agents-md-0460 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 460: Worker kısa transaction içinde `FOR UPDATE SKIP LOCKED` ile claim eder, state/lease yazar ve lock'u bırakır. GPU işi boyunca DB transaction/row lock tutulmaz. Cancel ancak native process gerçekten durup resource cleanup tamamlandıktan sonra `cancelled` olur.

--- agents-md-0462 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 462: ## 19. MinIO ownership

--- agents-md-0464 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 464: MinIO binary object owner'dır: input images, original videos, selected face crops, timeline/evidence artifacts. Object key'ler yalnız opaque ID ve teknik segment taşır; name/metadata/secrets içermez.

--- agents-md-0466 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 466: Worker yalnız finalize edilmiş, stat/size/checksum doğrulanmış canonical video objesini işler. Browser stream'i worker ve MinIO'ya iki kez tee edilmez. Video, image ve face-sample retention sınıfları ayrıdır. Source video TTL ile silinirken persistent identity sample crop'u kendiliğinden silinmez.

--- agents-md-0468 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 468: ## 20. Qdrant ownership

--- agents-md-0470 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 470: Qdrant derived ve rebuildable embedding index'tir. Point ID tam olarak `face_sample.sample_id`; vector 512-D; payload yalnız `sample_id`, `face_id`, active flag ve model/preprocess version gibi teknik alanlar taşır.

--- agents-md-0472 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 472: Qdrant name/metadata/history sahibi değildir. Search sonucu final karardan önce PostgreSQL identity/sample lifecycle ile doğrulanır. Collection model-versioned olur; model migration dual-read/dual-write veya rebuild planı olmadan yapılmaz.

--- agents-md-0474 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 474: ## 21. Cross-store consistency

--- agents-md-0476 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 476: PostgreSQL, MinIO ve Qdrant tek transaction paylaşmaz. Yeni sample akışı idempotent state machine'dir:

--- agents-md-0478 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 478: ```text

--- agents-md-0479 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 479: PG reserve pending_blob

--- agents-md-0480 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 480: -> deterministic MinIO upload + SHA verify

--- agents-md-0481 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 481: -> PG blob_ready + outbox

--- agents-md-0482 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 482: -> Qdrant idempotent upsert(sampleId)

--- agents-md-0483 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 483: -> PG indexed/active

--- agents-md-0484 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 484: -> result finalize

--- agents-md-0485 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 485: ```

--- agents-md-0487 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 487: Qdrant index tamamlanmadan sample recognition-ready görünmez. Partial failure için retry, compensation, orphan scan ve reconciliation integration testleri zorunludur.

--- agents-md-0489 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 489: ## 22. Image recognition workflow

--- agents-md-0491 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 491: `POST /faces/recognize` bütün yüzleri bağımsız işler. Invalid/corrupt/empty input structured error; no-face başarılı `faceCount=0` sonucudur. Her detection canonical align/embed/search/lifecycle validation'dan geçer.

--- agents-md-0493 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 493: Existing known aynı `faceId/known`; existing unnamed aynı `faceId/anonymous`; no valid match persistent sample tamamlandıktan sonra `new_anonymous` döner. Mixed known/anonymous/new-anonymous tek response'ta desteklenir.

--- agents-md-0495 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 495: ## 23. Enrollment, update ve delete

--- agents-md-0497 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 497: Enrollment iki explicit mode taşır:

--- agents-md-0499 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 499: 1. New identity: image + name + metadata; 0 yüz error, 2+ yüz explicit policy/error.

--- agents-md-0500 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 500: 2. Existing anonymous promotion: `faceId + name + metadata`; aynı faceId korunur.

--- agents-md-0502 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 502: Bir identity çok sayıda sample taşıyabilir. Update optimistic version kullanır. Delete önce identity'yi search dışında bırakır, sonra outbox ile Qdrant/MinIO cleanup yapar. History gereği hard cascade varsayılmaz; tombstone ve privacy policy açıkça tanımlanır. Duplicate identity merge canonical redirect ve audit ile yapılır.

--- agents-md-0504 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 504: ## 24. Video upload ve retention

--- agents-md-0506 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 506: API direct multipart video kabul etmeye devam eder. Internal UI büyük dosyada presigned multipart upload kullanabilir. Upload complete idempotent olur; backend container/codec, boyut, süre, checksum ve readability doğrular; job yalnız bundan sonra queued olur.

--- agents-md-0508 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 508: Browser local `File` için object URL ile anında preview gösterebilir. Refresh sonrası private MinIO object kısa ömürlü signed Range URL veya authorized proxy ile oynatılır. Incomplete multipart explicit abort ve stale cleanup ile temizlenir.

--- agents-md-0510 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 510: ## 25. Sampling, zaman ve bbox contract'ı

--- agents-md-0512 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 512: Sampling `every_n_frames` veya `frames_per_second` olarak request/config üzerinden seçilebilir. İlk correctness fixture'larında every-frame kullanılır. Canonical zaman integer `pts_ns + time_base`; `frame/fps` tek zaman kaynağı değildir.

--- agents-md-0514 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 514: BBox canonical formatı ve inclusivity dondurulur; API original display-space pixel koordinatı döndürür. Rotation, sample/display aspect ratio, letterbox ve downscale reverse mapping test edilir. Interpolated/held overlay actual detection gibi sunulmaz; provenance alanı taşır.

--- agents-md-0516 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 516: ## 26. Video aggregation

--- agents-md-0518 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 518: Frame-level evidence doğrudan final identity değildir. Tracklet boyunca quality-selected, temporally diverse embedding'ler robust/quality-weighted şekilde birleştirilir; top-1, top-2, margin, threshold ve kullanılan evidence kaydedilir.

--- agents-md-0520 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 520: `firstSeen/lastSeen` PTS tabanlıdır. `totalDuration`, appearance interval toplamıdır; aradaki görünmediği süreyi kapsamaz. Person-level result faceId, public trackId, raw tracklet listesi, appearances ve processed-frame detections erişimi taşır.

--- agents-md-0522 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 522: ## 27. Crop ve sample politikası

--- agents-md-0524 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 524: Her frame crop olarak saklanmaz. İlk baseline: canonical video identity başına en fazla 5 candidate ve en fazla 3 aktif başlangıç sample; exact değer config/calibration ile belirlenir.

--- agents-md-0526 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 526: Minimum face size, blur, pose, occlusion, landmark geometry, alignment residual, detector score ve temporal diversity değerlendirilir. L2-normalized ArcFace embedding normu image quality olarak kullanılamaz. Existing known identity'ye video sample otomatik eklemek gallery poisoning riski nedeniyle ayrı güçlü gate gerektirir.

--- agents-md-0528 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 528: ## 28. Internal UI ve overlay extension

--- agents-md-0530 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 530: UI ayrı service/container'dır ve backend API olmadan çalışmaz. Product playback original video üzerinde Canvas/SVG overlay'dir. `requestVideoFrameCallback().metadata.mediaTime`, `ResizeObserver`, DPR, fullscreen, seek, playback-rate, VFR ve `object-fit` offset'leri test edilir.

--- agents-md-0532 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 532: İsim her detection record'una bake edilmez. Immutable timeline `trackId/bbox/PTS`; mutable identity map `faceId/currentName/currentStatus/version` taşır. Rename sonrası eski video yeniden render edilmeden yeni isim gösterir. Bütün timeline tek seferde yüklenmez; zaman chunk'ları prefetch edilir.

--- agents-md-0534 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 534: ## 29. API, process, log ve history

--- agents-md-0536 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 536: Versioned OpenAPI contract zorunludur. Requirement endpointleri korunur: image recognize/enroll/detail/delete/history/process ve video recognize/job/status/result/cancel/appearances. Ek upload/playback/timeline/SSE endpointleri extension olarak açıkça dokümante edilir.

--- agents-md-0538 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 538: Process record, result, identity ve job persistence mandatory business data'dır. Yalnız auxiliary diagnostic logging/metrics best-effort olabilir. Logger failure ana inference'ı bozmaz; result persistence failure başarı gibi dönemez. History pagination ve immutable snapshot/current projection ayrımını destekler.

--- agents-md-0540 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 540: ## 30. Security ve privacy baseline

--- agents-md-0542 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 542: Input image/video ve face crop biyometrik veridir. Bucket'lar private, signed URL kısa ömürlü, service credentials least-privilege olur. Name/metadata object key, Qdrant payload, raw logs ve error response'a sızmaz. Enrollment/update/delete/merge authorization gerektirir.

--- agents-md-0544 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 544: Upload content type'a güvenilmez; gerçek container/codec/decode probe edilir. Size, duration, pixel/decompression ve concurrency limitleri config'ten gelir. Secrets hardcode/default boş olamaz. Qdrant public network'e auth/TLS olmadan açılmaz.

--- agents-md-0546 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 546: ## 31. Test-driven development

--- agents-md-0548 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 548: Production behavior ve bug fix sırası:

--- agents-md-0550 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 550: 1. Failing test veya minimal reproducer.

--- agents-md-0551 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 551: 2. Minimum implementation.

--- agents-md-0552 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 552: 3. Targeted unit test.

--- agents-md-0553 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 553: 4. Integration/contract test.

--- agents-md-0554 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 554: 5. Gerçek PostgreSQL/MinIO/Qdrant veya GPU runtime smoke.

--- agents-md-0555 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 555: 6. Lint/type/build.

--- agents-md-0556 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 556: 7. Diff/scope/review.

--- agents-md-0558 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 558: Mock, build, plugin registration, engine deserialize veya file existence gerçek runtime/correctness kanıtı değildir.

--- agents-md-0560 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 560: ## 32. Debugging, verification ve benchmark

--- agents-md-0562 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 562: Runtime failure'da `systematic-debugging` uygulanır: stuck stage belirlenir, buffer/meta/tensor/frame/PTS ve process lifetime gözlemlenir; rastgele timeout/pool/threshold değiştirilmez. Hung container/process temizlenir ve GPU allocation'ın process lifetime mı leak mi olduğu ayrılır.

--- agents-md-0564 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 564: Completion öncesi `verification-before-completion` uygulanır. Benchmark warmup, tekrar, median/p95, hardware UUID, engine SHA, config ve raw JSON report içerir. CPU tracker replay, GPU observation, storage-disabled E2E ve full E2E ayrı benchmark'lanır.

--- agents-md-0566 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 566: ## 33. Reference-first ve provenance

--- agents-md-0568 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 568: Implementasyon model hafızasından yazılmaz. Önce `opensourcereferences/references.md` içinden official docs ve pinned upstream source seçilir; ilgili gerçek symbol/call path okunur; sonra failing test yazılır.

--- agents-md-0570 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 570: Adapte edilen her source için URL, commit/tag, erişim tarihi, repository/per-file license, adapte edilen symbol, yapılan değişiklik, reddedilen alternatif ve local parity gate `REFERENCE_DECISION_LOG.md` içine yazılır. Paper veya README tek başına production contract değildir. Code license ile model-weight license ayrı doğrulanır.

--- agents-md-0572 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 572: ## 34. MCP ve skill accountability

--- agents-md-0574 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 574: Yeni sprint/multi-file discovery'de `codebase-memory-mcp`; version-sensitive library davranışında `context7`; upstream repository mimarisi/symbol path'inde `deepwiki`; eksik/current primary source aramasında `exa`; API runtime acceptance'ta `postman`; gerçek UI E2E'de `playwright` kullanılır. GitHub plugin/MCP varsa aktif repo ve upstream source doğrulamasında tercih edilir. `21st` ve Ruflo kullanılmaz.

--- agents-md-0576 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 576: Skill sırası göreve göre uygulanır:

--- agents-md-0578 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 578: - `using-superpowers`: workflow governance;

--- agents-md-0579 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 579: - `brainstorming`: yeni architecture/product kararları;

--- agents-md-0580 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 580: - `writing-plans`: multi-file implementation planı;

--- agents-md-0581 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 581: - `executing-plans`: onaylanmış plan;

--- agents-md-0582 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 582: - `test-driven-development`: production behavior;

--- agents-md-0583 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 583: - `systematic-debugging`: failure/root cause;

--- agents-md-0584 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 584: - `verification-before-completion`: bütün completion claim'leri;

--- agents-md-0585 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 585: - `receiving-code-review` / `requesting-code-review`: review lifecycle.

--- agents-md-0587 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 587: Finalde her MCP ve kullanılan skill için gerçekten ne yaptığı veya neden skipped olduğu yazılır. Çağrılmayan araç `used` gösterilmez.

--- agents-md-0589 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 589: ## 35. Sprint, review ve completion sözleşmesi

--- agents-md-0591 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 591: Her sprint cohesive, çalışan bir vertical outcome veya açık teknik gate üretir. Report-only sprint açılmaz. Sprint sonunda `CURRENT_SPRINT.md` ve `IMPLEMENTATION_DETAILS.md` güncellenir; meaningful implementation için `docs/implementation/review_packages/SPRINT-<NNN>-CODE-REVIEW-PACKAGE.md` hazırlanır.

--- agents-md-0593 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 593: Completion verdict yalnız `PASS`, `PARTIAL`, `BLOCKED` veya `NOT_TESTED` olur. Final cevapta çalışan kullanıcı davranışı, exact validation komutları, raw sonuç özeti, changed-source map, known limitations, MCP/skill accountability ve tek önerilen sonraki sprint bulunur. Kanıtsız `production-ready`, `GPU-only`, `600 FPS`, `fully optimized` veya `accuracy verified` denmez.

--- agents-md-0594 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 594: ## 36. Memory / context kullanımı — sadece açık talimatla

--- agents-md-0596 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 596: Memory araçları (prompt-memory-mcp, mem0-mcp vb.) yalnızca kullanıcı açıkça "bunu hatırla", "kaydet" veya "memory'ye al" dediğinde çalışır. Otomatik snapshot, periyodik retrieve veya session başı context dump yapma.

--- agents-md-0598 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 598: ### Kurallar

--- agents-md-0600 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 600: 1. **Açık talimat yoksa çağırma**: Session başlangıcında, her turda, compaction'da veya sabit aralıklarla otomatik `store_memory` / `search_memory` / `index_sessions` çağrısı yapma.

--- agents-md-0601 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 601: 2. **Kullanıcı "hatırla/kaydet" dediğinde**: Kısa ve öz bir `Memory` veya `Decision` kaydet; gereksiz uzunlukta olmasın.

--- agents-md-0602 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 602: 3. **Araç seçimi**: `prompt-memory-mcp` Claude Code tarafında açık kalır. OpenCode tarafında kapalıdır. Kullanıcı hangi MCP'yi işaret ederse onu kullan; aktif olmayanı zorla çalıştırma.

--- agents-md-0604 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 604: Bu kural kesindir; otomatik hafıza çağrıları atlanmadan devam edilemez.

--- agents-md-0606 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 606: ## 37. Context recovery (compaction / startup / resume)

--- agents-md-0608 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 608: Session `startup`, `resume` ve `/compact` sonrası context sıfırlanma riski vardır. Bu nedenle `~/.claude/settings.json` içinde `SessionStart` hook'uyla `get-full-context-after-compact` otomatik çalıştırılır. Hook şunları yapar:

--- agents-md-0610 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 610: 1. `python3 /home/user/.claude/skills/get_full_context/get_full_context.py` çalıştırılır; DB'deki tüm kullanıcı mesajları, Decision/özet node'ları ve dosya başlıkları konuşmaya eklenir.

--- agents-md-0611 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 611: 2. Tam digest ayrıca `/home/user/.claude/projects/-home-user-Workspace-MergenVisionPhase2v2/get_full_context_latest.txt` dosyasına kaydedilir.

--- agents-md-0612 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 612: 3. Kod/implementation bağlamı gerekiyorsa `mcp__codebase-memory-mcp__get_architecture(project="home-user-Workspace-MergenVisionPhase2v2", aspects=["overview"])` çağrılır.

--- agents-md-0614 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 614: 

--- agents-md-0002 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 2: 

--- agents-md-0004 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 4: 

--- agents-md-0006 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 6: 

--- agents-md-0008 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 8: 

--- agents-md-0010 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 10: 

--- agents-md-0012 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 12: 

--- agents-md-0020 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 20: 

--- agents-md-0022 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 22: 

--- agents-md-0024 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 24: 

--- agents-md-0026 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 26: 

--- agents-md-0028 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 28: 

--- agents-md-0030 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 30: 

--- agents-md-0041 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 41: 

--- agents-md-0043 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 43: 

--- agents-md-0045 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 45: 

--- agents-md-0047 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 47: 

--- agents-md-0055 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 55: 

--- agents-md-0057 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 57: 

--- agents-md-0059 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 59: 

--- agents-md-0061 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 61: 

--- agents-md-0065 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 65: 

--- agents-md-0067 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 67: 

--- agents-md-0069 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 69: 

--- agents-md-0071 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 71: 

--- agents-md-0083 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 83: 

--- agents-md-0085 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 85: 

--- agents-md-0087 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 87: 

--- agents-md-0089 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 89: 

--- agents-md-0091 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 91: 

--- agents-md-0093 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 93: 

--- agents-md-0096 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 96: 

--- agents-md-0098 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 98: 

--- agents-md-0100 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 100: 

--- agents-md-0102 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 102: 

--- agents-md-0108 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 108: 

--- agents-md-0110 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 110: 

--- agents-md-0112 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 112: 

--- agents-md-0114 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 114: 

--- agents-md-0116 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 116: 

--- agents-md-0118 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 118: 

--- agents-md-0120 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 120: 

--- agents-md-0122 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 122: 

--- agents-md-0124 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 124: 

--- agents-md-0129 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 129: 

--- agents-md-0131 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 131: 

--- agents-md-0133 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 133: 

--- agents-md-0135 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 135: 

--- agents-md-0141 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 141: 

--- agents-md-0143 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 143: 

--- agents-md-0145 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 145: 

--- agents-md-0147 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 147: 

--- agents-md-0149 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 149: 

--- agents-md-0151 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 151: 

--- agents-md-0153 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 153: 

--- agents-md-0155 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 155: 

--- agents-md-0157 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 157: 

--- agents-md-0159 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 159: 

--- agents-md-0163 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 163: 

--- agents-md-0165 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 165: 

--- agents-md-0167 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 167: 

--- agents-md-0169 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 169: 

--- agents-md-0171 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 171: 

--- agents-md-0173 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 173: 

--- agents-md-0175 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 175: 

--- agents-md-0177 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 177: 

--- agents-md-0179 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 179: 

--- agents-md-0181 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 181: 

--- agents-md-0183 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 183: 

--- agents-md-0192 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 192: 

--- agents-md-0194 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 194: 

--- agents-md-0196 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 196: 

--- agents-md-0198 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 198: 

--- agents-md-0200 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 200: 

--- agents-md-0202 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 202: 

--- agents-md-0204 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 204: 

--- agents-md-0207 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 207: 

--- agents-md-0209 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 209: 

--- agents-md-0211 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 211: 

--- agents-md-0213 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 213: 

--- agents-md-0215 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 215: 

--- agents-md-0217 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 217: 

--- agents-md-0219 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 219: 

--- agents-md-0221 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 221: 

--- agents-md-0223 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 223: 

--- agents-md-0225 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 225: 

--- agents-md-0227 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 227: 

--- agents-md-0229 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 229: 

--- agents-md-0231 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 231: 

--- agents-md-0233 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 233: 

--- agents-md-0235 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 235: 

--- agents-md-0237 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 237: 

--- agents-md-0239 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 239: 

--- agents-md-0241 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 241: 

--- agents-md-0243 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 243: 

--- agents-md-0245 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 245: 

--- agents-md-0247 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 247: 

--- agents-md-0249 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 249: 

--- agents-md-0251 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 251: 

--- agents-md-0253 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 253: 

--- agents-md-0255 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 255: 

--- agents-md-0263 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 263: 

--- agents-md-0265 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 265: 

--- agents-md-0267 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 267: 

--- agents-md-0269 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 269: 

--- agents-md-0271 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 271: 

--- agents-md-0273 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 273: 

--- agents-md-0275 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 275: 

--- agents-md-0277 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 277: 

--- agents-md-0279 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 279: 

--- agents-md-0281 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 281: 

--- agents-md-0283 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 283: 

--- agents-md-0292 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 292: 

--- agents-md-0294 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 294: 

--- agents-md-0296 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 296: 

--- agents-md-0298 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 298: 

--- agents-md-0300 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 300: 

--- agents-md-0302 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 302: 

--- agents-md-0304 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 304: 

--- agents-md-0306 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 306: 

--- agents-md-0314 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 314: 

--- agents-md-0316 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 316: 

--- agents-md-0318 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 318: 

--- agents-md-0320 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 320: 

--- agents-md-0322 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 322: 

--- agents-md-0324 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 324: 

--- agents-md-0335 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 335: 

--- agents-md-0337 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 337: 

--- agents-md-0339 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 339: 

--- agents-md-0341 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 341: 

--- agents-md-0349 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 349: 

--- agents-md-0351 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 351: 

--- agents-md-0353 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 353: 

--- agents-md-0355 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 355: 

--- agents-md-0359 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 359: 

--- agents-md-0361 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 361: 

--- agents-md-0363 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 363: 

--- agents-md-0365 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 365: 

--- agents-md-0377 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 377: 

--- agents-md-0379 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 379: 

--- agents-md-0381 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 381: 

--- agents-md-0383 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 383: 

--- agents-md-0385 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 385: 

--- agents-md-0387 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 387: 

--- agents-md-0390 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 390: 

--- agents-md-0392 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 392: 

--- agents-md-0394 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 394: 

--- agents-md-0396 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 396: 

--- agents-md-0402 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 402: 

--- agents-md-0404 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 404: 

--- agents-md-0406 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 406: 

--- agents-md-0408 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 408: 

--- agents-md-0410 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 410: 

--- agents-md-0412 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 412: 

--- agents-md-0414 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 414: 

--- agents-md-0416 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 416: 

--- agents-md-0418 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 418: 

--- agents-md-0423 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 423: 

--- agents-md-0425 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 425: 

--- agents-md-0427 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 427: 

--- agents-md-0429 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 429: 

--- agents-md-0435 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 435: 

--- agents-md-0437 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 437: 

--- agents-md-0439 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 439: 

--- agents-md-0441 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 441: 

--- agents-md-0443 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 443: 

--- agents-md-0445 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 445: 

--- agents-md-0447 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 447: 

--- agents-md-0449 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 449: 

--- agents-md-0451 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 451: 

--- agents-md-0453 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 453: 

--- agents-md-0457 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 457: 

--- agents-md-0459 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 459: 

--- agents-md-0461 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 461: 

--- agents-md-0463 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 463: 

--- agents-md-0465 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 465: 

--- agents-md-0467 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 467: 

--- agents-md-0469 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 469: 

--- agents-md-0471 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 471: 

--- agents-md-0473 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 473: 

--- agents-md-0475 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 475: 

--- agents-md-0477 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 477: 

--- agents-md-0486 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 486: 

--- agents-md-0488 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 488: 

--- agents-md-0490 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 490: 

--- agents-md-0492 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 492: 

--- agents-md-0494 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 494: 

--- agents-md-0496 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 496: 

--- agents-md-0498 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 498: 

--- agents-md-0501 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 501: 

--- agents-md-0503 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 503: 

--- agents-md-0505 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 505: 

--- agents-md-0507 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 507: 

--- agents-md-0509 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 509: 

--- agents-md-0511 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 511: 

--- agents-md-0513 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 513: 

--- agents-md-0515 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 515: 

--- agents-md-0517 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 517: 

--- agents-md-0519 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 519: 

--- agents-md-0521 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 521: 

--- agents-md-0523 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 523: 

--- agents-md-0525 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 525: 

--- agents-md-0527 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 527: 

--- agents-md-0529 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 529: 

--- agents-md-0531 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 531: 

--- agents-md-0533 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 533: 

--- agents-md-0535 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 535: 

--- agents-md-0537 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 537: 

--- agents-md-0539 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 539: 

--- agents-md-0541 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 541: 

--- agents-md-0543 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 543: 

--- agents-md-0545 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 545: 

--- agents-md-0547 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 547: 

--- agents-md-0549 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 549: 

--- agents-md-0557 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 557: 

--- agents-md-0559 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 559: 

--- agents-md-0561 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 561: 

--- agents-md-0563 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 563: 

--- agents-md-0565 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 565: 

--- agents-md-0567 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 567: 

--- agents-md-0569 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 569: 

--- agents-md-0571 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 571: 

--- agents-md-0573 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 573: 

--- agents-md-0575 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 575: 

--- agents-md-0577 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 577: 

--- agents-md-0586 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 586: 

--- agents-md-0588 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 588: 

--- agents-md-0590 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 590: 

--- agents-md-0592 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 592: 

--- agents-md-0595 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 595: 

--- agents-md-0597 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 597: 

--- agents-md-0599 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 599: 

--- agents-md-0603 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 603: 

--- agents-md-0605 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 605: 

--- agents-md-0607 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 607: 

--- agents-md-0609 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 609: 

--- agents-md-0613 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 613: 4. Elde edilen özet üzerinden konuşmaya kaldığı yerden devam edilir.

--- agents-md-0615 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 615: **Hedefli büyük retrieval:** Eğer belirli bir konuya ait çok sayıda satır (örneğin 1000 satırlık bir prompt) DB'den tek seferde çekilmek isteniyorsa `mcp__prompt-memory__search_memory` yerine `mcp__prompt-memory__retrieve_memory` tool’u, `/recall` skill’i veya doğrudan SQLite script'i kullanılır; MCP limitlerine ve sayfalamaya takılmaz:

--- agents-md-0616 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 616: 

--- agents-md-0617 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 617: ```text

--- agents-md-0618 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 618: mcp__prompt-memory__retrieve_memory(query="MergenVision")

--- agents-md-0619 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 619: mcp__prompt-memory__retrieve_memory(tag="MergenVision", label="Memory")

--- agents-md-0620 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 620: mcp__prompt-memory__retrieve_memory(query="worker heartbeat")

--- agents-md-0621 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 621: ```

--- agents-md-0622 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 622: 

--- agents-md-0623 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 623: Claude slash command olarak:

--- agents-md-0624 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 624: 

--- agents-md-0625 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 625: ```bash

--- agents-md-0626 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 626: /recall MergenVision

--- agents-md-0627 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 627: /recall --tag MergenVision

--- agents-md-0628 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 628: /recall --query "worker heartbeat"

--- agents-md-0629 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 629: /recall --tag MergenVision --output /tmp/tum_memoryler.txt

--- agents-md-0630 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 630: ```

--- agents-md-0631 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 631: 

--- agents-md-0632 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 632: Slash command arkasında şu script çalışır:

--- agents-md-0633 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 633: `python3 /home/user/.claude/skills/get_full_context/prompt_memory_retrieve_all.py <args>`

--- agents-md-0634 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 634: 

--- agents-md-0635 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 635: Ayrıca `python3 /home/user/.claude/skills/get_full_context/get_full_context.py "MergenVision faceId"` komutu FTS5 ile eşleşen tüm node'ları tam içerikleriyle döndürür.

--- agents-md-0636 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 636: 

--- agents-md-0637 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 637: Bu kural Section 36'daki "açık talimat" kuralının bir istisnasıdır; context recovery açıkça AGENTS.md ile zorunlu tutulmuştur.

--- agents-md-0638 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, AGENTS.md — Engineering Constitution ---
Dosya: AGENTS.md | Satır 638: 

--- tool-prompt-memory-retrieval [Memory] tags: compaction-retrieve, tools, MergenVision, prompt-memory ---
Prompt-memory'den büyük içerikleri tek seferde çekmek için kullanılan araçlar:

1. `python3 /home/user/.claude/skills/get_full_context/get_full_context.py`
   - Argümansız: tüm kullanıcı mesajları, karar/özet node'ları, dosya başlıklarını döndürür.
   - Argümanlı (ör. `"MergenVision faceId"`): FTS5 ile eşleşen tüm node'ları tam içerikleriyle döndürür, limit 10.000.

2. `python3 /home/user/.claude/skills/get_full_context/prompt_memory_retrieve_all.py`
   - Doğrudan SQLite/FTS5 okur, MCP limitlerine takılmaz, içeriği kesmez.
   - Parametreler:
     - `--query "kelime1 kelime2"` — FTS5 araması
     - `--tag MergenVision` — tag filtresi
     - `--label Memory|Decision` — label filtresi
     - `--limit N` — default 10.000
     - `--output /tmp/dosya.txt` — dosyaya kaydet

3. Hook: `~/.claude/hooks/get-full-context-after-compact`
   - `startup`, `resume`, `compact` sonrası otomatik çalışır.
   - `get_full_context.py` çıktısını konuşmaya basar ve `/home/user/.claude/projects/-home-user-Workspace-MergenVisionPhase2v2/get_full_context_latest.txt` dosyasına kaydeder.

Kullanıcı "1000 satırlık prompt gibi tek seferde DB'den okunsun" istiyor. Bu durumda `prompt_memory_retrieve_all.py` veya argümanlı `get_full_context.py` kullanılmalı; `mcp__prompt-memory__search_memory` default limit 10 döndürdüğü için yeterli olmaz.

--- decision-persistence-0004-constraint-bug [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karat/tespit: `video_tracklet` migration unique constraint hatası hâlâ veritabanında.
- `0004` migration `(job_id, tracklet_ordinal)` unique; doğrusu `(job_id, track_id, tracklet_ordinal)`.
- Sonraki `cf0441294c5f` migration `upgrade()` içinde `pass` yapmış.
- Eylem: yeni migration ile constraint düzelt; mevcut üretim veritabanı varsa alembic düzeltmesi planla.

--- decision-worker-main-test-coverage-gap [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karar/tespit: worker orchestration katmanında doğrudan test yok.
- `backend/app/worker/video_worker_main.py` içinde `_process_one_job`, `_run_native_worker`, `_compress_artifact_bundle`, `_subsample_frames` gibi fonksiyonlar sadece E2E acceptance’ta dolaylı test ediliyor.
- Lease/fencing, cancel, retry, artifact upload hataları için erken geri bildirim zayıf.
- Eylem: native süreç ve artifact yönetimini mock’layan unit/integration testleri ekle.

--- decision-frontend-overlay-full-fetch [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karar/tespit: frontend overlay frame’lerini tüm video için tek seferde istiyor.
- `JobDetailPage` `useVideoOverlayFrames(jobId, 0, undefined)` çağrısı backend’den bitiş zamanı sınırı koymadan tüm frame detection’larını getirir.
- Uzun/high-fps videolarda bant genişliği ve bellek sorunu oluşur.
- Eylem: oynatıcı viewport/zaman penceresine göre chunked istek yap; query key’i request ile uyumlu hale getir.

--- decision-native-bundle-not-persisted [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karar/tespit: native worker artifact’leri kalıcı object store’a yazılmıyor.
- `observations.pb.zst`, `track_templates.pb.zst`, `crops/*.webp` sadece geçici `work_dir`’de üretilip okunuyor; sonrasında siliniyor.
- MinIO’ya sadece `result/manifest.json` ve public overlay yazılıyor.
- Eylem: debug/audit/reprocess senaryoları için artifact’leri MinIO’da tutma/kısa süreli archive politikası belirle.

--- decision-cannot-link-not-generated [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karar/tespit: cannot-link kısıtı tüketiliyor ama hiçbir yerde üretilmiyor.
- AGENTS.md bölüm 14: aynı anda görülen farklı yüzler arasında cannot-link olmalı.
- Kod sadece `tracklet.__dict__.get("cannot_link_track_ids")` okuyor; bu seti dolduran bir yer yok.
- Risk: aynı frame’deki farklı kişiler yanlışlıkla aynı canonical track/faceId altında birleşebilir.
- Eylem: tracker/detection mapper’dan cannot-link seti üret ve `CanonicalTrack` alanında da kullan.

--- decision-observation-id-job-prefix-inconsistency [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karar/tespit: `observation_id` üretiminde `job_id` öneki tutarsız.
- `backend/native/video_worker/include/mv/video/detection_mapper.hpp` ve proto yorumu `job_id:presentation_index:ordinal` bekler.
- `backend/native/video_worker/src/video_face_pipeline.cpp` içindeki gerçek mapping `:presentation_index:ordinal` şeklinde, yani `job_id` eksik.
- Eylem: production mapping’i detection_mapper.hpp ile uyumlu hale getir; test ekleyerek sabitle.

--- decision-worker-heartbeat-recovery-missing [Decision] tags: compaction-retrieve, MergenVision, codebase-decision, risk ---
Karar/tespit: `video_worker_main.py` heartbeat ve recover_expired_leases çağrılarını bağlamamış.
- `video_job_queue.heartbeat` ve `recover_expired_leases` metotları var ama üretim worker’ında veya dış scheduler’da çağrılmıyor.
- Uzun videolarda 1800 sn lease süresi yetmeyebilir; asılı kalan `processing` job’lar otomatik geri alınmaz.
- Eylem: worker döngüsüne periyodik heartbeat ve startup’ta recover ekle, veya ayrı bir janitor process tanımla.

--- summary-tests-coverage [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Test katmanı özeti:
- Unit: bundle reader, observation/template reader, crop provider, tracking, reconciliation, identity resolution, persistence, phase2 step0 contract.
- Integration: upload/job API, job queue, video identity persistence, processing/result API, repositories, Qdrant model version.
- Native: `image_runtime` surface/safety, C++ decode smoke, sequence contract, real GPU batching smoke.
- Frontend E2E: Playwright smoke/upload.
- Research video reference lab: CPU/CUDA oracle, chunk invariance, synthetic/real model smoke, Friends.mp4 acceptance.
- Güçlü alanlar: okuma-serileştirme, tracking/reconciliation, identity resolution, API contract.
- Eksik/zayıf: worker orchestration (`video_worker_main.py`) doğrudan testi, overlay service izole testi, bundle writer↔reader round-trip, frontend job tamamlanma UI testleri, GPU smoke çevre bağımlılığı.

--- summary-native-cpp-pipeline [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Native C++/CUDA video pipeline özeti:
- Pipeline: GStreamer/NVDEC/NVMM → pad probe → `FrameEnvelope`/`DeviceImageView` → batch assembler → `VideoFacePipeline::infer_detector_batch` → RetinaFace R50 → postprocess → recognition (5-point align + warp + GlintR100 + L2) → `NaiveTracker` → protobuf/zst artifact.
- Tüm GPU işler tek `cudaStream_t` üzerinde async; sync noktaları detector count/recognizer D2H.
- `NaiveTracker` geçici: center-distance greedy, 50px threshold, 5 misses; production değil.
- CPU fallback yok; desteklenmeyen format exception.
- Riskler:
  - `observation_id` C++ pipeline’da `job_id` öneki olmadan üretiliyor; `detection_mapper.hpp` ve proto yorumuyla çelişki.
  - Recognition quality gate `true` sabit; gerçek kalite filtresi yok.
  - Native worker binary aynı test dosyasından derleniyor.
  - `manifest.json`’da `input_sha256` argüman alınıp kullanılmıyor.

--- summary-serialization-protobuf [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Infrastructure serialization & protobuf artifact katmanı özeti:
- Contract truth: `backend/contracts/video_observation_v1.proto`, `video_track_template_v1.proto`.
- Writer: C++ `backend/native/video_worker/` (`ArtifactState`), delimited protobuf stream + zstd seviye 3.
- Reader: `backend/app/infrastructure/serialization/video_observation_reader.py`, `video_track_template_reader.py`, `native_bundle_reader.py`.
- Native artifact’ler geçici `work_dir`’de üretilir, Python worker tarafından okunur, sonra silinir; **MinIO’ya upload edilmez**.
- `manifest.json` footer sayaçlarla doğrulanır; crop’lar WebP 112×112.
- Riskler:
  - `video_observation_writer.py` yok; yazma C++’da.
  - Reader tüm artifact’i `read_bytes()` ile memory’e alır.
  - Native worker main `#ifdef MV_VIDEO_WORKER` altında `tests/real_batching_smoke.cpp` içinde.

--- summary-persistence-repositories [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Persistence / repositories katmanı özeti:
- Ana dosya: `backend/app/infrastructure/persistence/sqlalchemy/repositories/video_repositories.py`.
- Repository’ler: asset, job, track, tracklet, appearance interval, track sample, timeline chunk, idempotency, face identity/sample, process record, recognition result, outbox event, job queue, unit of work.
- `IdentityStorageLifecycleService` cross-store (PG ↔ MinIO ↔ Qdrant) lifecycle ve compensasyon sağlar.
- `VideoUploadService` staging/canonical upload, idempotency claim, outbox event üretir.
- Önemli riskler:
  - Migration `0004`’te `video_tracklet` unique constraint hatalı `(job_id, tracklet_ordinal)`; sonraki migration `upgrade`’de `pass`.
  - Outbox consumer/background worker yok.
  - Optimistic locking sadece `FaceIdentityRepository`’de tutarlı; video asset/job update’lerinde version kontrolü yok.
  - `VideoOverlayService` MinIO upload sonrası ayrı UoW ile PG chunk yazar; orphan object riski.
  - `session.py` `NullPool` kullanıyor.

--- summary-domain-tracking-reconciliation [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Domain tracking & reconciliation özeti:
- `backend/app/application/services/video_tracking_service.py` IoU + max_gap ile `RawTracklet` oluşturur, kalite bazlı template seçer.
- `video_reconciliation_service.py` tracklet’leri `CanonicalTrack`’lere birleştirir: cosine threshold 0.6, zamansal çakışma engeli, cannot-link kısıtı.
- `video_identity_resolution_service.py`: Qdrant arama, top2 margin guard, mevcut kimlik eşleştirme veya yeni `new_anonymous`.
- `video_track_persistence_service.py` job bazlı idempotent PG yazımı.
- Invariant: aynı `faceId` çakışan zaman aralıklarına sahip iki canonical track’e atanamaz.
- Riskler:
  - `cannot_link_track_ids` seti hiçbir yerde doldurulmuyor; AGENTS.md bölüm 14 intent’i uygulanmamış.
  - `CanonicalTrack.cannot_link_track_ids` alanı kullanılmıyor; reconcile tracklet’in `__dict__`’sine bakıyor.
  - Temsilci embedding güncellemesi ağırlıksız ortalama.
  - `VideoReconciliationService` için test kapsamı zayıf/eksik.

--- summary-worker-orchestration [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Worker orchestration & application services özeti:
- `backend/app/worker/video_worker_main.py` ana döngüde `claim_next` ile job alır, `mv_video_worker` native binary’i çalıştırır, artifact’leri okuyup `VideoProcessingService.process` çağırır.
- `backend/app/infrastructure/persistence/sqlalchemy/video_job_queue.py`: `FOR UPDATE SKIP LOCKED`, lease UUIDv7, heartbeat, cancel, retry, recover_expired_leases.
- `VideoProcessingService.process`: iki UoW transaction arasında uzun hesaplama (tracking → reconciliation → identity → persistence → overlay → manifest).
- Kritik riskler:
  - `heartbeat` metodu var ama worker hiç çağırmıyor; default lease 1800 sn.
  - `recover_expired_leases` hiçbir scheduler tarafından çağrılmıyor.
  - Native process çalışırken cancel yok.
  - Worker `update_stage` çağırmıyor; progress daima 0/100.
  - Finalize transaction lease token fencing yapmıyor; race riski.

--- summary-frontend-api-ui [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
Frontend/API/UI modülü özeti:
- `frontend/src/api/videos.ts` React Query hook'ları: upload, job/jobResult polling, cancel, retry, people, appearances, timeline, overlay frames, playback URL.
- Akış: `VideoPage` upload → `JobDetailPage` polling → tamamlandığında `VideoOverlayPlayer`, `TrackListPanel`, `AppearanceTimeline`.
- Backend kontrat: `/api/v1/videos` REST, job state machine, `Idempotency-Key`, 202 Accepted async model.
- Overlay player: `<video>` + `<canvas>`, `requestVideoFrameCallback`, box/label çizimi, known/anonymous filtre, seçili track vurgusu.
- Önemli riskler:
  - `useVideoOverlayFrames` tüm video için tek seferde çekiliyor (`endPtsNs=undefined`); uzun videolarda bellek/yükleme riski.
  - Query key ile API çağrısı arasında `end_pts_ns=0` tutarsızlığı.
  - `playback_video` backend tüm video object’ini memory’e alıyor (`object_store.get`).
  - Cancel/retry sonrası cache invalidation eksik.

--- codebase-architecture-overview [Memory] tags: compaction-retrieve, MergenVision, codebase-summary ---
MergenVisionPhase2v2 kod tabanı genel görünümü (codebase-memory-mcp):
- Proje graph'ı: 1313 node, 3966 edge; codebase-memory-mcp’de proje adı `home-user-Workspace-MergenVisionPhase2v2`.
- Diller: Python 33 dosya, TypeScript 5 dosya, C++ 4 dosya, TOML 1, YAML 1.
- Paketler: `native` (244 node), `app` (224), `tests` (90), `Makefile` (59).
- Giriş noktaları: `backend/app/worker/video_worker_main.py:main`, frontend `frontend/src/api/videos.ts` hook'ları, `frontend/src/pages/VideoPage.tsx`.
- Cluster'lar: video worker main, detection/recognition, repository, bundle reader/writer, tracker, pipeline, protobuf domain'leri.
- Layer tespiti: `tests` entry, `app` ve `native` internal, `contracts` core.

--- summary-prompt10-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt10.txt — Phase 2 Video Identity Vertical Slice instruction özeti:
- Önceki "image-only Sprint 02" non-goal'ını supersede eder; video ürününü end-to-end tamamla.
- Akış: POST video upload → PG/MinIO → async GPU worker → native pipeline → track template → conservative reconciliation → persistent global faceId → PG/MinIO/Qdrant → result/timeline API → React overlay.
- Native runtime sadece compact evidence üretir; known/anonymous kararı ve faceId Python control plane'de.
- Detector temporal batch default 16; recognizer max batch 32; embedding 512-D L2-normalize; annotated MP4 yok.
- Quality-aware track template: temporal diversity, outlier rejection, quality-weighted centroid, L2 renormalize.
- Conservative reconciliation: cannot-link mutlak (co-occurring/overlap), strong cosine evidence, one faceId per canonical cluster.
- React overlay: <video> + <canvas>, requestVideoFrameCallback, source↔display transform.
- Test hedefleri: phase2-m6-native-full-observation, phase2-m6-track-template, phase2-m6-track-reconcile, phase2-m7-video-identity, phase2-m7-video-worker-e2e, phase2-m7-video-api, phase2-m8-video-ui-overlay, phase2-video-e2e-acceptance.
- Yasak: trackId==faceId, her raw track için ayrı identity, co-occurring merge, Qdrant sonucunu PG doğrulamasız kabul, PG embedding, NVENC.

--- summary-prompt9-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt9.txt — M5.1 RetinaFace 0 detection fix instruction özeti:
- Kök neden: decoder NV12 surface RGBA CUDA kernel'e veriliyor. Önce surface contract'ını kanıtla.
- Aşama A: RGBA oracle path (nvvideoconvert → RGBA NVMM → streammux → appsink → mevcut RGBA kernel).
- Aşama B: production fused NV12 kernel (NV12→RGB/BGR, letterbox, normalize, NCHW) tek launch.
- Colorimetry/range negotiated caps'den alınır; BT.709 hardcode yasak.
- 15 test hedefi: NV12 reddi, RGBA pitch, NV12 plane offset, BT.601/709, odd dimensions, letterbox, image runtime parity, RGBA vs NV12 parity, batch mapping, exact 300 accepted frames, partial batch, known-face acceptance.
- Make target'ları: phase2-m5-surface-contract, phase2-m5-known-face-oracle, phase2-m5-nv12-preprocess-parity, phase2-m5-retinaface-runtime, phase2-m5-real-gpu-batching.
- Yasak: threshold düşürme, model değiştirme, CPU fallback, Friends ilk 16 frame'den sonuç çıkarma, NV12 uchar4 okuma, tensor dump production'da bırakma.

--- summary-prompt7-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt7.txt — M5/M6 Native GPU Video Pipeline detay instruction özeti:
- Kesin pipeline: filesrc → qtdemux → h264parse → nvv4l2decoder → nvstreammux → appsink → C++ FacePipeline → NativeTracker → ObservationWriter.
- nvinfer/nvtracker/nvdsosd kullanılmaz; nvstreammux sadece temporal batch ve metadata için.
- nvv4l2decoder cudadec-memtype=0 (device), output NV12 NVMM. System memory → VIDEO_GPU_DECODER_MEMORY_CONTRACT_FAILED.
- CUDA async ömür: buffer retain, aynı stream, event synchronize; frame başı cudaDeviceSynchronize yasak.
- Tracker: native C++ ByteTrack benzeri, Kalman, Hungarian, IoU gating, same-frame cannot-link, deterministic local track key.
- Worker lease/fencing: lease_token; stale worker finalize edemez.
- Test gates: runtime gate, native build, GPU decode smoke (32/64 frame), GPU face smoke (300 frame), tracker unit tests, real track smoke, worker/storage integration, full E2E.
- Yasak: CPU fallback, JPEG/image API round-trip, frame başı sync, annotated MP4, PG embedding, published migration rewrite.

--- summary-prompt6-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt6.txt — M5 Real Native GPU Video Pipeline instruction özeti:
- GStreamer/NVDEC GPU decode → RetinaFace R50 → CUDA 5-point alignment → GlintR100 → GPU L2 → native tracker → protobuf/zst observation → PG track persistence.
- Python control plane / C++ CUDA data plane ayrımı. Python’a full frame/tensor/NVMM geçmez.
- TensorRT profiles: RetinaFace 1/4/8, GlintR100 1/8/32.
- Common mv_face_pipeline kütüphanesi; image/video aynı core.
- Artifact streaming yazılır; tüm video RAM'de biriktirilmez.
- Test hedefleri: runtime gate, native build, GPU decode smoke, GPU face smoke, tracker tests, real track smoke, worker integration, full E2E.
- Yasak: CPU decode/fallback, OpenCV/PIL, annotated MP4/NVENC, PG embedding, Qdrant PII, migration rewrite.

--- summary-prompt5-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt5.txt — M2 correction + M3/M4/M5 autonomous instruction özeti:
- M2 preflight: fresh-checkout reproducibility (.gitignore models/ unanchored → ORM ignore), 0003/0004 migration rewrite yasak, 0005 ekle, idempotency (video bytes hash), durable upload, outbox, ffprobe errors, API semantics, Makefile recipe'leri.
- M3 job lease/retry/worker: atomic claim (FOR UPDATE SKIP LOCKED), lease_token fencing, heartbeat, bounded concurrency, cancel propagation.
- M4 common device FacePipeline: DeviceImageView + FacePipeline; image/video aynı native core. Yasak: JPEG round-trip, OpenCV/Pillow decode, CPU fallback, Python’a NVMM/full frame.
- M5 real NVIDIA video observation: C++/GStreamer/DeepStream, protobuf/zst artifact, runtime compatibility gate. No accuracy/600 FPS claim.
- Yasak: lab import, frontend feature, migration rewrite, git/volume/model/driver değişikliği.

--- summary-prompt3-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt3.txt — Video Reference Lab correction instruction özeti:
- Labı adli/forensic şekilde düzeltmek; üretim değil, doğruluk oracle'ı.
- Veri modeli: Face observation → Raw tracklet → Canonical track → Display label. Tracker ID kalıcı kimlik değil.
- InsightFace buffalo_l tek seferlik; ticari olmayan araştırma. Model lisans uyarısı.
- ByteTrack lifecycle, chunk invariance, sahne kesimi, cannot-link, complete-link.
- Kalite kontroller önce; threshold düşürerek preprocessing hatası gizlenemez.
- 27 bölümlük REPORT.md ve review package; dürüst FAIL/BLOCKED/NOT_RUN etiketleri.
- Yasak: üretim kaynağına dokunma, ünlü/galeri veriseti indirme, commit/push.

--- summary-prompt2-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt2.txt — Sprint 002 Offline Video Reference Lab instruction özeti:
- İzole Python/ONNX referans laboratuvarı; üretim değil, ilerideki GPU pipeline için oracle.
- Friends.mp4 bir kez işlenir; frames/observations/embeddings dondurulur.
- ByteTrack IoU + HybridFaceByteTracker; scene cut; tracklet template aggregation; complete-link reconciliation; cannot-link mutlak.
- Kalite: Laplacian variance, bbox size, interocular mesafe, landmark geometry, alignment residual. Embedding normu kalite sayılmaz.
- Chunk invariance: 1/8/17/64 chunk boyutlarında aynı observation→tracklet mapping ve canonical üyelik.
- Etiket uydurulamaz; yetersiz ground truth → PARTIAL_NEEDS_HUMAN_LABELS.
- Yasak: C++/CUDA/DeepStream, PostgreSQL/MinIO/Qdrant lifecycle, FastAPI, production UI, gerçek anonymous persistence, model indirme, biometrik artifact Git'e ekleme.

--- summary-prompt-txt [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
prompt.txt — Phase 1 Sprint 01 correction instruction özeti:
- Mevcut Sprint 01 implementation’ındaki doğrulanmış hataları TDD ile düzeltme; yeni sprint başlatma.
- Odak: test resource isolation, fail-closed resource guard, fresh-clone reproducibility (.gitignore models/ unanchored problemi), CI workflow.
- Candidate selection: Qdrant top_k, raw score descending, PG active doğrulama, stale-first valid-second, empty max yasak.
- Score semantics: raw cosine threshold karşılaştırması; persist confidence [0,1] clamp.
- Cross-store failure: PG reserve → MinIO → Qdrant → PG finalize; her adım hatasında reverse compensation, completed/result veya orphan vector bırakılamaz.
- Optimistic locking: update_with_expected_version, ConcurrentUpdateError.
- Yasak: sprint 02, FastAPI endpoint, UI, GPU/video, git commit/push, volume silme.

--- summary-plan-md [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
PLAN.md — Phase 1 Sprint 01 Minimal Identity Storage Foundation planı özeti:
- Hedef: gerçek PostgreSQL/MinIO/Qdrant üzerinde kalıcı kimlik yaşam döngüsünü kanıtlamak.
- Dört tablo: face_identity, face_sample, process_record, recognition_result.
- State machine'ler: FaceIdentity (anonymous/known), FaceSample (pending/active/failed/inactive), ProcessRecord (processing/completed/failed), RecognitionResult immutable snapshot.
- IdentityStorageLifecycleService: store_new_identity (PG reserve → MinIO upload → Qdrant upsert → PG finalize), recognize_existing (Qdrant query → PG active check → result), enroll_identity (optimistic locking ile anonymous→known).
- 17 adımlı TDD planı ve Makefile acceptance target'ları.
- Sprint 01 dışı: outbox, saga, reconciliation, inference_profile, API/UI, gerçek detection/GPU, video.
- Geçmiş plan; mevcut repo Phase 2'ye evrilmiş durumda.

--- summary-references-md [Memory] tags: compaction-retrieve, MergenVision, reference-summary ---
MergenVision referans kataloğu (opensourcereferences/references.md) özeti:
- Reference-first engineering rehberi; liste kendisi requirement değil.
- Her adaptasyon öncesinde REFERENCE_DECISION_LOG.md’ye karar kaydı zorunlu: Decision ID, URL, commit/tag, erişim tarihi, lisanslar, adopted/rejected behavior, local modifikasyon, failing test, parity command, limitation.
- master/main körlemesine referans verilmez; README/paper tek başına contract değil; paper threshold production threshold olmaz.
- Code license ile pretrained model/dataset license ayrı incelenir. Build/engine deserialize/output file oluşması correctness kanıtı değildir.
- MCP yönlendirme: active repo → codebase-memory-mcp; version-sensitive API → context7; upstream mimari → deepwiki; eksik kaynak → exa; API runtime → postman; UI E2E → playwright; UI kod üretimi → 21st yasak.
- Reference package completion checklist 13 maddeden oluşur; tamamlanmadan "DeepStream supported", "ByteTrack correct", "600 FPS" denemez.

--- architectureplan-md-0668 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bir sonraki adım kod yazmak değil: bu mimariyi requirement amendment + ERD + API contract + lifecycle state machine + ilk E2E acceptance olarak dondurmak. Bunlar onaylandıktan sonra implementation agent’a kapsamı değiştirme alanı bırakmadan Sprint 1’den başlayabiliriz.

--- architectureplan-md-0666 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bence bu karar setiyle baştan kurarsak önceki C++/DeepStream bataklığına tekrar düşmeyiz. Mevcut projeden yalnız kanıtlanmış RetinaFace/TensorRT, alignment testleri, engine artifact kontrolleri ve benchmark araçlarını seçerek taşırız; broken tracker, skeleton recognizer ve product render topolojisini taşımayız.

--- architectureplan-md-0664 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Batch başlangıç default’u 4; benchmark ile 1/2/8 seçilebilir: evet.

--- architectureplan-md-0663 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- İlk doğruluk sürümünde inference bütün framelerde; sampling sonradan configurable: evet.

--- architectureplan-md-0662 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Tracking trackletId, global kimlik faceId: evet.

--- architectureplan-md-0661 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Detection timeline zaman chunk’ları: evet.

--- architectureplan-md-0660 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Identity başına en fazla 3–5 best shot: evet.

--- architectureplan-md-0659 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Historical decision + current identity ayrı alanlar: evet.

--- architectureplan-md-0658 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- new_anonymous job-time result: evet.

--- architectureplan-md-0657 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- faceId immutable UUIDv7: evet.

--- architectureplan-md-0656 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Native katman yalnız GPU hot path: evet.

--- architectureplan-md-0655 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Python tracking/reconciliation: evet.

--- architectureplan-md-0654 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Worker finalized MinIO object’i okur: evet.

--- architectureplan-md-0653 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Browser → worker canlı çift stream: hayır.

--- architectureplan-md-0652 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Original video + frontend overlay: evet.

--- architectureplan-md-0651 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Product annotated MP4: hayır.

--- architectureplan-md-0650 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Internal React UI: evet, requirement amendment ile.

--- architectureplan-md-0648 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Benim önerdiğim varsayılanlar:

--- architectureplan-md-0646 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Şimdi dondurmamız gereken kararlar

--- architectureplan-md-0644 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bu geçerse elimizde gerçekten ürün vardır.

--- architectureplan-md-0642 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- No-face video completed + personCount=0 olur.

--- architectureplan-md-0641 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Container restart sonrası bütün bilgiler korunur.

--- architectureplan-md-0640 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- /faces/{faceId}/appearances iki videoyu ve zaman aralıklarını döner.

--- architectureplan-md-0639 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Eski video yeniden encode edilmeden Gunther label’ı gösterir.

--- architectureplan-md-0638 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- faceId değişmez.

--- architectureplan-md-0637 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- UI’da bu identity “Gunther” olarak promote edilir.

--- architectureplan-md-0636 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- İkinci videoda aynı anonymous aynı faceId ile anonymous döner.

--- architectureplan-md-0635 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- En fazla 3–5 kaliteli crop MinIO’ya yazılır.

--- architectureplan-md-0634 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Diğer kişi için tek faceId oluşturulur; 300 frame için 300 identity değil.

--- architectureplan-md-0633 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Rachel known görünür.

--- architectureplan-md-0632 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Worker Rachel ve başka bir kişiyi bulur.

--- architectureplan-md-0631 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- UI orijinal videoyu oynatır.

--- architectureplan-md-0630 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Friends videosu UI’dan yüklenir.

--- architectureplan-md-0629 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- personId + faceId + sampleId oluşur.

--- architectureplan-md-0628 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Rachel’ın fotoğrafı enrollment edilir.

--- architectureplan-md-0626 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bu tek senaryo geçmeden “sistem çalışıyor” demeyelim:

--- architectureplan-md-0624 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
İlk gerçek E2E kabul senaryosu

--- architectureplan-md-0622 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Throughput/latency/recall ölçümleri

--- architectureplan-md-0621 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Multi-GPU job ownership

--- architectureplan-md-0620 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- 10M benchmark harness

--- architectureplan-md-0619 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Oracle import/sync boundary

--- architectureplan-md-0618 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Privacy

--- architectureplan-md-0617 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Cross-store partial failure

--- architectureplan-md-0616 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Restart/retry/cancel/retention

--- architectureplan-md-0614 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 7 — Full acceptance ve scale

--- architectureplan-md-0612 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Appearance history

--- architectureplan-md-0611 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Merge duplicate identity

--- architectureplan-md-0610 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Anonymous → Rachel

--- architectureplan-md-0609 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Timeline/person sidebar

--- architectureplan-md-0608 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Canvas bbox/name overlay

--- architectureplan-md-0607 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Original video playback

--- architectureplan-md-0606 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Job progress via SSE

--- architectureplan-md-0605 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Local preview

--- architectureplan-md-0604 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Upload

--- architectureplan-md-0602 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 6 — Internal UI

--- architectureplan-md-0600 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Best-shot selection ve crop pass

--- architectureplan-md-0599 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- FirstSeen/lastSeen/appearance/totalDuration

--- architectureplan-md-0598 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Cross-scene same-face reconciliation

--- architectureplan-md-0597 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Known/anonymous/new_anonymous

--- architectureplan-md-0596 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Qdrant top-K

--- architectureplan-md-0595 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Quality-weighted track embedding

--- architectureplan-md-0594 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- ByteTrack-benzeri tracklet continuity

--- architectureplan-md-0593 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Temporal ordering

--- architectureplan-md-0591 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 5 — Python tracking ve reconciliation

--- architectureplan-md-0589 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Tracker ve render yok

--- architectureplan-md-0588 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Python’a yalnız kompakt embedding/metadata

--- architectureplan-md-0587 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Versioned observation artifact

--- architectureplan-md-0586 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Original bbox coordinate

--- architectureplan-md-0585 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- GlintR100 embedding

--- architectureplan-md-0584 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Five-landmark align

--- architectureplan-md-0583 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- RetinaFace

--- architectureplan-md-0582 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- NVDEC/GStreamer

--- architectureplan-md-0580 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 4 — GPU observation worker

--- architectureplan-md-0578 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- No-face completed sonucu

--- architectureplan-md-0577 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Playback URL

--- architectureplan-md-0576 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Job state, progress, cancel, retry

--- architectureplan-md-0575 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- MinIO retention

--- architectureplan-md-0574 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Format/codec/size/duration validation

--- architectureplan-md-0573 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Multipart/presigned upload

--- architectureplan-md-0571 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 3 — Video upload ve async jobs

--- architectureplan-md-0569 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bu geçmeden video başlamaz.

--- architectureplan-md-0567 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Merge/alias audit

--- architectureplan-md-0566 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Anonymous → Rachel promote

--- architectureplan-md-0565 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Aynı yüzün ikinci request’te aynı faceId dönmesi

--- architectureplan-md-0564 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Best-shot crop persistence

--- architectureplan-md-0563 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- known / new_anonymous / anonymous

--- architectureplan-md-0562 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Image multi-face recognition

--- architectureplan-md-0561 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Person photo/sample

--- architectureplan-md-0560 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Person enrollment

--- architectureplan-md-0558 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 2 — Image recognition ve anonymous lifecycle

--- architectureplan-md-0556 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Health/readiness

--- architectureplan-md-0555 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Docker Compose persistence

--- architectureplan-md-0554 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Cross-store outbox/reconciliation

--- architectureplan-md-0553 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- National-ID encryption + lookup HMAC + masked display

--- architectureplan-md-0552 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- UUIDv7 IDs

--- architectureplan-md-0551 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Qdrant collection/index lifecycle

--- architectureplan-md-0550 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- MinIO private buckets

--- architectureplan-md-0549 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- PostgreSQL migrations

--- architectureplan-md-0547 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 1 — Identity ve storage foundation

--- architectureplan-md-0545 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Acceptance fixtures

--- architectureplan-md-0544 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Retention/privacy

--- architectureplan-md-0543 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Storage ownership

--- architectureplan-md-0542 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- API/OpenAPI contract

--- architectureplan-md-0541 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Upload contract

--- architectureplan-md-0540 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- ERD

--- architectureplan-md-0539 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- ID ve status semantiği

--- architectureplan-md-0538 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- UI requirement amendment

--- architectureplan-md-0536 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Kod yok:

--- architectureplan-md-0534 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Sprint 0 — Contract freeze

--- architectureplan-md-0532 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Her şeyi tek dev promptta birden yaptırmak yine skeleton ve false-green test üretir. Ama mikro-sprint de yapmayacağız. Her aşama çalışan bir dikey sonuç üretecek.

--- architectureplan-md-0530 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
10. Uygulama sırası

--- architectureplan-md-0528 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Rachel’ın adı değiştiğinde binlerce bbox record yeniden yazılmaz. UI yalnız küçük identity map’i tekrar çeker.

--- architectureplan-md-0526 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0525 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
}

--- architectureplan-md-0524 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  }

--- architectureplan-md-0523 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    "identityVersion": 8

--- architectureplan-md-0522 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    "currentName": "Rachel",

--- architectureplan-md-0521 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    "currentStatus": "known",

--- architectureplan-md-0520 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    "faceId": "F100",

--- architectureplan-md-0519 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "T17": {

--- architectureplan-md-0518 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
{

--- architectureplan-md-0517 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```json

--- architectureplan-md-0515 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Identity map ayrı olur:

--- architectureplan-md-0513 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0512 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
}

--- architectureplan-md-0511 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "detectorScore": 0.98

--- architectureplan-md-0510 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "bbox": [640, 220, 180, 180],

--- architectureplan-md-0509 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "trackletId": "T17",

--- architectureplan-md-0508 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "ptsNs": 5005000000,

--- architectureplan-md-0507 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
{

--- architectureplan-md-0506 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```json

--- architectureplan-md-0504 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Observation’a isim gömmeyeceğiz:

--- architectureplan-md-0502 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
gibi 5–15 saniyelik chunk’lar halinde MinIO’da olabilir.

--- architectureplan-md-0500 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0499 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
jobs/{jobId}/overlay/chunk-000001.json.gz

--- architectureplan-md-0498 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
jobs/{jobId}/overlay/chunk-000000.json.gz

--- architectureplan-md-0497 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
jobs/{jobId}/overlay/manifest.json

--- architectureplan-md-0496 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0494 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Detaylı overlay timeline:

--- architectureplan-md-0492 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
tutulur.

--- architectureplan-md-0490 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Identity relation

--- architectureplan-md-0489 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Appearance interval

--- architectureplan-md-0488 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Tracklet

--- architectureplan-md-0487 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Kişi özeti

--- architectureplan-md-0486 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Job ve progress

--- architectureplan-md-0484 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
PostgreSQL’de:

--- architectureplan-md-0482 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Uzun videodaki bütün bbox’ları tek JSON response’a koymayacağız.

--- architectureplan-md-0480 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
9. Frontend overlay verisi

--- architectureplan-md-0478 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Yani “crop MinIO’ya gitti ama database kayıt olmadı” durumu normal kabul edilmeyecek.

--- architectureplan-md-0476 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Orphan MinIO object’leri cleanup job ile bulunur.

--- architectureplan-md-0475 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Başarısız outbox işlemleri reconcile edilir.

--- architectureplan-md-0474 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Qdrant tamamlanmadan sample ready görünmez.

--- architectureplan-md-0473 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Retry duplicate identity/sample oluşturmaz.

--- architectureplan-md-0472 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Qdrant point ID tam olarak sampleId olur.

--- architectureplan-md-0471 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- MinIO object key deterministiktir.

--- architectureplan-md-0470 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- origin_job_id + origin_tracklet_id unique constraint olur.

--- architectureplan-md-0469 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- ID’ler önceden ve deterministik üretilir.

--- architectureplan-md-0467 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Kurallar:

--- architectureplan-md-0465 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0464 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    F --> G["Job result finalized"]

--- architectureplan-md-0463 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    E --> F["PG sample indexed"]

--- architectureplan-md-0462 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    D --> E["Qdrant idempotent upsert"]

--- architectureplan-md-0461 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    C --> D["PG blob_ready + outbox"]

--- architectureplan-md-0460 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    B --> C["MinIO crop upload"]

--- architectureplan-md-0459 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    A["Tracklet final evidence"] --> B["PG identity/sample reserve"]

--- architectureplan-md-0458 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```flowchart TD

--- architectureplan-md-0456 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
PostgreSQL, MinIO ve Qdrant tek transaction paylaşamaz. Bu nedenle yeni anonymous oluşturma şu state machine ile ilerlemeli:

--- architectureplan-md-0454 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
8. Cross-store transaction

--- architectureplan-md-0452 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bir identity’nin birden fazla sample vector’ü olabilir; sonuçlar faceId bazında gruplanır. Qdrant birden fazla vector/sample temsilini destekler: Qdrant vectors.

--- architectureplan-md-0450 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Qdrant’a isim, ulusal kimlik, departman veya başka PII yazılmaz.

--- architectureplan-md-0448 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0447 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
}

--- architectureplan-md-0446 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "active": true

--- architectureplan-md-0445 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "preprocess_version": "...",

--- architectureplan-md-0444 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "model_version": "...",

--- architectureplan-md-0443 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "face_id": "...",

--- architectureplan-md-0442 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "sample_id": "...",

--- architectureplan-md-0441 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
{

--- architectureplan-md-0440 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```json

--- architectureplan-md-0438 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Minimal payload:

--- architectureplan-md-0436 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0435 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
}

--- architectureplan-md-0434 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "distance": "cosine"

--- architectureplan-md-0433 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "vector": "512-D normalized embedding",

--- architectureplan-md-0432 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "point_id": "face_sample.sample_id",

--- architectureplan-md-0431 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
{

--- architectureplan-md-0430 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```json

--- architectureplan-md-0428 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bir point:

--- architectureplan-md-0426 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Qdrant

--- architectureplan-md-0424 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Orijinal video retention sonunda silinebilir. Kimlik sample crop’u ayrı retention politikasına sahiptir; video silindi diye otomatik silinmez.

--- architectureplan-md-0422 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Kullanıcının orijinal dosya adı

--- architectureplan-md-0421 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Oracle person ID

--- architectureplan-md-0420 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Ulusal kimlik numarası

--- architectureplan-md-0419 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Soyadı

--- architectureplan-md-0418 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Rachel adı

--- architectureplan-md-0416 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Object key’lerde şunlar bulunmaz:

--- architectureplan-md-0414 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0413 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
jobs/{jobId}/evidence/observations.jsonl

--- architectureplan-md-0412 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
jobs/{jobId}/overlay/chunk-000001.json.gz

--- architectureplan-md-0411 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
face-samples/{faceId}/{sampleId}/crop.webp

--- architectureplan-md-0410 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
videos/{videoId}/source/original

--- architectureplan-md-0409 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0407 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Örnek opaque key’ler:

--- architectureplan-md-0405 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
MinIO

--- architectureplan-md-0403 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Embedding veya video binary PostgreSQL’e yazılmaz.

--- architectureplan-md-0401 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- identity merge/redirect kaydı

--- architectureplan-md-0400 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- cross-store outbox

--- architectureplan-md-0399 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- video_detection veya detection artifact index’i

--- architectureplan-md-0398 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- video_appearance

--- architectureplan-md-0397 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- video_tracklet

--- architectureplan-md-0396 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- video_person

--- architectureplan-md-0395 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- video_job

--- architectureplan-md-0394 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- video_asset

--- architectureplan-md-0392 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Video genişletmesi için onaylanacak tablolar:

--- architectureplan-md-0390 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- process_event

--- architectureplan-md-0389 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- recognition_result

--- architectureplan-md-0388 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- face_sample

--- architectureplan-md-0387 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- person_photo

--- architectureplan-md-0386 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- inference_profile

--- architectureplan-md-0385 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- process_record

--- architectureplan-md-0384 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- face_identity

--- architectureplan-md-0383 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- person

--- architectureplan-md-0381 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Phase 1 çekirdeği:

--- architectureplan-md-0379 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
PostgreSQL

--- architectureplan-md-0377 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| Qdrant | Rebuildable embedding index |

--- architectureplan-md-0376 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| MinIO | Binary object’ler |

--- architectureplan-md-0375 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| PostgreSQL | Business ve relational source of truth |

--- architectureplan-md-0374 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
|---|---|

--- architectureplan-md-0373 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| Sistem | Sahip olduğu veri |

--- architectureplan-md-0371 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
7. Üç storage’ın sahipliği

--- architectureplan-md-0369 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Decode-only ikinci pass hızlıdır ve sistemi karmaşık bidirectional IPC’den korur. Sistem doğru çalışınca tek-pass crop retention optimizasyonu yapılabilir.

--- architectureplan-md-0367 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Pass B: GPU crop extractor videoyu hızlıca tekrar decode eder ve yalnız seçilen yüzleri çıkarır.

--- architectureplan-md-0366 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Python track/reconcile yapar ve saklanacak frame/detection ID’lerini seçer.

--- architectureplan-md-0365 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Pass A: GPU worker detection + alignment + embedding evidence üretir.

--- architectureplan-md-0363 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
İki pass kullanabiliriz:

--- architectureplan-md-0361 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
İlk implementation için temiz yöntem

--- architectureplan-md-0359 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Known bir kişiye ait her video karesini otomatik gallery’ye eklemeyeceğiz. Yanlış recognition gallery poisoning yaratır. Yeni known crop’lar önce candidate olabilir veya çok sıkı multi-frame consensus sonrası aktif edilir.

--- architectureplan-md-0357 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Başlangıç için identity başına en fazla 3–5 temporally diverse best shot saklamak mantıklı.

--- architectureplan-md-0355 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
bakılacak.

--- architectureplan-md-0353 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Occlusion var mı?

--- architectureplan-md-0352 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Önceki örneklerden yeterince farklı mı?

--- architectureplan-md-0351 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Detector confidence yüksek mü?

--- architectureplan-md-0350 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Landmark alignment geçerli mi?

--- architectureplan-md-0349 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Pose uygun mu?

--- architectureplan-md-0348 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Blur düşük mü?

--- architectureplan-md-0347 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Yüz yeterince büyük mü?

--- architectureplan-md-0345 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Her frame’i crop olarak saklamayacağız. Bir track boyunca:

--- architectureplan-md-0343 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0342 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    VIDEO_TRACKLET ||--o{ VIDEO_DETECTION : contains

--- architectureplan-md-0341 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    VIDEO_PERSON ||--o{ VIDEO_TRACKLET : groups

--- architectureplan-md-0340 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    VIDEO_JOB ||--o{ VIDEO_PERSON : produces

--- architectureplan-md-0339 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    FACE_IDENTITY ||--o{ FACE_SAMPLE : contains

--- architectureplan-md-0338 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    PERSON o|--o| FACE_IDENTITY : owns

--- architectureplan-md-0337 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```erDiagram

--- architectureplan-md-0335 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bir identity’nin birden fazla örneği olur:

--- architectureplan-md-0333 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- `face_sample.sample_id` = tek bir crop / embedding örneği

--- architectureplan-md-0332 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- `face_identity.face_id` = kalıcı biyometrik kimlik

--- architectureplan-md-0330 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Crop’un primary key’i faceId olmayacak.

--- architectureplan-md-0328 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
6. Face crop nasıl saklanmalı?

--- architectureplan-md-0326 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
soruları ayrı cevaplanır.

--- architectureplan-md-0324 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- UI: Bu yüz bugün kim olarak biliniyor?

--- architectureplan-md-0323 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Audit: Sistem video işlendiği gün ne karar verdi?

--- architectureplan-md-0321 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Böylece:

--- architectureplan-md-0319 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0318 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
}

--- architectureplan-md-0317 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "faceId": "F100"

--- architectureplan-md-0316 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "currentName": "Rachel",

--- architectureplan-md-0315 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "currentStatus": "known",

--- architectureplan-md-0314 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  "decisionStatusAtProcessing": "new_anonymous",

--- architectureplan-md-0313 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
{

--- architectureplan-md-0312 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```json

--- architectureplan-md-0310 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
İki alan saklayacağız:

--- architectureplan-md-0308 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Tarihsel sonuç

--- architectureplan-md-0306 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Canonical sonuç F20/Rachel olur.

--- architectureplan-md-0305 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Audit yapılabilir.

--- architectureplan-md-0304 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Qdrant sample’ları çöpe gitmez.

--- architectureplan-md-0303 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Eski URL ve faceId sorguları çalışır.

--- architectureplan-md-0302 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Eski appearance kayıtları kaybolmaz.

--- architectureplan-md-0300 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
F100 silinmez; alias/redirect olarak tutulur. Böylece:

--- architectureplan-md-0298 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0297 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
F100 → canonical F20

--- architectureplan-md-0296 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0294 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
varsa, rename değil merge yapılır:

--- architectureplan-md-0292 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- F100 = yanlışlıkla yeni anonymous oluşmuş identity

--- architectureplan-md-0291 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- F20 = mevcut Rachel identity

--- architectureplan-md-0289 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Elimizde:

--- architectureplan-md-0287 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Rachel zaten sistemde varsa

--- architectureplan-md-0285 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Primary key asla Rachel olmaz.

--- architectureplan-md-0283 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
demektir.

--- architectureplan-md-0281 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0280 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    person_id -> Rachel person kaydı

--- architectureplan-md-0279 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
face_identity F100

--- architectureplan-md-0278 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0276 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
“faceId’yi Rachel diye değiştirmek” gerçekte:

--- architectureplan-md-0274 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
faceId hiçbir zaman değişmez.

--- architectureplan-md-0272 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0271 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
name = Rachel

--- architectureplan-md-0270 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
status = known

--- architectureplan-md-0269 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
faceId = F100

--- architectureplan-md-0268 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0266 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Video C:

--- architectureplan-md-0264 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0263 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
F100 kişisini Rachel'a bağladı

--- architectureplan-md-0262 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0260 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Kullanıcı:

--- architectureplan-md-0258 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0257 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
status = anonymous

--- architectureplan-md-0256 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
faceId = F100

--- architectureplan-md-0255 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0253 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Video B:

--- architectureplan-md-0251 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0250 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
status = new_anonymous

--- architectureplan-md-0249 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
faceId = F100

--- architectureplan-md-0248 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0246 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Video A:

--- architectureplan-md-0244 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0243 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    Known --> Known: İsim veya metadata değişti

--- architectureplan-md-0242 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    NewAnonymous --> Known: Doğrudan promote

--- architectureplan-md-0241 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    Anonymous --> Known: Kullanıcı kişiye bağladı

--- architectureplan-md-0240 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    NewAnonymous --> Anonymous: Persist ve index tamamlandı

--- architectureplan-md-0239 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    [*] --> NewAnonymous: İlk güvenilir eşleşmeyen yüz

--- architectureplan-md-0238 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```stateDiagram-v2

--- architectureplan-md-0236 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
En önemli karar: new_anonymous kalıcı database state’i değildir; o job sırasında verilen sonuçtur.

--- architectureplan-md-0234 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
5. known / anonymous / new_anonymous döngüsü

--- architectureplan-md-0232 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Tracker hata yaparsa recognition/reconciliation ikinci savunma katmanı olur. Ama “kesinlikle aynı kişi” matematiksel olarak tracker’dan kanıtlanmaz; gallery embedding dağılımı, top-1/top-2 margin, çoklu kaliteli frame consensus’u ve gerektiğinde kullanıcı doğrulamasıyla karar verilir.

--- architectureplan-md-0230 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Crossing, occlusion, scene cut ve re-entry fixture’larıyla doğrulayacağız.

--- architectureplan-md-0229 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Python’da test edilebilir ByteTrack-benzeri tracking kullanacağız.

--- architectureplan-md-0228 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- NvDCF kullanmak zorunda olmayacağız.

--- architectureplan-md-0227 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- C++ custom tracker yazmayacağız.

--- architectureplan-md-0225 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
İlk sürümde:

--- architectureplan-md-0223 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Dolayısıyla batch yalnız GPU inference throughput optimizasyonudur; temporal tracking semantiğini değiştirmez.

--- architectureplan-md-0221 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Sonraki batch aynı state üzerinden devam eder.

--- architectureplan-md-0220 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Batch bittiğinde tracker state’ini sıfırlamaz.

--- architectureplan-md-0219 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Tracker’a frame frame verir.

--- architectureplan-md-0218 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Observation’ları sourceId, PTS, frameNumber ile sıralar.

--- architectureplan-md-0216 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Python:

--- architectureplan-md-0214 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- quality

--- architectureplan-md-0213 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- embedding

--- architectureplan-md-0212 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- detectorScore

--- architectureplan-md-0211 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- landmarks

--- architectureplan-md-0210 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- bbox

--- architectureplan-md-0209 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- PTS

--- architectureplan-md-0208 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- frameNumber

--- architectureplan-md-0207 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- sourceId

--- architectureplan-md-0205 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Her observation şu bilgiyi taşır:

--- architectureplan-md-0203 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Detector batch 8 çalışabilir; tracker batch üzerinde paralel state güncellemez.

--- architectureplan-md-0201 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Batch sıralamayı bozmayacak

--- architectureplan-md-0199 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0198 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    R --> C["Canonical faceId"]

--- architectureplan-md-0197 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    S --> R["PostgreSQL lifecycle validation"]

--- architectureplan-md-0196 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    E --> S["Qdrant top-K samples"]

--- architectureplan-md-0195 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    T --> E["Quality-weighted embeddings"]

--- architectureplan-md-0194 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    D["Frame detections"] --> T["Python tracklets"]

--- architectureplan-md-0193 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```flowchart TD

--- architectureplan-md-0191 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Birbirinden kopmuş T1 ve T81 aynı faceId altında birleşmeli mi?

--- architectureplan-md-0189 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Reconciliation’ın görevi:

--- architectureplan-md-0187 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bu tracklet hangi kalıcı face identity’ye ait?

--- architectureplan-md-0185 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Recognition’ın görevi:

--- architectureplan-md-0183 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Yakın zamanlı framelerdeki kutular aynı fiziksel yüz mü?

--- architectureplan-md-0181 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Yani tracker’ın görevi:

--- architectureplan-md-0179 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- T81 → faceId F_RACHEL

--- architectureplan-md-0178 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- T1 → faceId F_RACHEL

--- architectureplan-md-0176 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Recognition/reconciliation şunu yapar:

--- architectureplan-md-0174 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- frame 7345–7600 → tracklet T81

--- architectureplan-md-0173 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- frame 301–7000 → Rachel görünmüyor

--- architectureplan-md-0172 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- frame 1–300 → tracklet T1

--- architectureplan-md-0170 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Örneğin:

--- architectureplan-md-0168 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Rachel frame 1’de görünüp frame 7345’te tekrar geldiğinde aynı tracker ID’sini korumak zorunda değiliz.

--- architectureplan-md-0166 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- `faceId` = videolar ve sahneler boyunca kalıcı biyometrik kimlik

--- architectureplan-md-0165 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- `trackletId` = bu video içindeki kesintisiz fiziksel takip parçası

--- architectureplan-md-0163 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Burada iki kavramı kesin olarak ayırıyoruz:

--- architectureplan-md-0161 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
4. Tracking ve “Rachel neden sonda yine Rachel?” meselesi

--- architectureplan-md-0159 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Orijinal video hiçbir zaman değiştirilmez.

--- architectureplan-md-0158 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Overlay hâlâ frontend tarafındadır; annotated video üretilmez.

--- architectureplan-md-0157 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Codec uyumsuzsa yalnız playback derivative oluşturulur.

--- architectureplan-md-0156 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Container uyumsuzsa playback için lossless remux üretilir.

--- architectureplan-md-0155 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Browser uyumlu MP4 ise orijinal dosya oynatılır.

--- architectureplan-md-0153 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
AVI/MOV kabul etmekle browser’ın bunları oynatabilmesi aynı şey değil. Politika şu olur:

--- architectureplan-md-0151 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Böylece letterbox/pillarbox olsa bile kutu doğru yerde kalır.

--- architectureplan-md-0149 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0148 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
offsetY = (containerHeight - videoHeight * scale) / 2

--- architectureplan-md-0147 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
offsetX = (containerWidth  - videoWidth  * scale) / 2

--- architectureplan-md-0145 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
            containerHeight / videoHeight)

--- architectureplan-md-0144 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
scale = min(containerWidth / videoWidth,

--- architectureplan-md-0143 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```javascript

--- architectureplan-md-0141 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
BBox backend’den orijinal video koordinatında gelir. UI yalnız CSS ölçüsüne dönüştürür:

--- architectureplan-md-0139 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
ile yapılır. Bu API gösterilen video frame’iyle senkron callback verir: MDN requestVideoFrameCallback.

--- architectureplan-md-0137 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0136 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
});

--- architectureplan-md-0135 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
  drawOverlay(metadata.mediaTime);

--- architectureplan-md-0134 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
video.requestVideoFrameCallback((_, metadata) => {

--- architectureplan-md-0133 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```javascript

--- architectureplan-md-0131 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Overlay çizimi timeupdate ile değil, mümkün olduğunda:

--- architectureplan-md-0129 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Playback için MinIO’dan kısa ömürlü presigned GET URL üretilir. Browser seek yapabilsin diye Range ve 206 Partial Content desteklenir.

--- architectureplan-md-0127 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Storage’a ikinci büyük video yazılmaz.

--- architectureplan-md-0126 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Bbox ve recognition hataları çok daha kolay debug edilir.

--- architectureplan-md-0125 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Aynı video farklı overlay versiyonlarıyla gösterilebilir.

--- architectureplan-md-0124 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Rachel’ın adı sonradan değişince video yeniden render edilmez.

--- architectureplan-md-0123 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- NVENC, nvstreamdemux, OSD ve render backpressure ortadan kalkar.

--- architectureplan-md-0122 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Görüntü kalitesi kaybolmaz.

--- architectureplan-md-0120 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bunun ciddi avantajları var:

--- architectureplan-md-0118 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
kullanacak.

--- architectureplan-md-0116 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0115 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
+ timeline

--- architectureplan-md-0114 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
+ identity map

--- architectureplan-md-0113 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
+ canvas/SVG overlay

--- architectureplan-md-0112 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Original video

--- architectureplan-md-0111 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0109 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
UI:

--- architectureplan-md-0107 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Production çıktısı annotated MP4 olmayacak.

--- architectureplan-md-0105 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
3. Videoyu yeniden encode etmeyeceğiz

--- architectureplan-md-0103 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bu endpoint multipart videoyu API üzerinden MinIO’ya stream edip aynı internal upload-finalize-job akışına bağlanabilir. UI ise büyük dosyalarda presigned multipart yolunu kullanır.

--- architectureplan-md-0101 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0100 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
POST /api/v1/videos/recognize

--- architectureplan-md-0099 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0097 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
API istemcileri için requirement’taki endpoint’i de koruruz:

--- architectureplan-md-0095 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Sayfa yenilenirse MinIO’daki video oynatılır.

--- architectureplan-md-0094 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Overlay sonuçları geldikçe local preview üzerine bile çizilebilir.

--- architectureplan-md-0093 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Upload finalize olunca worker başlar.

--- architectureplan-md-0092 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Arka planda tek upload MinIO’ya gider.

--- architectureplan-md-0091 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Dosyayı seçtiği anda local preview başlar.

--- architectureplan-md-0089 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Yani kullanıcı açısından:

--- architectureplan-md-0087 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0086 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
video.src = previewUrl;

--- architectureplan-md-0085 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
const previewUrl = URL.createObjectURL(file);

--- architectureplan-md-0084 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```javascript

--- architectureplan-md-0082 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Ama kullanıcı upload sürerken videoyu hemen izleyebilir:

--- architectureplan-md-0080 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
MinIO’nun presigned upload mekanizması browser’ın storage credential almadan private bucket’a yükleme yapmasını destekliyor: MinIO JavaScript SDK.

--- architectureplan-md-0078 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0077 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    Worker->>PG: Progress ve sonuç

--- architectureplan-md-0076 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    Worker->>MinIO: Tamamlanmış videoyu oku

--- architectureplan-md-0075 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    Worker->>PG: Job claim

--- architectureplan-md-0074 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    API->>PG: Video ready, job pending

--- architectureplan-md-0073 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    API->>MinIO: Size ve checksum doğrula

--- architectureplan-md-0072 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    UI->>API: Upload complete

--- architectureplan-md-0071 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    UI->>MinIO: Videoyu tek kez yükle

--- architectureplan-md-0070 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    API-->>UI: Presigned multipart URL

--- architectureplan-md-0069 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    API->>PG: videoId ve jobId ayır

--- architectureplan-md-0068 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    UI->>API: Upload session oluştur

--- architectureplan-md-0066 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    participant Worker

--- architectureplan-md-0065 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    participant PG

--- architectureplan-md-0064 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    participant MinIO

--- architectureplan-md-0063 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    participant API

--- architectureplan-md-0062 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    participant UI

--- architectureplan-md-0061 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```sequenceDiagram

--- architectureplan-md-0059 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Önerdiğim güvenli akış:

--- architectureplan-md-0057 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Bazı MP4/MOV dosyaları dosyanın sonundaki metadata’yı görmeden açılamaz.

--- architectureplan-md-0056 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Cancel ve resume karmaşıklaşır.

--- architectureplan-md-0055 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Worker’ın hızı browser upload’ına backpressure yapabilir.

--- architectureplan-md-0054 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Browser bandwidth’i iki kat kullanılır.

--- architectureplan-md-0053 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Retry sırasında hangi byte dizisinin işlendiği belirsizleşir.

--- architectureplan-md-0052 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- MinIO başarılı olurken worker kopabilir.

--- architectureplan-md-0051 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
- Worker eksik MP4 okumaya başlayabilir.

--- architectureplan-md-0049 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
İki farklı consumer olduğunda:

--- architectureplan-md-0047 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Senin “browser byte stream’i hem MinIO’ya hem workera yollasın” düşüncen UX açısından anlaşılır ama production açısından riskli.

--- architectureplan-md-0045 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
2. Upload akışını nasıl yapalım?

--- architectureplan-md-0043 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
C++ yalnız gerçekten GPU’da olması gereken işi yapacak.

--- architectureplan-md-0041 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Böylece daha önceki problem ortadan kalkıyor: C++ içinde tracker, gallery lifecycle, rename, persistence, history gibi business mantıkları yazmıyoruz.

--- architectureplan-md-0039 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| Python | Tracking, tracklet aggregation, identity matching, lifecycle ve persistence orchestration |

--- architectureplan-md-0038 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| C++/CUDA/TensorRT | NVDEC, detect, landmarks, align, embedding |

--- architectureplan-md-0037 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| Qdrant | Yeniden üretilebilir 512-D face embedding index’i |

--- architectureplan-md-0036 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| MinIO | Orijinal video, fotoğraf, seçilmiş face crop ve büyük evidence artifact’ları |

--- architectureplan-md-0035 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| PostgreSQL | Kişi, kimlik, job, track, appearance ve audit source of truth |

--- architectureplan-md-0034 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| FastAPI | İş akışı, authorization, job yönetimi, API contract |

--- architectureplan-md-0033 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| React UI | Upload, job progress, video playback, bbox overlay, identity düzenleme |

--- architectureplan-md-0032 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
|---|---|

--- architectureplan-md-0031 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
| Katman | Sorumluluk |

--- architectureplan-md-0029 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Bölümlerin görevleri net olacak:

--- architectureplan-md-0027 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```

--- architectureplan-md-0026 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    MINIO --> UI

--- architectureplan-md-0025 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    PG --> UI

--- architectureplan-md-0024 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    PY --> QD["Qdrant"]

--- architectureplan-md-0023 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    PY --> MINIO

--- architectureplan-md-0022 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    PY --> PG

--- architectureplan-md-0021 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    EVIDENCE --> PY["Python tracking + reconciliation"]

--- architectureplan-md-0020 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    WORKER --> EVIDENCE["Detection + embedding evidence"]

--- architectureplan-md-0019 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    MINIO --> WORKER

--- architectureplan-md-0018 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    PG --> WORKER["GPU worker"]

--- architectureplan-md-0017 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    UI --> MINIO["MinIO original video"]

--- architectureplan-md-0016 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    API --> PG["PostgreSQL"]

--- architectureplan-md-0015 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
    UI["React UI"] --> API["FastAPI control plane"]

--- architectureplan-md-0014 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
```flowchart TD

--- architectureplan-md-0012 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
1. Yeni ana topoloji

--- architectureplan-md-0010 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Kullanıcı videoyu bir kez yükler; sistem orijinal videoyu değiştirmeden saklar, GPU worker yüz observation’larını üretir, Python track/identity kararlarını verir, PostgreSQL–MinIO–Qdrant’a güvenli şekilde kaydeder ve UI orijinal videonun üzerine bbox/isim overlay’i çizer.

--- architectureplan-md-0008 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Yeni sistemin tek cümlelik hedefi

--- architectureplan-md-0006 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Çünkü requirement zaten her işlenen frame için orijinal çözünürlükte bbox istiyor ve “istemci videonun üzerine çizsin” diyor. Annotated MP4 üretme zorunluluğu yok.

--- architectureplan-md-0004 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Requirement dokümanında “UI olmayacak” yazıyor. Senin güncel kararın bunu değiştiriyor. Bunu resmi bir requirement amendment olarak kaydedelim; onun dışında önerdiğin frontend-overlay yaklaşımı requirement’a çok iyi uyuyor.

--- architectureplan-md-0003 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
“Phase 1 yapılmış varsayalım” demeyelim. Phase 2’nin ihtiyaç duyduğu kimlik, storage, güvenlik ve image-recognition çekirdeğini gerçekten kuralım.

--- architectureplan-md-0001 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, architectureplan.md — Onaylanmış Mimari ---
Abi evet, bence doğru yön tam olarak bu. Fakat iki noktayı baştan düzeltelim:

--- current-sprint-md-0133 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
(Previous Sprint 02 package must not be modified.)

--- current-sprint-md-0132 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
Final package: `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md`

--- current-sprint-md-0130 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
## Review Package

--- current-sprint-md-0128 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
Next: React canvas overlay + Playwright acceptance (`make phase2-video-e2e-acceptance`).

--- current-sprint-md-0126 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- mevcut Playwright artifact’ları fresh-checkout product PASS kanıtı değildir

--- current-sprint-md-0125 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- E2E harici `../../lfw/...` dataset’ine bağımlı

--- current-sprint-md-0124 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- frontend/E2E `GET /faces?search=...` çağırıyor; backend list endpoint’i yok

--- current-sprint-md-0123 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- frontend `POST /faces/enroll` çağırıyor; backend path-param enroll kullanıyor

--- current-sprint-md-0121 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
in a later explicit UI gate):

--- current-sprint-md-0120 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
is frozen for backend/native work. Known UI contract drift (to be resolved

--- current-sprint-md-0119 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
Frontend source in the working tree is an **unrelated Phase 1 UI baseline** and

--- current-sprint-md-0117 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
`app.infrastructure.uuid7.generate_uuid7()` as required.

--- current-sprint-md-0116 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
All `uuid.uuid4()` / `uuid4()` usages in backend source/tests were replaced with

--- current-sprint-md-0114 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m8-video-result` — 1 passed

--- current-sprint-md-0113 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m7-video-identity` — 8 passed

--- current-sprint-md-0112 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m6-track-reconcile` — 6 passed

--- current-sprint-md-0111 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m6-track-template` — 11 passed

--- current-sprint-md-0110 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m6-native-full-observation` — PASSED (6665 frames, 9020 detections, 150 raw tracks, 9020 embeddings, 385.53 FPS, L2 norm 1.0)

--- current-sprint-md-0109 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m5-video-observation` — contract test passed; real GPU smoke green

--- current-sprint-md-0108 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m4-device-pipeline` — 5 passed, 2 skipped (native tests skip on host)

--- current-sprint-md-0107 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-m3-worker-control` — 9 passed

--- current-sprint-md-0106 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-control-plane` — 30 passed

--- current-sprint-md-0105 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-migrations` — 9 passed

--- current-sprint-md-0104 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `make phase2-step0-static` — green (ruff + mypy)

--- current-sprint-md-0102 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
Closed gates:

--- current-sprint-md-0100 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
DeepStream/GPU container. Milestones 6, 7 and 8 are closed.

--- current-sprint-md-0099 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
and blocked until the common device FacePipeline is built inside the pinned

--- current-sprint-md-0098 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
is in place; the C++/GStreamer native worker and real NVIDIA smoke are **NOT_RUN**

--- current-sprint-md-0097 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
native GPU implementation remains open. Milestone 5 protobuf observation contract

--- current-sprint-md-0096 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
IN PROGRESS — Milestones 0–3 closed. Milestone 4 Python port contract closed;

--- current-sprint-md-0094 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
## Status

--- current-sprint-md-0092 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Production-polished public UI (the internal React overlay is in scope as the Phase 2 client)

--- current-sprint-md-0091 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- National ID / Oracle / 10M-person scope

--- current-sprint-md-0090 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- 600 FPS / throughput claims without full measurement context

--- current-sprint-md-0089 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Annotated MP4 as primary product

--- current-sprint-md-0088 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Raw NVMM surface mapped into Python

--- current-sprint-md-0087 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Full-frame OpenCV/PIL production decode

--- current-sprint-md-0086 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Frame-by-frame JPEG round-trip to image API

--- current-sprint-md-0085 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- CPU inference fallback

--- current-sprint-md-0084 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- SCRFD / other recognizer

--- current-sprint-md-0083 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Model family change

--- current-sprint-md-0081 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
## Out of Scope

--- current-sprint-md-0079 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `research/video_reference_lab/**` is frozen and must not change

--- current-sprint-md-0078 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Product output: original video + time-synchronized overlay metadata (not annotated MP4)

--- current-sprint-md-0077 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Python metadata tracker first; C++ tracker rewrite only with profiling evidence

--- current-sprint-md-0076 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- UUIDv7 for all persistent opaque IDs

--- current-sprint-md-0075 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- `requestId` per HTTP call, `processId` per business operation, `jobId` per async GPU execution

--- current-sprint-md-0074 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Public JSON field names: camelCase via Pydantic aliases

--- current-sprint-md-0073 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Canonical API prefix: `/api/v1`

--- current-sprint-md-0072 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Result manifest key: `videos/{videoId}/jobs/{jobId}/result/manifest.json`

--- current-sprint-md-0071 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Public timeline key: `videos/{videoId}/jobs/{jobId}/timeline/{sequence}.jsonl.zst`

--- current-sprint-md-0070 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Video observation artifact key: `videos/{videoId}/jobs/{jobId}/observations/{sequence}.pb.zst`

--- current-sprint-md-0069 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Video source key: `videos/{videoId}/source/original`

--- current-sprint-md-0068 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Face crop MinIO key (unchanged): `faces/{faceId}/{sampleId}/aligned.webp`

--- current-sprint-md-0067 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Qdrant collection (unchanged): `face_samples_retinaface_r50_glintr100_v1`

--- current-sprint-md-0066 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
  - `backend/artifacts/models/glintr100.onnx`

--- current-sprint-md-0065 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
  - `backend/artifacts/models/retinaface_r50_dynamic.onnx`

--- current-sprint-md-0064 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
- Models (unchanged):

--- current-sprint-md-0062 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
## Binding Decisions

--- current-sprint-md-0060 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
further work in the affected area only.

--- current-sprint-md-0059 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
automatically followed by the next; hard stops from the master prompt block

--- current-sprint-md-0058 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
No gate gets `PASS` on mock/placeholder/fake adapter evidence. Each gate is

--- current-sprint-md-0056 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 9 | Client overlay, security & acceptance | React canvas overlay + Playwright on real backend + `make phase2-video-e2e-acceptance` | pending |

--- current-sprint-md-0055 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 8 | Worker/job integration, result/timeline API | person summary + appearances + timeline + API routes | ✅ `make phase2-m8-video-result` green (1 passed) |

--- current-sprint-md-0054 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 7 | Video identity resolution & persistence | reuse lifecycle service, canonical→faceId, PG/MinIO/Qdrant sample persistence | ✅ `make phase2-m7-video-identity` green (8 passed) |

--- current-sprint-md-0053 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 6 | Python tracking & reconciliation | ByteTrack-style + identity resolution | ✅ `make phase2-m6-track-template` green (11 passed), `make phase2-m6-track-reconcile` green (6 passed) |

--- current-sprint-md-0052 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 5 | DeepStream/GStreamer GPU observation worker | C++/GStreamer native worker + real NVIDIA smoke | ✅ `make phase2-m6-native-full-observation` green; 6665 frames, 9020 detections/tracks/embeddings |

--- current-sprint-md-0051 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 5 | DeepStream/GStreamer GPU observation worker | protobuf contract + observation schema | ✅ contract file + schema test green |

--- current-sprint-md-0050 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 4 | Common native device face pipeline | Python `DeviceImageView` + `FacePipeline` port | ✅ host contract (`phase2-m4-device-pipeline`: 5 passed); native GPU impl verified |

--- current-sprint-md-0049 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 3 | Job lease/retry/worker control | PG lease queue + claim/cancel/retry | ✅ `make phase2-m3-worker-control` green (9 passed) |

--- current-sprint-md-0048 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 2 | Video upload/finalization/async job API | `GET /api/v1/videos/{videoId}` + job status + cancel + retry + result 409 | ✅ `make phase2-control-plane` green |

--- current-sprint-md-0047 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 2 | Video upload/finalization/async job API | `POST /api/v1/videos/recognize` + idempotency | ✅ `make phase2-control-plane` green (30 passed) |

--- current-sprint-md-0046 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 1 | PostgreSQL video control plane | `make phase2-migrations` target + regression pass | ✅ 9 passed |

--- current-sprint-md-0045 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 1 | PostgreSQL video control plane | Migrations `0003_video_control_plane`, `0004_video_results` | ✅ upgraded + schema tests green |

--- current-sprint-md-0044 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | `make phase2-step0-closure` | ✅ verified in pinned TensorRT container (31 passed) |

--- current-sprint-md-0043 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.8 Step 0 automated acceptance Makefile targets | ✅ Makefile targets added; native tests skipped on host |

--- current-sprint-md-0042 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.7 Native safety fixes (GIL, RAII slot, abort removal, alignment status, model profile, exact profiles, Dockerfile digest, engine build script) | ✅ code; GPU verification pending container |

--- current-sprint-md-0041 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.6 Qdrant `model_version` filter + collection contract validation | ✅ code + integration test |

--- current-sprint-md-0040 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.5 Bounded input validation (JPEG magic, dimensions/pixels) | ✅ code + unit tests |

--- current-sprint-md-0039 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.4 Delete/detail/history semantics on real PostgreSQL | ✅ code + integration test |

--- current-sprint-md-0038 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.3 Image orchestration guarded lifecycle + failure persistence | ✅ code + unit tests |

--- current-sprint-md-0037 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.2 Health/readiness endpoints with real dependency checks | ✅ code + unit tests |

--- current-sprint-md-0036 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| 0 | Image closure & native safety | 0.1 Canonical `/api/v1` API contract + requestId + safe errors | ✅ code + unit tests |

--- current-sprint-md-0035 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
|---|-----------|----------|--------|

--- current-sprint-md-0034 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
| # | Milestone | Sub-gate | Status |

--- current-sprint-md-0032 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
## Phase 2 Milestone Ledger

--- current-sprint-md-0030 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
implementation.

--- current-sprint-md-0029 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
the previous Phase 1 scope freeze; the user explicitly authorized Phase 2

--- current-sprint-md-0028 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
`requirements/videorequirements.md` (video) remain binding. This update supersedes

--- current-sprint-md-0027 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
All requirements in `requirements/ProjectRequirements.md` (image) and

--- current-sprint-md-0025 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
```

--- current-sprint-md-0024 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> FastAPI response

--- current-sprint-md-0023 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> paginated public overlay timeline

--- current-sprint-md-0022 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> person-level summary + appearances

--- current-sprint-md-0021 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> identity/storage persistence

--- current-sprint-md-0020 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> Python tracker + reconciliation

--- current-sprint-md-0019 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> compact protobuf/zstd observation artifacts

--- current-sprint-md-0018 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> GlintR100 512-D L2-normalized embedding

--- current-sprint-md-0017 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> CUDA five-point alignment

--- current-sprint-md-0016 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> RetinaFace R50 detection + landmarks

--- current-sprint-md-0015 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> common device-resident FacePipeline

--- current-sprint-md-0014 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> NVIDIA decoder (GStreamer/DeepStream/NVDEC/NVMM)

--- current-sprint-md-0013 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> async PostgreSQL job claim/lease

--- current-sprint-md-0012 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
-> MinIO canonical object

--- current-sprint-md-0011 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
encoded video

--- current-sprint-md-0010 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
```text

--- current-sprint-md-0008 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
The end-to-end chain must be:

--- current-sprint-md-0006 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
GPU image identity vertical slice.

--- current-sprint-md-0005 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
Build the complete Phase 2 video face-recognition product on top of the existing

--- current-sprint-md-0003 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
## Objective

--- current-sprint-md-0001 [Memory] tags: compaction-retrieve, line-by-line, MergenVision, CURRENT_SPRINT.md — Phase 2 Sprint Durumu ---
# Current Sprint: Phase 2 — Complete Video Recognition Product

--- VideoFaceGpuLab [Project] tags: - ---


--- prjgoal-L545 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 545: Şimdi bu ürün bağlamını aklında tutarak onaylanmış Sprint 02 planını Build Mode’da uygula.

--- prjgoal-L543 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 543: deme.

--- prjgoal-L541 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 541: - 600 FPS

--- prjgoal-L540 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 540: - fully optimized

--- prjgoal-L539 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 539: - video-ready

--- prjgoal-L538 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 538: - GPU-only E2E

--- prjgoal-L537 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 537: - accuracy verified

--- prjgoal-L536 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 536: - production-ready

--- prjgoal-L534 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 534: Kanıt olmadan:

--- prjgoal-L532 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 532: - SPRINT-002-CODE-REVIEW-PACKAGE.md yolu.

--- prjgoal-L531 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 531: - Bilinen sınırlamalar.

--- prjgoal-L530 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 530: - Labın değişmediğinin kanıtı.

--- prjgoal-L529 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 529: - CPU fallback kullanılmadığının kanıtı.

--- prjgoal-L528 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 528: - Performance p50/p95/p99, yalnız informational.

--- prjgoal-L527 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 527: - Batch parity sonuçları.

--- prjgoal-L526 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 526: - Docker runtime evidence.

--- prjgoal-L525 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 525: - API/OpenAPI evidence.

--- prjgoal-L524 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 524: - PostgreSQL/MinIO/Qdrant evidence.

--- prjgoal-L523 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 523: - Multi-face sonucu.

--- prjgoal-L522 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 522: - Single-face lifecycle sonucu.

--- prjgoal-L521 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 521: - No-face sonucu.

--- prjgoal-L520 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 520: - Exact validation commands ve exit results.

--- prjgoal-L519 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 519: - Dynamic profiles.

--- prjgoal-L518 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 518: - RetinaFace ve GlintR100 model/engine SHA.

--- prjgoal-L517 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 517: - CUDA/TensorRT versions.

--- prjgoal-L516 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 516: - Gerçek GPU identity.

--- prjgoal-L515 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 515: - Değişen dosyalar.

--- prjgoal-L514 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 514: - PASS/PARTIAL/BLOCKED verdict.

--- prjgoal-L512 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 512: Şunları getir:

--- prjgoal-L510 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 510: Sprint sonunda yalnız “done” deme.

--- prjgoal-L508 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 508: ==================================================

--- prjgoal-L507 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 507: 14. BİTİRDİĞİNDE NE GETİRECEKSİN?

--- prjgoal-L506 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 506: ==================================================

--- prjgoal-L504 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 504: Fixture yokluğu bütün source implementation’ı durdurma nedeni değildir. Yapılabilen her şeyi tamamla; yalnız real-face acceptance’ı BLOCKED_REAL_GPU_FIXTURES olarak bırak.

--- prjgoal-L502 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 502: - Real face fixture yok ve real-face acceptance aşamasına geldin.

--- prjgoal-L501 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 501: - System dependency/model download gerekiyor.

--- prjgoal-L500 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 500: - Dynamic batch model tarafından gerçekten desteklenmiyor.

--- prjgoal-L499 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 499: - Model artifact/contract geçersiz.

--- prjgoal-L498 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 498: - Host driver container stack’i desteklemiyor.

--- prjgoal-L497 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 497: - Pinned NVIDIA container çalışmıyor.

--- prjgoal-L496 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 496: - Çakışan user changes.

--- prjgoal-L495 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 495: - Yanlış repository/base.

--- prjgoal-L493 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 493: Yalnız şu gerçek blocker’larda dur:

--- prjgoal-L491 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 491: Normal implementation/test/debug adımlarında kullanıcıdan mikro-onay isteme.

--- prjgoal-L489 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 489: Bu kararlar verilmiştir.

--- prjgoal-L487 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 487: - Eski planı silip silmeyeceğin

--- prjgoal-L486 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 486: - Labı ne yapacağımız

--- prjgoal-L485 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 485: - Qdrant collection

--- prjgoal-L484 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 484: - Dynamic batch profile

--- prjgoal-L483 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 483: - DeepStream kullanımı

--- prjgoal-L482 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 482: - Build container seçimi

--- prjgoal-L481 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 481: - Model seçimi

--- prjgoal-L479 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 479: Tekrar şu kararları sorma:

--- prjgoal-L477 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 477: Onaylanmış Sprint 02 planını uygula.

--- prjgoal-L475 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 475: Artık Build Mode’dasın.

--- prjgoal-L473 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 473: ==================================================

--- prjgoal-L472 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 472: 13. BUILD MODE DAVRANIŞIN

--- prjgoal-L471 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 471: ==================================================

--- prjgoal-L469 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 469: - Driver/system CUDA değiştirme

--- prjgoal-L468 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 468: - Model/dataset indirme

--- prjgoal-L467 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 467: - Eski plan/doküman silme

--- prjgoal-L466 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 466: - Eski Qdrant collection reset

--- prjgoal-L465 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 465: - Yeni PostgreSQL tablo

--- prjgoal-L464 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 464: - 10M-person platformu

--- prjgoal-L463 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 463: - Oracle

--- prjgoal-L462 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 462: - National ID

--- prjgoal-L461 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 461: - Bulk enrollment

--- prjgoal-L460 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 460: - Three-GPU

--- prjgoal-L459 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 459: - 600 FPS optimizasyonu

--- prjgoal-L458 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 458: - Accuracy benchmark

--- prjgoal-L457 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 457: - Blur/pose threshold calibration

--- prjgoal-L456 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 456: - React UI

--- prjgoal-L455 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 455: - Lab düzeltmesi

--- prjgoal-L454 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 454: - Labeling UI

--- prjgoal-L453 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 453: - Temporal aggregation

--- prjgoal-L452 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 452: - Tracklet merge

--- prjgoal-L451 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 451: - Tracking

--- prjgoal-L450 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 450: - NVDEC

--- prjgoal-L449 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 449: - GStreamer/DeepStream

--- prjgoal-L448 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 448: - Video upload/job

--- prjgoal-L447 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 447: - SCRFD veya başka model

--- prjgoal-L445 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 445: Bu sprintte yapma:

--- prjgoal-L443 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 443: ==================================================

--- prjgoal-L442 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 442: 12. SPRINT 02’DE YAPILMAYACAKLAR

--- prjgoal-L441 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 441: ==================================================

--- prjgoal-L439 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 439: Final raporda gerçek çalışan, çalışmayan ve hiç test edilmeyen davranışları açıkça ayır.

--- prjgoal-L437 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 437: Raw exception client’a dönmez.

--- prjgoal-L435 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 435: - CPU inference fallback yapma.

--- prjgoal-L434 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 434: - GPU request kapasitesi doluysa bounded overload error döner.

--- prjgoal-L433 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 433: - No-face ise bu failure değildir; completed faceCount=0 olur.

--- prjgoal-L432 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 432: - Result persistence başarısızsa completed response dönmez.

--- prjgoal-L431 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 431: - Qdrant upsert başarısızsa sample recognition-ready olmaz.

--- prjgoal-L430 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 430: - MinIO upload başarısızsa sample active olmaz.

--- prjgoal-L429 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 429: - Tensor output invalidse process failed.

--- prjgoal-L428 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 428: - GPU inference başarısızsa process failed.

--- prjgoal-L427 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 427: - JPEG decode başarısızsa process failed.

--- prjgoal-L425 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 425: Örnekler:

--- prjgoal-L423 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 423: Yanlış durumda sahte success dönme.

--- prjgoal-L421 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 421: ==================================================

--- prjgoal-L420 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 420: 11. FAILURE DAVRANIŞI

--- prjgoal-L419 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 419: ==================================================

--- prjgoal-L417 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 417: İlk acceptance aynı exact yüz görüntüsünün tekrar tanınmasını gösterebilir. Daha geniş accuracy/threshold calibration sonraki ayrı çalışma olacaktır.

--- prjgoal-L415 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 415: Bu sınırlamaları saklma fakat bunları bahane ederek implementation’ı durdurma.

--- prjgoal-L413 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 413: - Three-GPU scaling.

--- prjgoal-L412 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 412: - 600 FPS.

--- prjgoal-L411 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 411: - Production-scale throughput.

--- prjgoal-L410 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 410: - Blur/pose/occlusion eşiklerinin doğruluğu.

--- prjgoal-L409 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 409: - Optimal cosine threshold.

--- prjgoal-L408 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 408: - Genel precision/recall.

--- prjgoal-L407 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 407: - Her kamera ortamında doğru yüz tanıma.

--- prjgoal-L405 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 405: Şunları kanıtlayamaz:

--- prjgoal-L403 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 403: - Identity/storage lifecycle çalışıyor.

--- prjgoal-L402 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 402: - Batch 1 ve batch N tutarlı.

--- prjgoal-L401 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 401: - TensorRT embedding reference implementation ile uyumlu.

--- prjgoal-L400 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 400: - GlintR100 preprocessing doğru.

--- prjgoal-L399 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 399: - Alignment yönü doğru.

--- prjgoal-L398 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 398: - Landmark order doğru.

--- prjgoal-L397 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 397: - Bounding box original coordinate’e doğru dönüyor.

--- prjgoal-L396 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 396: - RetinaFace output’u doğru decode ediliyor.

--- prjgoal-L395 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 395: - Model gerçekten GPU’da çalışıyor.

--- prjgoal-L393 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 393: Bu sprint küçük fixture setiyle şunları kanıtlayabilir:

--- prjgoal-L391 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 391: ==================================================

--- prjgoal-L390 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 390: 10. MODEL DOĞRULUĞU KONUSUNDA DÜRÜSTLÜK

--- prjgoal-L389 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 389: ==================================================

--- prjgoal-L387 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 387: Mevcut RetinaFace ve GlintR100 modellerini doğru preprocess, doğru alignment ve doğru TensorRT runtime ile best-effort ürün sistemine bağlıyoruz.

--- prjgoal-L385 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 385: Biz yeni bir computer-vision araştırma problemi çözmeye çalışmıyoruz.

--- prjgoal-L383 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 383: Bu sprintin ilerlemesi lab doğruluğuna bağlı değildir.

--- prjgoal-L381 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 381: Lab yalnız ileride bir native modülün bbox/alignment/quality davranışı şüpheli olduğunda debugging/reference aracı olabilir.

--- prjgoal-L379 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 379: - Labeling UI üzerinde çalışma.

--- prjgoal-L378 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 378: - Ground-truth problemi çözmeye çalışma.

--- prjgoal-L377 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 377: - Friends sonuçlarını acceptance sayma.

--- prjgoal-L376 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 376: - Product dependency yapma.

--- prjgoal-L375 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 375: - Çalıştırma.

--- prjgoal-L374 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 374: - Labı değiştirme.

--- prjgoal-L372 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 372: Bu sprintte:

--- prjgoal-L370 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 370: research/video_reference_lab ürün değildir.

--- prjgoal-L368 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 368: ==================================================

--- prjreq-L1 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 1: Face Recognition API

--- prjreq-L2 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 2: Proje Gereksinimleri

--- prjreq-L4 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 4: Giriş

--- prjreq-L6 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 6: Bu proje, görüntüler üzerinden yüz tanıma yapan, yalnızca API olarak çalışan bir servisin geliştirilmesini kapsar. Sistem, kendisine gönderilen bir görüntüdeki tüm yüzleri tespit eder, her yüzü sistemde tanımlı olan kimlikle (face ID) eşleştirir ve daha önce görülmüş yüzleri tutarlı biçimde aynı kimlikle döner. Sistemde kayıtlı yüzler ile daha önce görülmüş ancak isimlendirilmemiş anonim yüzler ayrı durumlarda raporlanır; böylece istemci, bir yüzün tanınıp tanınmadığını ve kim olduğunu net olarak ayırt edebilir.

--- prjreq-L8 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 8: Servis ayrıca güçlü bir izlenebilirlik katmanı sağlar: her API çağrısına benzersiz bir işlem kimliği (process ID) atanır, bu kimlik zaman damgası ve işlem detaylarıyla birlikte loglanır ve geriye dönük sorgulanabilir. Bu sayede belirli bir yüzün hangi işlemlerde ve ne zaman göründüğü takip edilebilir.

--- prjreq-L10 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 10: İlk etapta yalnızca tek görüntü üzerinden çalışacak şekilde tasarlanan sistem, ileride video ve canlı akış gibi yeni giriş tiplerine genişletilebilecek bir mimariyi hedeflemelidir.

--- prjreq-L12 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 12: 1. Görüntü Girişi

--- prjreq-L13 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 13:     • API, her request içerisinde bir görüntü kabul edebilmeli.

--- prjreq-L14 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 14:     • Görüntünün geçerli/desteklenen bir formatta olduğu doğrulanmalı.

--- prjreq-L15 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 15:     • Görüntü okunamıyor, bozuk veya boş ise anlamlı bir hata dönülmeli.

--- prjreq-L16 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 16:     • Görüntüde hiç yüz bulunamadığı durum ayrı bir sonuç olarak ele alınmalı. Hata değil, "yüz bulunamadı" cevabı dönülmeli.

--- prjreq-L18 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 18: 2. Face Detection

--- prjreq-L19 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 19:     • Görüntü içindeki tüm yüzler tespit edilmeli

--- prjreq-L20 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 20:     • Her tespit edilen yüz için görüntü üzerindeki konum bilgisi (bounding box vb.) dönülmeli.

--- prjreq-L21 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 21:     • Aynı görüntüde birden fazla yüz varsa hepsi bağımsız olarak işlenmeli.

--- prjreq-L23 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 23: 3. Face Recognition

--- prjreq-L24 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 24:     • Tespit edilen her yüz için kalıcı bir kimlik (face ID) belirlenmeli; anonim yüzlerin de face ID'si olur.

--- prjreq-L25 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 25:     • Daha önce kayıtlı bir yüzle eşleşiyorsa, her zaman aynı face ID dönülmeli.

--- prjreq-L26 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 26:     • Eşleşme bir benzerlik/eşik mantığına göre yapılmalı; eşik altında kalan yüzler "tanınmadı" sayılmalı.

--- prjreq-L27 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 27:     • Her yüz için tanınma durumu (status) belirlenmeli ve dönülmeli:

--- prjreq-L28 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 28:         ◦ known: sistemde enroll edilmiş, isim/metadata'sı olan yüz.

--- prjreq-L29 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 29:         ◦ anonymous: daha önce görülmüş, mevcut ama isimlendirilmemiş anonim kayıt.

--- prjreq-L30 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 30:         ◦ new_anonymous: bu istekte ilk kez görülen, yeni oluşturulan anonim kayıt.

--- prjreq-L31 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 31:     • İsim ve ek metadata yalnızca known durumunda dolu olmalı; anonim durumlarda boş/null kalmalı.

--- prjreq-L32 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 32:     • Bir görüntüde hem bilinen hem anonim yüzler aynı anda bulunabilmeli ve her biri ayrı sonuçlanmalı.

--- prjreq-L34 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 34: 4. Bilinmeyen Yüzlerin Saklanması

--- prjreq-L35 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 35:     • Mevcut kayıtlarla eşleşmeyen yüzler için otomatik olarak yeni bir anonim kimlik oluşturulmalı (new_anonymous).

--- prjreq-L36 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 36:     • Bu anonim kayıt veritabanına eklenmeli ve sonraki isteklerde aynı yüz tekrar gelirse aynı ID ile anonymous durumunda tanınmalı.

--- prjreq-L37 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 37:     • Anonim kayıtlar, kişisel bilgi (isim vb.) olmadan yalnızca tanıma için gerekli verilerle saklanmalı.

--- prjreq-L38 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 38:     • Anonim bir kimlik daha sonra enroll ile isimlendirilebilmeli; bu durumda aynı face ID korunarak status known'a geçmeli.

--- prjreq-L40 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 40: 5. Veritabanı / Kayıt Yönetimi

--- prjreq-L41 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 41:     • Tanınan yüzlere ait kimlik verilerini(face ID) saklayacak kalıcı bir veri yapısı olmalı.

--- prjreq-L42 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 42:     • Yeni yüz ekleme / isimlendirme işlemi desteklenmeli (kayıt/enrollment).

--- prjreq-L43 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 43:     • Enroll edilen kayıtta isim ve ek metadata saklanabilmeli.

--- prjreq-L44 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 44:     • Mevcut bir kimliğin sorgulanması mümkün olmalı.

--- prjreq-L45 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 45:     • Bir kimliğin silinmesi/güncellenmesi desteklenmeli.

--- prjreq-L46 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 46:     • Aynı kişiye ait birden fazla yüz örneği saklanabilmeli (zamanla tanıma doğruluğunu artırmak için).

--- prjreq-L48 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 48: 6. Recognition - İşlem Takibi

--- prjreq-L49 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 49:     • Her API çağrısı için sistem tarafından benzersiz bir process ID üretilmeli.

--- prjreq-L50 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 50:     • Bu process ID her zaman response içinde dönülmeli.

--- prjreq-L51 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 51:     • Process ID unique olmalı.

--- prjreq-L52 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 52:     • Bir process ID ile o işlemin sonradan tekrar sorgulanabilmesi mümkün olmalı.

--- prjreq-L54 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 54: 7. Recognition - İşlem Loglama

--- prjreq-L55 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 55:     • Her process; process ID, zaman damgası (timestamp) ve task detayı ile birlikte loglanmalı.

--- prjreq-L56 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 56:     • Task detayı en az: işlem tipi, işlenen yüz sayısı, tespit edilen face ID'ler ve status bilgilerini içermeli.

--- prjreq-L57 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 57:     • Loglar kalıcı olarak saklanmalı ve sorgulanabilir olmalı.

--- prjreq-L58 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 58:     • Loglama, ana işlemin başarısını engellememeli (hata olsa bile işlem sonucu dönmeli).

--- prjreq-L60 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 60: 8. Geçmiş / İlişki Sorgulama

--- prjreq-L61 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 61:     • Belirli bir face ID'nin daha önce hangi process'lerde ve ne zaman göründüğü sorgulanabilmeli.

--- prjreq-L62 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 62:     • Sonuç, ilgili process ID'leri ve zaman damgalarını içermeli.

--- prjreq-L63 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 63:     • Belirli bir process ID'ye ait detayların geri çağrılması mümkün olmalı.

--- prjreq-L65 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 65: 9. API Davranışı

--- prjreq-L66 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 66:     • Sadece API olarak çalışmalı; herhangi bir kullanıcı arayüzü olmayacak.

--- prjreq-L67 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 67:     • Her endpoint için input/output contract’ı tanımlanmalı.

--- prjreq-L68 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 68:     • Cevaplar yapısal ve tutarlı bir formatta dönülmeli (örn. process ID + yüz listesi + her yüz için ID, status, isim, konum, skor).

--- prjreq-L69 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 69:     • Hata durumları standart ve ayırt edilebilir şekilde raporlanmalı.

--- prjreq-L71 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 71: 10. Örnek API Endpoint'leri

--- prjreq-L72 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 72:     • POST /faces/recognize – Request ile gönderilen görüntü için tespit edilen yüzleri (face ID, status, isim, konum, skor) ve process ID'yi döner.

--- prjreq-L73 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 73:     • POST /faces/enroll – Bir yüzü/kişiyi isimle kaydeder; mevcut anonim ID'yi isimlendirebilir.

--- prjreq-L74 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 74:     • GET /faces/{faceId} – Bir face ID'nin detaylarını (status, isim, metadata) döner.

--- prjreq-L75 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 75:     • DELETE /faces/{faceId} – Bir face ID'yi siler.

--- prjreq-L76 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 76:     • GET /faces/{faceId}/history – Bir face ID'nin geçmiş process'lerini ve zamanlarını döner.

--- prjreq-L77 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 77:     • GET /processes/{processId} – Bir process'in detaylarını ve sonucunu döner.

--- prjreq-L78 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 78: 11. Sonuç İçeriği

--- prjreq-L79 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 79:     • Her işlenmiş görüntü için: process ID, tespit edilen yüz sayısı.

--- prjreq-L80 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 80:     • Her yüz için:

--- prjreq-L81 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 81:         ◦ faceId – her zaman dolu (anonim de olsa).

--- prjreq-L82 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 82:         ◦ status – known / anonymous / new_anonymous.

--- prjreq-L83 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 83:         ◦ name – yalnızca known durumunda dolu, diğerlerinde null.

--- prjreq-L84 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 84:         ◦ metadata – kayıtlı kişiye ait ek bilgiler (varsa), anonimde boş.

--- prjreq-L85 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 85:         ◦ boundingBox – konum bilgisi.

--- prjreq-L86 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 86:         ◦ confidence – eşleşme güven skoru.

--- prjreq-L88 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 88: 12. Deployment – Docker

--- prjreq-L89 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 89:     • API, Docker üzerinde çalışabilecek şekilde paketlenmeli

--- prjreq-L90 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 90:     • Projede bir Dockerfile bulunmalı ve image sorunsuz build edilebilmeli.

--- prjreq-L91 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 91:     • Container ayağa kalktığında API herhangi bir manuel ek adım olmadan çalışır durumda olmalı. 

--- prjreq-L92 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 92:     • Yapılandırma (port, eşik değeri, veri yolu vb.) ortam değişkenleri (environment variables) ile dışarıdan verilebilmeli. 

--- prjreq-L93 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 93:     • Kalıcı veriler container yeniden başlatıldığında kaybolmamalı

--- prjreq-L94 [Memory] tags: line-by-line, MergenVision, prjreq ---
Dosya: ProjectRequirements.md | Satır 94:     • Birden fazla servis gerekiyorsa docker-compose ile tüm sistem ayağa kaldırılabilmeli.

--- videoreq-L1 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 1: Face Recognition API

--- videoreq-L2 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 2: Video – Ek Gereksinimler

--- videoreq-L4 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 4: Giriş

--- videoreq-L6 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 6: Bu doküman, tek görüntü üzerinden çalışan mevcut yüz tanıma servisinin video inputlarını da işleyebilecek şekilde genişletilmesini kapsar. Sistem, kendisine gönderilen bir videoyu kareler (frame) halinde işler, her karedeki yüzleri tespit eder, mevcut tanıma mantığıyla (known / anonymous / new_anonymous) kimliklendirir ve ardışık karelerde aynı kişiyi tutarlı biçimde takip eder. Çıktı, kare kare değil; videoda görünen kişiler bazında özetlenir. Böylece istemci, bir videoda kimlerin bulunduğunu, her kişinin ne zaman ve ne kadar süre göründüğünü net olarak görebilir.

--- videoreq-L8 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 8: Mevcut izlenebilirlik katmanı (process ID, loglama, geçmiş sorgulama) korunur ve video bağlamına genişletilir: her video işlemi bir işe (job) bağlanır, video metadata'sı (süre, fps, kare sayısı) loglanır ve bir yüzün hangi videonun hangi anında göründüğü geriye dönük sorgulanabilir.

--- videoreq-L10 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 10: Mevcut görüntü tabanlı tüm gereksinimler geçerliliğini korur; bu doküman yalnızca video inputuna özgü ek gereksinimleri tanımlar. Canlı akış (RTSP, webcam vb.) bu sürümün kapsamı dışındadır ancak mimari ileride bu girdi tipine genişletilebilecek şekilde tasarlanmalıdır.

--- videoreq-L12 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 12: 1. Video Input

--- videoreq-L13 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 13:     • API, request içerisinde bir video kabul edebilmeli

--- videoreq-L14 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 14:     • Videonun geçerli/desteklenen bir formatta olduğu doğrulanmalı (örn. mp4, avi, mov).

--- videoreq-L15 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 15:     • Video okunamıyor, bozuk veya boş ise anlamlı bir hata dönülmeli.

--- videoreq-L16 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 16:     • Videoda hiç yüz bulunamadığı durum hata değil; ayrı bir sonuç ("yüz bulunamadı") olarak ele alınmalı.

--- videoreq-L17 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 17:     • Maksimum dosya boyutu ve/veya süre limiti tanımlanmalı; limit aşıldığında net bir hata dönülmeli.

--- videoreq-L18 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 18:     • Gönderilen input video, belirlenen bir saklama süresi (retention) boyunca sistemde tutulmalı; bu süre sonunda otomatik olarak silinmeli.

--- videoreq-L19 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 19:     • Saklama süresi ve saklama yolu/dizini yapılandırılabilir olmalı (environment variable).

--- videoreq-L20 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 20:     • Saklanan videoya, ilgili job/process ID üzerinden erişilebilmeli (örn. sonradan yeniden işleme veya doğrulama için).

--- videoreq-L21 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 21:     • Bu limitler (boyut, süre, desteklenen formatlar, saklama süresi) yapılandırılabilir olmalı (environment variable).

--- videoreq-L22 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 22: 2. Frame Çıkarma / Örnekleme (Sampling)

--- videoreq-L23 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 23:     • Videodan işlenecek kareler bir örnekleme stratejisine göre seçilmeli (örn. her N karede bir veya saniyede X kare).

--- videoreq-L24 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 24:     • Örnekleme oranı, request parametresi ve/veya environment variable ile dışarıdan ayarlanabilmeli.

--- videoreq-L25 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 25:     • Her işlenen kare için video içi zaman bilgisi (timestamp / saniye ve kare numarası) tutulmalı.

--- videoreq-L26 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 26:     • Her karenin işlenmesi, mevcut tek görüntü tanıma mantığıyla aynı kurallara tabi olmalı (detection + recognition).

--- videoreq-L27 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 27: 3. Face Tracking

--- videoreq-L28 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 28:     • Ardışık karelerde görünen aynı yüz takip edilmeli ve aynı kişiye bir track ID atanmalı.

--- videoreq-L29 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 29:     • Track ID, mevcut kalıcı face ID ile ilişkilendirilmeli: tracking "karelerdeki bu yüz aynı obje mi", recognition "bu obje kim" sorusunu cevaplamalı.

--- videoreq-L30 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 30:     • Bir track boyunca karelerde farklı tanıma sonuçları çıkarsa (kimi karede known, kimi anonymous), güven/çoğunluk bazlı tek bir nihai karara bağlanmalı.

--- videoreq-L31 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 31: 4. Sonuç Toplama (Aggregation)

--- videoreq-L32 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 32:     • Çıktı kare kare değil; video içinde görünen benzersiz kişiler (track/face ID) bazında özetlenmeli.

--- videoreq-L33 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 33:     • Her kişi için en az: ilk göründüğü an, son göründüğü an, toplam görünme süresi ve göründüğü kare/zaman aralıkları dönülmeli.

--- videoreq-L34 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 34:     • Özet bilgisine ek olarak, her kişi için işlenen her karedeki bbox detayı da dönülmeli: kare numarası, video içi timestamp ve o karedeki bounding box. Bu detay, istemcinin video üzerine bbox çizebilmesi için gereklidir.

--- videoreq-L35 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 35:     • Her kişi için mevcut tanıma alanları korunmalı: faceId, status (known / anonymous / new_anonymous), name (yalnızca known'da dolu), metadata, confidence.

--- videoreq-L36 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 36:     • Videoda hem bilinen hem anonim kişiler aynı anda bulunabilmeli ve her biri ayrı sonuçlanmalı.

--- videoreq-L37 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 37:     • İlk kez görülen yüzler için mevcut new_anonymous mantığı işlemeli; bu anonim kayıtlar veritabanına eklenmeli ve sonraki videolarda aynı yüz tekrar gelirse aynı ID ile tanınmalı.

--- videoreq-L38 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 38: 5. Bounding Box ve Koordinat Sistemi

--- videoreq-L39 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 39:     • Sistem performans için videoyu küçülterek (downscale) işleyebilir; ancak dönülen tüm bounding box koordinatları orijinal video çözünürlüğüne göre verilmeli.

--- videoreq-L40 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 40:     • Yani işleme sırasında ölçekleme yapılsa bile, koordinat dönüşümü API tarafında yapılmalı; istemci herhangi bir ölçekleme/oran düzeltmesi yapmak zorunda kalmamalı.

--- videoreq-L41 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 41:     • Bounding box, istemcinin doğrudan orijinal video karesi üzerine çizebileceği şekilde dönülmeli (örn. x, y, width, height ya da sol/üst/sağ/alt köşe koordinatları net tanımlanmalı).

--- videoreq-L42 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 42: 6. Asenkron İşlem / Job Yönetimi

--- videoreq-L43 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 43:     • Video işleme uzun sürebileceğinden, işlem asenkron yürütülmeli: istek anında bir job ID dönülmeli, sonuç hemen beklenmemeli.

--- videoreq-L44 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 44:     • Job durumu sorgulanabilmeli: pending / processing / completed / failed ve mümkünse ilerleme yüzdesi.

--- videoreq-L45 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 45:     • İşlem tamamlandığında sonuç ayrı bir çağrı ile alınabilmeli.

--- videoreq-L46 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 46:     • Bir job'un iptal edilebilmesi desteklenmeli.

--- videoreq-L47 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 47: 7. İşlem Takibi ve Loglama

--- videoreq-L48 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 48:     • Mevcut process ID ve loglama mantığı korunmalı; her video işlemi bir process/job ile ilişkilendirilmeli.

--- videoreq-L49 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 49:     • Process detayına video metadata'sı eklenmeli: video süresi, fps, toplam kare sayısı ve işlenen kare sayısı.

--- videoreq-L50 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 50:     • Task detayı en az: işlem tipi (video), işlenen kişi sayısı, tespit edilen face ID'ler ve status bilgilerini içermeli.

--- videoreq-L51 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 51:     • Loglar kalıcı olarak saklanmalı ve sorgulanabilir olmalı; loglama ana işlemin başarısını engellememeli.

--- videoreq-L52 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 52: 8. Geçmiş / İlişki Sorgulama (Genişletme)

--- videoreq-L53 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 53:     • Belirli bir face ID'nin daha önce hangi videolarda ve o videoların hangi anlarında (timestamp) göründüğü sorgulanabilmeli.

--- videoreq-L54 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 54:     • Sonuç; ilgili process/job ID'leri, video referansları ve zaman bilgilerini içermeli.

--- videoreq-L55 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 55:     • Belirli bir job ID'ye ait video sonucunun ve detaylarının geri çağrılması mümkün olmalı.

--- videoreq-L56 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 56: 9. API Davranışı

--- videoreq-L57 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 57:     • Sadece API olarak çalışmalı; herhangi bir kullanıcı arayüzü olmayacak.

--- videoreq-L58 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 58:     • Her yeni endpoint için input/output contract'ı tanımlanmalı.

--- videoreq-L59 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 59:     • Cevaplar yapısal ve tutarlı bir formatta dönülmeli (örn. job ID + kişi listesi + her kişi için faceId, status, isim, görünme zamanları, kare bazlı kutu detayları, skor).

--- videoreq-L60 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 60:     • Hata durumları standart ve ayırt edilebilir şekilde raporlanmalı.

--- videoreq-L61 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 61:     • Video sonuç formatı, mevcut görüntü sonuç formatıyla uyumlu/tutarlı olmalı.

--- videoreq-L62 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 62: 10. Örnek API Endpoint'leri

--- videoreq-L63 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 63:     • POST /videos/recognize – Gönderilen videoyu işlemek üzere bir job oluşturur ve job ID döner.

--- videoreq-L64 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 64:     • GET /videos/jobs/{jobId} – Bir job'un durumunu (pending / processing / completed / failed) ve ilerlemesini döner.

--- videoreq-L65 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 65:     • GET /videos/jobs/{jobId}/result – Tamamlanmış bir job'un kişi bazlı sonucunu (faceId, status, isim, görünme zamanları, skor) döner.

--- videoreq-L66 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 66:     • DELETE /videos/jobs/{jobId} – Devam eden bir job'u iptal eder.

--- videoreq-L67 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 67:     • GET /faces/{faceId}/appearances – Bir face ID'nin hangi videolarda ve hangi anlarda göründüğünü döner.

--- videoreq-L68 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 68: 11. Sonuç İçeriği

--- videoreq-L69 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 69: Her işlenmiş video için: job/process ID, video metadata'sı (süre, fps, çözünürlük, işlenen kare sayısı) ve tespit edilen benzersiz kişi sayısı.

--- videoreq-L70 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 70: Her kişi için:

--- videoreq-L71 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 71:     • faceId – her zaman dolu (anonim de olsa).

--- videoreq-L72 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 72:     • trackId – kişinin video içindeki takip kimliği.

--- videoreq-L73 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 73:     • status – known / anonymous / new_anonymous.

--- videoreq-L74 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 74:     • name – yalnızca known durumunda dolu, diğerlerinde null.

--- videoreq-L75 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 75:     • metadata – kayıtlı kişiye ait ek bilgiler (varsa), anonimde boş.

--- videoreq-L76 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 76:     • firstSeen / lastSeen – kişinin videoda ilk ve son görüldüğü an.

--- videoreq-L77 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 77:     • appearances – kişinin göründüğü zaman aralıkları (başlangıç/bitiş timestamp ve kare bilgisi).

--- videoreq-L78 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 78:     • detections – kişinin işlenen her karedeki detayı; her biri: frame numarası, timestamp ve orijinal çözünürlüğe göre boundingBox. İstemci bu listeyi kullanarak video üzerine kutu çizebilir.

--- videoreq-L79 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 79:     • confidence – nihai eşleşme güven skoru.

--- videoreq-L80 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 80: 12. Örnek Response (JSON)

--- videoreq-L81 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 81: Aşağıdaki örnek, tamamlanmış bir video işleme job'unun sonucunu (GET /videos/jobs/{jobId}/result) temsil eder. Alan adları ve yapı bilgilendirme amaçlıdır; nihai contract uygulamada netleştirilmelidir. Bounding box koordinatları orijinal video çözünürlüğüne göredir.

--- videoreq-L83 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 83: {

--- videoreq-L84 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 84:   "jobId": "job_8f3c1a2e",

--- videoreq-L85 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 85:   "processId": "proc_5d9b7c10",

--- videoreq-L86 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 86:   "status": "completed",

--- videoreq-L87 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 87:   "video": {

--- videoreq-L88 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 88:     "duration": 42.5,

--- videoreq-L89 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 89:     "fps": 30,

--- videoreq-L90 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 90:     "width": 1920,

--- videoreq-L91 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 91:     "height": 1080,

--- videoreq-L92 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 92:     "totalFrames": 1275,

--- videoreq-L93 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 93:     "processedFrames": 128,

--- videoreq-L94 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 94:     "samplingRate": "every_10th_frame"

--- videoreq-L95 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 95:   },

--- videoreq-L96 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 96:   "personCount": 2,

--- videoreq-L97 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 97:   "persons": [

--- videoreq-L98 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 98:     {

--- videoreq-L99 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 99:       "faceId": "face_001",

--- videoreq-L100 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 100:       "trackId": "track_a1",

--- videoreq-L101 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 101:       "status": "known",

--- videoreq-L102 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 102:       "name": "Ahmet Yilmaz",

--- videoreq-L103 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 103:       "metadata": { "department": "Engineering" },

--- videoreq-L104 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 104:       "firstSeen": 1.2,

--- videoreq-L105 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 105:       "lastSeen": 12.8,

--- videoreq-L106 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 106:       "totalDuration": 11.6,

--- videoreq-L107 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 107:       "confidence": 0.94,

--- videoreq-L108 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 108:       "appearances": [

--- videoreq-L109 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 109:         { "start": 1.2, "end": 12.8, "startFrame": 36, "endFrame": 384 }

--- videoreq-L110 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 110:       ],

--- videoreq-L111 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 111:       "detections": [

--- videoreq-L112 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 112:         {

--- videoreq-L113 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 113:           "frame": 36,

--- videoreq-L114 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 114:           "timestamp": 1.2,

--- videoreq-L115 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 115:           "boundingBox": { "x": 640, "y": 220, "width": 180, "height": 180 },

--- videoreq-L116 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 116:           "confidence": 0.93

--- videoreq-L117 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 117:         },

--- videoreq-L118 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 118:         {

--- videoreq-L119 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 119:           "frame": 46,

--- videoreq-L120 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 120:           "timestamp": 1.53,

--- videoreq-L121 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 121:           "boundingBox": { "x": 648, "y": 224, "width": 182, "height": 181 },

--- videoreq-L122 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 122:           "confidence": 0.95

--- videoreq-L123 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 123:         }

--- videoreq-L124 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 124:       ]

--- videoreq-L125 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 125:     },

--- videoreq-L126 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 126:     {

--- videoreq-L127 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 127:       "faceId": "face_117",

--- videoreq-L128 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 128:       "trackId": "track_b2",

--- videoreq-L129 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 129:       "status": "new_anonymous",

--- videoreq-L130 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 130:       "name": null,

--- videoreq-L131 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 131:       "metadata": {},

--- videoreq-L132 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 132:       "firstSeen": 3.0,

--- videoreq-L133 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 133:       "lastSeen": 9.4,

--- videoreq-L134 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 134:       "totalDuration": 6.4,

--- videoreq-L135 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 135:       "confidence": 0.81,

--- videoreq-L136 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 136:       "appearances": [

--- videoreq-L137 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 137:         { "start": 3.0, "end": 9.4, "startFrame": 90, "endFrame": 282 }

--- videoreq-L138 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 138:       ],

--- videoreq-L139 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 139:       "detections": [

--- videoreq-L140 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 140:         {

--- videoreq-L141 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 141:           "frame": 90,

--- videoreq-L142 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 142:           "timestamp": 3.0,

--- videoreq-L143 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 143:           "boundingBox": { "x": 1100, "y": 300, "width": 160, "height": 160 },

--- videoreq-L144 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 144:           "confidence": 0.80

--- videoreq-L145 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 145:         }

--- videoreq-L146 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 146:       ]

--- videoreq-L147 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 147:     }

--- videoreq-L148 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 148:   ]

--- videoreq-L149 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 149: }

--- videoreq-L151 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 151: 13. Performans ve Ölçeklenme

--- videoreq-L152 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 152:     • Frame işleme paralel/batch olarak yürütülebilmeli (örn. kuyruk + worker mimarisi).

--- videoreq-L153 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 153:     • Eşzamanlı job sayısı, timeout ve kaynak limitleri yapılandırılabilir olmalı.

--- videoreq-L154 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 154:     • Örnekleme oranı, sistemin makul sürede yanıt verebilmesi için ayarlanabilir tutulmalı.

--- videoreq-L155 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 155: 14. Deployment – Docker

--- videoreq-L156 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 156:     • Video işleme bileşeni mevcut API ile birlikte Docker üzerinde çalışabilecek şekilde paketlenmeli.

--- videoreq-L157 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 157:     • Ayrı bir worker servisi gerekiyorsa, tüm sistem docker-compose ile tek seferde ayağa kaldırılabilmeli.

--- videoreq-L158 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 158:     • Tüm yapılandırılabilir parametreler (örnekleme oranı, eşik değeri, dosya/süre limitleri, video saklama süresi ve yolu, veri yolu, port, eşzamanlı job sayısı, timeout vb.) environment variable ile dışarıdan verilebilmeli; kod içinde sabit (hard-coded) değer bulunmamalı.

--- videoreq-L159 [Memory] tags: line-by-line, videoreq, MergenVision ---
Dosya: videorequirements.md | Satır 159:     • İşlenen videolara, job sonuçlarına ve anonim kayıtlara ait kalıcı veriler container yeniden başlatıldığında kaybolmamalı.

--- prjgoal-L1 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 1: PROJE BAĞLAMI — ÖNCE BUNU ANLA, SONRA IMPLEMENTASYONA BAŞLA

--- prjgoal-L3 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 3: Bu bölüm teknik görev listesinden üstündür. Buradaki amaç, yalnız dosyaları tamamlaman değil, ortaya çıkarmaya çalıştığımız ürünü anlamandır.

--- prjgoal-L5 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 5: ==================================================

--- prjgoal-L6 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 6: 1. MERGENVISION NEDİR?

--- prjgoal-L7 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 7: ==================================================

--- prjgoal-L9 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 9: MergenVision basit bir “yüz modeli çalıştırma demosu” değildir.

--- prjgoal-L11 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 11: Amacımız fotoğraf ve ileride video üzerinden çalışan, kalıcı kimlik yönetimi bulunan bir yüz tanıma platformu oluşturmaktır.

--- prjgoal-L13 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 13: Kullanıcı açısından ürünün temel sorusu şudur:

--- prjgoal-L15 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 15: “Bu görüntüde kimler var, sistem bunları daha önce gördü mü, gördüyse aynı kimliği koruyabiliyor mu?”

--- prjgoal-L17 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 17: Sistem yalnız cosine score döndürmeyecek. Kalıcı faceId, enrollment, geçmiş, process takibi ve storage lifecycle sağlayacaktır.

--- prjgoal-L19 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 19: Bağlayıcı gereksinimler:

--- prjgoal-L21 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 21: - requirements/ProjectRequirements.md

--- prjgoal-L22 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 22: - requirements/videorequirements.md

--- prjgoal-L24 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 24: İkisini tamamen oku. Bu açıklama onları değiştirmez; ürünün niyetini anlamana yardım eder.

--- prjgoal-L26 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 26: ==================================================

--- prjgoal-L27 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 27: 2. PHASE 1: FOTOĞRAF TABANLI KİMLİK SİSTEMİ

--- prjgoal-L28 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 28: ==================================================

--- prjgoal-L30 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 30: Phase 1’in kullanıcı akışı şöyledir:

--- prjgoal-L32 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 32: 1. Kullanıcı bir fotoğraf gönderir.

--- prjgoal-L33 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 33: 2. Sistem input’un gerçekten desteklenen ve decode edilebilir bir görüntü olduğunu doğrular.

--- prjgoal-L34 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 34: 3. Fotoğraftaki bütün yüzleri bulur.

--- prjgoal-L35 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 35: 4. Her yüzü bağımsız olarak tanımaya çalışır.

--- prjgoal-L36 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 36: 5. Tek API çağrısı için tek processId üretir.

--- prjgoal-L37 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 37: 6. Her yüz için bounding box ve identity sonucu döner.

--- prjgoal-L38 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 38: 7. Sonuçlar ve process geçmişi daha sonra sorgulanabilir.

--- prjgoal-L40 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 40: Bir görüntüde üç yüz varsa:

--- prjgoal-L42 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 42: - Üç ayrı process oluşturulmaz.

--- prjgoal-L43 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 43: - Bir process oluşturulur.

--- prjgoal-L44 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 44: - Üç recognition_result kaydı oluşur.

--- prjgoal-L45 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 45: - Her yüzün sonucu bağımsız olabilir.

--- prjgoal-L47 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 47: Örneğin aynı response içinde:

--- prjgoal-L49 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 49: - Bir kişi known

--- prjgoal-L50 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 50: - Bir kişi anonymous

--- prjgoal-L51 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 51: - Bir kişi new_anonymous

--- prjgoal-L53 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 53: olabilir.

--- prjgoal-L55 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 55: No-face bir sistem hatası değildir.

--- prjgoal-L57 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 57: Geçerli fakat yüz bulunmayan görüntü:

--- prjgoal-L59 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 59: - HTTP success döner.

--- prjgoal-L60 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 60: - process completed olur.

--- prjgoal-L61 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 61: - faceCount=0 olur.

--- prjgoal-L62 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 62: - faces=[] olur.

--- prjgoal-L64 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 64: Boş, bozuk veya desteklenmeyen dosya ise structured error döner. Raw CUDA, TensorRT veya decoder exception’ı kullanıcıya gösterilmez.

--- prjgoal-L66 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 66: ==================================================

--- prjgoal-L67 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 67: 3. IDENTITY SEMANTİĞİ

--- prjgoal-L68 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 68: ==================================================

--- prjgoal-L70 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 70: Üç recognition result status’u vardır:

--- prjgoal-L72 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 72: - new_anonymous

--- prjgoal-L73 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 73: - anonymous

--- prjgoal-L74 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 74: - known

--- prjgoal-L76 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 76: Bunları birbirine karıştırma.

--- prjgoal-L78 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 78: new_anonymous:

--- prjgoal-L80 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 80: Sistem bu yüzü daha önce tanımamıştır ve ilk defa kalıcı bir faceId oluşturmuştur.

--- prjgoal-L82 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 82: Bu yalnız ilk recognition sonucunun snapshot status’udur.

--- prjgoal-L84 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 84: Persistent face_identity state’i new_anonymous değildir. Oluşturulan identity PostgreSQL’de anonymous olarak tutulur.

--- prjgoal-L86 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 86: anonymous:

--- prjgoal-L88 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 88: Sistem bu yüzü daha önce görmüştür.

--- prjgoal-L90 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 90: Aynı kalıcı faceId bulunmuştur fakat kullanıcı henüz bu kimliğe isim vermemiştir.

--- prjgoal-L92 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 92: known:

--- prjgoal-L94 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 94: Aynı faceId kullanıcı tarafından enroll edilmiştir. Sonuç name ve metadata içerebilir.

--- prjgoal-L96 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 96: Beklenen lifecycle:

--- prjgoal-L98 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 98: İlk istek:

--- prjgoal-L99 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 99: new_anonymous

--- prjgoal-L100 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 100: faceId = A

--- prjgoal-L102 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 102: Aynı yüz ikinci kez:

--- prjgoal-L103 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 103: anonymous

--- prjgoal-L104 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 104: faceId = A

--- prjgoal-L106 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 106: Kullanıcı enroll eder:

--- prjgoal-L107 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 107: faceId = A korunur

--- prjgoal-L108 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 108: identity known olur

--- prjgoal-L110 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 110: Aynı yüz tekrar:

--- prjgoal-L111 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 111: known

--- prjgoal-L112 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 112: faceId = A

--- prjgoal-L113 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 113: name/metadata döner

--- prjgoal-L115 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 115: Enrollment yeni faceId oluşturmamalıdır.

--- prjgoal-L117 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 117: Eski recognition result’ları daha sonra değiştirilmemelidir. İlk sonuç sonsuza kadar new_anonymous snapshot’ı olarak kalır. Enrollment geçmişi yeniden yazmaz.

--- prjgoal-L119 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 119: ==================================================

--- prjgoal-L120 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 120: 4. BU SPRINTİN GERÇEK SONUCU

--- prjgoal-L121 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 121: ==================================================

--- prjgoal-L123 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 123: Şu anda Sprint 02’de gerçek GPU image identity vertical slice’ını yapıyoruz.

--- prjgoal-L125 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 125: Çalışması gereken zincir:

--- prjgoal-L127 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 127: JPEG

--- prjgoal-L128 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 128: → gerçek NVIDIA GPU runtime

--- prjgoal-L129 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 129: → nvJPEG decode

--- prjgoal-L130 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 130: → CUDA preprocessing

--- prjgoal-L131 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 131: → TensorRT RetinaFace R50

--- prjgoal-L132 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 132: → CUDA RetinaFace decode/NMS/landmarks

--- prjgoal-L133 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 133: → CUDA five-point face alignment

--- prjgoal-L134 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 134: → TensorRT GlintR100

--- prjgoal-L135 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 135: → CUDA L2-normalized 512-D embedding

--- prjgoal-L136 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 136: → Python application service

--- prjgoal-L137 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 137: → PostgreSQL

--- prjgoal-L138 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 138: → MinIO

--- prjgoal-L139 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 139: → Qdrant

--- prjgoal-L140 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 140: → FastAPI response

--- prjgoal-L142 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 142: Kullanılacak modeller dondurulmuştur:

--- prjgoal-L144 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 144: Detector:

--- prjgoal-L145 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 145: backend/artifacts/models/retinaface_r50_dynamic.onnx

--- prjgoal-L147 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 147: Recognizer:

--- prjgoal-L148 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 148: backend/artifacts/models/glintr100.onnx

--- prjgoal-L150 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 150: Başka modele geçme.

--- prjgoal-L152 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 152: Bu sprintin sonunda senior’a şu gerçek demo gösterilebilmelidir:

--- prjgoal-L154 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 154: 1. API çalıştırılır.

--- prjgoal-L155 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 155: 2. No-face JPEG gönderilir ve başarılı boş sonuç alınır.

--- prjgoal-L156 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 156: 3. Tek yüzlü JPEG gönderilir.

--- prjgoal-L157 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 157: 4. Sistem new_anonymous ve bir faceId döndürür.

--- prjgoal-L158 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 158: 5. Aynı JPEG tekrar gönderilir.

--- prjgoal-L159 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 159: 6. Sistem aynı faceId ile anonymous döndürür.

--- prjgoal-L160 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 160: 7. faceId enroll edilir.

--- prjgoal-L161 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 161: 8. Aynı JPEG tekrar gönderilir.

--- prjgoal-L162 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 162: 9. Sistem aynı faceId ile known döndürür.

--- prjgoal-L163 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 163: 10. Multi-face JPEG gönderilir.

--- prjgoal-L164 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 164: 11. Bütün yüzler tek processId altında bağımsız sonuçlanır.

--- prjgoal-L165 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 165: 12. Process ve face history endpoint’lerinden sonuçlar okunur.

--- prjgoal-L166 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 166: 13. PostgreSQL, MinIO ve Qdrant restart sonrasında kayıtlar korunur.

--- prjgoal-L168 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 168: Yalnız engine dosyası üretmek bu sprinti tamamlamaz.

--- prjgoal-L170 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 170: Yalnız detector bounding box çıktısı üretmek tamamlamaz.

--- prjgoal-L172 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 172: Yalnız sentetik embedding testi tamamlamaz.

--- prjgoal-L174 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 174: Gerçek HTTP → gerçek GPU → gerçek storage lifecycle aynı zincirde çalışmalıdır.

--- prjgoal-L176 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 176: ==================================================

--- prjgoal-L177 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 177: 5. STORAGE’LARIN ROLÜ

--- prjgoal-L178 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 178: ==================================================

--- prjgoal-L180 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 180: PostgreSQL business source-of-truth’tür.

--- prjgoal-L182 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 182: Şunları tutar:

--- prjgoal-L184 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 184: - face identity

--- prjgoal-L185 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 185: - current known/anonymous state

--- prjgoal-L186 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 186: - process

--- prjgoal-L187 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 187: - sample lifecycle

--- prjgoal-L188 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 188: - immutable recognition result

--- prjgoal-L189 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 189: - history

--- prjgoal-L191 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 191: PostgreSQL’e yazılmayacak:

--- prjgoal-L193 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 193: - image binary

--- prjgoal-L194 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 194: - crop binary

--- prjgoal-L195 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 195: - embedding vector

--- prjgoal-L197 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 197: MinIO binary owner’dır.

--- prjgoal-L199 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 199: Bu sprintte aligned face crop saklar:

--- prjgoal-L201 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 201: faces/{faceId}/{sampleId}/aligned.webp

--- prjgoal-L203 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 203: Object key içinde isim, metadata veya kişisel bilgi bulunmaz.

--- prjgoal-L205 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 205: Qdrant derived vector index’tir.

--- prjgoal-L207 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 207: Şunları tutar:

--- prjgoal-L209 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 209: - 512-D GlintR100 embedding

--- prjgoal-L210 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 210: - sample_id

--- prjgoal-L211 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 211: - face_id

--- prjgoal-L212 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 212: - active

--- prjgoal-L213 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 213: - model_version

--- prjgoal-L215 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 215: Qdrant’a yazılmayacak:

--- prjgoal-L217 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 217: - name

--- prjgoal-L218 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 218: - identity metadata

--- prjgoal-L219 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 219: - MinIO object key

--- prjgoal-L220 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 220: - history

--- prjgoal-L221 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 221: - raw image

--- prjgoal-L223 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 223: Qdrant point ID tam olarak face_sample.sample_id olmalıdır.

--- prjgoal-L225 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 225: Eski synthetic vector’larla yeni GlintR100 vector’ları karıştırılmamalıdır. Bu nedenle yeni collection kullanılacaktır:

--- prjgoal-L227 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 227: face_samples_retinaface_r50_glintr100_v1

--- prjgoal-L229 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 229: Eski collection silinmeyecek veya resetlenmeyecektir.

--- prjgoal-L231 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 231: ==================================================

--- prjgoal-L232 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 232: 6. GPU HOT PATH NEDEN BÖYLE TASARLANIYOR?

--- prjgoal-L233 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 233: ==================================================

--- prjgoal-L235 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 235: Python business orchestration içindir.

--- prjgoal-L237 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 237: Python’ın görevi:

--- prjgoal-L239 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 239: - API

--- prjgoal-L240 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 240: - process yönetimi

--- prjgoal-L241 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 241: - identity kararı

--- prjgoal-L242 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 242: - PostgreSQL

--- prjgoal-L243 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 243: - MinIO

--- prjgoal-L244 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 244: - Qdrant

--- prjgoal-L245 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 245: - error mapping

--- prjgoal-L246 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 246: - response oluşturma

--- prjgoal-L248 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 248: Native C++/CUDA runtime’ın görevi:

--- prjgoal-L250 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 250: - decode

--- prjgoal-L251 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 251: - preprocess

--- prjgoal-L252 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 252: - detect

--- prjgoal-L253 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 253: - NMS

--- prjgoal-L254 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 254: - landmarks

--- prjgoal-L255 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 255: - alignment

--- prjgoal-L256 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 256: - embedding

--- prjgoal-L257 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 257: - L2 normalization

--- prjgoal-L259 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 259: Production path’te Python/OpenCV/Pillow ile full image decode ve resize istemiyoruz.

--- prjgoal-L261 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 261: Raw TensorRT detector output’unu NumPy’ya taşıyıp CPU postprocess yapmak istemiyoruz.

--- prjgoal-L263 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 263: Full decoded frame’i GPU’dan CPU’ya geri taşımak istemiyoruz.

--- prjgoal-L265 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 265: Python’a yalnız şu compact sonuçlar çıkmalıdır:

--- prjgoal-L267 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 267: - original image dimensions

--- prjgoal-L268 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 268: - bounding boxes

--- prjgoal-L269 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 269: - five landmarks

--- prjgoal-L270 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 270: - detector confidence

--- prjgoal-L271 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 271: - 512-D normalized embedding

--- prjgoal-L272 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 272: - küçük aligned crop

--- prjgoal-L273 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 273: - timing/model metadata

--- prjgoal-L275 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 275: Native runtime identity kararı vermez.

--- prjgoal-L277 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 277: Native taraf:

--- prjgoal-L279 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 279: - known demez

--- prjgoal-L280 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 280: - anonymous demez

--- prjgoal-L281 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 281: - faceId üretmez

--- prjgoal-L282 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 282: - name/metadata bilmez

--- prjgoal-L284 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 284: Bunlar application/business layer kararıdır.

--- prjgoal-L286 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 286: ==================================================

--- prjgoal-L287 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 287: 7. DYNAMIC BATCH’İN AMACI

--- prjgoal-L288 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 288: ==================================================

--- prjgoal-L290 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 290: Current image API bir request’te bir fotoğraf işler.

--- prjgoal-L292 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 292: Buna rağmen engine’leri yalnız batch=1’e kilitlemiyoruz.

--- prjgoal-L294 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 294: RetinaFace profile:

--- prjgoal-L296 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 296: - min=1

--- prjgoal-L297 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 297: - opt=4

--- prjgoal-L298 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 298: - max=8

--- prjgoal-L299 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 299: - spatial size 640×640

--- prjgoal-L301 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 301: GlintR100 profile:

--- prjgoal-L303 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 303: - min=1

--- prjgoal-L304 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 304: - opt=8

--- prjgoal-L305 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 305: - max=32

--- prjgoal-L306 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 306: - crop size 112×112

--- prjgoal-L308 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 308: Şimdiki image API RetinaFace’e batch=1 verir.

--- prjgoal-L310 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 310: Dynamic detector batch’in amacı gelecekte video pipeline’da birden fazla decoded frame’i aynı native core’a verebilmektir.

--- prjgoal-L312 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 312: GlintR100 batch’in amacı bir veya birden fazla görüntüde bulunan bütün yüz crop’larını verimli biçimde embed etmektir.

--- prjgoal-L314 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 314: 32’den fazla crop varsa deterministic şekilde chunk edilir; yüzler sessizce atılmaz.

--- prjgoal-L316 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 316: Batch çalışırken detection ile embedding association kaybolmamalıdır.

--- prjgoal-L318 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 318: Her detection şunları korumalıdır:

--- prjgoal-L320 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 320: - hangi source image/frame’den geldiği

--- prjgoal-L321 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 321: - detection index

--- prjgoal-L322 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 322: - bbox

--- prjgoal-L323 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 323: - landmarks

--- prjgoal-L324 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 324: - embedding

--- prjgoal-L326 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 326: ==================================================

--- prjgoal-L327 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 327: 8. GELECEKTE VİDEO NASIL EKLENECEK?

--- prjgoal-L328 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 328: ==================================================

--- prjgoal-L330 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 330: videorequirements.md gelecekteki video ürününü tarif eder.

--- prjgoal-L332 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 332: Video tarafında kullanıcı:

--- prjgoal-L334 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 334: - Video upload edecek.

--- prjgoal-L335 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 335: - Async jobId alacak.

--- prjgoal-L336 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 336: - Job status/progress sorgulayacak.

--- prjgoal-L337 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 337: - Sistem videoyu örnekleyerek işleyecek.

--- prjgoal-L338 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 338: - PTS/time-base doğru tutulacak.

--- prjgoal-L339 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 339: - Aynı kişiye ait ardışık detection’lar track/tracklet olacak.

--- prjgoal-L340 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 340: - Track boyunca en kaliteli yüz örnekleri seçilecek.

--- prjgoal-L341 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 341: - Track template embedding oluşturulacak.

--- prjgoal-L342 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 342: - Known/anonymous identity kararı verilecek.

--- prjgoal-L343 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 343: - Appearance interval/timeline üretilecek.

--- prjgoal-L344 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 344: - İstenirse annotated output oluşturulacak.

--- prjgoal-L345 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 345: - Cancel/retry/restart davranışı olacak.

--- prjgoal-L347 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 347: Fakat bunların hiçbiri Sprint 02’de implement edilmeyecek.

--- prjgoal-L349 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 349: Bu sprintin video açısından görevi, tekrar kullanılabilir bir native çekirdek sağlamaktır:

--- prjgoal-L351 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 351: Image bugün:

--- prjgoal-L352 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 352: nvJPEG → DeviceImageView → FacePipeline

--- prjgoal-L354 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 354: Video gelecekte:

--- prjgoal-L355 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 355: NVDEC/DeepStream → DeviceImageView batch → aynı FacePipeline

--- prjgoal-L357 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 357: Böylece video sprintinde detector, alignment ve recognizer’ı yeniden yazmayacağız. Yalnız video decode, batching, timestamps ve tracking katmanlarını ekleyeceğiz.

--- prjgoal-L359 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 359: TrackId ile faceId aynı şey değildir:

--- prjgoal-L361 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 361: - trackId yalnız bir video/job içindeki yerel hareket kimliğidir.

--- prjgoal-L362 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 362: - faceId global persistent identity’dir.

--- prjgoal-L364 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 364: Native detector trackId veya faceId üretmemelidir.

--- prjgoal-L366 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 366: ==================================================

--- prjgoal-L367 [Memory] tags: line-by-line, MergenVision, prjgoal ---
Dosya: ProjectGoalandContext.md | Satır 367: 9. LABIN ROLÜ

--- prompt-phase2-full-slice [Memory] tags: mergenvision, compaction-retrieve, prompt, phase2 ---
prompt10.txt (Phase 2 full video identity vertical slice) özeti:
- Önceki Sprint 02 image-only non-goal’ı supersede eder; aktif görev Phase 2 video ürününü end-to-end tamamlamak.
- Native baseline korunur: Friends.mp4 6665 frame, 9020 detection, 150 raw track, batch 16 ~529 FPS; default detector temporal batch = 16.
- Zincir: POST video upload -> PG video_asset/job -> MinIO original -> worker lease -> native GPU (GStreamer/NVDEC, RetinaFace, alignment, GlintR100, L2) -> quality tracks -> per-track templates -> conservative reconciliation -> canonical video identities -> Qdrant search -> PG lifecycle -> known/anonymous/new_anonymous -> PG track/appearance persistence -> MinIO observation artifact -> result/timeline API -> React canvas overlay.
- Annotated MP4 yok; NVENC yok; UI orijinal video + metadata overlay.
- Native runtime known/anonymous kararı vermez, faceId oluşturmaz; sadece compact visual evidence üretir.
- Application layer reconcile, template, Qdrant arama, PG lifecycle doğrulama, faceId kararı verir.

--- prompt-tool-discipline [Memory] tags: mergenvision, compaction-retrieve, prompt, tools ---
prompt9.txt (Tool / MCP disiplini) özeti:
- Kod yazmadan önce ve her büyük karar noktasında aktif araç kullan.
- codebase-memory-mcp / mem0: yeni görev, context reseti, refactor, test hatasında repo call graph/context ara.
- context7: FastAPI, SQLAlchemy, Alembic, GStreamer, DeepStream, TensorRT, CUDA, Protobuf, MinIO, Qdrant version-sensitive API doğrula.
- deepwiki + gerçek upstream source: NVIDIA DeepStream, GStreamer, ByteTrack, RetinaFace, TensorRT pattern adapte etmeden önce.
- exa: official/upstream source eksikse.
- postman: API endpoint acceptance; playwright: UI overlay E2E.
- 21st ve Ruflo kesinlikle yasak.
- Her final raporda araçları USED/SKIPPED/FORBIDDEN_NOT_USED formatında yaz; göstermelik çağrı yapma.

--- prompt-zero-retinafix [Memory] tags: mergenvision, compaction-retrieve, prompt, debug ---
prompt8.txt (Zero RetinaFace detection systematic fix) özeti:
- Root cause: NVDEC NV12 NvBufSurface, RGBA kabul eden CUDA kernel’e veriliyor; pitch 2048 NV12 Y plane gösteriyor.
- Hemen threshold düşürme; önce surface contract’ını kanıtla: caps, colorFormat, planeParams, pitch.
- Aşama A: GPU RGBA oracle path (nvvideoconvert -> RGBA NVMM -> nvstreammux -> appsink -> mevcut RGBA kernel) ile root cause kanıtla.
- Aşama B: production fused NV12 path; yeni kernel mergenvision_preprocess_detector_nv12_batch: Y/UV plane pointer, pitch, width/height, color matrix, bilinear sampling, letterbox, channel reorder, normalization.
- Model preprocess contract’ını image runtime’dan doğrula: RGB/BGR, NCHW/NHWC, pixel range, normalize, letterbox fill.
- Bilinen yüzle kısa H.264 debug fixture oluştur; detector correctness oracle olarak kullan.

--- prompt-m5-native-gpu [Memory] tags: mergenvision, compaction-retrieve, prompt, native-gpu ---
prompt6.txt (M5 real native GPU video pipeline) özeti:
- Mevcut M0-M4 foundation üzerinde gerçek NVIDIA GPU video data plane implemente et.
- Zincir: POST video upload -> MinIO asset + PG video_job -> worker lease/claim -> GStreamer/NVDEC GPU decode -> RetinaFace R50 -> CUDA 5-point alignment -> GlintR100 -> GPU L2 -> tracker -> protobuf/zst observation artifact -> PG track/tracklet persistence.
- Python control plane; C++/CUDA/TensorRT data plane. Python’a full decoded frame, JPEG, raw tensor, NVMM surface, NumPy/OpenCV/PIL geçmez.
- Yasak: cv2.VideoCapture, OpenCV/PIL decode, FFmpeg software fallback, ONNX CPU fallback, InsightFace FaceAnalysis, frame→JPEG→image API, sessiz CPU fallback.
- Container/runtime gate: mergenvision/deepstream-dev:9.0 image inspect, nvidia-smi, engine manifest (ONNX/engine SHA, profile, versions).
- Dynamic TensorRT profile: RetinaFace 1/4/8 x 3x640x640; GlintR100 1/8/32 x 3x112x112.

--- prompt-m2-m3-m4-m5 [Memory] tags: mergenvision, compaction-retrieve, prompt, phase2 ---
prompt5.txt (M2 correction + M3 job lease/worker + M4 device pipeline + M5 real video observation) özeti:
- Verdict PARTIAL; M2 local çalışıyor ama fresh-checkout/idempotency/failure safety açık.
- P0: backend/app/infrastructure/persistence/sqlalchemy/models/video_asset.py, video_job.py, idempotency_record.py GitHub’da yok çünkü root .gitignore models/ kuralı unanchored; düzelt.
- Video upload idempotency: _request_hash video bytes içermiyor; same key + different video conflict olmalı.
- Asset dış storage işleminden sonra DB’ye ekleniyor -> orphan risk; sırayı düzelt.
- 0005 migration ile process terminal states, video_job lease/attempt constraints, result_manifest zorunlulukları ekle.
- M3/M4/M5 aynı görev içinde otonom ilerle; research/video_reference_lab frozen kalacak.
- Frontend freeze; backend/native işine frontend feature ekleme.

--- prompt-project-purpose-lab [Memory] tags: mergenvision, compaction-retrieve, prompt, data-model ---
prompt3.txt (Önce bunu anla — proje amacı) özeti:
- Nihai ürün: kullanıcı video yükler, MinIO’ya kaydedilir, worker analiz eder, her karedeki yüzler bulunur, aynı yüz tracklet altında toplanır, parçalı tracklet’ler canonical track altında birleştirilir, gallery eşleşmesiyle etiketlenir, frontend orijinal video üzerine overlay çizer.
- Veri modeli hiyerarşisi: Video frame -> Face observation -> Raw tracklet -> Canonical track -> Gallery decision -> Known/Unknown.
- Tracker, recognizer ve reconciliation farklı soruları cevaplar; görevlerini karıştırma.
- Video lab, GPU hot path’ten önce yapılır çünkü aynı anda decoder/CUDA/TensorRT/tracker/gallery ile uğraşmak hatayı lokalize edilemez yapar.
- Önce immutable frame/observation/embeddings artifact üret; sonra tracking/reconciliation sadece bunları okusun.
- Friends.mp4 sadece "video okundu" testi değil; bbox, landmark, alignment, embedding, tracker, reconciliation, gallery kararlarını gözle/sayısal doğrula.

--- prompt-video-reference-lab [Memory] tags: mergenvision, compaction-retrieve, prompt, video-lab ---
prompt2.txt (Sprint 002 video reference lab) özeti:
- Kullanıcı onaylı izole Python referans/doğruluk laboratuvarı; production GPU pipeline değil, oracle görevi.
- Friends.mp4’ü Python/ONNX referans path ile bir kez işle; observation’ları dondur.
- Tracking ve reconciliation’ı tekrar decode/inference yapmadan replay et.
- Raw tracklet, canonical track ve persistent faceId ayrımını kanıtla.
- Görsel contact sheet, diagnostic, overlay metadata ve debug annotated video üret.
- research/video_reference_lab/ altında izole proje; backend/pyproject.toml değiştirme.
- Scope: Python video decoding, detection/alignment/embedding, frozen artifact, tracking, reconciliation, gallery matching, ground-truth eval, visual diagnostics.
- Yasak: PostgreSQL schema, migration, MinIO/Qdrant lifecycle, FastAPI, C++/CUDA, GStreamer, React, job orchestration.

--- prompt-phase1-correction [Memory] tags: mergenvision, compaction-retrieve, prompt, phase1 ---
prompt.txt (Phase 1 Sprint 01 forensic correction) özeti:
- Mevcut Sprint 01 storage-foundation correction’ı uygula; yeni sprint başlatma.
- Hard stop: repo origin/HEAD kontrolü, commit/push/history rewrite/volume silme yasak.
- Zorunlu okuma: AGENTS.md, CURRENT_SPRINT.md, requirements, architectureplan, references, Makefile, docker-compose, pyproject, domain/ports, IdentityStorageLifecycleService, SQLAlchemy repositories, Alembic migrations, adapters, tests.
- Görev sınırı: sadece dört tablo (face_identity, process_record, face_sample, recognition_result); yeni tablo, FastAPI endpoint, UI, GPU, video ekleme.
- Dedicated test compose (docker-compose.test.yml) ve fail-closed resource guard oluştur; test cleanup sadece dedicated namespace üzerinde.
- Forward repo hygiene: .gitignore’daki models/ kuralını daralt; production ORM modülleri ignore edilmemeli.
- Synchronous compensation zorunlu; yakalanan hata sahte completed/orphan vector bırakamaz.

--- agents-process-completion [Memory] tags: mergenvision, compaction-retrieve, agents ---
AGENTS.md süreç ve completion kuralları:
- TDD sırası: failing test -> min implementasyon -> unit test -> integration test -> PG/MinIO/Qdrant/GPU smoke -> lint/type -> review.
- Reference-first çalış; model hafızasından uydurma. Reference decision log tut.
- MCP disiplini: codebase-memory (discovery), context7 (version-sensitive API), deepwiki (upstream mimari), exa (primary source), postman (API acceptance), playwright (UI E2E); 21st ve Ruflo yasak.
- Skill akışı: using-superpowers -> brainstorming -> writing-plans -> executing-plans -> test-driven-development -> systematic-debugging -> verification-before-completion -> requesting-code-review.
- Her sprint cohesive vertical outcome üretir; report-only sprint olmaz.
- Completion verdict sadece PASS/PARTIAL/BLOCKED/NOT_TESTED; kanıtsız üretim-ready/600 FPS/fully optimized denmez.
- Memory/context kullanımı: sadece kullanıcı açıkça hatırla dediğinde; otomatik snapshot yasak.

--- agents-architecture-rules [Memory] tags: mergenvision, compaction-retrieve, agents ---
AGENTS.md mimari kuralları:
- Python control plane: FastAPI, domain, orchestration, PostgreSQL, MinIO, Qdrant, tracking, reconciliation.
- Native data plane: GStreamer/DeepStream, NVDEC/NVMM, CUDA preprocess/alignment, TensorRT detector/recognizer.
- UI sadece versioned API tüketir; domain outer layer’a bağımlı olmaz.
- GPU hot path: encoded video -> GStreamer -> NVDEC -> DeepStream batch -> RetinaFace -> alignment -> GlintR100 -> L2 -> compact metadata. Full-frame CPU decode, NumPy postprocess, sessiz CPU fallback yasaktır.
- Tracklet/track/faceId ayrımı: rawTrackletId (kesintisiz segment), trackId (video içi canonical kişi grubu), faceId (kalıcı global identity), detectionId (tek frame observation).
- ID’ler UUIDv7; Idempotency-Key desteklenir.
- PostgreSQL authoritative; MinIO binary; Qdrant rebuildable vector index. Cross-store consistency idempotent state machine ile sağlanır.
- Image workflow: POST /faces/recognize bağımsız işler; mixed known/anonymous/new_anonymous tek response’ta.

--- agents-core-governance [Memory] tags: mergenvision, compaction-retrieve, agents ---
AGENTS.md çekirdek governance:
- Ürün misyonu: görüntü/video çoklu yüz tespiti + kalıcı faceId; ilk new_anonymous, sonra anonymous, enroll sonrası known.
- Source-of-truth sırası: kullanıcı kararı -> ProjectRequirements -> videorequirements -> onaylı architecture -> CURRENT_SPRINT -> vendor docs -> eski repo sadece lessons-learned.
- Ürün sınırı: backend API-first; React UI sadece kontrollü extension; çıktı orijinal video + overlay metadata, annotated MP4 değil.
- Implementation sırası atlanmaz: req freeze -> PG/MinIO/Qdrant -> image vertical slice -> enrollment -> video upload/job -> native GPU extraction -> tracking/reconciliation -> persistence -> aggregation/overlay -> UI/E2E.
- Her görev başlangıcı: repo/branch doğrula, AGENTS.md oku, CURRENT_SPRINT/requirements oku, dirty worktree koru, codebase-memory ile call graph keşfet.

--- ctx-projectgoalandcontext [Memory] tags: mergenvision, compaction-retrieve, context ---
ProjectGoalandContext.md özet:
- MergenVision: fotoğraf/video üzerinden kalıcı kimlik yönetimi olan yüz tanıma platformu.
- Ürün sorusu: "Bu görüntüde kimler var, sistem bunları daha önce gördü mü, aynı kimliği koruyabiliyor mu?"
- Phase 1 fotoğraf akışı: input validate -> tüm yüzleri bul -> her yüzü tanı -> tek processId -> bbox + identity sonucu.
- Lifecycle: new_anonymous (ilk istek) -> anonymous (sonraki) -> known (enroll); recognition result’lar immutable.
- Storage rolleri: PostgreSQL business source-of-truth (identity, process, result, history); MinIO binary; Qdrant derived vector index.
- GPU hot path: nvJPEG decode -> CUDA preprocess -> TensorRT RetinaFace R50 -> CUDA landmarks/alignment -> TensorRT GlintR100 -> CUDA L2 -> compact metadata CPU’ya geçer.
- Python control plane, native C++/CUDA data plane; native runtime identity kararı vermez.

--- req-videorequirements [Memory] tags: mergenvision, compaction-retrieve, requirements ---
requirements/videorequirements.md özet:
- Video input: mp4/avi/mov, dosya boyut/süre limiti, retention süresi saklanır.
- Frame sampling every_n_frames veya frames_per_second; her karenin timestamp/frame numarası tutulur.
- Face tracking: ardışık karelerde aynı yüze trackId; trackId ≠ faceId.
- Sonuç kişi bazında özetlenir: firstSeen, lastSeen, totalDuration, appearances, detections.
- Bounding box koordinatları orijinal video çözünürlüğüne göre döner.
- Asenkron job: POST /videos/recognize -> jobId; GET /videos/jobs/{jobId}/status, /result, DELETE iptal.
- Loglama video metadata (süre, fps, işlenen kare) ve face status’leri içerir.
- Deployment docker-compose, tüm parametreler env variable.

--- req-projectrequirements [Memory] tags: mergenvision, compaction-retrieve, requirements ---
requirements/ProjectRequirements.md özet:
- API-only yüz tanıma servisi; her request tek görüntü kabul eder.
- Her yüz için kalıcı faceId üretilir; status known / anonymous / new_anonymous.
- new_anonymous: ilk kez görülen yüz, sonraki eşleşmeler anonymous, enroll sonrası known olur.
- Bilinmeyen yüzler otomatik anonim kayıt oluşturulur; sonraki isteklerde aynı faceId ile tanınır.
- Endpointler: POST /faces/recognize, /faces/enroll, GET/DELETE /faces/{faceId}, /faces/{faceId}/history, /processes/{processId}.
- Her çağrıya benzersiz processId; işlem ve yüz geçmişi sorgulanabilir.
- No-face durumu success ama faceCount=0; bozuk/desteklenmeyen input structured error.
- Docker üzerinde çalışacak, env-based config, kalıcı veri container restartında korunacak.

--- hava-durumu-2026-07-18 [Memory] tags: weather, personal ---
rüzgar var hava soğuk

# Toplam 1936 node döndü. (limit: 100000)
