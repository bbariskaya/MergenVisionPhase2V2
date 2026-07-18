# Phase 1 Isolated Native GPU Bulk Enrollment & Benchmark Plan

## 1. Hedef (tek cümle)

MergenVisionPhase2v2'nin mevcut image/video kaynaklarına hiç dokunmadan, MergenVisionDemo'daki çalışan native GPU batch enrollment mimarisini kaynak üzerinden inceleyip `phase1/gpu_bulk_enrollment/` altında ayrı bir engine/package/CLI/Docker/test/benchmark altyapısı olarak yeniden kullanmak ve adaptasyonu tamamlamak.

## 2. Kısıtlar (binding)

- **Yazılabilir tek source prefix:** `phase1/gpu_bulk_enrollment/**`
- **Runtime artifact'ler:** `.artifacts/phase1_gpu_bulk_enrollment/**` veya package içi gitignored `artifacts/`
- **Read-only / protected:** `backend/**`, `frontend/**`, `research/**`, `requirements/**`, `docs/**` (root), `migrations/**`, `docker-compose.gpu.yml`, root `Makefile`, mevcut Dockerfile'lar, mevcut testler, MergenVisionDemo'nun tamamı.
- **Model'ler:** `backend/artifacts/models/retinaface_r50_dynamic.onnx` ve `backend/artifacts/models/glintr100.onnx` — read-only, başka model yok.
- **Model-only qwen kullanımı**, Anthropic override yok.
- **Ruflo / 21st forbidden.**
- Her büyük adımda `phase2-untouched-gate` çalıştırılacak.

## 3. Kaynaklanmış Gerçekler

### 3.1 Mevcut Phase 2 contract (read-only)

- **PersonOrm** (`backend/infrastructure/persistence/sqlalchemy/models/person.py`): `person_id` PK, `display_name`, `person_metadata` JSONB, `is_active`, `version >= 1`, `created_at/updated_at/deleted_at`.
- **FaceIdentityOrm** (`backend/infrastructure/persistence/sqlalchemy/models/face_identity.py`): `face_identity_id` PK, `person_id` nullable, `display_name` nullable, `status` enum, `is_active`, `redirect_to_face_id` nullable, check constraint'ler (known → person+display_name, redirect → inactive, active/deleted consistency).
- **FaceSampleOrm** (`backend/infrastructure/persistence/sqlalchemy/models/face_sample.py`): `sample_id` PK, `face_identity_id`, `person_id`, `object_key`, `bucket`, `model_version`, `preprocess_version`, `embedding_model`, `detector_model`, `bbox`, `landmarks`, `quality_score`, `status` (pending/active/failed/inactive), `activated_at`, check constraint'ler (active → bucket/object_key/activated_at not null).
- **MinIO key contract** (`backend/infrastructure/storage/minio_adapter.py`): teknik UUID object key, SHA-256 metadata, idempotent put, bucket zorunlu.
- **Qdrant contract** (`backend/infrastructure/vectors/qdrant_adapter.py`): 512-D, distance=cosine, payload index `face_id`/`active`/`model_version`, point ID = `str(sample_id)`, payload `sample_id`, `face_id`, `active`, `model_version`.
- **Settings** (`backend/infrastructure/config.py`): `model_version` zorunlu; `model_profile_path`, `detector_engine_path`, `recognizer_engine_path`, `gpu_device_id`, `inference_slot_count` mevcut. `preprocess_version` settings'ta yok ama `model_profile` JSON'da `preprocess_version` alanı var.
- **Native runtime** (`backend/native/image_runtime/`): mevcut `ExecutionSlot::infer_jpeg` tek görüntü inference'ı yapıyor. Batch path için MergenVisionDemo'nun native kernel/Python sarmalayıcısı adapte edilecek; mevcut `image_runtime` **değiştirilmeyecek**.

### 3.2 MergenVisionDemo referans (read-only)

- **GpuFacePipeline.extract_batch** (`backend/app/ml/gpu/face_pipeline.py:336-556`): encoded JPEG bytes listesi alır; decode→preprocess→detector→GPU postprocess→largest pick→GPU alignment→recognizer→GPU L2→compact sonuçlar.
- **BufferArena** (`backend/app/ml/gpu/buffer_arena.py`): shape/dtype anahtarlı GPU buffer havuzu, CUDA event fencing, lease acquire/release.
- **TrtDeviceEngine** (`backend/app/ml/gpu/trt_device_engine.py`): TensorRT engine wrapper, device seçimi, input/output profile.
- **BulkEnrollmentService** (`backend/app/services/bulk_enrollment.py:85-782`): producer/consumer queues, batched identity upsert, MinIO upload, PG bulk upsert, Qdrant batch upsert, deterministic ID'ler, checkpoint/commit progress.
- **Native module** (`backend/native/mergenvision_gpu/`): `pybind11` + `scikit-build-core`, CUDA kernels (decode, NMS, alignment, L2, argsort, warp, scale/compact).

### 3.3 Uyumsuzluk / adaptasyon kararları

- Phase 2 `Person.create` şu an `uuid4()` kullanıyor; Phase 1 deterministic UUIDv7/namespace tabanlı ID üretip Phase 2 schema'sına yazacak.
- Phase 2 `FaceIdentity` known/redirect/pending lifecycle'ına uygun kayıt üretilecek.
- Phase 2 `FaceSample.status` pending→active state machine korunacak.
- MinIO key: `enrollments/{person_id}/{sample_id}` formatına uygun teknik UUID key üretilecek.
- Qdrant point ID = `sample_id` (UUID), payload Phase 2 adapter'ının beklediği alanları taşıyacak.
- ONNX'ler Phase 2'nin mevcut `backend/artifacts/models/` altından read-only okunacak; engine build `phase1` artifact path'ine yazılacak.

## 4. 30 Adım Özeti

| Adım | Konu | Çıkış Kriteri |
|------|------|---------------|
| 1 | Repo baseline + protected tree manifest + prompt-memory context | Manifest var, gate green |
| 2 | Reference repo doğrulama + source map | `docs/REFERENCE_SOURCE_MAP.md` ilk hali |
| 3 | Exact reference flow çıkarımı | Buffer/path tahmin kalmayacak |
| 4 | Current target contracts (PG/MinIO/Qdrant/model) | Records Phase 2 tarafından aranabilir |
| 5 | Phase 1 package scaffold | Empty package import/tests green, gate green |
| 6 | Reference code adaptation | New package standalone configure edilebilir |
| 7 | Model contract inspection (ONNX) | Schema-valid model profile, SHA mismatch fail-fast |
| 8 | Separate engine builder | Engine deserialize + inference smoke |
| 9 | Batched JPEG decode | Valid/corrupt/mixed/no-face testleri |
| 10 | Batched RetinaFace | source_index, dimensions, letterbox, reverse mapping korunur |
| 11 | GPU detector postprocess | Decode/NMS/landmarks GPU'da, deterministic ordering |
| 12 | Batched alignment | 5 landmark, ArcFace template, 112×112, batch parity |
| 13 | Batched GlintR100 | Deterministic chunks, 512-D finite, GPU L2, norm≈1 |
| 14 | Full `extract_batch` native contract | N input → N result, per-image error isolation, multi-face quarantine |
| 15 | Scalar reference parity | Demo vs Phase1 batch=1 vs batch=N parity metrikleri |
| 16 | Dataset manifest reader | Read-only bind mount, relative paths, traversal koruması, SHA |
| 17 | Bounded producer/consumer | Bounded queues, backpressure, cancellation/failure testleri |
| 18 | Multi-GPU worker ownership | 1 worker = 1 GPU, stable sharding, hash() yasak, önce 1 GPU sonra 3 GPU |
| 19 | Admission & quality | Ayrı reason codes, subject içi duplicate/outlier/diversity |
| 20 | Deterministic identity mapping | source_namespace + external_subject_key + image SHA + model_version idempotency |
| 21 | PostgreSQL bulk persistence | Bulk upsert, bounded tx, existing reuse, pending→active |
| 22 | MinIO persistence | Bounded concurrent puts, teknik key, stat verification |
| 23 | Qdrant batch upsert | Existing collection, 512-D, point_id=sample_id, minimal payload |
| 24 | Cross-store finalization | IDs reserved → PG pending → MinIO → Qdrant → PG active → checkpoint |
| 25 | Small real E2E fixture | ≥2 person, multi-image, no-face, corrupt, duplicate, multi-face, holdout |
| 26 | Phase 1 → Phase 2 continuity | Black-box: Phase 2 image API/video pipeline aynı face_id/person_id döndürür mü |
| 27 | Benchmark matrix | Compute-only, one-GPU E2E, three-GPU E2E, scalar baseline, batch matrix |
| 28 | Long-run, retry, cleanup | Restart resume, duplicate yok, GPU memory baseline, container cleanup |
| 29 | Fresh checkout/container reproducibility | Pinned base digest, source'tan native build, engine build/load, small fixture |
| 30 | Final full acceptance + review | Tüm `make` hedefleri green, `phase2-untouched-gate` green, final rapor |

## 5. Public Native API Hedefi

```python
pipeline = GpuFacePipeline(...)
results = pipeline.extract_batch(image_bytes_list, source_keys=source_keys)
```

Her sonuç:
- `source_index`, `source_key`, `original_width`, `original_height`, `status`, `rejection_reason`, `faces[]`

Her yüz:
- `source_index`, `detection_ordinal`, `bbox_original`, `landmarks_original`, `detector_score`, `quality_primitives`, `embedding[512]`, `embedding_norm`, `crop_bytes`, `model_version`, `preprocess_version`

Durumlar: `accepted`, `no_face`, `multi_face`, `rejected`, `decode_error`, `inference_error`. Batch association korunur.

## 6. CLI Hedefi

- `mv-phase1-bulk inspect-models --profile config/model_profile.json`
- `mv-phase1-bulk build-engines --profile config/model_profile.json`
- `mv-phase1-bulk validate-manifest --dataset-root /dataset --manifest /dataset/manifest.jsonl`
- `mv-phase1-bulk enroll --dataset-root /dataset --manifest /dataset/manifest.jsonl --workers 1 --gpu-devices 0 --batch-size 16 --resume`
- `mv-phase1-bulk benchmark --dataset-root /dataset --manifest /dataset/benchmark.jsonl --gpu-devices 0,1,2 --batch-matrix 1,2,4,8,16 --runs 3`
- `mv-phase1-bulk reconcile --run-id <run-id>`
- `mv-phase1-bulk report --run-id <run-id>`

## 7. Test Matrisi Hedefi

- Unit: manifest, key normalization, sharding, ID reservation, admission, duplicate, queue backpressure, reporting.
- Native contract: source index, batch=1/N parity, partial batch, mixed-size, no-face, corrupt JPEG, multi-face, max crops, finite normalized embedding.
- Integration: real PG, MinIO, Qdrant, pending→active, partial failure, retry, resume, conflict.
- GPU: real TensorRT, real model artifacts, batch >1, no CPU fallback, memory cleanup, multiple runs.
- Acceptance: import, holdout recognition, Phase 2 image lookup, Phase 2 video lookup, one/three GPU benchmark, protected tree unchanged.

## 8. Riskler ve Olası Blocker'lar

| Risk | Etki | Azaltma |
|------|------|---------|
| ONNX batch>1 desteği yetersiz çıkarsa | Adım 10 blocker | Önce `onnx` shape inspection; dynamic batch profili ile TensorRT build denenecek |
| Phase 2 engine'leri Phase 1'e uyarlanamazsa | Adaptasyon risk | MergenVisionDemo'nun `mergenvision_gpu` kernel'leri ayrı native module olarak yeniden build edilecek |
| Phase 2 schema `preprocess_version` bekliyorsa ama settings'ta yok | Data contract risk | `model_profile.json`'dan `preprocess_version` alınacak, FaceSample kayıtlarına yazılacak |
| Person `uuid4()` mevcut `create` kullanıyor; Phase 1 deterministic ID | Idempotency risk | Phase 1 kendi deterministic UUID üretimini kullanır, mevcut `create` değiştirilmez |
| 3 GPU worker sharding deterministic olmazsa | Scale risk | Stable shard key (HMAC/SHA + mod), `hash()` yasak |
| Dataset/fixture eksikliği | Real E2E blocker | Unit/contract/integration testler kaynak fixture'larla ilerler; real E2E fixture eksikliği BLOCKED olarak raporlanır |

## 9. Onay Beklenen Kararlar

Plan modu onayı sonrası `ExitPlanMode` ile kullanıcı onayına sunulur. Onay alındığında Build Mode'da Adım 1'den başlanır. Kullanıcıdan adım adım mikro-onay istenmez; yalnızca gerçek blocker durumlarında durulur.

## 10. İlk Eylemler (Build Mode Adım 1)

1. `git status --short`, `git diff --name-only`, protected tree SHA manifest üret.
2. `phase1/gpu_bulk_enrollment/scripts/check_phase2_untouched.sh` gate şablonunu oluştur ve çalıştır.
3. `prompt-memory-mcp` üzerinden `MergenVision` → `MergenVisionDemo` ve `MergenVisionPhase2` context'ini yükle; yeni `phase1_gpu_bulk_enrollment` başlangıç checkpoint'i oluştur.
4. `phase1/gpu_bulk_enrollment/` boş scaffold'u oluştur; empty import testi.
