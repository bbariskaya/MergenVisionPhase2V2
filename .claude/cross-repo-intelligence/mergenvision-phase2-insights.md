# MergenVisionPhase2 Çapraz Repo İstihbarat Raporu

> **Kaynak proje:** `MergenVisionPhase2` (`/home/user/Workspace/MergenVisionPhase2`)  
> **Hedef proje:** `MergenVisionPhase2v2` (`/home/user/Workspace/MergenVisionPhase2v2`)  
> Bu rapordaki her kalıp, **ancak açıkça aktarıldığında** Phase2v2'de varsayılabilir; aksi halde harici referans olarak değerlendirilmelidir.

---

## Executive Summary

MergenVisionPhase2, yüklenen video dosyaları üzerinde çevrimdışı (offline) yüz tespiti, takip etme ve tanıma yapan bir doğruluk/performans laboratuvarıdır. Mevcut sürümde Python kontrol katmanı oldukça incedir; FastAPI uç noktaları henüz yerleşik değildir, gerçek iş yükünü `backend/native/` altındaki GStreamer/DeepStream/CUDA/TensorRT çalışanı yürütür. İş mantığı "ports & adapters" ve "domain-driven" katmanlara ayrılmıştır; takip kanıtlarını video boyunca birleştiren çevrimdışı reconciliation, Python domain katmanında uygulanmıştır. Kalıcılık (PostgreSQL, MinIO, Qdrant) gereksinim dokümanlarında tanımlıdır ancak kod tabanında henüz mevcut değildir; sonuçlar şimdilik yerel `output_dir` altındaki JSONL/JSON dosyalarına yazılmaktadır.

---

## Architecture

- **Katman haritası:** `backend/README.md`’ye göre:  
  `api` (FastAPI router’lar - gelecek sprint) → `application/services` (use-case) → `domain` (nesneler/kurallar) → `ports` (protokoller) → `infrastructure` (adaptörler) → `native` (C++/CUDA GPU data plane).
- **Giriş noktaları:**
  - `backend/app/cli.py` — `detect` komutu, CLI üzerinden tek video işler.
  - `backend/native/worker/main.cpp` — gerçek GPU çalışanı (tek video, tek GPU).
- **Frontend:** React 18 + Vite + TypeScript + React Router + TanStack Query; şimdilik sadece mock API ile çalışan UI prototipi.
- **Çalışma zamanı:** Derin öğrenme çalışanı NVIDIA DeepStream 9.0 (`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`) konteynerinde çalıştırılır; derleme `mergenvision/deepstream-dev:9.0` konteyneriyle yapılır.
- **Not:** `backend/app/api/__init__.py` yalnızca bir FastAPI yer tutucu açıklaması içerir; canlı HTTP API yoktur.

---

## Bulk Processing

- **Toplu kayıt (bulk enrollment) API’si yoktur.** Kimlik galerisi `backend/native/artifacts/gallery/gallery_centroids.json` gibi statik bir JSON dosyası olarak yüklenir.
- **Toplu çıkarım (batch inference) native çalışanda vardır:**
  - `WorkerOptions.batch_size` (`backend/native/worker/main.cpp:40`) ve `--batch-size N` argümanıyla ayarlanır.
  - `nvstreammux` üzerinde `batch-size` ve `batched-push-timeout` yapılandırılır.
  - `RetinaFacePostproc::processBatch` (`backend/native/worker/retinaface_postproc.cpp:218`) tek seferde çoklu karede NMS/landmark çıkarır.
- **Kısıt:** Tracker (`nvtracker`) `batch_size > 1` ile çalışmaz; `MV_ALLOW_TRACKER_BATCH` ortam değişkeni olmadan çalışan reddeder. `--mode fast` tracker’ı tamamen devre dışı bırakır.
- `Makefile`’da batch doğrulama hedefleri vardır: `backend-batch-parity`, `backend-batch-determinism`, `backend-batch-benchmark`.

---

## GPU / Native Runtime

- **Pipeline (ana hat):**  
  `filesrc` → `qtdemux` → `h264parse` → `nvv4l2decoder` (NVDEC) → `nvstreammux` → `nvdspreprocess` → `nvdsretinaface` → (`nvtracker`) → `nvvideoconvert` → `mvfacerecognizer` → `fakesink` veya render kolu.
- **Özel GStreamer eklentileri:**
  - `gst-nvdsretinaface`: TensorRT RetinaFace-R50 motorunu çalıştırır (`retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine`).
  - `gst-mvfacerecognizer`: Yüz kırpma, 5-nokta hizalama, TensorRT GlintR100 embedding, L2 normalize ve galeri eşleştirme yapar.
  - `gst-mvfacetracker`: ByteTrack tabanlı yerel iz (`tracklet`) üretimi için yer tutucu/yardımcı eklenti.
- **CUDA çekirdekleri (`backend/native/kernels/`):** `retinaface_decode`, `argsort`, `nms`, `scale_clip_compact_xy`, `l2_normalize`, `similarity_transform`, `warp_align`, `warp_align_rgba_pitch`.
- **Hot-path sözleşmesi:** Tam detector çıkış tensörü CPU’ya toplu çekilmez; yalnızca NMS sonrası yoğunlaştırılmış metadata (bbox, landmark, skor) kopyalanır (`retinaface_postproc.cpp` Stage 1/2).
- **İzleme:** `mv::tracking::ByteTracker` ve `mv::tracking::MultiSourceTracker`; embedding tabanlı maliyet matrisi destekli, Kalman filtresi + IOU + görünüm benzerliği.

---

## Persistence & Storage

- **Mevcut kodda kalıcı veri deposu yoktur.** Çalışan çıktıları şu dosyaları üretir:
  - `<output_dir>/detections.jsonl` — kare kare tespit ve tanıma meta verileri.
  - `<output_dir>/tracks.json` — tracker sonrası ham izler (tracker açıksa).
  - `<output_dir>/run_manifest.json` — GPU, sürücü, CUDA, DeepStream, batch, süre istatistikleri.
- **Hedef mimari (gereksinim ve `AGENTS.md`’de tanımlı):**
  - PostgreSQL: iş/kişilik/kayıt kaynağı.
  - MinIO: video ve ikili nesne sahibi.
  - Qdrant: yeniden oluşturulabilir embedding indeksi.
- İstenen kalıcılık özellikleri: deterministik ID’ler, idempotent retry, açık durum makinesi, sınırlı batch/eşzamanlılık, hata olayı, telafi/reconciliation, kısmi başarısızlık testleri.
- `new_anonymous` kayıtların reconciliation tamamlanmadan kalıcılaştırılmaması gerektiği vurgulanır.

---

## Identity Model

- **Domain modelleri (`backend/app/domain/video_tracking.py`):**
  - `RecognitionObservation` — tek karedeki tanıma kanıtı (embedding, kalite, poz, keskinlik, top1/top2 benzerlik).
  - `TrackletEvidence` — kesintisiz iz ve içerdiği gözlemler.
  - `CanonicalVideoPerson` — video-geneli birleştirilmiş kişi: `video_person_id`, `face_id`, `status`, `name`, `tracklet_ids`, `appearances`, `best_shot`.
  - `ReconciliationConfig` — eşikler, minimum gözlem, benzerlik sınırları, kümeleme eşiği, görünme boşluğu.
- **Durumlar:** `known`, `anonymous`, `new_anonymous` (`frontend/src/api/contracts.ts`’te de `unknown` eklenmiştir).
- **Galeri (`backend/native/recognition/gallery.cpp/.h`):**
  - JSON şema: `schema_version`, `identities` objesi; her kimlik `canonical_face_id`, `display_name`, `[512] centroid`.
  - Merkezler yükleme sırasında L2 normalize edilir, SHA-256 hash’i hesaplanır.
  - Eşleştirme saf CPU üzerindedir; `top1`/`top2` kosinüs benzerliği ve marj hesaplar.
- **Reconciliation (`backend/app/application/services/reconcile_video_identities.py`):**
  - Önce bilinen galeriye, sonra anonim galeriye karşı eşleştirir.
  - Bilinmeyen izleri tam bağlantılı (complete-link) kümeleme ile birleştirir; `cannot-link` kurallarıyla çakışan izleri ayırır.
  - Deterministik sıralama ve eşikler kullanır.

---

## API & Worker Orchestration

- **FastAPI henüz implemente edilmemiştir.** `backend/app/api/routers/` boş, `backend/app/api/__init__.py` yalnızca açıklamadır.
- **İstenen uç noktalar (`requirements/phase2requirements.md` ve `frontend/src/api/`):**
  - `POST /videos/recognize` — video yükle, job oluştur.
  - `GET /videos/jobs/{jobId}` — durum/ilerleme.
  - `GET /videos/jobs/{jobId}/result` — kişi bazlı sonuç.
  - `DELETE /videos/jobs/{jobId}` — iptal.
  - `GET /faces/{faceId}/appearances` — yüzün göründüğü videolar/anlar.
- **Mevcut iş akışı:**
  - `RunVideoDetectionService` (`backend/app/application/services/run_video_detection.py`) → `NativeWorkerPort` protokolü.
  - `SubprocessNativeWorkerAdapter` (`backend/app/infrastructure/native_worker/subprocess_adapter.py`) → `NativeDetectorClient` ile Docker komutu oluşturur, stdout’tan JSON/anahtar-değer satırlarını ayrıştırır.
  - Çalışan konteyneri her iş için bir kez başlatılır; işlem başına tek GPU, tek video.
- **CLI:** `python -m app.cli detect --video ... --output ... --host-gpu N` aynı zinciri çalıştırır.
- **Frontend mock:** `frontend/src/api/mock/` içinde bellek içi mağaza ve iş simülatörü bulunur; gerçek backend olmadan UI geliştirilebilir.

---

## Recommendations for Phase2v2

1. **Aktarılabilir kalıplar:**
   - Ports & adapters + domain-driven katman ayrımı.
   - `NativeWorkerPort` protokolü ve `SubprocessNativeWorkerAdapter`’den esinlenilmiş çalışan sözleşmesi.
   - GPU hot-path sınırı: tam kare CPU kopyası yok, yalnızca yoğunlaştırılmış metadata çıkar.
   - `RecognitionObservation`/`TrackletEvidence`/`CanonicalVideoPerson` domain modeli.
   - Deterministik, marj-tabanlı reconciliation mantığı.
   - İş durum makinesi (`pending`/`processing`/`completed`/`failed`/`cancelled`).

2. **Dikkatli adapte edilmesi gereken kalıplar:**
   - Native çalışanın tek video/tek GPU kısıtı; Phase2v2’de eşzamanlı çoklu iş veya kuyruk gerekiyorsa ayrı tasarlanmalıdır.
   - Tracker + batch_size > 1 çelişkisi; Phase2v2 batch stratejisi buna göre karar vermelidir.
   - `mvfacerecognizer`’ın tam kare üzerindeki RGBA NVMM bağımlılığı; farklı bir pipeline mimarisi kullanılacaksa hizalama çekirdekleri yeniden doğrulanmalıdır.

3. **Kaçınılması gerekenler:**
   - Mevcutta FastAPI uç noktası olmadığından, Phase2v2’de onları "zaten var" varsaymayın.
   - PostgreSQL/MinIO/Qdrant entegrasyonunun Phase2’de kodlanmadığını unutmayın; gereksinim olarak iyi ama çalışan bir örneği yok.
   - `build2/` ve `build_docker/` gibi derleme artefaktlarını asıl kaynak olarak almayın; `CMakeLists.txt` ve kaynak dosyalar geçerlidir.
   - `new_anonymous` kayıtların reconciliation öncesinde kalıcılaştırılması veri tutarsızlığına yol açar; idempotent upsert tasarımı şart.

4. **Phase2v2 için önerilen öncelikler:**
   - Phase2’deki domain modeli ve native sözleşmeyi bir "referans uygulama" olarak tutun, ancak Phase2v2’nin kendi kalıcılık ve API katmanlarını sıfırdan kurun.
   - Gerçek GPU çalışanının doğruluğunu ölçmeden önce detector/recognizer/alignment parity testlerini (Phase2’nin test matrisi) taşıyın veya karşılaştırın.
   - Çoklu depo tutarlılığı (iş durumu, video nesnesi, embedding vektörü) için Saga/transaction outbox benzeri bir model planlayın.
