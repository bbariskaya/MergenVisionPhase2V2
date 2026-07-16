# MergenVision Open-Source and Official Reference Catalog

Son güncelleme: 2026-07-16

Bu katalog MergenVision image/video face-recognition ürününde reference-first engineering için kullanılır. Liste requirement değildir. Bir linkin burada bulunması kodunun kopyalanmasını, model ağırlığının kullanılmasını veya davranışının production'a uygun olduğunu göstermez.

Her implementation kararı için önce official documentation, sonra pinned upstream source, sonra local failing test/parity gate sırası izlenir.

---

## 0. Bu dosya nasıl kullanılmalı?

Her referans adaptasyonundan önce aşağıdaki kayıt `docs/implementation/REFERENCE_DECISION_LOG.md` içine eklenmelidir:

```text
Decision ID:
Local feature/symbol:
Reference URL:
Repository commit/tag:
Access date:
Repository license:
Per-file SPDX/license:
Model-weight/data license:
Inspected upstream files/symbols:
Behavior adopted:
Behavior explicitly rejected:
Local modifications:
Failing test/reproducer:
Parity/runtime acceptance command:
Known limitation:
```

Zorunlu kurallar:

1. `master`/`main` branch körlemesine referans verilmez; commit/tag pinlenir.
2. README tek başına source contract değildir; ilgili symbol/call path okunur.
3. Paper threshold'u local production threshold olarak alınmaz.
4. Code license ile pretrained model/dataset license ayrı incelenir.
5. Build, plugin load, engine deserialize veya output file oluşması correctness kanıtı değildir.
6. Official docs kullanılan exact package/image/runtime version ile eşleşmiyorsa limitation yazılır.
7. NVIDIA SDK EULA'sı ile sample source lisansı birbirine genellenmez.

---

## 1. Yerel requirement ve architecture kaynakları

### `requirements/ProjectRequirements.md`

Bakılacak konular:

- image input validation;
- multi-face detection;
- immutable `faceId`;
- `known / anonymous / new_anonymous` semantiği;
- anonymous persistence ve aynı faceId ile enrollment;
- multiple face samples;
- process ID, logs ve history;
- API-only davranış;
- Docker ve restart persistence.

Bu dosya şu kararları tek başına tanımlamaz:

- PostgreSQL/MinIO/Qdrant seçimi;
- model, threshold veya embedding aggregation;
- authentication/authorization;
- exact delete/merge/idempotency semantiği;
- internal UI extension.

### `requirements/videorequirements.md`

Bakılacak konular:

- image requirements'ın video için korunması;
- upload validation ve retention;
- sampling;
- frame index ve timestamp;
- tracking ve faceId ilişkisi;
- person-level aggregation;
- original-coordinate bbox;
- async job/progress/cancel;
- appearances/history;
- Docker Compose ve configurable limits.

Bu dosya annotated MP4 istemez. Tam tersine bbox detayının istemci tarafından video üzerine çizileceğini açıklar. UI ise açıkça scope dışıdır; internal UI kullanıcı kararıyla extension olarak kaydedilmelidir.

---

## 2. DeepStream 9 ana dokümantasyonu

### DeepStream Development Guide

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/>

Kullanım:

- Kurulu DeepStream SDK sürümünün plugin/API davranışına giriş.
- DeepStream'in GStreamer tabanlı pipeline ve `NvDsBatchMeta` modelini anlamak.
- C/C++ sample path'lerini bulmak.

Kanıtlamaz:

- Local container image'in exact patch-level davranışını.
- Custom plugin'in metadata/tensor lifetime doğruluğunu.
- Temporal frame batching'in bütün downstream elementlerle uyumlu olduğunu.

### Gst-nvstreammux — legacy mux

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html>

Bakılacak konular:

- `batch-size`;
- `batched-push-timeout`;
- source pad/index;
- original resolution ve PTS'nin `NvDsFrameMeta` içindeki yeri;
- `NvDsBatchMeta` oluşturma;
- partial batch/EOS davranışı.

Yerel gate:

- batch 1/2/4/8 actual batch histogramı;
- frame number/PTS uniqueness ve sırası;
- partial final batch;
- EOS clean completion;
- batch semantic parity.

Kanıtlamaz:

- Aynı source'un ardışık karelerinden temporal batch oluşturmanın NvTracker gibi bütün downstream pluginlerde desteklendiğini.
- Daha yüksek batch'in E2E throughput'u artıracağını.

### Gst-nvstreammux New

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux2.html>

Bakılacak konular:

- New mux config-file davranışı;
- adaptive batching;
- round-robin collection;
- per-source frame rate ve max-same-source-frames;
- mux output metadata.

Kurallar:

- Legacy/new mux environment flag ile rastgele değiştirilmez.
- Seçilen mux exact container runtime'da fixture ve benchmark ile dondurulur.
- Unsupported/unknown GObject property sessizce set edilmez; property existence kontrol edilir.

### Gst-nvdspreprocess

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspreprocess.html>

Bakılacak konular:

- `network-input-shape` full tensor contract'ı;
- full-frame ve ROI modes;
- color format, channel order, scaling ve normalization;
- tensor buffer pool;
- `GstNvDsPreProcessBatchMeta`;
- custom transform library interface;
- target unique IDs.

Yerel gate:

- frame identity;
- OpenCV/InsightFace oracle ile preprocessing comparison;
- exact same native tensor üzerinde ONNX vs TensorRT;
- tensor batch ile frame meta count eşleşmesi;
- pool exhaustion/error cleanup.

Kanıtlamaz:

- Model-specific normalization'ın doğru olduğunu.
- Tensorin doğru frame/ROI ile eşleştiğini.

### Gst-nvinfer

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html>

Bakılacak konular:

- PGIE/SGIE modes;
- batched NV12/RGBA input;
- `input-tensor-from-meta`;
- ROI/tensor meta;
- secondary classifier caching;
- custom parser interface;
- output tensor metadata.

Kullanım kararı:

- RetinaFace veya Glint recognition için stock `nvinfer`/SGIE feasibility gate'i burada incelenir.
- Variable face count ve per-object result mapping fixture ile güvenilir değilse custom native recognizer tercih edilir.

Kanıtlamaz:

- Object metadata ile variable face batch result mapping'inin bizim pipeline'da doğru olduğunu.
- Custom ArcFace alignment matematiğini.

### DeepStream metadata

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html>

Bakılacak konular:

- `NvDsBatchMeta -> NvDsFrameMeta -> NvDsObjectMeta` hierarchy;
- meta pools;
- `NvDsUserMeta`;
- copy/release callback;
- upstream/downstream transform behavior.

Yerel gate:

- copy/release callbacks under buffer copy/demux;
- ASAN/Compute Sanitizer where applicable;
- no use-after-free/double-free;
- frame/object ownership under partial batches.

Kanıtlamaz:

- Bizim custom pointer lifetime'ımızın doğru olduğunu.

### NvBufSurface API

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/structNvBufSurfaceParams.html>

Bakılacak konular:

- width/height;
- pitch;
- layout;
- color format;
- `dataPtr`;
- batch index ve memory type.

Yerel gate:

- pitch-aware CUDA sampling;
- RGBA/NV12 contract;
- no assumption `pitch == width * channels`;
- device pointer type validation.

### Gst-nvstreamdemux

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreamdemux.html>

Kullanım:

- Yalnız debug render/encoded-output branch gerektiğinde batch'i source frame'lere ayırma.
- Parent batch lifetime ve downstream child-buffer retention davranışını anlamak.

Product kararı:

- Ana observation pipeline'da demux/render/NVENC bulunmaz.
- UI original video + overlay kullanır.

### Gst-nvtracker

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html>

Bakılacak konular:

- NvSORT/NvDCF/NvDeepSORT requirements;
- input format/memory requirements;
- multi-stream batching;
- low-level `NvMOT` API;
- past-frame and terminated-track data.

Kritik sınır:

- Dokümandaki batched tracking esas olarak multiple stream/source frame batching'ini anlatır.
- Aynı videonun ardışık sekiz karesini tek batch'te NvDCF'ye vermek için resmi temporal-order guarantee varsayılmaz.
- Bu projede ilk tercih Python metadata tracker'dır.

### DeepStream C/C++ sample apps

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html>

Bakılacak örnekler:

- `deepstream-test1`: single-source decode/mux/infer;
- `deepstream-test2`: detector/tracker/secondary classifier;
- `deepstream-test3`: multi-source;
- `deepstream-test4`: message metadata;
- `deepstream-app`: production-like configuration patterns.

Kurallar:

- Sample topology kopyalanmadan önce kullanılan DeepStream version ile source path doğrulanır.
- Sample'da çalışması bizim custom tensor/meta correctness'imizin kanıtı değildir.

---

## 3. DeepStream Python ve Service Maker

### NVIDIA DeepStream Python Apps

Repository: <https://github.com/NVIDIA-AI-IOT/deepstream_python_apps>

Bakılacak konular:

- GStreamer bus/EOS/error handling;
- pad probe ve metadata traversal;
- Python binding build/version matrix;
- custom user meta sample'ları.

Version notu:

- Repository DeepStream 9 için Python 3.12/GStreamer 1.24.x matrisini belirtir.
- `pyds` bindings deprecated durumundadır; yeni wheel dağıtımı yerine Service Maker yönü önerilmektedir.

Product kararı:

- Python full frame/tensor hot path'e sokulmaz.
- Pipeline'ın native C++ worker içinde kalması tercih edilir.
- Python orchestration, compact observation consumption ve business logic sahibidir.

### PyServiceMaker Pipeline API

URL: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_intro_to_pipeline_api.html>

Kullanım:

- Gelecekte native graph orchestration'ını Python'a taşımak düşünülürse official direction olarak incelenir.
- Current C++ worker'ı sebepsiz yeniden yazma gerekçesi değildir.

---

## 4. GStreamer core referansları

### GStreamer documentation root

URL: <https://gstreamer.freedesktop.org/documentation/>

Exact installed GStreamer version ile docs davranışı karşılaştırılır.

### `queue`

URL: <https://gstreamer.freedesktop.org/documentation/coreelements/queue.html>

Bakılacak konular:

- ayrı streaming thread;
- max-size-buffers/bytes/time;
- leaky modes;
- overrun/underrun signals;
- current level observability;
- EOS behavior.

Kanıtlamaz:

- Queue eklemenin performans problemini otomatik çözdüğünü.
- Buffer pool/lifetime probleminin ortadan kalktığını.

### `appsink`

URL: <https://gstreamer.freedesktop.org/documentation/app/appsink.html>

Kullanım:

- Native application'ın buffer/sample tüketmesi;
- bounded `max-buffers`, drop/leaky behavior;
- pull-sample lifecycle.

Product sınırı:

- Python appsink üzerinden NVMM frame/NumPy taşımak yasaktır.
- Native collector yalnız compact metadata protocol'u üretir.

### `appsrc`

URL: <https://gstreamer.freedesktop.org/documentation/app/appsrc.html>

Kullanım:

- Test fixture veya ileride external encoded input besleme.
- Ana browser-upload tasarımı için gerekli değildir.

### GstBaseTransform

URL: <https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html>

Bakılacak konular:

- start/stop/set_caps/transform_ip;
- in-place transform;
- passthrough;
- caps negotiation;
- buffer allocation.

Kanıtlamaz:

- CUDA stream/tensor lifetime correctness.

### GstMeta

URL: <https://gstreamer.freedesktop.org/documentation/gstreamer/gstmeta.html>

Bakılacak konular:

- init/free/transform callbacks;
- buffer copy ve custom meta ownership.

DeepStream `NvDsUserMeta` ile aynı şey değildir; ikisinin boundary'si ayrı incelenir.

### GStreamer Bus

URL: <https://gstreamer.freedesktop.org/documentation/application-development/basics/bus.html>

Bakılacak konular:

- EOS;
- ERROR/WARNING;
- state messages;
- application main-loop integration.

Done yalnız EOS ve bütün mandatory persistence tamamlandıktan sonra ilan edilir.

### GStreamer plugin licensing FAQ

URL: <https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/general.html>

Kullanım:

- Core/plugin family licensing araştırması.
- Kullanılan codec/plugin/dependency lisansı ayrıca kaydedilir; “GStreamer LGPL” bütün plugin stack'e genellenmez.

---

## 5. TensorRT 10.14 referansları

### TensorRT Documentation

URL: <https://docs.nvidia.com/deeplearning/tensorrt/10.14.1/>

Exact container/runtime sürümüyle eşleşmelidir.

### Dynamic Shapes

URL: <https://docs.nvidia.com/deeplearning/tensorrt/10.14.1/inference-library/work-dynamic-shapes.html>

Bakılacak konular:

- optimization profiles;
- min/opt/max shape;
- runtime input shape;
- context/profile selection;
- output shape resolution;
- `enqueueV3`.

Local gate:

- batch 1/partial/max;
- invalid shape rejection;
- binding/tensor name/dtype checks;
- repeated context use;
- concurrent context ownership.

Kanıtlamaz:

- ONNX/FP32 ile semantic parity.

### TensorRT Best Practices

URL: <https://docs.nvidia.com/deeplearning/tensorrt/10.14.1/performance/best-practices.html>

Bakılacak konular:

- `trtexec` warmup/profiling;
- CUDA Graph;
- throughput/latency distinction;
- Nsight integration;
- data transfer and synchronization.

Kritik sınır:

- `trtexec` FPS yalnız izole engine throughput'udur.
- Decode, preprocess, metadata, tracking, Qdrant, persistence veya UI dahil değildir.

### NVIDIA TensorRT source repository

Repository: <https://github.com/NVIDIA/TensorRT>

License: <https://github.com/NVIDIA/TensorRT/blob/main/LICENSE>

Bakılacak konular:

- pinned sample code;
- plugin registration/lifecycle;
- buffer management;
- ONNX parser sample patterns.

NVIDIA runtime/SDK binary lisansı repository source lisansından ayrı kaydedilir.

---

## 6. CUDA, profiling ve memory correctness

### CUDA C++ Programming Guide

URL: <https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>

Bakılacak konular:

- device/context ownership;
- stream semantics;
- asynchronous execution;
- events;
- memory visibility/lifetime;
- CUDA Graph.

### CUDA C++ Best Practices Guide

URL: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

Bakılacak konular:

- host-device transfer azaltma;
- pinned memory;
- coalesced access;
- overlap;
- occupancy ve ölçüm.

### Compute Sanitizer

URL: <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>

Kullanım:

- out-of-bounds;
- race;
- uninitialized memory;
- misalignment;
- invalid synchronization.

Performans veya ML correctness kanıtı değildir.

### Nsight Systems

URL: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>

Kullanım:

- CUDA API/kernel/memcpy timeline;
- CPU/GPU overlap;
- unintended synchronize;
- full-frame D2H tespiti;
- queue/backpressure zaman çizelgesi.

No-D2H gözlemi bbox/identity accuracy kanıtı değildir.

### CUDA Samples

Repository: <https://github.com/NVIDIA/cuda-samples>

License: <https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE>

Yalnız pinned commit ve per-file license doğrulamasıyla adaptasyon yapılır.

---

## 7. RetinaFace detection referansları

### RetinaFace paper

URL: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf>

Bakılacak konular:

- single-stage detection;
- five landmarks;
- multi-scale feature design.

Kanıtlamaz:

- Local ONNX output tensor name/shape;
- anchor order;
- variances;
- NMS;
- resize/letterbox reverse mapping;
- local dataset accuracy.

### InsightFace RetinaFace source

Repository path: <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>

Bakılacak konular:

- anchor generation;
- bbox decode;
- landmark decode;
- variances;
- NMS;
- resize mapping.

Exact commit pinlenir. Local ONNX/export'un aynı config olduğuna artifact inspection olmadan karar verilmez.

### InsightFace Python detector wrappers

Repository: <https://github.com/deepinsight/insightface>

Kullanım:

- CPU/reference oracle;
- input normalization ve output decode comparison.

Production dependency olarak full-frame Python/OpenCV yolu kullanılmaz.

---

## 8. ArcFace, GlintR100 ve alignment

### ArcFace paper

URL: <https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf>

Bakılacak konular:

- additive angular margin;
- normalized embedding space;
- cosine similarity interpretation.

Kanıtlamaz:

- Production threshold;
- anonymous lifecycle;
- top-K/margin policy;
- video aggregation;
- model weight license.

### InsightFace ArcFace ONNX wrapper

URL: <https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py>

Bakılacak konular:

- input tensor inspection;
- mean/std selection;
- `norm_crop`;
- channel order;
- cosine computation.

Exact commit ve local ONNX graph contract doğrulanır. Wrapper'ın default'u bütün exported models için varsayılmaz.

### InsightFace five-point face alignment

URL: <https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py>

Bakılacak konular:

- landmark order;
- 112x112 ArcFace template;
- similarity transform direction;
- border behavior;
- CPU oracle.

Local gate:

- CUDA vs oracle pixel MAE/p95/max;
- contact sheet;
- embedding cosine parity;
- pitch, border, pixel-center ve interpolation cases;
- degenerate landmark rejection.

### ArcFace Torch training/export README

URL: <https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/README.md>

Kullanım:

- Training/export provenance araştırması.
- Local pretrained model license'ı otomatik sağlamaz.

### InsightFace Model Zoo

URL: <https://github.com/deepinsight/insightface/tree/master/model_zoo>

Bakılacak konular:

- Model-pack composition;
- detector/recognizer mapping;
- artifact naming.

`glintr100.onnx` dosya adı tek başına model provenance değildir. Şunlar kaydedilir:

- source URL/model pack;
- exact SHA-256;
- graph tensor names/shapes/dtypes;
- training dataset statement;
- license/use restrictions;
- preprocessing/alignment contract.

### Glint360K / Partial FC paper

URL: <https://arxiv.org/abs/2010.05222>

Kullanım:

- Glint360K ve large-class training provenance.

Kanıtlamaz:

- Dataset/model weight commercial-use permission;
- target-domain accuracy;
- local ONNX identity.

### InsightFace licensing warning

Repository: <https://github.com/deepinsight/insightface>

Kritik kural:

- InsightFace source code license ile pretrained model/model-pack kullanım şartları ayrıdır.
- ONNX'i TensorRT engine'e çevirmek ağırlığın lisansını değiştirmez.
- License/provenance onayı olmadan model production-ready ilan edilmez.

---

## 9. Tracking: ByteTrack ve assignment

### ByteTrack paper

URL: <https://arxiv.org/abs/2110.06864>

Bakılacak konular:

- High-score ve low-score detections ile iki aşamalı association;
- lost/removed track lifecycle;
- MOT evaluation yaklaşımı.

Kanıtlamaz:

- Face-domain threshold;
- sparse sampling correctness;
- shot-cut/re-entry persistent identity;
- 600 FPS Python implementation;
- Rachel'ın videonun sonunda aynı faceId olması.

### FoundationVision ByteTrack repository

Repository: <https://github.com/FoundationVision/ByteTrack>

License: <https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE>

İncelenecek dosyalar pinned commit'te kaydedilir:

- `byte_tracker.py`;
- `matching.py`;
- Kalman filter;
- track state lifecycle;
- detection score partitioning.

Adaptasyon kuralları:

- low-score candidates tracker'a ulaşır;
- raw tracker ID canonical faceId olmaz;
- explicit gating ve deterministic tie-break eklenir;
- frame/PTS gap handling ve shot-cut policy eklenir;
- aynı source state'i sequential mutate edilir.

### SciPy linear sum assignment

URL: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>

Kullanım:

- Rectangular cost assignment;
- min/max direction;
- installed-version behavior.

Kanıtlamaz:

- Gating;
- Kalman prediction;
- unmatched lifecycle;
- deterministic equal-cost tie policy;
- identity correctness.

### Norfair

Repository: <https://github.com/tryolabs/norfair>

Kullanım:

- Python metadata tracker API/lifecycle tasarımı için alternatif reference.
- License, exact commit ve performance profile doğrulanmadan production dependency seçilmez.

### MOTChallenge metrics

URL: <https://motchallenge.net/>

Kullanım:

- ID switch, IDF1/HOTA/MOTA kavramlarını anlamak.

Face-specific fixture ve product identity correctness'in yerine geçmez.

---

## 10. Face quality ve template aggregation

### InsightFace IJB-C evaluator

URL: <https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/onnx_ijbc.py>

Bakılacak konular:

- Image-to-template aggregation;
- faceness weighting;
- normalized template representation.

Kanıtlamaz:

- Aynı weighting'in video product için calibrated olduğunu.

### MagFace paper

URL: <https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.pdf>

Önemli ders:

- Embedding normunun quality ile ilişkisi MagFace'in özel training objective'inin sonucudur.
- L2-normalized GlintR100/ArcFace embedding normu quality olarak kullanılamaz.

### SER-FIQ paper

URL: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_Embedding_Robustness_CVPR_2020_paper.pdf>

Kullanım:

- Embedding robustness tabanlı quality araştırması.

Frozen deterministic TensorRT pipeline'a doğrudan/ucuz uygulanabilirlik varsayılmaz.

### CAFace

Repository: <https://github.com/mk-minchul/caface>

Paper: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/ea35a58ee3da13c01a69df2a819386b3-Abstract-Conference.html>

Kullanım:

- Large probe set cluster/aggregate araştırması;
- Order-invariant aggregation fikri.

CAFace/AdaFace model-specific weights veya features GlintR100'e doğrudan taşınmaz.

### Baseline aggregation kararı

İlk production baseline şeffaf ve test edilebilir olmalıdır:

1. Minimum bbox size.
2. Detector confidence.
3. Landmark geometry/alignment residual.
4. Pose/blur/occlusion gate.
5. Temporal diversity.
6. L2-normalized embeddings.
7. Robust outlier rejection.
8. Quality-weighted average.
9. Centroid re-normalization.
10. Top-1/top-2/margin evidence.
11. Target fixture calibration.

Araştırma aggregation modeli ölçüm olmadan baseline yerine konmaz.

---

## 11. PostgreSQL queue, consistency ve storage

### PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED`

URL: <https://www.postgresql.org/docs/16/sql-select.html>

Kullanım:

- Multiple workers arasında queue-like job claim.

Doğru contract:

1. Kısa transaction.
2. Pending row seç/lock.
3. `processing`, lease owner/expiry ve attempt yaz.
4. Commit ve lock'u bırak.
5. GPU work transaction dışında.

Kanıtlamaz:

- Exactly-once execution;
- heartbeat/retry;
- idempotent external side effects.

### PostgreSQL explicit locking

URL: <https://www.postgresql.org/docs/16/explicit-locking.html>

Bakılacak konular:

- Row/table locks;
- deadlocks;
- lock lifetime.

GPU job boyunca transaction veya row lock tutulmaz.

### PostgreSQL transaction isolation

URL: <https://www.postgresql.org/docs/16/transaction-iso.html>

Bakılacak konular:

- `READ COMMITTED` behavior;
- concurrent updates;
- retry requirements.

### `pg_locks`

URL: <https://www.postgresql.org/docs/16/view-pg-locks.html>

Kullanım:

- Test/operation sırasında blocker/holder inceleme.

### PostgreSQL JSON types

URL: <https://www.postgresql.org/docs/16/datatype-json.html>

Kullanım:

- Bounded name/metadata/config snapshots.

Kullanılmaması gereken:

- Bütün video detection timeline'ını tek JSONB row'a koymak.

### PostgreSQL table partitioning

URL: <https://www.postgresql.org/docs/16/ddl-partitioning.html>

Kullanım:

- Eğer per-detection relational storage gerçekten seçilirse job/time partition araştırması.

İlk tasarımda detailed overlay timeline MinIO chunk artifact olabilir.

### PostgreSQL advisory locks

URL: <https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS>

Kullanım:

- Concurrent same-unknown creation için bounded coordination araştırması.

Biometric equality kanıtı değildir; lock bucket seçimi ve second search uygulama sorumluluğudur.

---

## 12. SQLAlchemy 2 ve Alembic

### SQLAlchemy `with_for_update`

URL: <https://docs.sqlalchemy.org/20/core/selectable.html#sqlalchemy.sql.expression.Select.with_for_update>

Bakılacak konular:

- `skip_locked=True`;
- generated SQL;
- PostgreSQL dialect behavior.

Gerçek PostgreSQL integration testi olmadan queue PASS değildir.

### SQLAlchemy Session transactions

URL: <https://docs.sqlalchemy.org/en/latest/orm/session_transaction.html>

Bakılacak konular:

- Transaction ownership;
- context managers;
- rollback;
- nested transaction/savepoint.

MinIO/Qdrant çağrıları PostgreSQL transaction'ına dahil olmaz.

### SQLAlchemy asyncio

URL: <https://docs.sqlalchemy.org/20/orm/extensions/asyncio.html>

Kural:

- Her concurrent task ayrı `AsyncSession` kullanır.
- Session task'lar arasında paylaşılmaz.
- Blocking SDK çağrısı otomatik async olmaz.

### Alembic tutorial

URL: <https://alembic.sqlalchemy.org/en/latest/tutorial.html>

### Alembic autogenerate

URL: <https://alembic.sqlalchemy.org/en/latest/autogenerate.html>

Kural:

- Autogenerate output elle incelenir.
- Rename, constraint, data migration ve rollback otomatik doğru kabul edilmez.
- Empty DB upgrade, populated DB upgrade ve mümkünse downgrade gerçek PostgreSQL'de test edilir.

---

## 13. MinIO / S3-compatible object storage

### MinIO Python SDK

URL: <https://docs.min.io/aistor/developers/sdk/python/api/>

Bakılacak konular:

- `put_object` streaming;
- `presigned_get_object`;
- `presigned_put_object`;
- expiry;
- stat;
- multipart operations.

Presigned URL bearer credential'dır; verildikten sonra business authorization sağlamaz.

### MinIO JavaScript SDK

URL: <https://docs.min.io/aistor/developers/sdk/javascript/api/>

Kullanım:

- Internal UI upload/session tasarımı;
- browser/mobile direct upload için signed URL patterns.

Browser'a MinIO access key/secret verilmez.

### MinIO S3 API compatibility

URL: <https://docs.min.io/aistor/developers/s3-api-compatibility/>

Bakılacak konular:

- CreateMultipartUpload;
- UploadPart;
- CompleteMultipartUpload;
- AbortMultipartUpload;
- version-specific deviations.

Exact pinned MinIO image üzerinde contract test gerekir.

### MinIO thresholds

URL: <https://docs.min.io/aistor/reference/aistor-server/thresholds/>

Bakılacak konular:

- Maximum part count;
- part-size ranges;
- object limits.

Optimal part size local network/browser/RAM benchmark'ıyla seçilir.

### MinIO CORS

URL: <https://docs.min.io/aistor/administration/cors-configuration/>

Bakılacak konular:

- Allowed origins;
- methods;
- request/exposed headers;
- browser upload/playback.

CORS authentication değildir.

### MinIO object lifecycle management

URL: <https://docs.min.io/aistor/administration/object-lifecycle-management/>

Kullanım:

- Original video retention;
- debug artifact expiry;
- prefix/tag policy.

PostgreSQL identity/job lifecycle ile atomic consistency sağlamaz. Application reconciliation gerekir.

### MinIO security/access policy

URL: <https://docs.min.io/aistor/administration/iam/access/>

Kullanım:

- Least-privilege service accounts;
- Get/Put/multipart permissions;
- private buckets.

### AWS multipart upload overview

URL: <https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html>

Kullanım:

- S3 multipart semantics;
- part retry;
- ETag/complete concepts.

MinIO exact compatibility current image'de doğrulanır. Multipart ETag normal SHA-256 varsayılmaz.

---

## 14. Qdrant

### Points and upsert

URL: <https://qdrant.tech/documentation/manage-data/points/>

Bakılacak konular:

- UUID point IDs;
- batch upsert;
- same-ID idempotent overwrite;
- `wait=true`;
- ordering/ack semantics.

PostgreSQL/MinIO/Qdrant atomic transaction sağlamaz.

### Collections

URL: <https://qdrant.tech/documentation/manage-data/collections/>

Kullanım:

- 512-D cosine collection;
- vector configuration;
- optimizer/HNSW baseline.

Face threshold veya model accuracy kanıtı değildir.

### Vectors

URL: <https://qdrant.tech/documentation/manage-data/vectors/>

Kullanım:

- Named/multiple vectors;
- data types;
- representation options.

İlk baseline one point per `face_sample` kullanır; premature multivector tasarım ölçümsüz yapılmaz.

### Filtering

URL: <https://qdrant.tech/documentation/search/filtering/>

Bakılacak konular:

- `must/should/must_not`;
- active/model-version filters.

Final lifecycle PostgreSQL'den yine doğrulanır.

### Indexing

URL: <https://qdrant.tech/documentation/manage-data/indexing/>

Kullanım:

- Sık filtrelenen technical payload fields;
- ingest öncesi payload index.

### Bulk upload

URL: <https://qdrant.tech/documentation/manage-data/bulk-upload/>

Kullanım:

- Rebuild/migration throughput;
- bounded batch writes.

### Snapshots

URL: <https://qdrant.tech/documentation/snapshots/>

Kullanım:

- Snapshot create/restore;
- node-specific concerns.

PostgreSQL/MinIO ile point-in-time consistent backup otomatik değildir.

### Security

URL: <https://qdrant.tech/documentation/security/>

Kritik not:

- Self-hosted Qdrant varsayılanında güvenli public exposure varsayılmaz.
- Private network/bind, API key ve gerekirse TLS konfigüre edilir.
- Application authorization yerine geçmez.

### Embedding model migration

URL: <https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/>

Kullanım:

- ID-preserving upsert;
- dual representation/collection migration;
- model-version transition.

---

## 15. FastAPI, Pydantic ve OpenAPI

### FastAPI request files

URL: <https://fastapi.tiangolo.com/tutorial/request-files/>

Bakılacak konular:

- `bytes` bütün dosyayı memory'ye alır;
- `UploadFile` spooled-file behavior;
- multipart contract.

Large video upload'un bounded/production-safe olduğunu tek başına kanıtlamaz.

### FastAPI UploadFile reference

URL: <https://fastapi.tiangolo.com/reference/uploadfile/>

Kullanım:

- file handle/read/seek;
- content type/size metadata.

Client `Content-Type` güvenilir validation değildir; probe gerekir.

### FastAPI streaming responses

URL: <https://fastapi.tiangolo.com/advanced/custom-response/>

Kullanım:

- Chunk iterator;
- streaming response lifecycle;
- cancellation behavior.

Automatic HTTP Range support varsayılmaz.

### FastAPI Server-Sent Events

URL: <https://fastapi.tiangolo.com/tutorial/server-sent-events/>

Version notu:

- Native SSE support kullanılan exact FastAPI version'da doğrulanır.
- Daha eski version'da Starlette/third-party behavior ayrı değerlendirilir.

SSE yalnız notification/progress kanalıdır; source-of-truth PostgreSQL'dir. Poll fallback bulunur.

### Starlette Responses

URL: <https://www.starlette.io/responses/>

Bakılacak konular:

- `FileResponse` Range/206/416 behavior;
- streaming semantics.

### FastAPI dependency injection

URL: <https://fastapi.tiangolo.com/tutorial/dependencies/>

Kullanım:

- Application service/authorization/config boundary.

Router içinde business logic veya direct repository query yapılmaz.

### Pydantic Settings

URL: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>

Kullanım:

- Env config validation;
- secret/config source;
- invalid startup hard-fail.

### OpenAPI Specification

URL: <https://spec.openapis.org/oas/latest.html>

Kullanım:

- Versioned request/response/error contracts;
- multipart/JSON modes;
- cursor pagination;
- generated client contract tests.

---

## 16. Browser video, upload ve overlay

### URL.createObjectURL

URL: <https://developer.mozilla.org/en-US/docs/Web/API/URL/createObjectURL_static>

### URL.revokeObjectURL

URL: <https://developer.mozilla.org/en-US/docs/Web/API/URL/revokeObjectURL_static>

Kullanım:

- Kullanıcının seçtiği local `File` için upload tamamlanmadan immediate preview.
- Component cleanup sırasında URL revoke.

### HTTP Range requests

URL: <https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests>

Bakılacak konular:

- `Accept-Ranges`;
- `Range`;
- `206 Partial Content`;
- `Content-Range`;
- `416`.

MinIO/reverse-proxy/CORS zincirinin Range header'larını gerçekten koruduğu E2E test edilir.

### requestVideoFrameCallback

URL: <https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback>

Bakılacak konular:

- `metadata.mediaTime`;
- presented frames;
- display dimensions;
- compositor-synchronized overlay.

Her encoded frame için callback veya native PTS ile otomatik exact match garantisi değildir.

### HTMLMediaElement currentTime

URL: <https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/currentTime>

Kullanım:

- Seek/pause/playback-state fallback.

Canonical backend zaman `pts_ns` olur; `currentTime * fps` ile frame türetilmez.

### ResizeObserver

URL: <https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver>

Kullanım:

- Responsive/fullscreen video container ölçüsü.

Letterbox, rotation, SAR ve DPR matematiğini otomatik çözmez.

### Canvas API

URL: <https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API>

Kullanım:

- Bbox/label overlay;
- DPR-aware drawing;
- clear/redraw lifecycle.

### CORS

URL: <https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS>

Kullanım:

- Browser preflight;
- allowed/exposed headers;
- MinIO upload/playback.

CORS authorization değildir.

### EventSource / SSE

URL: <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events>

Bakılacak konular:

- reconnect;
- event IDs;
- retry;
- browser connection limits.

Missed event persistence/replay source-of-truth'u değildir.

### Media Source Extensions

URL: <https://developer.mozilla.org/en-US/docs/Web/API/Media_Source_Extensions_API>

Kullanım:

- Yalnız ileride progressive/streaming derivative playback gerektiğinde.
- İlk file-upload product flow için gerekli değildir.

---

## 17. Container, GPU ve deployment

### Docker Compose GPU support

URL: <https://docs.docker.com/compose/how-tos/gpu-support/>

Bakılacak konular:

- GPU device reservation;
- `capabilities: [gpu]`;
- `device_ids` vs `count`;
- one-worker/one-GPU ownership.

Doğru GPU UUID/inference kanıtı değildir.

### NVIDIA Container Toolkit — GPU enumeration

URL: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html>

Bakılacak konular:

- `NVIDIA_VISIBLE_DEVICES`;
- GPU index/UUID;
- `compute`, `utility`, `video` driver capabilities.

Host GPU index ile container logical index aynı varsayılmaz. Worker manifest'te host-assigned UUID ve container logical ID kaydeder.

### NVIDIA Container Toolkit install guide

URL: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

Kullanım:

- Host/runtime setup reference.

Agent kullanıcı onayı olmadan system toolkit/driver değiştirmez.

### Docker Compose startup order

URL: <https://docs.docker.com/compose/how-tos/startup-order/>

Bakılacak konular:

- `depends_on.condition: service_healthy`;
- initial dependency order.

Runtime dependency failure recovery sağlamaz.

### Docker Compose services reference

URL: <https://docs.docker.com/reference/compose-file/services/>

Bakılacak konular:

- healthcheck;
- restart;
- environment/secrets;
- volumes;
- resource constraints.

Health ile readiness ayrılır. Process alive olmak model/storage ready demek değildir.

---

## 18. FFmpeg/ffprobe ve media validation

### FFmpeg documentation

URL: <https://ffmpeg.org/documentation.html>

Kullanım:

- `ffprobe` container/codec/duration/frame metadata;
- acceptance artifact validation;
- browser-playable derivative/remux investigation.

Product hot path decode'u FFmpeg CPU fallback'a çevirmek için kullanılmaz.

### ffprobe documentation

URL: <https://ffmpeg.org/ffprobe.html>

Bakılacak konular:

- structured JSON output;
- streams/format;
- duration, time base, frame count limitations;
- codec/container validation.

Reported frame count her VFR/container'da exact varsayılmaz; worker decoded count ile karşılaştırılır.

### ISO Base Media File Format notu

MP4/MOV incremental processing; seek, indexing ve `moov` placement gibi container-specific davranışlar exact fixture ile test edilir. Browser stream'ini worker'a yarım dosya olarak vermeme kararının ana nedeni canonical finalized object, retry ve checksum bütünlüğüdür.

---

## 19. Security references

### OWASP File Upload Cheat Sheet

URL: <https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html>

Kullanım:

- Extension/MIME/content validation;
- size limits;
- opaque names;
- quarantine;
- storage isolation;
- authorization.

Codec/decode correctness veya malware-free guarantee değildir.

### OWASP REST Security Cheat Sheet

URL: <https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html>

Kullanım:

- AuthN/AuthZ;
- input validation;
- error sanitization;
- rate limits;
- HTTPS;
- audit.

### OWASP API Security Top 10

URL: <https://owasp.org/API-Security/>

Kullanım:

- Object-level authorization;
- resource consumption;
- inventory/versioning;
- unsafe consumption risk review.

### NIST face-recognition evaluation resources

URL: <https://pages.nist.gov/frvt/html/frvt11.html>

Kullanım:

- Face-recognition evaluation concepts ve demographic/performance risk farkındalığı.

Local model/data accuracy veya compliance certification değildir.

---

## 20. Testing, contracts ve observability

### pytest

URL: <https://docs.pytest.org/en/stable/>

Kullanım:

- Unit/integration fixtures;
- markers;
- parametrization;
- failure diagnostics.

### Hypothesis

URL: <https://hypothesis.readthedocs.io/en/latest/>

Kullanım:

- State-machine/property tests;
- bbox/coordinate edge cases;
- idempotency/state transition inputs.

GPU parity yerine geçmez.

### Postman documentation

URL: <https://learning.postman.com/docs/introduction/overview/>

Kullanım:

- Frozen OpenAPI contract üzerinde gerçek service API collection/acceptance.

Mock collection gerçek PostgreSQL/MinIO/Qdrant/GPU E2E değildir.

### Playwright documentation

URL: <https://playwright.dev/docs/intro>

Kullanım:

- Upload;
- progress/error/no-face states;
- video seek/play/pause/fullscreen;
- overlay alignment;
- identity rename ve history flow;
- fatal console errors.

Screenshot tek başına coordinate correctness değildir; numeric bbox/rendered rect assertions ve visual fixture gerekir.

### OpenTelemetry

URL: <https://opentelemetry.io/docs/>

Kullanım:

- requestId/processId/jobId correlation;
- API/worker stage latency;
- queue/backlog metrics;
- storage call tracing.

Business source-of-truth veya mandatory process history değildir.

---

## 21. Performance ve correctness kanıt hiyerarşisi

En zayıftan en güçlüye:

1. Source okunması.
2. Build/link.
3. `gst-inspect` plugin registration.
4. Engine deserialize.
5. Unit test.
6. Exact-tensor ONNX/TensorRT parity.
7. Native pipeline short smoke.
8. Frame/PTS/tensor identity gate.
9. Long-video deterministic semantic output.
10. GPU/CPU profiling ve memory correctness.
11. PostgreSQL/MinIO/Qdrant partial-failure integration.
12. Full API/UI E2E.
13. Target-data accuracy/calibration.
14. Sustained full-product performance.

Yanlış çıkarımlar:

- `gst-inspect` -> recognition çalışıyor değildir.
- Build success -> runtime success değildir.
- Engine deserialize -> inference parity değildir.
- Equal detection count -> bbox/landmark parity değildir.
- MP4 oluşması -> render/identity correctness değildir.
- `nvidia-smi` -> zero-copy değildir.
- `trtexec 600 FPS` -> full video API 600 FPS değildir.
- ByteTrack paper -> bizim yüz fixture'ımızda correct tracking değildir.
- Unit test -> EOS/backpressure/memory lifetime değildir.
- Output JSON oluşması -> persistent identity lifecycle değildir.

---

## 22. Proje için zorunlu reference decision kayıtları

En az şu kararlar ayrı kayıt olmalıdır:

1. RetinaFace artifact/source/license.
2. RetinaFace tensor names/shapes/anchors/variances.
3. RetinaFace preprocessing.
4. RetinaFace postprocess/NMS.
5. ArcFace/GlintR100 artifact/source/license.
6. Five-landmark order/template.
7. CUDA alignment transform/interpolation/border.
8. L2 normalization.
9. Cosine/top-K/margin.
10. Face quality baseline.
11. Template/tracklet aggregation.
12. ByteTrack upstream commit/license/adaptation.
13. Assignment/gating/lifecycle.
14. DeepStream legacy/new mux choice.
15. nvdspreprocess vs custom preprocess.
16. nvinfer SGIE vs custom recognizer.
17. Native observation protocol.
18. PostgreSQL queue/lease.
19. MinIO multipart/finalization/retention.
20. Qdrant collection/model migration.
21. Browser playback/Range/overlay.
22. Delete/merge/history semantics.

---

## 23. MCP yönlendirme tablosu

| İhtiyaç | Öncelikli araç | Çıktı nasıl doğrulanır? |
|---|---|---|
| Active repo architecture/callers/tests | `codebase-memory-mcp` | Filesystem source + `rg` + tests |
| Version-sensitive library/API | `context7` | Exact pinned version official docs |
| Upstream GitHub architecture/symbol path | `deepwiki` | Upstream file/commit/license |
| Current/niche primary source | `exa` | Official vendor/paper/upstream URL |
| Active repo/upstream comparison | GitHub plugin/MCP | Local checkout/source hash |
| Real API acceptance | `postman` | Running service + persisted state |
| Real UI E2E | `playwright` | Running UI/API/storage + assertions |
| UI code generation | `21st` | Forbidden; kullanılmaz |

Araç unavailable ise `used` denmez. Fallback ve limitation açıkça yazılır.

---

## 24. Reference package completion checklist

Bir sprintin reference çalışması tamamlanmış sayılmadan önce:

- [ ] Requirement path ve behavior map yazıldı.
- [ ] Official docs exact version kontrol edildi.
- [ ] Upstream commit/tag pinlendi.
- [ ] Repository ve per-file license kontrol edildi.
- [ ] Model weight/data license ayrı kontrol edildi.
- [ ] İlgili upstream source symbol'leri okundu.
- [ ] Adapted/rejected behavior yazıldı.
- [ ] Failing test/reproducer var.
- [ ] Local parity/runtime command var.
- [ ] Artifact SHA/model/preprocess contract kayıtlı.
- [ ] Security/privacy etkisi yazıldı.
- [ ] Performance claim kapsamı doğru etiketlendi.
- [ ] Reference Decision Log güncellendi.

Bu checklist tamamlanmadan “reference implementation'a uyuyor”, “DeepStream supported”, “ByteTrack correct”, “ArcFace compatible”, “GPU-only” veya “600 FPS” completion claim'i yapılamaz.