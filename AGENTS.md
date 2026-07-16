# MergenVision Engineering Constitution

Bu dosya MergenVision repository'sinde çalışan bütün insan ve AI agent'lar için kalıcı çalışma sözleşmesidir. Kullanıcının güncel açık kararı bu dosyadan üstündür. Sprint'e özel hedefler `docs/implementation/CURRENT_SPRINT.md`, kaynak adaptasyon kararları `docs/implementation/REFERENCE_DECISION_LOG.md` içinde tutulur.

## 1. Ürün misyonu

Sistem önce görüntülerde, ardından videolarda çoklu yüz tespiti ve kalıcı yüz kimliği üretir. Her detected yüzün immutable bir `faceId` değeri olur. İlk karşılaşma `new_anonymous`, daha sonraki eşleşme `anonymous`, aynı `faceId` isimlendirildikten sonra `known` sonucunu üretir. Video genişletmesi aynı identity karar motorunu kullanır; tracking yalnız temporal sürekliliği, recognition kalıcı kimliği cevaplar.

## 2. Source-of-truth sırası

Çelişki halinde sıra şöyledir:

1. Kullanıcının güncel açık kararı.
2. `requirements/ProjectRequirements.md` içindeki image/identity gereksinimleri.
3. `requirements/videorequirements.md` içindeki additive video gereksinimleri.
4. Kullanıcı tarafından onaylanmış architecture/ADR belgeleri.
5. `docs/implementation/CURRENT_SPRINT.md`.
6. Official vendor documentation ve pinned upstream source.
7. Eski repository kodları ve raporları yalnız lessons-learned kaynağıdır.

Eski client özetleri; Oracle, 10M kişi, national ID veya başka ek kapsamları kullanıcı yeniden onaylamadıkça ürün requirement'ı sayılmaz.

## 3. Onaylanmış ürün sınırı

Backend bağımsız ve API-first çalışır. Internal React UI, kullanıcı tarafından istenen kontrollü bir extension'dır; backend olmadan çalışamaz ve business/ML/storage mantığı içermez. Product output yeniden encode edilmiş annotated MP4 değil, orijinal video + zaman senkronlu overlay metadata'sıdır. Annotated MP4 yalnız debug/acceptance artifact'ı olarak ayrıca istenirse üretilebilir.

## 4. Zorunlu implementation sırası

Sıra atlanmaz:

1. Requirement/contract/ERD/state-machine freeze.
2. PostgreSQL + MinIO + Qdrant foundation.
3. Image `new_anonymous -> anonymous` vertical slice.
4. Aynı `faceId` ile enrollment -> `known`, update/delete/history.
5. Video upload, retention, async job, cancel/retry.
6. Native GPU video observation extraction.
7. Python temporal tracking ve identity reconciliation.
8. Video best-shot/new-anonymous persistence.
9. Person-level aggregation, appearances ve overlay API.
10. Internal UI, hardening, performance ve full E2E.

Image identity lifecycle gerçek storage üzerinde geçmeden video recognition PASS ilan edilmez.

## 5. Her görevde zorunlu başlangıç

Kod veya doküman değiştirmeden önce:

1. Repository root, branch, HEAD ve `git status --short` doğrulanır.
2. Bu dosya tamamen okunur.
3. `CURRENT_SPRINT.md`, ilgili requirements, architecture ve testler okunur.
4. Dirty worktree'deki kullanıcı değişiklikleri belirlenir ve korunur.
5. Multi-file/sprint işinde `codebase-memory-mcp` ile gerçek caller/callee ve test path'leri keşfedilir; filesystem source ile doğrulanır.
6. İlgili official docs/upstream source `opensourcereferences/references.md` üzerinden seçilir.
7. Plan current sprint ile çelişiyorsa implementation başlatılmaz.

## 6. Repository ve değişiklik güvenliği

Aktif repository dışında yazma yapılmaz. Eski repository'ler read-only referanstır. Kullanıcı istemeden `git add`, commit, push, merge, history rewrite, tracked file silme, model/dataset indirme, Docker volume silme veya system CUDA/driver değişikliği yapılmaz. Machine-specific absolute runtime path production source'a yazılmaz.

## 7. Katman sınırı

- Python control plane: FastAPI, contracts, domain kararları, orchestration, PostgreSQL, MinIO, Qdrant, tracking, reconciliation, history.
- Native data plane: GStreamer/DeepStream, NVDEC/NVMM, CUDA preprocess/alignment, TensorRT detector/recognizer, compact observation emission.
- UI: yalnız versioned API'leri tüketir.

Domain outer layer'a bağımlı olmaz. API doğrudan SQL/Qdrant/MinIO/GPU çağırmaz. Infrastructure business karar sahibi değildir. Tracker kalıcı identity sahibi değildir.

## 8. Production GPU hot path

Hedef video yolu:

```text
encoded video
-> GStreamer graph
-> NVIDIA decoder / NVMM
-> DeepStream batching
-> CUDA/TensorRT RetinaFace
-> CUDA five-point alignment
-> TensorRT ArcFace/Glint embedding
-> GPU L2 normalization
-> compact metadata/embedding CPU boundary
```

Production hot path'te full-frame OpenCV/PIL/NumPy decode, CPU resize, raw tensor NumPy postprocess, frame başına zorunlu device synchronize ve sessiz CPU inference fallback yasaktır.

## 9. GStreamer ve DeepStream kararı

Bu bir “GStreamer mı DeepStream mi?” seçimi değildir: DeepStream, GStreamer tabanlı NVIDIA data plane'dir. Graph GStreamer ile kurulur; `nvv4l2decoder`, `nvstreammux`, `nvdspreprocess`, `nvinfer` veya doğrulanmış custom native elementler gibi DeepStream/NVIDIA bileşenleri kullanılır.

İlk product graph'ında render branch, `nvstreamdemux`, OSD, encoder ve filesink zorunlu değildir. Bunlar GPU observation throughput'unu etkilememelidir. Python NVMM surface map etmez.

## 10. Frame batch ve face batch ayrımı

- Frame batch: detector throughput'u için ardışık input frame'leri.
- Face batch: bir frame batch içinde bulunan değişken sayıdaki yüzlerin recognizer input'u.

Bu iki batch aynı şey değildir. Detector `batch=8` iken recognizer batch'i o sekiz frame'deki geçerli yüz sayısı olabilir. Dynamic TensorRT profile min/opt/max, actual batch, partial final batch ve EOS ayrı test edilir. Batch sonucu frame/PTS sırası kaybedilemez.

## 11. CPU boundary ve backpressure

CPU'ya yalnız şu kompakt kayıtlar çıkar:

- source ID, frame index, PTS/time base;
- original-resolution bbox ve landmarks;
- detector score ve quality metrikleri;
- 512-D embedding veya seçilmiş embedding evidence;
- model/preprocess/profile kimliği.

Full frame, NV12/RGBA surface, detector input tensor veya bütün raw TRT outputs CPU'ya taşınmaz. Native producer bounded ring buffer/queue kullanır. Queue davranışı açıkça `backpressure`, kontrollü drop veya job failure olarak tanımlanır; sessiz sınırsız RAM büyümesi yasaktır.

## 12. Performance iddiası disiplini

`trtexec` engine FPS, detector-only pipeline FPS, GPU observation FPS ve full E2E FPS ayrı metriklerdir. “600 FPS” denebilmesi için hardware, video, codec, batch, sampling, tracker, recognizer, persistence, model SHA ve ölçüm kapsamı yazılmalıdır.

600 FPS yaklaşık 1.67 ms/frame bütçedir. Python tracker bu bütçe içinde varsayımla PASS sayılamaz. Frozen observation replay; p50/p95/p99 latency, queue depth, backlog, drop, RSS ve sustained-duration raporu gerekir. Detector-only sayı full product performansı değildir.

## 13. Python tracker sözleşmesi

İlk tercih Python metadata tracker'dır; C++ rewrite yalnız profiling kanıtıyla yapılır. Her source'un tracker state'i tek sıralı consumer tarafından `PTS/frame` sırasıyla mutate edilir. Batch içindeki frame'ler sırayla tracker'a verilir; batch sınırında state resetlenmez. Kaynak/job seviyesinde paralellik olabilir, aynı source tracker state'inde paralel update olamaz.

ByteTrack adapte edilirse low-score detection ikinci association aşamasına ulaşmalıdır. NMS/threshold ile bütün low-score adayları önceden silinip sonra “ByteTrack kullanılıyor” denilemez.

## 14. Tracklet, track ve identity ayrımı

- `rawTrackletId`: Kesintisiz temporal tracker segmenti.
- `trackId`: Bir video içinde reconciliation sonrası canonical kişi grubu.
- `faceId`: Bütün image/video request'leri boyunca kalıcı biyometrik identity.
- `detectionId`: Tek processed-frame observation.

Scene cut, uzun kayıp veya yeniden giriş yeni raw tracklet üretir. Birinci ve son sahnedeki Rachel'ın aynı `faceId` olması tracker değil, embedding evidence + reconciliation sonucudur. Aynı anda görünen iki farklı yüz için cannot-link uygulanır.

## 15. Identity status semantiği

`new_anonymous` yalnız identity'nin yaratıldığı process/job sonucudur; persistent identity type değildir.

```text
ilk unmatched process -> result=new_anonymous, identity=anonymous
sonraki match          -> result=anonymous, identity=anonymous
aynı faceId enroll     -> sonraki result=known, identity=known
```

`faceId`, sample ID'leri ve geçmiş sonuçlar rename/enroll sırasında değişmez. Immutable `statusAtProcessing/nameAtProcessing` ile mutable `currentStatus/currentName` ayrılır.

## 16. ID ve concurrency kuralları

Persistent ID'ler opaque UUIDv7 olur: `faceId`, `sampleId`, `processId`, `videoId`, `jobId`, `trackId`, `trackletId`. Her HTTP çağrısı `requestId`; business operation `processId`; async video execution `jobId` taşır.

Retry için `Idempotency-Key` desteklenir. Aynı key duplicate process, face identity, MinIO object veya Qdrant point oluşturamaz. Concurrent same-unknown race'i için ikinci vector search, bounded lock/reconciliation ve merge yolu bulunur.

## 17. PostgreSQL ownership

PostgreSQL authoritative business source-of-truth'tür. En az identity/sample/process/result/event/inference-profile ve video asset/job/person/tracklet/appearance/timeline-index/outbox lifecycle'larını taşır. Embedding ve image/video binary PostgreSQL'e yazılmaz.

Historical recognition result immutable snapshot'tır. Current face identity ayrı projection'dır. Uzun per-frame timeline tek JSONB row'a veya unbounded API response'a gömülmez.

## 18. Video job state ve cancellation

`active` boolean tek source-of-truth olamaz. En az şu state'ler bulunur:

```text
pending, processing, cancelling, completed, failed, cancelled
```

`cancellation_requested_at`, `lease_owner`, `lease_expires_at`, `heartbeat_at`, `attempt_no` tutulur. İstenirse `is_active = state IN (pending, processing, cancelling)` derived/generated alanı veya partial index olarak eklenebilir.

Worker kısa transaction içinde `FOR UPDATE SKIP LOCKED` ile claim eder, state/lease yazar ve lock'u bırakır. GPU işi boyunca DB transaction/row lock tutulmaz. Cancel ancak native process gerçekten durup resource cleanup tamamlandıktan sonra `cancelled` olur.

## 19. MinIO ownership

MinIO binary object owner'dır: input images, original videos, selected face crops, timeline/evidence artifacts. Object key'ler yalnız opaque ID ve teknik segment taşır; name/metadata/secrets içermez.

Worker yalnız finalize edilmiş, stat/size/checksum doğrulanmış canonical video objesini işler. Browser stream'i worker ve MinIO'ya iki kez tee edilmez. Video, image ve face-sample retention sınıfları ayrıdır. Source video TTL ile silinirken persistent identity sample crop'u kendiliğinden silinmez.

## 20. Qdrant ownership

Qdrant derived ve rebuildable embedding index'tir. Point ID tam olarak `face_sample.sample_id`; vector 512-D; payload yalnız `sample_id`, `face_id`, active flag ve model/preprocess version gibi teknik alanlar taşır.

Qdrant name/metadata/history sahibi değildir. Search sonucu final karardan önce PostgreSQL identity/sample lifecycle ile doğrulanır. Collection model-versioned olur; model migration dual-read/dual-write veya rebuild planı olmadan yapılmaz.

## 21. Cross-store consistency

PostgreSQL, MinIO ve Qdrant tek transaction paylaşmaz. Yeni sample akışı idempotent state machine'dir:

```text
PG reserve pending_blob
-> deterministic MinIO upload + SHA verify
-> PG blob_ready + outbox
-> Qdrant idempotent upsert(sampleId)
-> PG indexed/active
-> result finalize
```

Qdrant index tamamlanmadan sample recognition-ready görünmez. Partial failure için retry, compensation, orphan scan ve reconciliation integration testleri zorunludur.

## 22. Image recognition workflow

`POST /faces/recognize` bütün yüzleri bağımsız işler. Invalid/corrupt/empty input structured error; no-face başarılı `faceCount=0` sonucudur. Her detection canonical align/embed/search/lifecycle validation'dan geçer.

Existing known aynı `faceId/known`; existing unnamed aynı `faceId/anonymous`; no valid match persistent sample tamamlandıktan sonra `new_anonymous` döner. Mixed known/anonymous/new-anonymous tek response'ta desteklenir.

## 23. Enrollment, update ve delete

Enrollment iki explicit mode taşır:

1. New identity: image + name + metadata; 0 yüz error, 2+ yüz explicit policy/error.
2. Existing anonymous promotion: `faceId + name + metadata`; aynı faceId korunur.

Bir identity çok sayıda sample taşıyabilir. Update optimistic version kullanır. Delete önce identity'yi search dışında bırakır, sonra outbox ile Qdrant/MinIO cleanup yapar. History gereği hard cascade varsayılmaz; tombstone ve privacy policy açıkça tanımlanır. Duplicate identity merge canonical redirect ve audit ile yapılır.

## 24. Video upload ve retention

API direct multipart video kabul etmeye devam eder. Internal UI büyük dosyada presigned multipart upload kullanabilir. Upload complete idempotent olur; backend container/codec, boyut, süre, checksum ve readability doğrular; job yalnız bundan sonra queued olur.

Browser local `File` için object URL ile anında preview gösterebilir. Refresh sonrası private MinIO object kısa ömürlü signed Range URL veya authorized proxy ile oynatılır. Incomplete multipart explicit abort ve stale cleanup ile temizlenir.

## 25. Sampling, zaman ve bbox contract'ı

Sampling `every_n_frames` veya `frames_per_second` olarak request/config üzerinden seçilebilir. İlk correctness fixture'larında every-frame kullanılır. Canonical zaman integer `pts_ns + time_base`; `frame/fps` tek zaman kaynağı değildir.

BBox canonical formatı ve inclusivity dondurulur; API original display-space pixel koordinatı döndürür. Rotation, sample/display aspect ratio, letterbox ve downscale reverse mapping test edilir. Interpolated/held overlay actual detection gibi sunulmaz; provenance alanı taşır.

## 26. Video aggregation

Frame-level evidence doğrudan final identity değildir. Tracklet boyunca quality-selected, temporally diverse embedding'ler robust/quality-weighted şekilde birleştirilir; top-1, top-2, margin, threshold ve kullanılan evidence kaydedilir.

`firstSeen/lastSeen` PTS tabanlıdır. `totalDuration`, appearance interval toplamıdır; aradaki görünmediği süreyi kapsamaz. Person-level result faceId, public trackId, raw tracklet listesi, appearances ve processed-frame detections erişimi taşır.

## 27. Crop ve sample politikası

Her frame crop olarak saklanmaz. İlk baseline: canonical video identity başına en fazla 5 candidate ve en fazla 3 aktif başlangıç sample; exact değer config/calibration ile belirlenir.

Minimum face size, blur, pose, occlusion, landmark geometry, alignment residual, detector score ve temporal diversity değerlendirilir. L2-normalized ArcFace embedding normu image quality olarak kullanılamaz. Existing known identity'ye video sample otomatik eklemek gallery poisoning riski nedeniyle ayrı güçlü gate gerektirir.

## 28. Internal UI ve overlay extension

UI ayrı service/container'dır ve backend API olmadan çalışmaz. Product playback original video üzerinde Canvas/SVG overlay'dir. `requestVideoFrameCallback().metadata.mediaTime`, `ResizeObserver`, DPR, fullscreen, seek, playback-rate, VFR ve `object-fit` offset'leri test edilir.

İsim her detection record'una bake edilmez. Immutable timeline `trackId/bbox/PTS`; mutable identity map `faceId/currentName/currentStatus/version` taşır. Rename sonrası eski video yeniden render edilmeden yeni isim gösterir. Bütün timeline tek seferde yüklenmez; zaman chunk'ları prefetch edilir.

## 29. API, process, log ve history

Versioned OpenAPI contract zorunludur. Requirement endpointleri korunur: image recognize/enroll/detail/delete/history/process ve video recognize/job/status/result/cancel/appearances. Ek upload/playback/timeline/SSE endpointleri extension olarak açıkça dokümante edilir.

Process record, result, identity ve job persistence mandatory business data'dır. Yalnız auxiliary diagnostic logging/metrics best-effort olabilir. Logger failure ana inference'ı bozmaz; result persistence failure başarı gibi dönemez. History pagination ve immutable snapshot/current projection ayrımını destekler.

## 30. Security ve privacy baseline

Input image/video ve face crop biyometrik veridir. Bucket'lar private, signed URL kısa ömürlü, service credentials least-privilege olur. Name/metadata object key, Qdrant payload, raw logs ve error response'a sızmaz. Enrollment/update/delete/merge authorization gerektirir.

Upload content type'a güvenilmez; gerçek container/codec/decode probe edilir. Size, duration, pixel/decompression ve concurrency limitleri config'ten gelir. Secrets hardcode/default boş olamaz. Qdrant public network'e auth/TLS olmadan açılmaz.

## 31. Test-driven development

Production behavior ve bug fix sırası:

1. Failing test veya minimal reproducer.
2. Minimum implementation.
3. Targeted unit test.
4. Integration/contract test.
5. Gerçek PostgreSQL/MinIO/Qdrant veya GPU runtime smoke.
6. Lint/type/build.
7. Diff/scope/review.

Mock, build, plugin registration, engine deserialize veya file existence gerçek runtime/correctness kanıtı değildir.

## 32. Debugging, verification ve benchmark

Runtime failure'da `systematic-debugging` uygulanır: stuck stage belirlenir, buffer/meta/tensor/frame/PTS ve process lifetime gözlemlenir; rastgele timeout/pool/threshold değiştirilmez. Hung container/process temizlenir ve GPU allocation'ın process lifetime mı leak mi olduğu ayrılır.

Completion öncesi `verification-before-completion` uygulanır. Benchmark warmup, tekrar, median/p95, hardware UUID, engine SHA, config ve raw JSON report içerir. CPU tracker replay, GPU observation, storage-disabled E2E ve full E2E ayrı benchmark'lanır.

## 33. Reference-first ve provenance

Implementasyon model hafızasından yazılmaz. Önce `opensourcereferences/references.md` içinden official docs ve pinned upstream source seçilir; ilgili gerçek symbol/call path okunur; sonra failing test yazılır.

Adapte edilen her source için URL, commit/tag, erişim tarihi, repository/per-file license, adapte edilen symbol, yapılan değişiklik, reddedilen alternatif ve local parity gate `REFERENCE_DECISION_LOG.md` içine yazılır. Paper veya README tek başına production contract değildir. Code license ile model-weight license ayrı doğrulanır.

## 34. MCP ve skill accountability

Yeni sprint/multi-file discovery'de `codebase-memory-mcp`; version-sensitive library davranışında `context7`; upstream repository mimarisi/symbol path'inde `deepwiki`; eksik/current primary source aramasında `exa`; API runtime acceptance'ta `postman`; gerçek UI E2E'de `playwright` kullanılır. GitHub plugin/MCP varsa aktif repo ve upstream source doğrulamasında tercih edilir. `21st` ve Ruflo kullanılmaz.

Skill sırası göreve göre uygulanır:

- `using-superpowers`: workflow governance;
- `brainstorming`: yeni architecture/product kararları;
- `writing-plans`: multi-file implementation planı;
- `executing-plans`: onaylanmış plan;
- `test-driven-development`: production behavior;
- `systematic-debugging`: failure/root cause;
- `verification-before-completion`: bütün completion claim'leri;
- `receiving-code-review` / `requesting-code-review`: review lifecycle.

Finalde her MCP ve kullanılan skill için gerçekten ne yaptığı veya neden skipped olduğu yazılır. Çağrılmayan araç `used` gösterilmez.

## 35. Sprint, review ve completion sözleşmesi

Her sprint cohesive, çalışan bir vertical outcome veya açık teknik gate üretir. Report-only sprint açılmaz. Sprint sonunda `CURRENT_SPRINT.md` ve `IMPLEMENTATION_DETAILS.md` güncellenir; meaningful implementation için `docs/implementation/review_packages/SPRINT-<NNN>-CODE-REVIEW-PACKAGE.md` hazırlanır.

Completion verdict yalnız `PASS`, `PARTIAL`, `BLOCKED` veya `NOT_TESTED` olur. Final cevapta çalışan kullanıcı davranışı, exact validation komutları, raw sonuç özeti, changed-source map, known limitations, MCP/skill accountability ve tek önerilen sonraki sprint bulunur. Kanıtsız `production-ready`, `GPU-only`, `600 FPS`, `fully optimized` veya `accuracy verified` denmez.

Sistem önce görüntülerde, ardından videolarda çoklu yüz tespiti ve kalıcı yüz kimliği üretir. Her detected yüzün immutable bir `faceId` değeri olur. İlk karşılaşma `new_anonymous`, daha sonraki eşleşme `anonymous`, aynı `faceId` isimlendirildikten sonra `known` sonucunu üretir. Video genişletmesi aynı identity karar motorunu kullanır; tracking yalnız temporal sürekliliği, recognition kalıcı kimliği cevaplar.

## 2. Source-of-truth sırası

Çelişki halinde sıra şöyledir:

1. Kullanıcının güncel açık kararı.
2. `requirements/ProjectRequirements.md` içindeki image/identity gereksinimleri.
3. `requirements/videorequirements.md` içindeki additive video gereksinimleri.
4. Kullanıcı tarafından onaylanmış architecture/ADR belgeleri.
5. `docs/implementation/CURRENT_SPRINT.md`.
6. Official vendor documentation ve pinned upstream source.
7. Eski repository kodları ve raporları yalnız lessons-learned kaynağıdır.

Eski client özetleri; Oracle, 10M kişi, national ID veya başka ek kapsamları kullanıcı yeniden onaylamadıkça ürün requirement'ı sayılmaz.

## 3. Onaylanmış ürün sınırı

Backend bağımsız ve API-first çalışır. Internal React UI, kullanıcı tarafından istenen kontrollü bir extension'dır; backend olmadan çalışamaz ve business/ML/storage mantığı içermez. Product output yeniden encode edilmiş annotated MP4 değil, orijinal video + zaman senkronlu overlay metadata'sıdır. Annotated MP4 yalnız debug/acceptance artifact'ı olarak ayrıca istenirse üretilebilir.

## 4. Zorunlu implementation sırası

Sıra atlanmaz:

1. Requirement/contract/ERD/state-machine freeze.
2. PostgreSQL + MinIO + Qdrant foundation.
3. Image `new_anonymous -> anonymous` vertical slice.
4. Aynı `faceId` ile enrollment -> `known`, update/delete/history.
5. Video upload, retention, async job, cancel/retry.
6. Native GPU video observation extraction.
7. Python temporal tracking ve identity reconciliation.
8. Video best-shot/new-anonymous persistence.
9. Person-level aggregation, appearances ve overlay API.
10. Internal UI, hardening, performance ve full E2E.

Image identity lifecycle gerçek storage üzerinde geçmeden video recognition PASS ilan edilmez.

## 5. Her görevde zorunlu başlangıç

Kod veya doküman değiştirmeden önce:

1. Repository root, branch, HEAD ve `git status --short` doğrulanır.
2. Bu dosya tamamen okunur.
3. `CURRENT_SPRINT.md`, ilgili requirements, architecture ve testler okunur.
4. Dirty worktree'deki kullanıcı değişiklikleri belirlenir ve korunur.
5. Multi-file/sprint işinde `codebase-memory-mcp` ile gerçek caller/callee ve test path'leri keşfedilir; filesystem source ile doğrulanır.
6. İlgili official docs/upstream source `opensourcereferences/references.md` üzerinden seçilir.
7. Plan current sprint ile çelişiyorsa implementation başlatılmaz.

## 6. Repository ve değişiklik güvenliği

Aktif repository dışında yazma yapılmaz. Eski repository'ler read-only referanstır. Kullanıcı istemeden `git add`, commit, push, merge, history rewrite, tracked file silme, model/dataset indirme, Docker volume silme veya system CUDA/driver değişikliği yapılmaz. Machine-specific absolute runtime path production source'a yazılmaz.

## 7. Katman sınırı

- Python control plane: FastAPI, contracts, domain kararları, orchestration, PostgreSQL, MinIO, Qdrant, tracking, reconciliation, history.
- Native data plane: GStreamer/DeepStream, NVDEC/NVMM, CUDA preprocess/alignment, TensorRT detector/recognizer, compact observation emission.
- UI: yalnız versioned API'leri tüketir.

Domain outer layer'a bağımlı olmaz. API doğrudan SQL/Qdrant/MinIO/GPU çağırmaz. Infrastructure business karar sahibi değildir. Tracker kalıcı identity sahibi değildir.

## 8. Production GPU hot path

Hedef video yolu:

```text
encoded video
-> GStreamer graph
-> NVIDIA decoder / NVMM
-> DeepStream batching
-> CUDA/TensorRT RetinaFace
-> CUDA five-point alignment
-> TensorRT ArcFace/Glint embedding
-> GPU L2 normalization
-> compact metadata/embedding CPU boundary
```

Production hot path'te full-frame OpenCV/PIL/NumPy decode, CPU resize, raw tensor NumPy postprocess, frame başına zorunlu device synchronize ve sessiz CPU inference fallback yasaktır.

## 9. GStreamer ve DeepStream kararı

Bu bir “GStreamer mı DeepStream mi?” seçimi değildir: DeepStream, GStreamer tabanlı NVIDIA data plane'dir. Graph GStreamer ile kurulur; `nvv4l2decoder`, `nvstreammux`, `nvdspreprocess`, `nvinfer` veya doğrulanmış custom native elementler gibi DeepStream/NVIDIA bileşenleri kullanılır.

İlk product graph'ında render branch, `nvstreamdemux`, OSD, encoder ve filesink zorunlu değildir. Bunlar GPU observation throughput'unu etkilememelidir. Python NVMM surface map etmez.

## 10. Frame batch ve face batch ayrımı

- Frame batch: detector throughput'u için ardışık input frame'leri.
- Face batch: bir frame batch içinde bulunan değişken sayıdaki yüzlerin recognizer input'u.

Bu iki batch aynı şey değildir. Detector `batch=8` iken recognizer batch'i o sekiz frame'deki geçerli yüz sayısı olabilir. Dynamic TensorRT profile min/opt/max, actual batch, partial final batch ve EOS ayrı test edilir. Batch sonucu frame/PTS sırası kaybedilemez.

## 11. CPU boundary ve backpressure

CPU'ya yalnız şu kompakt kayıtlar çıkar:

- source ID, frame index, PTS/time base;
- original-resolution bbox ve landmarks;
- detector score ve quality metrikleri;
- 512-D embedding veya seçilmiş embedding evidence;
- model/preprocess/profile kimliği.

Full frame, NV12/RGBA surface, detector input tensor veya bütün raw TRT outputs CPU'ya taşınmaz. Native producer bounded ring buffer/queue kullanır. Queue davranışı açıkça `backpressure`, kontrollü drop veya job failure olarak tanımlanır; sessiz sınırsız RAM büyümesi yasaktır.

## 12. Performance iddiası disiplini

`trtexec` engine FPS, detector-only pipeline FPS, GPU observation FPS ve full E2E FPS ayrı metriklerdir. “600 FPS” denebilmesi için hardware, video, codec, batch, sampling, tracker, recognizer, persistence, model SHA ve ölçüm kapsamı yazılmalıdır.

600 FPS yaklaşık 1.67 ms/frame bütçedir. Python tracker bu bütçe içinde varsayımla PASS sayılamaz. Frozen observation replay; p50/p95/p99 latency, queue depth, backlog, drop, RSS ve sustained-duration raporu gerekir. Detector-only sayı full product performansı değildir.

## 13. Python tracker sözleşmesi

İlk tercih Python metadata tracker'dır; C++ rewrite yalnız profiling kanıtıyla yapılır. Her source'un tracker state'i tek sıralı consumer tarafından `PTS/frame` sırasıyla mutate edilir. Batch içindeki frame'ler sırayla tracker'a verilir; batch sınırında state resetlenmez. Kaynak/job seviyesinde paralellik olabilir, aynı source tracker state'inde paralel update olamaz.

ByteTrack adapte edilirse low-score detection ikinci association aşamasına ulaşmalıdır. NMS/threshold ile bütün low-score adayları önceden silinip sonra “ByteTrack kullanılıyor” denilemez.

## 14. Tracklet, track ve identity ayrımı

- `rawTrackletId`: Kesintisiz temporal tracker segmenti.
- `trackId`: Bir video içinde reconciliation sonrası canonical kişi grubu.
- `faceId`: Bütün image/video request'leri boyunca kalıcı biyometrik identity.
- `detectionId`: Tek processed-frame observation.

Scene cut, uzun kayıp veya yeniden giriş yeni raw tracklet üretir. Birinci ve son sahnedeki Rachel'ın aynı `faceId` olması tracker değil, embedding evidence + reconciliation sonucudur. Aynı anda görünen iki farklı yüz için cannot-link uygulanır.

## 15. Identity status semantiği

`new_anonymous` yalnız identity'nin yaratıldığı process/job sonucudur; persistent identity type değildir.

```text
ilk unmatched process -> result=new_anonymous, identity=anonymous
sonraki match          -> result=anonymous, identity=anonymous
aynı faceId enroll     -> sonraki result=known, identity=known
```

`faceId`, sample ID'leri ve geçmiş sonuçlar rename/enroll sırasında değişmez. Immutable `statusAtProcessing/nameAtProcessing` ile mutable `currentStatus/currentName` ayrılır.

## 16. ID ve concurrency kuralları

Persistent ID'ler opaque UUIDv7 olur: `faceId`, `sampleId`, `processId`, `videoId`, `jobId`, `trackId`, `trackletId`. Her HTTP çağrısı `requestId`; business operation `processId`; async video execution `jobId` taşır.

Retry için `Idempotency-Key` desteklenir. Aynı key duplicate process, face identity, MinIO object veya Qdrant point oluşturamaz. Concurrent same-unknown race'i için ikinci vector search, bounded lock/reconciliation ve merge yolu bulunur.

## 17. PostgreSQL ownership

PostgreSQL authoritative business source-of-truth'tür. En az identity/sample/process/result/event/inference-profile ve video asset/job/person/tracklet/appearance/timeline-index/outbox lifecycle'larını taşır. Embedding ve image/video binary PostgreSQL'e yazılmaz.

Historical recognition result immutable snapshot'tır. Current face identity ayrı projection'dır. Uzun per-frame timeline tek JSONB row'a veya unbounded API response'a gömülmez.

## 18. Video job state ve cancellation

`active` boolean tek source-of-truth olamaz. En az şu state'ler bulunur:

```text
pending, processing, cancelling, completed, failed, cancelled
```

`cancellation_requested_at`, `lease_owner`, `lease_expires_at`, `heartbeat_at`, `attempt_no` tutulur. İstenirse `is_active = state IN (pending, processing, cancelling)` derived/generated alanı veya partial index olarak eklenebilir.

Worker kısa transaction içinde `FOR UPDATE SKIP LOCKED` ile claim eder, state/lease yazar ve lock'u bırakır. GPU işi boyunca DB transaction/row lock tutulmaz. Cancel ancak native process gerçekten durup resource cleanup tamamlandıktan sonra `cancelled` olur.

## 19. MinIO ownership

MinIO binary object owner'dır: input images, original videos, selected face crops, timeline/evidence artifacts. Object key'ler yalnız opaque ID ve teknik segment taşır; name/metadata/secrets içermez.

Worker yalnız finalize edilmiş, stat/size/checksum doğrulanmış canonical video objesini işler. Browser stream'i worker ve MinIO'ya iki kez tee edilmez. Video, image ve face-sample retention sınıfları ayrıdır. Source video TTL ile silinirken persistent identity sample crop'u kendiliğinden silinmez.

## 20. Qdrant ownership

Qdrant derived ve rebuildable embedding index'tir. Point ID tam olarak `face_sample.sample_id`; vector 512-D; payload yalnız `sample_id`, `face_id`, active flag ve model/preprocess version gibi teknik alanlar taşır.

Qdrant name/metadata/history sahibi değildir. Search sonucu final karardan önce PostgreSQL identity/sample lifecycle ile doğrulanır. Collection model-versioned olur; model migration dual-read/dual-write veya rebuild planı olmadan yapılmaz.

## 21. Cross-store consistency

PostgreSQL, MinIO ve Qdrant tek transaction paylaşmaz. Yeni sample akışı idempotent state machine'dir:

```text
PG reserve pending_blob
-> deterministic MinIO upload + SHA verify
-> PG blob_ready + outbox
-> Qdrant idempotent upsert(sampleId)
-> PG indexed/active
-> result finalize
```

Qdrant index tamamlanmadan sample recognition-ready görünmez. Partial failure için retry, compensation, orphan scan ve reconciliation integration testleri zorunludur.

## 22. Image recognition workflow

`POST /faces/recognize` bütün yüzleri bağımsız işler. Invalid/corrupt/empty input structured error; no-face başarılı `faceCount=0` sonucudur. Her detection canonical align/embed/search/lifecycle validation'dan geçer.

Existing known aynı `faceId/known`; existing unnamed aynı `faceId/anonymous`; no valid match persistent sample tamamlandıktan sonra `new_anonymous` döner. Mixed known/anonymous/new-anonymous tek response'ta desteklenir.

## 23. Enrollment, update ve delete

Enrollment iki explicit mode taşır:

1. New identity: image + name + metadata; 0 yüz error, 2+ yüz explicit policy/error.
2. Existing anonymous promotion: `faceId + name + metadata`; aynı faceId korunur.

Bir identity çok sayıda sample taşıyabilir. Update optimistic version kullanır. Delete önce identity'yi search dışında bırakır, sonra outbox ile Qdrant/MinIO cleanup yapar. History gereği hard cascade varsayılmaz; tombstone ve privacy policy açıkça tanımlanır. Duplicate identity merge canonical redirect ve audit ile yapılır.

## 24. Video upload ve retention

API direct multipart video kabul etmeye devam eder. Internal UI büyük dosyada presigned multipart upload kullanabilir. Upload complete idempotent olur; backend container/codec, boyut, süre, checksum ve readability doğrular; job yalnız bundan sonra queued olur.

Browser local `File` için object URL ile anında preview gösterebilir. Refresh sonrası private MinIO object kısa ömürlü signed Range URL veya authorized proxy ile oynatılır. Incomplete multipart explicit abort ve stale cleanup ile temizlenir.

## 25. Sampling, zaman ve bbox contract'ı

Sampling `every_n_frames` veya `frames_per_second` olarak request/config üzerinden seçilebilir. İlk correctness fixture'larında every-frame kullanılır. Canonical zaman integer `pts_ns + time_base`; `frame/fps` tek zaman kaynağı değildir.

BBox canonical formatı ve inclusivity dondurulur; API original display-space pixel koordinatı döndürür. Rotation, sample/display aspect ratio, letterbox ve downscale reverse mapping test edilir. Interpolated/held overlay actual detection gibi sunulmaz; provenance alanı taşır.

## 26. Video aggregation

Frame-level evidence doğrudan final identity değildir. Tracklet boyunca quality-selected, temporally diverse embedding'ler robust/quality-weighted şekilde birleştirilir; top-1, top-2, margin, threshold ve kullanılan evidence kaydedilir.

`firstSeen/lastSeen` PTS tabanlıdır. `totalDuration`, appearance interval toplamıdır; aradaki görünmediği süreyi kapsamaz. Person-level result faceId, public trackId, raw tracklet listesi, appearances ve processed-frame detections erişimi taşır.

## 27. Crop ve sample politikası

Her frame crop olarak saklanmaz. İlk baseline: canonical video identity başına en fazla 5 candidate ve en fazla 3 aktif başlangıç sample; exact değer config/calibration ile belirlenir.

Minimum face size, blur, pose, occlusion, landmark geometry, alignment residual, detector score ve temporal diversity değerlendirilir. L2-normalized ArcFace embedding normu image quality olarak kullanılamaz. Existing known identity'ye video sample otomatik eklemek gallery poisoning riski nedeniyle ayrı güçlü gate gerektirir.

## 28. Internal UI ve overlay extension

UI ayrı service/container'dır ve backend API olmadan çalışmaz. Product playback original video üzerinde Canvas/SVG overlay'dir. `requestVideoFrameCallback().metadata.mediaTime`, `ResizeObserver`, DPR, fullscreen, seek, playback-rate, VFR ve `object-fit` offset'leri test edilir.

İsim her detection record'una bake edilmez. Immutable timeline `trackId/bbox/PTS`; mutable identity map `faceId/currentName/currentStatus/version` taşır. Rename sonrası eski video yeniden render edilmeden yeni isim gösterir. Bütün timeline tek seferde yüklenmez; zaman chunk'ları prefetch edilir.

## 29. API, process, log ve history

Versioned OpenAPI contract zorunludur. Requirement endpointleri korunur: image recognize/enroll/detail/delete/history/process ve video recognize/job/status/result/cancel/appearances. Ek upload/playback/timeline/SSE endpointleri extension olarak açıkça dokümante edilir.

Process record, result, identity ve job persistence mandatory business data'dır. Yalnız auxiliary diagnostic logging/metrics best-effort olabilir. Logger failure ana inference'ı bozmaz; result persistence failure başarı gibi dönemez. History pagination ve immutable snapshot/current projection ayrımını destekler.

## 30. Security ve privacy baseline

Input image/video ve face crop biyometrik veridir. Bucket'lar private, signed URL kısa ömürlü, service credentials least-privilege olur. Name/metadata object key, Qdrant payload, raw logs ve error response'a sızmaz. Enrollment/update/delete/merge authorization gerektirir.

Upload content type'a güvenilmez; gerçek container/codec/decode probe edilir. Size, duration, pixel/decompression ve concurrency limitleri config'ten gelir. Secrets hardcode/default boş olamaz. Qdrant public network'e auth/TLS olmadan açılmaz.

## 31. Test-driven development

Production behavior ve bug fix sırası:

1. Failing test veya minimal reproducer.
2. Minimum implementation.
3. Targeted unit test.
4. Integration/contract test.
5. Gerçek PostgreSQL/MinIO/Qdrant veya GPU runtime smoke.
6. Lint/type/build.
7. Diff/scope/review.

Mock, build, plugin registration, engine deserialize veya file existence gerçek runtime/correctness kanıtı değildir.

## 32. Debugging, verification ve benchmark

Runtime failure'da `systematic-debugging` uygulanır: stuck stage belirlenir, buffer/meta/tensor/frame/PTS ve process lifetime gözlemlenir; rastgele timeout/pool/threshold değiştirilmez. Hung container/process temizlenir ve GPU allocation'ın process lifetime mı leak mi olduğu ayrılır.

Completion öncesi `verification-before-completion` uygulanır. Benchmark warmup, tekrar, median/p95, hardware UUID, engine SHA, config ve raw JSON report içerir. CPU tracker replay, GPU observation, storage-disabled E2E ve full E2E ayrı benchmark'lanır.

## 33. Reference-first ve provenance

Implementasyon model hafızasından yazılmaz. Önce `opensourcereferences/references.md` içinden official docs ve pinned upstream source seçilir; ilgili gerçek symbol/call path okunur; sonra failing test yazılır.

Adapte edilen her source için URL, commit/tag, erişim tarihi, repository/per-file license, adapte edilen symbol, yapılan değişiklik, reddedilen alternatif ve local parity gate `REFERENCE_DECISION_LOG.md` içine yazılır. Paper veya README tek başına production contract değildir. Code license ile model-weight license ayrı doğrulanır.

## 34. MCP ve skill accountability

Yeni sprint/multi-file discovery'de `codebase-memory-mcp`; version-sensitive library davranışında `context7`; upstream repository mimarisi/symbol path'inde `deepwiki`; eksik/current primary source aramasında `exa`; API runtime acceptance'ta `postman`; gerçek UI E2E'de `playwright` kullanılır. GitHub plugin/MCP varsa aktif repo ve upstream source doğrulamasında tercih edilir. `21st` ve Ruflo kullanılmaz.

Skill sırası göreve göre uygulanır:

- `using-superpowers`: workflow governance;
- `brainstorming`: yeni architecture/product kararları;
- `writing-plans`: multi-file implementation planı;
- `executing-plans`: onaylanmış plan;
- `test-driven-development`: production behavior;
- `systematic-debugging`: failure/root cause;
- `verification-before-completion`: bütün completion claim'leri;
- `receiving-code-review` / `requesting-code-review`: review lifecycle.

Finalde her MCP ve kullanılan skill için gerçekten ne yaptığı veya neden skipped olduğu yazılır. Çağrılmayan araç `used` gösterilmez.

## 35. Sprint, review ve completion sözleşmesi

Her sprint cohesive, çalışan bir vertical outcome veya açık teknik gate üretir. Report-only sprint açılmaz. Sprint sonunda `CURRENT_SPRINT.md` ve `IMPLEMENTATION_DETAILS.md` güncellenir; meaningful implementation için `docs/implementation/review_packages/SPRINT-<NNN>-CODE-REVIEW-PACKAGE.md` hazırlanır.

Completion verdict yalnız `PASS`, `PARTIAL`, `BLOCKED` veya `NOT_TESTED` olur. Final cevapta çalışan kullanıcı davranışı, exact validation komutları, raw sonuç özeti, changed-source map, known limitations, MCP/skill accountability ve tek önerilen sonraki sprint bulunur. Kanıtsız `production-ready`, `GPU-only`, `600 FPS`, `fully optimized` veya `accuracy verified` denmez.