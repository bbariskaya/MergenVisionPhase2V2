Abi, origin’i detaylı inceledim. Toplu revert yapmayın: image/video GPU çekirdeği büyük ölçüde hâlâ duruyor. Sorun, etrafındaki contract ve ürün katmanının dağılmış olması.

En kritik bulgular:

Frontend /people çağırıyor ama backend’de People API/table yok; UI kesin kırık.
Çözüm yeni Person şeması değil: canonical kimlik yine face_identity/faceId.
Bulk run’daki 1.310 failure yanlış; 655 rejection iki kere sayılmış.
Gerçek enrollment batch-256 değil; subject subject çalıştığı için çoğu GPU çağrısı batch 1–2.
12.578/12.578 reconcile yalnız storage existence kanıtı; image/video recognition continuity kanıtı değil.
Bulk komutu kayıtları yazdıktan sonra CUDA context teardown sırasında abort/segfault oluyor.
Video enrollment sonrası historical snapshot korunuyor ama current name/status UI’a yeniden yansıtılmıyor.
Worker acceptance target’ı gerçek worker/GPU zincirini tam başlatmadığı için şu an “full E2E” denemez.
LFW manifest/logları subject isimleri ve absolute path’lerle commit edilmiş.
CURRENT_SPRINT kendi içinde green/NOT_RUN çelişkileri taşıyor.
Canonical ürün gereksinimleri hâlâ image requirements ve video requirements; Person/redirect planları stale.
İncelediğim mevcut origin HEAD: 6f1a527.

Hazırladığım prompt 615 satır ve şunları içeriyor:

Önce editsiz, uzun forensic Plan Mode
c555c9a4 ve 2cfde196 üzerinden cerrahi recovery
Phase 1/Phase 2 requirements restorasyonu
FaceIdentity merkezli tek identity mimarisi
Frontend/API enrollment düzeltmesi
Bulk CUDA teardown, batching, lifecycle, resume ve multi-GPU
Worker heartbeat/fencing/cancel/artifact/streaming düzeltmeleri
Bulk → held-out image → gerçek video → UI’da aynı faceId acceptance
İzole GPU video quality/tracker lab
96 maddelik Build Mode sırası
Gerçek full-E2E acceptance ve hard stops
Memory MCP, codebase-memory, Context7, DeepWiki, Postman ve Playwright disiplini

Tam prompt:

MERGENVISION_BIG_PICTURE_RECOVERY_PROMPT.md

Yeni agent’a dosyanın tamamını tek mesaj olarak ver. İlk turda mutlaka şu iki satırla durmasını bekle:

PLAN_READY_FOR_HUMAN_REVIEW
NO_SOURCE_FILES_CHANGED

Planı görmeden Build Mode’a geçirme; bu sefer önce bütün sistemi doğru şekilde dondursun.


# OpenCode Master Prompt — MergenVision Big-Picture Recovery

> Aşağıdaki metni yeni OpenCode agent oturumuna eksiksiz ver.

## ROL VE ÇALIŞMA MODU

Sen MergenVisionPhase2V2 repository'sini kurtaran senior staff/principal engineer'sin. Görevin tek bir testi yeşile çevirmek, yalnız bulk enrollment'ı bitirmek veya geçmiş commit'i körlemesine geri almak değildir. Görevin, çalışan native GPU çekirdeklerini koruyarak Phase 1 image identity ürünü, Phase 2 video identity ürünü, bulk dataset enrollment ve React UI'ı tekrar tek, tutarlı, doğrulanabilir bir ürün haline getirmektir.

Bu oturumun ilk aşaması **PLAN MODE**'dur. İlk aşamada production source, migration, test, config, doküman veya artifact değiştirme. Önce uzun ve kanıta dayalı forensic inceleme yap; aşağıda istenen recovery planını sun ve human review bekle. Kullanıcı açıkça `BUILD MODE'A GEÇ` demeden implementasyona başlama.

Plan kabul edilip Build Mode'a geçildiğinde mikro-onay istemeden milestone sırasıyla ilerle. Yalnız gerçek bir destructive işlem, migration/schema kararı, secret, model/dataset indirme, system CUDA/driver değişikliği veya git publish yetkisi gerekiyorsa dur.

Aktif repository:

```text
/home/user/Workspace/MergenVisionPhase2v2
origin: https://github.com/bbariskaya/MergenVisionPhase2V2
current audited origin HEAD: 6f1a52735d90b0a49aad4bdf73b968337937ccc6
```

Tarihsel read-only recovery anchor'ları:

```text
c555c9a4  # çalışan Phase 1 image identity baseline, “phase 1 done”
2cfde196  # Person contamination öncesi image + video ürün baseline'ı
f81f9f4   # daha sonraki ara durum; yalnız diff/provenance incelemesi için
6f1a527   # mevcut HEAD; kaynak gerçekliği fakat otomatik olarak ürün doğrusu değil
```

**Toplu reset/revert/cherry-pick yapma.** Bu commit'leri salt-okunur karşılaştırma ve symbol-level recovery kaynağı olarak kullan. Current HEAD'de korunması gereken yeni düzeltmeler ve izole bulk package var. Cerrahi restorasyon yap.

---

## 1. ÜRÜNÜN GERÇEK AMACI

MergenVision bir model demosu değildir. Kullanıcıya şu sorunun kalıcı cevabını verir:

> “Bu fotoğraf veya videoda kimler var; sistem bu yüzü daha önce gördü mü; aynı global `faceId` fotoğraf, video ve dataset enrollment boyunca korunuyor mu?”

Tek global identity aggregate şu an `face_identity` / `face_id`'dir.

- `faceId`: fotoğraf, video, bulk dataset ve UI boyunca kalıcı global identity.
- `trackId`: yalnız bir video/job içindeki yerel temporal track kimliği.
- `sampleId`: bir identity'ye ait belirli embedding/crop örneği; Qdrant point ID ile aynıdır.
- `processId`: business operation.
- `jobId`: async video execution.
- `requestId`: tek HTTP request tracing.

Identity sonucu:

- `new_anonymous`: ilk kez görülüp yeni global `faceId` oluşturulan recognition snapshot'ı.
- `anonymous`: daha önce görülmüş, aynı `faceId`, henüz isimlendirilmemiş identity.
- `known`: aynı `faceId` enroll edilmiş; name/metadata vardır.

Persistent `face_identity.status` yalnız `anonymous|known` olabilir. `new_anonymous` yalnız immutable recognition snapshot status'udur.

Beklenen continuity:

```text
ilk fotoğraf          -> new_anonymous, faceId=A
aynı fotoğraf         -> anonymous,     faceId=A
UI/API enrollment     -> known,         faceId=A
aynı fotoğraf         -> known,         faceId=A
aynı kişi videoda     -> known,         faceId=A
bulk seed holdout     -> known,         seeded faceId=A
```

Enrollment yeni `faceId` üretmez. Historical recognition/video snapshot'ları yeniden yazılmaz. Buna karşılık UI, historical snapshot yanında identity'nin **current** durumunu da gösterebilmelidir.

---

## 2. BAĞLAYICI ÜRÜN KARARLARI

Bu kararları tekrar kullanıcıya sorma:

1. **Canonical identity `face_identity`'dir.** Bu recovery sırasında `person` tablosu, `person_id`, Person CRUD, redirect/alias migration veya ikinci identity aggregate yaratma. `people` kelimesi UI presentation terimi olabilir; backend truth `face_id`'dir.
2. `newmission.md`, eski TARGET_CONTRACT ve Person/redirect planları requirement değildir; stale/superseded olarak değerlendir.
3. Phase 1 image ve Phase 2 video aynı PostgreSQL identity/sample lifecycle'ını ve aynı Qdrant collection'ı kullanır.
4. Qdrant collection: `face_samples_retinaface_r50_glintr100_v1`.
5. Qdrant point ID tam olarak `face_sample.sample_id`; payload yalnız teknik alanlar (`sample_id`, `face_id`, `active`, `model_version`). Name/metadata yok.
6. PostgreSQL business source-of-truth; MinIO binary owner; Qdrant derived vector index.
7. Modeller değişmeyecek:
   - `backend/artifacts/models/retinaface_r50_dynamic.onnx`
   - `backend/artifacts/models/glintr100.onnx`
8. Production inference'ta CPU fallback, PIL/OpenCV full-frame decode/resize, FFmpeg crop extraction ve frame-by-frame image API round-trip yasak.
9. Video output annotated MP4 değildir. Orijinal video + timeline metadata + React canvas overlay kullanılacak.
10. Existing Phase 2 native GPU hot path sıfırdan yazılmayacak. Önce baseline ve parity ile korunacak.
11. Bulk package ayrı data plane/package olarak kalabilir; Phase 2 native source'una performans bahanesiyle müdahale etmez. Fakat aynı identity/storage/model/preprocess contract'ına yazmalıdır.
12. Bulk aligned crop kararı:
    - key: `faces/{face_id}/{sample_id}/aligned.jpg`
    - content type: `image/jpeg`
    - encode: GPU-backed nvImageCodec/nvJPEG; PIL/FFmpeg yok.
13. Existing image/video online flow'un `.webp` crop'larını değiştirme. `.jpg` ve `.webp` birlikte yaşayabilir; DB `object_key` authoritative'dir. API doğru media type döndürmelidir.
14. Internal React UI mandatory'dir. Legacy requirement docs içindeki “UI olmayacak” cümlesi bu kullanıcı kararıyla supersede edilmiştir.
15. `research/video_reference_lab/**` frozen. Quality/tracker deneyleri için ayrı `research/gpu_video_lab/**` veya plan review'da onaylanan izole path kullanılabilir; production dependency yapılamaz.
16. Git add/commit/push, history rewrite, destructive reset, live PG/MinIO/Qdrant silme/reset yok.

---

## 3. ŞU ANKİ REPOSITORY GERÇEĞİ — BUNLARI DOĞRULA

Origin audit'te görülen gerçekleri source üzerinden yeniden doğrula; körlemesine kabul etme:

### Korunması muhtemel çalışan çekirdekler

- Phase 1 image API/backend/native GPU zinciri current HEAD'de hâlâ mevcut.
- Phase 2 video upload/job/worker/native inference/tracking/result/UI çekirdeğinin büyük kısmı current HEAD'de hâlâ mevcut.
- `backend/native/video_worker` real Friends smoke'ta 6.665 frame, 9.020 detection/embedding ve 150 raw track üretmişti.
- React video overlay source'u (`VideoOverlayPlayer`) current HEAD'de mevcut.
- Bulk package gerçek LFW run'da 12.578 accepted sample'ı PG/MinIO/Qdrant'a yazmış ve existence reconcile 12.578/12.578 göstermişti.
- Post-`2cfde196` korunması gereken küçük düzeltmeler:
  - `process_record.py` terminal timestamp fix'i,
  - `video_worker_main.py` `VideoObservationFrame` import fix'i,
  - `backend/native/image_runtime/src/model_profile.cpp` içindeki gerekli pybind STL include fix'i; source diff ve test ile gerçekten gerekli olduğunu doğrula.

### Current HEAD'deki bilinen P0 drift/bulgular

1. Backend `faces`, `processes`, `videos` router'larını mount ediyor; `/api/v1/people` router'ı yok.
2. Frontend `PeoplePage.tsx` ve `api/people.ts`, olmayan `/api/v1/people*` endpoint'lerini çağırıyor; bu kesin 404 contract drift'idir.
3. Frontend `/people/:personId` linkleri üretiyor fakat route yok.
4. Enroll/UI types `person_id` bekleyebiliyor; backend canonical response bunu dönmüyor.
5. Çözüm Person tablosunu geri getirmek değildir. `PeoplePage`, `/api/v1/faces` identity directory'sine bağlanmalıdır.
6. Video result/overlay, processing-time `status_at_processing/name_at_processing` snapshot'ını döndürüyor olabilir. Enrollment sonrası current name/status UI'da güncellenmiyor.
7. Current projection şu şekilde ayrılmalıdır:
   - immutable: `statusAtProcessing`, `nameAtProcessing`
   - mutable read-time projection: `currentStatus`, `currentName`, `currentMetadata`
8. Video enrollment sonrası people/overlay/timeline React Query cache'leri invalidate edilmelidir.
9. `CURRENT_SPRINT.md` milestone ledger ile Status paragrafı çelişiyor: bir yerde native M4/M5 green, başka yerde NOT_RUN/open.
10. `requirements/phase1requirements.md` ve `requirements/phase2requirements.md` origin'de yok; yalnız legacy `ProjectRequirements.md` ve `videorequirements.md` var.
11. `newmission.md`, TARGET_CONTRACT ve bazı prompt/review docs Person/redirect şeması anlatıyor fakat current DB Person-free.
12. Root `.artifacts/phase1_gpu_bulk_enrollment/**` altında büyük LFW log/manifest/baseline dosyaları commit edilmiş; subject isimleri ve makine path'leri sızıyor.
13. `README.md`, bazı review package'lar ve package acceptance docs current source ve gerçek runtime durumuyla uyumsuz.

---

## 4. BULK ENROLLMENT KANITINI DOĞRU YORUMLA

Mevcut run değerli fakat **PARTIAL**'dır:

```text
dataset images: 13,233
accepted/persisted: 12,578
benchmark quarantine: 655
reconcile existence: 12,578 / 12,578
compute-only benchmark: ~363 img/s
end-to-end persistence: ~49 img/s
```

Şunları kanıtlar:

- GPU extraction gerçek iş yaptı.
- 12.578 active sample için PG/MinIO/Qdrant existence tutarlıydı.
- Nonzero GPU device stream bug'ının `cudaSetDevice` ile ilgili olduğu doğrulandı.

Şunları kanıtlamaz:

- clean process exit,
- gerçek cross-subject batch-256 enrollment,
- multi-GPU,
- resume,
- embedding parity,
- held-out image recognition,
- video known continuity,
- accuracy/calibrated threshold,
- semantic cross-store correctness.

Bilinen source bug'ları:

1. `GpuFacePipeline.close()` `_jpeg_encoder` ve native/TensorRT/CUDA ownership'i deterministik kapatmıyor. Run sonunda `context is destroyed`, abort veya segfault görüldü.
2. Close idempotent değil veya destructor ikinci kez close edebilir. Correct device set + synchronization + dependency-aware teardown test edilmelidir.
3. Detector score async D2H kopyasından sonra synchronize edilmeden `score_h[0]` okunuyor olabilir.
4. Batch decode herhangi bir exception'da per-image fallback yapıyor. Systemic CUDA/context/OOM error'ı corrupt JPEG gibi gizlenmemeli. Yalnız açıkça sınıflandırılmış per-entry media error per-image rejection olabilir.
5. JPEG encoder backend selection fail-closed değil; default fallback'in CPU codec seçmediği kanıtlanmıyor.
6. Crop encode Python loop'unda yüz başına çağrılıyor; batch encoder API/support gerçek source ve official docs üzerinden araştırılmalı.
7. CLI `for bundle in bundles` ile kişi kişi `extract_batch` çağırıyor. LFW çoğunlukla 1–2 fotoğraf/subject olduğundan `--batch-size 256`, farklı subject'leri aynı GPU microbatch'ine sokmuyor.
8. Benchmark cross-subject batch kullanıyor; enrollment path kullanmıyor. İki throughput aynı şey değildir.
9. `persist_bundle()` rejected'ları `persisted.failed` içine zaten koyuyor; CLI tekrar `persisted.failed + rejected` yapıyor. 655 rejection'ın 1.310 diye raporlanmasının nedeni budur.
10. Accepted sample yokken known `face_identity` hazırlanabiliyor; active sample'sız boş known identity oluşma riski var.
11. Rejected sample satırı hazırlanmadığı halde `fail_samples_tx` update'i yapılabiliyor; rowcount/lifecycle gerçeği doğrulanmıyor.
12. Manifest ve bütün JPEG bytes dataset boyunca RAM'e yükleniyor; `queues.py` fiilen yok/boş olabilir.
13. `--gpu-devices` yalnız ilk device'ı kullanıyor; `--resume` mesaj basıp full rerun yapıyor.
14. GPU2'de çalışmak için ürün container'larının manuel durdurulması resource scheduling değildir.
15. Package model profile machine-specific absolute `/home/user/...` paths ve TensorRT 10.3 engine fingerprint taşıyor; backend/video runtime TensorRT 10.16 olabilir. Engine'leri birbirine kopyalama; runtime parity ölç.
16. Bulk docs/testlerden bazıları hâlâ `PersonRecord`, `person_id`, redirect ve WebP anlatıyor; current types ile uyumsuz.

---

## 5. PHASE 2 WORKER'DAKİ FALSE-E2E RİSKLERİ

Source'tan yeniden doğrula:

1. Source video MinIO'dan `get()` ile bütünü RAM'e alınıp diske yazılıyor olabilir; büyük video için bounded streaming yok.
2. Native subprocess boyunca heartbeat renew loop yok olabilir.
3. Cancel yalnız native worker başlamadan önce kontrol ediliyor olabilir; long-running native process'e bounded terminate/kill akışı yok.
4. Lease token/fencing token `VideoProcessingService` finalization/persistence katmanına taşınmıyor olabilir. Stale worker lease kaybettikten sonra side effect veya completed yazabilir.
5. Observation frames tek büyük Python listesine yükleniyor olabilir.
6. Sampling native full inference'tan sonra uygulanıp frame index yeniden numaralanıyor olabilir; original frame index/PTS association bozulmamalı.
7. Native observation/template artifacts yalnız tempte sıkıştırılıp MinIO'ya publish edilmeden temp dir ile siliniyor olabilir.
8. `VideoProcessingService` process completion ve job completion'ı iki ayrı UoW ile inconsistent finalize edebilir.
9. Result manifest `math.nan` yazıyor olabilir; strict JSON'da NaN yasak.
10. Playback Range endpoint'i tüm object'i RAM'e alıp slice ediyor olabilir; gerçek bounded range streaming değil.
11. Existing integration test fake embedding/fake crop/placeholder provider ile direct service çağrısı yapıyorsa buna E2E deme.
12. `make phase2-video-e2e-acceptance` worker container/process başlatmıyorsa bu target gerçek E2E değildir.
13. Required face appearance endpoint/contract, retention cleanup ve gerçek progress updates eksik olabilir.

Bu bulgular Phase 2'yi sıfırdan yazma gerekçesi değildir. Mevcut native hot path'i koru; orchestration, lifecycle ve gerçek acceptance çevresini düzelt.

---

## 6. ZORUNLU TOOL / MCP / MEMORY DİSİPLİNİ

Plan Mode'un ilk dakikalarında:

1. `prompt-memory-mcp` veya mevcut memory MCP ile bu repo/ürün için kayıtlı context'i ara.
2. Şu verified facts'i context olarak yükle: current HEAD, c555/2cf anchor'ları, FaceIdentity canonical kararı, Person migration'ın kaldırıldığı, bulk run sonuçları, teardown/failure-count bulguları.
3. Memory sonucu source yerine kesin kanıt sayma. Her iddiayı current files/DB/runtime ile doğrula.
4. Her milestone sonunda yalnız doğrulanmış karar/sonuç/blocker'ı memory'ye yaz. Dataset subject adı, secret, absolute private path, raw log veya spekülasyon yazma.
5. Memory MCP yoksa bunu açıkça `UNAVAILABLE` raporla; kullanmış gibi davranma.

Zorunlu kullanım:

- `codebase-memory-mcp`: current call graph, callers/tests, schema/repository discovery, context recovery.
- GitHub/origin read-only tools: current HEAD ve tarihsel commit file/diff incelemesi.
- `context7`: CUDA, TensorRT, nvImageCodec, CV-CUDA, FastAPI, SQLAlchemy/Alembic, MinIO, Qdrant, React Query, browser Range API gibi version-sensitive kararlar.
- `deepwiki`: MergenVisionDemo ve ilgili upstream architecture; tek başına kaynak sayma, gerçek file/source doğrula.
- `exa`/web: yalnız official/upstream primary docs eksikse.
- `postman`: gerçek API acceptance sırasında.
- `playwright`: gerçek backend + worker + UI acceptance sırasında.
- `21st`: `FORBIDDEN_NOT_USED`.
- Ruflo: `FORBIDDEN_NOT_USED`.

Read-only sibling/reference repos:

- `MergenVisionDemo`: bulk GPU data-plane pattern'i için gerçek source incele.
- Eski MergenVision repos: yalnız lessons learned/parity; production source diye blind copy yok.

Reference adapte edersen exact file, commit/tag, license ve yapılan değişikliği kaydet.

---

## 7. PLAN MODE — ZORUNLU FORENSIC ÇALIŞMA

İlk response'ta plan uydurma. Aşağıdaki 30 inceleme adımını gerçekten yap; sonra recovery planını sun.

### A. Repository ve provenance

1. `pwd`, repo root, `git remote -v`, branch, HEAD ve `git status --short` al.
2. Uncommitted user changes varsa koru; hiçbirini overwrite etme.
3. `git log --oneline --decorate -30` ile contamination zaman çizelgesini çıkar.
4. `git diff --stat/name-status 2cfde196..HEAD` ve `c555c9a4..HEAD` çıkar.
5. Her changed file'ı `KEEP / REPAIR / RESTORE_FROM_2CF / RESTORE_FROM_C555 / REMOVE_GENERATED / NEEDS_DECISION` olarak sınıflandır.
6. `f81f9f4..HEAD` ile surgical revert'in neyi sildiğini ve neyi yarım bıraktığını çıkar.
7. Applied Alembic revision'ı gerçek PostgreSQL'den read-only sorgula; migration dosyalarıyla karşılaştır. Applied migration rewrite etme.
8. PG table/constraint/index, MinIO bucket/key prefix ve Qdrant collection/schema/count inventory'si al; hiçbir şeyi silme.

### B. Requirement reconstruction

9. `requirements/ProjectRequirements.md` ve `requirements/videorequirements.md` tam oku.
10. `AGENTS.md`, `CURRENT_SPRINT.md`, architecture/implementation docs ve current API/OpenAPI'yi oku.
11. Stale docs (`newmission.md`, TARGET_CONTRACT, prompt files, old review packages) ile current source çelişkilerini listele.
12. Plan içinde iki **draft** canonical requirement dosyasının içeriğini tasarla:
    - `requirements/phase1requirements.md`: image identity + enrollment/history/storage + bulk seed + internal UI.
    - `requirements/phase2requirements.md`: Phase 1'i koruyan video upload/job/GPU track/identity/timeline/overlay/cancel/retry.
13. Legacy docs'ta UI yok maddesinin explicit user decision ile superseded olduğunu kaydet.
14. Requirement → source symbol → test → runtime evidence traceability matrix çıkar.
15. Requirement'ı source'tan uydurma. Açık gerçek ürün kararı yoksa `NEEDS_HUMAN_REVIEW` olarak işaretle.

### C. Identity/data model

16. Current Alembic + ORM + entity + repositories üzerinden gerçek ERD çıkar.
17. FaceIdentity/FaceSample/RecognitionResult/ProcessRecord ile video tables relation'larını doğrula.
18. Person/redirect kalıntılarını source, frontend, tests, docs, schemas içinde ara.
19. Snapshot/current projection modelini tasarla; historical row rewrite etmeyen read-time enrichment akışını göster.
20. Bulk deterministic ID → PG identity/sample → MinIO key → Qdrant point → image/video resolver call graph'ını çıkar.

### D. Runtime/data flow

21. Phase 1 image HTTP→GPU→stores→response call graph'ını source üzerinden çıkar.
22. Phase 2 upload→claim→native worker→artifact→tracking→identity→stores→API→UI call graph'ını çıkar.
23. Bulk manifest→batch→GPU→persistence→reconcile call graph'ını çıkar.
24. Her flow'da CPU/GPU boundary, synchronize, full-frame copy ve compact output boundary'lerini işaretle.
25. Native resource ownership/destructor graph'ını çıkar; teardown crash için minimal reproducer tasarla.
26. Worker lease/cancel/fencing/finalization state machine'ini çıkar.
27. UI route → API hook → backend route contract matrix çıkar.

### E. Kanıt sınıflandırması

28. Bütün Makefile targets/test suites'i `unit / contract / integration / native-GPU / real-E2E / fake-or-misnamed` sınıfına ayır.
29. Current HEAD'i değiştirmeden mümkün olan read-only/static testleri çalıştır; environment blocker'larını ayrı yaz.
30. Sonuçları `IMPLEMENTED_AND_PROVEN / IMPLEMENTED_NOT_PROVEN / PARTIAL / BROKEN / MISSING` truth table'ında özetle.

---

## 8. PLAN MODE ÇIKTISI — BU FORMATTA SUN VE DUR

Kod yazmadan önce tek, cohesive recovery planı sun. En az şu bölümler olsun:

1. **Verdict**: current product için PASS/PARTIAL/BROKEN ve gerekçe.
2. **Verified current truth**: çalışan image, video, bulk, UI parçaları.
3. **Regression map**: path/symbol bazında kesin kırıklar.
4. **Requirement matrix**: Phase 1 ve Phase 2 ayrı.
5. **Canonical architecture**: global faceId, trackId, sampleId, stores.
6. **Current ERD vs desired ERD**: yeni schema gerekiyor mu? Default cevap Person yok.
7. **API contract matrix**: backend route, frontend hook, status.
8. **State machines**: identity, sample, process, video job, lease/cancel/retry.
9. **Diff classification**: KEEP/REPAIR/RESTORE/REMOVE.
10. **Migration safety plan**.
11. **Data remediation plan**: mevcut 12.578 sample ve empty-known audit; default dry-run/read-only.
12. **Bulk runtime correction plan**.
13. **Video orchestration correction plan**.
14. **Frontend recovery plan**.
15. **Real acceptance matrix** ve exact commands.
16. **Risks/blockers**.
17. **Changed-file forecast**.
18. **Milestone order, rollback/stop conditions**.
19. **MCP/skill accountability**.
20. **Human review questions**: yalnız gerçekten architecture değiştiren en fazla 3 soru.

Bu planı sunduktan sonra **STOP**. Kullanıcı `BUILD MODE'A GEÇ` demeden edit yapma.

---

## 9. BUILD MODE — ONAYDAN SONRA UYGULAMA SIRASI

Plan onaylanınca aşağıdaki milestone'ları sırayla uygula. Her behavior için önce failing test/reproducer, sonra minimum implementation, sonra targeted test, sonra broader gate.

### Milestone 0 — Baseline freeze ve güvenli çalışma alanı

1. Current source ve live store inventory snapshot'ını PII'siz local artifact olarak kaydet.
2. c555/2cf native/image/video parity baseline tests ekle veya mevcutları çalıştır.
3. Product native hot path'i değiştirmeden önce output counts/hash/parity baseline al.
4. Applied migration hash'lerini kaydet; applied migration rewrite guard ekle.
5. Existing live data için destructive command guard koy.

### Milestone 1 — Canonical requirements ve doküman gerçeği

6. Human-approved content ile `phase1requirements.md` ve `phase2requirements.md` oluştur/restore et.
7. Legacy ProjectRequirements/videorequirements provenance'ını koru; superseded clauses'i açıkça işaretle.
8. `CURRENT_SPRINT.md` ledger/status çelişkisini gerçek rerun evidence ile düzelt.
9. `AGENTS.md` tekrar/bozuk/stale bölümlerini 2cf baseline + current approved decisions ile cerrahi temizle.
10. `newmission.md`, TARGET_CONTRACT ve prompt docs'i source-of-truth olmaktan çıkar; gerekirse `superseded` banner/archival classification yap.

### Milestone 2 — FaceIdentity merkezli frontend/API recovery

11. `PeoplePage.tsx`'i `/api/v1/faces` + face samples contract'ına bağla.
12. Phantom `api/people.ts`, Person types, `person_id` fields/query keys ve kırık `/people/:personId` linklerini kaldır.
13. `/people` UI route'u “identity directory” presentation olarak kalabilir; cards `/faces/{faceId}` açmalı.
14. Metadata-only “kişi yarat” davranışını kaldır. Identity image evidence olmadan yaratılmamalı.
15. Yeni bilinen kişi UI flow'u: image recognize → new anonymous faceId → aynı faceId enroll. Mevcut anonymous da aynı formdan enroll edilir.
16. Video track/person paneline anonymous için `Adlandır/Enroll` CTA ve known için face detail linki ekle.
17. Enroll success'te same faceId assertion yap.

### Milestone 3 — Historical snapshot + current identity projection

18. Video API response'unda processing snapshot ile current identity projection'ı ayır.
19. Current fields'i `video_track.face_id -> face_identity` üzerinden batch query ile resolve et; N+1 yapma.
20. Stored overlay artifact'te mutable display name/status source-of-truth yapma. Response read-time enrichment veya technical-only artifact kullan.
21. Enrollment historical snapshot'ı değiştirmesin; yalnız current projection değişsin.
22. Frontend enroll sonrası face, video people, appearances ve overlay/timeline query'lerini invalidate/refetch et.
23. Test: video anonymous → enroll → sidebar/overlay current known/name; statusAtProcessing hâlâ anonymous/new_anonymous; faceId aynı.

### Milestone 4 — Bulk native correctness ve clean shutdown

24. Repeated `create→warmup→extract→close` real GPU reproducer yaz; 50–100 cycle temiz exit ve memory baseline return ölç.
25. Her native object için explicit ownership/idempotent close ekle. Correct CUDA device set, outstanding work synchronization ve dependent resources before stream/context destruction sırasını official APIs/source ile doğrula.
26. `_jpeg_encoder` dahil bütün resource'ları deterministic kapat; `__del__` ikinci close'u harmless yap.
27. Detector score D2H synchronization race'ini düzelt.
28. Decode exceptions'i `per-entry invalid media` ve `systemic GPU/runtime` olarak ayır. Systemic error'da batch/worker fail-closed; silent per-image fallback yok.
29. nvImageCodec/nvJPEG encoder backend'ini explicit allowlist ve runtime evidence ile fail-closed yap. CPU encoder fallback yok.
30. GPU crop JPEG encode'u list/batch API destekliyorsa kullan; yoksa encode stage'i bounded/overlapped tasarla ve limitation'ı dürüst yaz.
31. Crop output için JPEG magic, 112×112 decode validation, SHA ve content type testi ekle.

### Milestone 5 — Bulk lifecycle doğruluğu

32. Rejected double-count bug'ını sampleId bazında tek source-of-truth ile düzelt.
33. Accounting invariant:

```text
discovered = accepted + rejected + failed
persisted <= accepted
distinct sample_id counts only
```

34. Accepted sample yoksa known face identity yaratma.
35. Rejected sample'ın PG lifecycle semantiğini açık seç: ya hiç sample row yok ve run journal rejection owner, ya valid failed row önce insert edilir. Hayali update yapma.
36. `ON CONFLICT DO NOTHING` sonrası existing row ownership/model/source/state doğrula.
37. pending→active/failed transitions state predicate + rowcount/fencing ile çalışsın.
38. Compensation yalnız current attempt'in yarattığı object/vector'u silsin.
39. Reconcile'ı existence ötesine genişlet: PG state, face ownership, MinIO key/content-type/SHA/JPEG dimensions, Qdrant point/payload/vector dimension/finite/L2/model version.
40. Existing LFW run için read-only `audit --dry-run` üret: empty-known identities, duplicate/error counts, missing objects/vectors. Kullanıcı onayı olmadan delete/repair yapma.

### Milestone 6 — Gerçek streaming cross-subject batching

41. Tüm manifest/image bytes'ı RAM'e yükleme. Streaming manifest reader ekle.
42. Photo work item association'ı koru:

```text
source_namespace
external_subject_key_hash
face_id
sample_id
image_path/reader token
ordinal
```

43. Farklı subject'lerden photo-level microbatch oluştur; GPU `extract_batch` gerçek batch N görsün.
44. Aynı subject'in bütün samples'ı doğru faceId'ye geri bağlansın.
45. Bounded reader→GPU→persistence queues ve backpressure ekle.
46. Compute ve batched MinIO/Qdrant/PG persistence overlap et; unbounded task yaratma.
47. Batch size 256'yı default sanma. 1/8/16/32/64/128/256 matrix'i free-memory/headroom ile ölç; safe sweet spot seç.
48. GPU OOM olduğunda silent data loss yok; bounded smaller-batch retry yalnız policy ile, systemic error journal'a yazılsın.

### Milestone 7 — Multi-GPU, idempotency, resume

49. Her worker process tam bir physical GPU'ya sahip olsun; host GPU UUID ve visible container index'i ayrı kaydet.
50. Sharding `stable_hash(face/subject id) mod worker_count`; Python `hash()` ve photoId sharding yok.
51. `--gpu-devices 0,1,2` gerçekten üç process/worker kullansın.
52. Başka product container'larını manuel stop etmek workflow olmasın. GPU admission/lease/free-memory headroom ve explicit resource scheduling yap.
53. Run journal/checkpoint processing boyunca durable yazılsın; yalnız finalde değil.
54. `--resume` gerçek resume etsin; completed samples skip, pending/failed policy ile reconcile/retry.
55. Rerun/resume duplicate face/sample/vector/object oluşturmasın.
56. Mid-run crash, MinIO failure, Qdrant failure, PG activation failure, worker death testleri ekle.

### Milestone 8 — Bulk ↔ image ↔ video embedding continuity

57. Aynı input/crop için bulk pipeline ve Phase 2 image runtime embedding parity ölç: dimension, finite, L2, cosine.
58. Bulk ve native video track-template parity için controlled fixture ölç.
59. TRT 10.3 bulk ile TRT 10.16 product engine'lerini birbirine kopyalama. Aynı ONNX/preprocess/alignment contract + measured output parity şart.
60. Model profile absolute machine path'lerini kaldır; env/repo-relative + runtime fingerprint kullan.
61. Enrollment dışı held-out LFW image'i `/api/v1/faces/recognize` ile gönder:
    - `known`
    - exact seeded `faceId`
62. Aynı kimliğin bulunduğu gerçek video job'ı çalıştır:
    - result `known`
    - exact same seeded `faceId`
63. Threshold düşürerek zorla geçirme. Önce preprocess/alignment/embedding distributions; sonra labeled calibration sonucu versioned config.

### Milestone 9 — Video worker orchestration correctness

64. MinIO source download bounded streaming/file writer olsun; full bytes RAM'e alma.
65. Native subprocess stdout/stderr bounded log handling; entire output RAM'e alma.
66. Native execution boyunca heartbeat renew, lease token ve fencing version taşı.
67. Cancel polling + SIGTERM grace + bounded SIGKILL + temp cleanup uygula.
68. Lease lost/stale worker identity/sample/track/manifest/completed side effect yazamasın.
69. Finalization öncesi cancel/lease/fencing recheck yap.
70. Original frame index, presentation index ve PTS/timebase korunmalı; sampling renumber etmemeli.
71. Sampling policy inference öncesi uygulanacaksa native pipeline contract ile yap; sonradan list filter'ı performance claim'i olmasın.
72. Observation/template/crop bundle'ını validate et, zstd compress et, canonical MinIO keys'e checksum/stat ile publish et.
73. Python dense observations bounded chunk/stream ile tüketilsin; tüm video listesi zorunlu olmasın.
74. Job progress decoded/processed/currentPTS/detections/stage olarak heartbeat ile güncellensin.
75. Strict JSON kullan; NaN/Infinity serialize etme.
76. Process/job/result manifest finalization consistency'sini transaction/fencing ile güçlendir.
77. Retention cleanup worker/command ve idempotent object cleanup ekle.
78. Playback gerçek Range/bounded streaming kullansın; storage key/credential public response'a sızmasın.

### Milestone 10 — Real UI/API acceptance

79. Existing video overlay player'ı koru; `requestVideoFrameCallback`, DPR, rotation, contain offsets, resize/fullscreen/seek testlerini çalıştır.
80. `/people` identity directory bulk-seeded known identities ve samples'ı göstersin.
81. Video sidebar/overlay known/anonymous filters ve enroll CTA çalışsın.
82. No-face image/video success empty state'i göster.
83. Upload/progress/completed/failed/cancelled/retry/loading/error states gerçek backend ile çalışsın.
84. Browser console fatal error ve 404 network contract drift olmasın.

### Milestone 11 — Repository hygiene ve güvenlik

85. Tracked LFW dataset/manifests/raw logs/baselines/screenshots/temp artifacts listesini çıkar.
86. Current tree'den generated artifacts'i normal commit diff ile untrack/ignore et; local user dataset'ini silme.
87. Geçmiş commit'ten veri kaldırmak history rewrite gerektiriyorsa kendin yapma; user'a ayrı security action olarak bildir.
88. Dataset subject adı/folder/path public log, object key, Qdrant payload veya report'a girmesin; HMAC/technical IDs kullan.
89. Dev compose credentials/ports'u production security kanıtı sayma. Local-dev ve hardened deployment profillerini ayır.
90. Mutating/admin bulk surface için auth/authorization threat model çıkar; scope onayına göre uygula.

### Milestone 12 — İzole GPU video lab (ürün continuity green olduktan sonra)

91. Production code'dan bağımsız `research/gpu_video_lab/**` tasarla.
92. Capture mode pinned DeepStream container ile real observation/template artifacts üretsin.
93. Replay/sweep mode capture edilmiş artifacts üzerinde CPU'da hızlı tracker/quality/template/reconciliation sweep yapsın.
94. Labeled küçük fixture seti ve metrikler:
    - fragmentation,
    - ID switches,
    - false merges,
    - cannot-link violations,
    - known match/false match,
    - track coverage,
    - best-shot quality.
95. Blur/brightness/face-size/border/pose/alignment residual/temporal diversity/outlier rejection parametrelerini sweep et.
96. Lab config'i production'a otomatik yazmasın. Yalnız versioned candidate report üretsin; measured improvement + parity + review sonrası promote edilsin.

---

## 10. GERÇEK ACCEPTANCE GATES

Existing target isimlerini source'tan doğrula. Fake test target'ını gerçek E2E diye bırakma; gerekiyorsa `contract` olarak yeniden adlandır ve yeni full target ekle.

Minimum gates:

```text
make recovery-static
make recovery-migrations
make phase1-image-regression
make phase1-bulk-native-clean-exit
make phase1-bulk-lifecycle
make phase1-bulk-cross-subject-batching
make phase1-bulk-resume-idempotency
make phase1-bulk-image-continuity
make phase2-native-regression
make phase2-worker-fencing-cancel
make phase2-video-known-continuity
make ui-contract
make ui-real-e2e
make mergenvision-full-e2e-acceptance
```

Final full chain aynı çalıştırmada şunları göstermeli:

1. Fresh checkout/config ile services ayağa kalkar.
2. No-face JPEG → completed, `faceCount=0`.
3. Single JPEG → `new_anonymous`, faceId A.
4. Aynı JPEG → `anonymous`, faceId A.
5. UI/API enroll → `known`, faceId A.
6. Multi-face JPEG → tek processId, bağımsız sonuçlar.
7. Bulk small deterministic dataset → clean exit 0, no segfault, GPU memory baseline return.
8. Bulk rerun/resume → duplicate counts yok.
9. Held-out image → `known`, exact seeded faceId B.
10. Real video upload → real async job → real native GPU worker.
11. Video result → seeded identity `known`, exact faceId B.
12. Anonymous video track UI'dan enroll → current known/name görünür, track snapshot immutable, faceId değişmez.
13. Original video + canvas overlay seek/resize/play works.
14. Cancel native process'i bounded durdurur; no canonical partial publish.
15. Retry/restart same job/data contract'ta duplicate side effect üretmez.
16. Stale lease holder completion yazamaz.
17. PG/MinIO/Qdrant cross-store semantic reconcile green.
18. Restart sonrası identities/jobs/results/samples korunur.

Her GPU/benchmark sonucunda üç metriği ayır:

- GPU compute-only throughput,
- end-to-end bulk storage throughput,
- full video worker/product latency/throughput.

Birini diğerinin performansı diye sunma.

---

## 11. HARD STOPS

Şunları yapma:

- `git reset --hard`, wholesale revert, broad checkout veya blind cherry-pick.
- Applied migration rewrite.
- Person/redirect/0006 migration'ını geri getirme.
- Existing PG/MinIO/Qdrant data/collection/bucket silme/reset.
- Phase 2 native hot path'i parity baseline olmadan değiştirme.
- Bulk için Phase 2 engine/source'u kopyalayıp bozma.
- Model family/change/download.
- CPU inference/decode/encode fallback.
- PIL, OpenCV veya FFmpeg crop provider.
- Annotated MP4/NVENC.
- `trackId = faceId`.
- Her raw track için ayrı identity yaratma.
- Fake crop/fake embedding testini E2E sayma.
- Native test skip'ini PASS sayma.
- `12578/12578 existence` sonucunu recognition continuity sayma.
- Threshold'u yalnız testi geçirmek için düşürme.
- Ground truth olmadan accuracy verified deme.
- Runtime artifact/dataset/log/engine/model/secret commit etme.
- User onayı olmadan history rewrite, commit veya push.
- 21st/Ruflo kullanma.

---

## 12. COMPLETION RAPORU

Her milestone sonunda kısa checkpoint; finalde tek denetlenebilir rapor ver:

1. `PASS / PARTIAL / BLOCKED / NOT_TESTED`.
2. Kullanıcı açısından gerçekten çalışan Phase 1 akışı.
3. Kullanıcı açısından gerçekten çalışan Phase 2 akışı.
4. Bulk→image→video exact same faceId evidence.
5. Current-vs-snapshot enrollment evidence.
6. Native runtime/container/GPU/CUDA/TRT/model/engine fingerprints.
7. Clean shutdown/memory evidence.
8. Cross-subject batch ve multi-GPU evidence.
9. PG rows/counts, MinIO keys/SHA/media type, Qdrant points/payload/norm.
10. Worker heartbeat/cancel/retry/fencing/restart evidence.
11. API/OpenAPI examples.
12. Playwright screenshots/results ve fatal console/network errors.
13. Exact commands + raw pass/fail/skip summaries.
14. Changed-file map.
15. Security/privacy review.
16. Known limitations ve tek sonraki öneri.
17. MCP/skill accountability; çağrılmayanı `used` gösterme.

Final review package:

```text
docs/implementation/review_packages/
MERGENVISION-BIG-PICTURE-RECOVERY-CODE-REVIEW-PACKAGE.md
```

Kanıt olmadan `production-ready`, `fully optimized`, `GPU-only E2E`, `accuracy verified`, `3-GPU scaled` veya `complete` deme.

---

## ŞİMDİ YAPACAĞIN İLK ŞEY

Plan Mode'da kal. Hiçbir dosyayı değiştirme. Önce memory/context'i yükle, origin/current HEAD'i ve historical anchors'ı doğrula, 30 maddelik forensic incelemeyi tamamla ve 20 bölümlü recovery planını sun. Planın sonunda açıkça:

```text
PLAN_READY_FOR_HUMAN_REVIEW
NO_SOURCE_FILES_CHANGED
```

yaz ve dur.


MERGENVISION BÜYÜK RESİM KURTARMA PLANI (PLAN MODU)
1. Verdict
Current product verdict: BROKEN with recoverable core.
Gerekçe:
- face_identity/faceId merkezli identity motoru, Phase 1 image API/backend/native GPU ve Phase 2 native video worker/tracker/reconciliation çekirdekleri halen mevcut ve çalışır durumda (gerçek Friends smoke: 6.665 frame, 9.020 detection/embedding, 150 raw track; LFW bulk 12.578 sample PG/MinIO/Qdrant existence reconcile geçti).
- Fakat ürün katmanları (API contract, UI, lifecycle accounting, worker orchestration, doküman/sprint doğruluğu, repo hijyeni) drift altında ve bütünsel acceptance geçmiyor.
- En kritik P0 kırıklar: UI /people → /api/v1/people mevcut değil, video snapshot/current projection ayrımı tutarsız, bulk lifecycle accounting hatalı, native teardown segfault, worker E2E/fencing/cancel gerçek değil, commit edilmiş dataset/log sızıntısı.
2. Verified Current Truth
Bileşen	Durum	Kanıt
Phase 1 image recognize/enroll/history/delete native GPU	Var, çalışıyor	backend/app/api/routes/faces.py, NativeImageRecognitionAdapter, c555/2cf sonrası unit/integration testleri
Phase 2 video upload/async job	Var	videos.py, video_upload_service.py
Phase 2 native GPU observation worker	Var, Friends smoke geçmiş	ledger: 6.665 frame/9.020 embedding/385.53 FPS
Phase 2 Python tracking/reconciliation	Var	video_tracking_service.py, video_track_persistence_service.py
Bulk GPU data-plane	Var; LFW 12.578 sample PG/MinIO/Qdrant existence eşleşiyor	phase1/gpu_bulk_enrollment/ run journal 14af8180...
React UI (video overlay player)	Var ama contract driftli	VideoOverlayPlayer var; /people kırık
PostgreSQL schema	Person yok; face_identity/sample/video modeli var	alembic_version = cf0441294c5f, table listesi
Qdrant collection	face_samples_retinaface_r50_glintr100_v1 (plan onaylı)	settings + bulk run
3. Regression Map
Path/Symbol	Sorun	Etki
frontend/src/api/people.ts:23	GET /api/v1/people çağrısı	404, PeoplePage boş/hata
frontend/src/pages/PeoplePage.tsx	person_id, create/update/delete Person hookleri	UI yeni kişi yaratamaz/enroll edemez
frontend/src/pages/PeoplePage.tsx:140+	/people/:personId route/link yok	kartlardan detay açılmaz
backend/app/api/main.py:333-335	yalnız faces/processes/videos mount, people route yok	API contract drift
backend/app/api/routes/videos.py + result schema	statusAtProcessing/nameAtProcessing + current projection karışmış olabilir	enroll sonrası UI eski isim gösterir
backend/app/infrastructure/persistence/sqlalchemy/models/video_track.py	snapshot/current ayrımı response katmanında değil	güncel isim/sabit faceId çelişkisi
phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/cli.py	rejected double-count, subject-subject batching	1.310 failed, gerçek batch-256 değil
phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/pipeline.py	close() deterministik değil, __del__ idempotent değil	process exit segfault
backend/app/worker/video_worker_main.py	heartbeat/fencing/cancel/artifact/streaming riskleri	“full E2E” değil
4. Requirement Matrix
Phase 1 (image) kanonik requirement domains:
- Persist face sample + embedding; immutable recognition result snapshot.
- Recognize → new_anonymous/anonymous/known sonuçları.
- Enroll/promote aynı faceId ile anonymous → known; history rewrite yok.
- Detail/history/delete; sample lifecycle active/inactive.
- Bulk dataset seed; same faceId held-out recognize.
- Internal UI identity directory + enroll flow.
Phase 2 (video) additive domains:
- Upload → canonical MinIO source object.
- Async job claim/lease/cancel/retry with fencing token.
- Native NVDEC/NVMM pipeline, compact observations, protobuf/zstd artifact.
- Python tracker + reconciliation; trackId local, faceId global.
- Person summary/appearances/timeline; overlay metadata (orijinal video + canvas, annotated MP4 değil).
- UI: upload progress, video player overlay, known/anonymous filter, enroll CTA.
Kaynak: requirements/ProjectRequirements.md, requirements/videorequirements.md; legacy newmission.md/TARGET_CONTRACT superseded. UI “mandatory” kararı, eski “UI olmayacak” cümlesini geçersiz kılıyor.
5. Canonical Architecture
┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐
│   Image     │   │   Bulk GPU   │   │     Video job       │
│   API       │   │  data-plane  │   │  + native worker    │
└──────┬──────┘   └──────┬───────┘   └──────────┬──────────┘
       │                  │                      │
       └──────────────────┼──────────────────────┘
                          ▼
               ┌────────────────────┐
               │  face_identity     │  (PG, canonical)
               │  face_sample       │
               │  recognition_result│  (immutable snapshot)
               │  video_track...    │  (snapshot + face_id FK)
               └─────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │      MinIO          │  faces/{faceId}/{sampleId}/aligned.{jpg|webp}
              │                     │  videos/{videoId}/source/original
              │                     │  videos/{videoId}/jobs/{jobId}/{observations,timeline,result}
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │  Qdrant             │  collection face_samples_retinaface_r50_glintr100_v1
              │  pointId=sampleId   │  payload: sample_id, face_id, active, model_version
              └─────────────────────┘
- faceId: tek global identity key.
- trackId: işlem anlık video local grouping.
- sampleId: embedding/crop evidence.
- Immuatable processing snapshot (statusAtProcessing, nameAtProcessing) ile read-time current projection (currentStatus, currentName, currentMetadata) ayrı.
6. Current ERD vs Desired ERD
Current PG tabloları: face_identity, face_sample, recognition_result, idempotency_record, process_record, process_event, outbox_event, video_asset, video_job, video_track, video_tracklet, video_track_sample, video_timeline_chunk, appearance_interval.
Person tablosu yok. Bu doğru.
Gereken değişiklikler:
- Schema migration gerekmez.
- video_track zaten face_id FK taşıyor. Current projection read-time batch query ile video_track.face_id → face_identity üzerinden eklenecek.
- face_sample zaten active, model_version, object_key, media_type taşıyor. Bulk .jpg ve online .webp coexistence’i destekler.
- Eğer video_track tablosunda status_at_processing/name_at_processing yoksa (şu an response schema seviyesinde olabilir), snapshot sütunlarını eklemek gerekir; plan onayı sonrası migration yazılır.
7. API Contract Matrix
UI route	Hook	Backend route	Status
/people	usePeople	GET /api/v1/people	MISSING
/people/:personId	usePerson	GET /api/v1/people/:id	MISSING
/people modal	useCreatePersonMutation	POST /api/v1/people	MISSING
/people batch	useCreatePeopleBatchMutation	POST /api/v1/people/batch	MISSING
/faces	yeni/refactor	GET /api/v1/faces + samples	var ama UI bağlanmamış
/faces/:faceId/enroll	mevcut EnrollPage	PATCH /api/v1/faces/{faceId}	mevcut, review edilecek
Video player overlay	video timeline/result hooks	GET /api/v1/videos/{videoId}/jobs/{jobId}/...	var, current projection eksik
Plan: phantom /api/v1/people kapatılacak; UI /api/v1/faces identity directory olarak bağlanacak.
8. State Machines
Identity lifecycle
anonymous ──enroll──> known
known ──update──> known (version++, name/meta değişir)
known ──delete──> inactive (deleted_at set, Qdrant passive)
Recognition result snapshot (immutable)
new_anonymous / anonymous / known  → process-specific, never rewritten
Sample lifecycle (PG)
pending_blob -> blob_ready -> indexed -> active
inactive (soft delete / update-delete)
Video job
pending -> processing -> cancelling/completed/failed/cancelled
lease: claim with lease_owner + lease_expires + heartbeat + fencing version
Worker cancel
cancel_requested -> native SIGTERM (grace) -> SIGKILL (bounded) -> cleanup -> cancelled
9. Diff Classification (2cfde196..HEAD)
Kategori	Örnekler
KEEP	Phase 2 video migrations, native video worker, tracker, video_track/timeline models, React overlay player, Phase 1 GPU native image runtime (c555/2cf temeli)
REPAIR	PeoplePage.tsx, api/people.ts, queryKeys.ts, types.ts, video result/current projection, bulk CLI lifecycle accounting, pipeline.py teardown, CURRENT_SPRINT.md, AGENTS.md
RESTORE_FROM_2CF	Frontend face-identity directory semantics (2cf öncesi /faces list/enroll bağlantısı)
REMOVE_GENERATED	.artifacts/phase1_gpu_bulk_enrollment/lfw_*, logs, manifests, baselines; backend/scripts/lfw_benchmark_progress.log; geçici run artifactları
NEEDS_DECISION	Aynı container’da TRT 10.3 bulk engine vs TRT 10.16 product engine parity; GPU scheduling/admission model; auth/authorization model
10. Migration Safety Plan
- Applied migration cf0441294c5f korunur; rewrite edilmez.
- Eğer snapshot sütunları (status_at_processing, name_at_processing) video_tracke eklenirse yeni migration yazılır; live veriye DEFAULT ile non-destructive.
- Çalışma öncesinde alembic current çıktısı ve PG alembic_version snapshot alınır.
- Hiçbir migration downgrade yapılmadan önce live store yedeği / PII'siz inventory kaydedilir.
- Person/redirect migration’ı geri getirilmez.
11. Data Remediation Plan
Mevcut 12.578 LFW sample için:
- Varsayılan dry-run, read-only.
- audit --dry-run çıktısı:
- boş known identity'ler,
- duplicate sample/object/vector,
- missing MinIO object veya Qdrant point,
- media type / SHA / JPEG dimension uyuşmazlıkları.
- Live veride repair (delete/update) kullanıcı onayı ve yedekleme sonrası.
- Subject adı, absolute path, raw log asla yeni commit/report’a girmez.
12. Bulk Runtime Correction Plan
Dosyalar:
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/cli.py
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/pipeline.py
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/persistence.py
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/queues.py
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/manifest.py
Yapılacaklar:
1. pipeline.py: deterministik idempotent close(); _jpeg_encoder/cvcuda/TRT/CUDA sıralı teardown; detector score D2H synchronize; device set + stream sync.
2. Decode exception: sadece belirli media error’ları per-image rejection; CUDA/OOM/systemic → batch fail-closed.
3. JPEG encoder backend explicit allowlist; CPU fallback yok, kanıt.
4. Crop encode batch/list API araştır; yoksa bounded overlap.
5. cli.py: rejected double-count düzelt; accepted/rejected/failed hesaplaması sampleId tek SOT.
6. manifest.py + queues.py: streaming reader; ram’e tüm bytes yüklenmez.
7. Cross-subject photo-level microbatch; extract_batch gerçek batch N görsün.
8. Multi-GPU process per device; stable hash sharding; resume/checkpoint; GPU admission/headroom.
9. Accounting invariant testleri.
13. Video Orchestration Correction Plan
Dosyalar:
- backend/app/worker/video_worker_main.py
- backend/app/application/services/video_processing_service.py
- backend/app/infrastructure/storage/minio_adapter.py
- backend/app/api/routes/videos.py
Yapılacaklar:
 1. MinIO source download: bounded streaming writer, RAM'e almama.
 2. Native subprocess stdout/stderr bounded log handling.
 3. Native execution süresince heartbeat renew; lease + fencing token taşı.
 4. Cancel: polling + SIGTERM grace + bounded SIGKILL + temp cleanup.
 5. Finalization öncesi cancel/lease/fencing recheck.
 6. Observation/artifact zstd + validate + publish MinIO’ya checksum/stat ile.
 7. Python observations stream/chunk tüketsin.
 8. Strict JSON; math.nan engelle.
 9. Playback Range: gerçek bounded range stream.
10. Progress updates: decoded/processed/currentPTS/detections/stage.
14. Frontend Recovery Plan
Dosyalar:
- frontend/src/pages/PeoplePage.tsx
- frontend/src/api/people.ts
- frontend/src/api/queryKeys.ts
- frontend/src/api/types.ts
- frontend/src/pages/FaceDetailPage.tsx
- frontend/src/pages/EnrollPage.tsx
- frontend/src/App.tsx (routing)
Yapılacaklar:
1. Phantom api/people.ts sil; yerine api/faces.ts identity directory hookleri (useFaces, useFace, useEnrollFaceMutation).
2. PeoplePage “Kişiler” olarak kalabilir ama /api/v1/faces listesini gösterir; kartlar /faces/{faceId} açar.
3. Metadata-only “Yeni Kişi Oluştur” kaldır; yeni known flow: recognize image → anonymous faceId → aynı faceId enroll.
4. EnrollPage/FaceDetailPage success’ta same faceId assertion.
5. Video sidebar’a anonymous track için Adlandır/Enroll CTA, known için face detail linki.
6. Enroll sonrası React Query cache invalidation: faces, video people, appearances, timeline, overlay.
7. UI route /people → /people presentation; route /people/:faceId kaldır, /faces/:faceId ekle.
15. Real Acceptance Matrix ve Exact Commands
Hedef gate listesi:
make recovery-static
make recovery-migrations
make phase1-image-regression
make phase1-bulk-native-clean-exit
make phase1-bulk-lifecycle
make phase1-bulk-cross-subject-batching
make phase1-bulk-resume-idempotency
make phase1-bulk-image-continuity
make phase2-native-regression
make phase2-worker-fencing-cancel
make phase2-video-known-continuity
make ui-contract
make ui-real-e2e
make mergenvision-full-e2e-acceptance
Full-chain acceptance senaryosu:
 1. No-face JPEG → /faces/recognize → faceCount=0.
 2. Single JPEG → new_anonymous, faceId=A.
 3. Same JPEG → anonymous, faceId=A.
 4. UI/API enroll → known, faceId=A.
 5. Multi-face JPEG → single processId, independent results.
 6. Bulk small dataset → exit 0, no segfault, memory baseline return.
 7. Bulk rerun/resume → no duplicates.
 8. Held-out LFW image → known, exact seeded faceId.
 9. Real video upload → async job → native GPU worker.
10. Video result → known, exact same seeded faceId.
11. Anonymous video track enroll from UI → current known/name visible; snapshot immutable; faceId unchanged.
12. Original video + canvas overlay seek/resize/play works.
13. Cancel → native process stops cleanly; no partial canonical publish.
14. Retry no duplicate side effects; stale lease holder cannot finalize.
15. Cross-store semantic reconcile green.
16. After restart: identities/jobs/results/samples preserved.
16. Risks/Blockers
Risk	Önlem
Bulk teardown segfault verileri etkilememesi ama worker stability riski	deterministik close + reproducer (M4)
TRT 10.3 bulk engine vs product engine mismatch	parity test, engine kopyalama yasak
GPU memory contention	admission/headroom + per-device worker, manual container stop yok
Frontend refactor regression	UI contract test + Playwright
Existing LFW log/manifest subject sızıntısı	untrack/ignore + güvenlik action bildirimi
Worker fencing/cancel karmaşıklığı	önce contract test, sonra native GPU test
17. Changed-File Forecast
Beklenen değişiklikler (build mode’da):
- frontend/src/api/people.ts → delete
- frontend/src/api/faces.ts → new
- frontend/src/api/queryKeys.ts, types.ts, client.ts → modify
- frontend/src/pages/PeoplePage.tsx, FaceDetailPage.tsx, EnrollPage.tsx → modify
- frontend/src/App.tsx → modify
- backend/app/api/routes/videos.py, backend/app/api/schemas.py → current projection
- backend/app/application/services/video_track_persistence_service.py → batch current projection
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/pipeline.py → teardown/idempotency
- phase1/gpu_bulk_enrollment/python/mv_phase1_bulk/cli.py, persistence.py, manifest.py, queues.py → lifecycle/batching/resume
- CURRENT_SPRINT.md, AGENTS.md, README.md, newmission.md → update/supersede
- .gitignore → artifact ignore
- Makefile → yeni acceptance targetleri
18. Milestone Order, Rollback/Stop Conditions
Build Mode milestone sırası:
0. Baseline freeze + store inventory snapshot + safety guard.
 1. Canonical requirements + doküman gerçeği.
 2. FaceIdentity merkezli frontend/API recovery.
 3. Historical snapshot + current identity projection.
 4. Bulk native correctness + clean shutdown.
 5. Bulk lifecycle doğruluğu.
 6. Gerçek streaming cross-subject batching.
 7. Multi-GPU, idempotency, resume.
 8. Bulk↔image↔video embedding continuity.
 9. Video worker orchestration correctness.
10. Real UI/API acceptance.
11. Repository hygiene + güvenlik.
12. İzole GPU video lab (ürün green olduktan sonra).
Hard stop / rollback koşulları:
- Herhangi bir milestone’un failing test/reproducer’ı giderilemezse dur.
- Migration rewrite, live data delete, model download, engine kopyalama talebi → user onayı.
- Threshold düşürerek test geçirmek → yasak.
- Native test skip → PASS sayma.
19. MCP/Skill Accountability
Bu plan modunda kullanılanlar:
- writing-plans: plan yapısı ve task disiplini.
- codebase-memory: PeoplePage, face_identity, video_track symbol keşfi.
- recall / prompt-memory-mcp: önceki MergenVision kararları ve bulgular.
- Git bash read-only: HEAD, status, diff, DB alembic/table inventory.
- read: PeoplePage.tsx, api/people.ts, main.py, CURRENT_SPRINT.md, face_identity.py, MERGENVISION_BIG_PICTURE_RECOVERY_REPORT.md.
Plan modunda çağrılmayanlar:
- context7, deepwiki, postman, playwright, exa — Build Mode acceptance ve library-specific kararlar için rezerve.
Yasak:
- 21st, Ruflo.
20. Human Review Questions
1. Frontend route kararı: /people URL’sini “identity directory” olarak koruyup kartları /faces/{faceId} detaya mı yönlendirelim, yoksa /people/:faceId URL’sini de /faces/:faceId ile eşitleyip tamamen /faces namespace’ine mi dönelim?
2. Bulk dataset remediasyonu: Mevcut 12.578 LFW sample için dry-run audit raporunu üretip kullanıcı onayı sonrası repair mi yapalım, yoksa sadece forward fix + yeni doğrulama runlarıyla mı ilerleyelim?
3. Video snapshot/current projection schema: video_track tablosuna immutable status_at_processing/name_at_processing sütunları eklemek için migration yazalım mı, yoksa response katmanında read-time projection ile mi yeterli?
PLAN_READY_FOR_HUMAN_REVIEW
NO_SOURCE_FILES_CHANGED

Planın yönü doğru; ama değişmeden Build Mode’a verirsen birkaç yeni hata çıkarır. Şu düzeltmeler bağlayıcı olmalı:

Actual enroll endpoint PATCH /faces/{id} değil, POST /api/v1/faces/{faceId}/enroll.
frontend/src/api/faces.ts zaten var; yeniden yaratılmayacak, korunup gerekirse düzeltilecek.
face_sample tablosunda model_version ve media_type kolonları yok. Format object_key uzantısından çözülüyor; hayali kolon eklenmeyecek.
Gerçek sample state machine pending → active|failed → inactive. pending_blob/blob_ready/indexed state’leri icat edilmeyecek.
Snapshot kolonları video track üzerinde zaten mevcut görünüyor. Yeni migration gerekmiyor.
Current identity projection persistence katmanına değil, VideoResultService/query katmanına konulmalı.
Normal test hatasında durmasın; sistematik debug etsin. Yalnız gerçek external blocker’da dursun.
Known identity güncellemesi unutulmuş: enroll’den ayrı olarak name/metadata update contract’ı gerekliyse FaceIdentity üzerinde PATCH /faces/{faceId} olarak uygulanmalı.
Video/image threshold drift’i ölçülmeden continuity testi koşturulmamalı.

Üç sorunun cevapları:

/people UI route’unu koruyun. Bu yalnız kullanıcı dostu “identity directory” sayfası olsun. Backend /people API yaratılmasın; /api/v1/faces kullansın. Kartlar /faces/{faceId} açsın. /people/:id yaratılmasın.
Önce mevcut 12.578 sample için salt-okunur dry-run audit, ardından forward fix ve küçük yeni doğrulama run’ı. Canlı kayıt repair/delete işlemi yalnız audit sonucu ve ayrı kullanıcı onayıyla.
Migration yazmayın. Existing status_at_processing/name_at_processing snapshot’larını koruyun; currentStatus/currentName/currentMetadata read-time batch projection olsun. Source incelemesi kolonların gerçekten olmadığını gösterirse implementation yapmadan migration tasarımını raporlasın.

Aşağıdakini agent’a gönder:

PLAN APPROVED WITH BINDING CORRECTIONS — ENTER BUILD MODE

Planın genel yönü onaylandı. Aşağıdaki düzeltmeler planın üzerindedir ve bağlayıcıdır.

1. IDENTITY / ROUTE KARARI

- Canonical identity yalnız face_identity / faceId.
- Person tablosu, person_id, redirect veya /api/v1/people backend API oluşturma.
- Frontend /people route’u kullanıcı dostu identity directory olarak korunacak.
- /people sayfası GET /api/v1/faces kullanacak.
- Kartlar /faces/{faceId} sayfasına gidecek.
- /people/:personId route’u oluşturulmayacak.
- frontend/src/api/people.ts ve phantom Person types/query keys kaldırılacak.
- frontend/src/api/faces.ts zaten mevcut; yeniden yaratma. Mevcut hookları koru/refactor et.

2. API CONTRACT DÜZELTMESİ

Actual enrollment endpoint:

  POST /api/v1/faces/{faceId}/enroll

Bunu PATCH ile karıştırma.

Anonymous → known enrollment bu endpoint üzerinden ve aynı faceId korunarak yapılır.

Known identity name/metadata update requirement’ı mevcut API’de yoksa enrollment’tan ayrı değerlendir:

  PATCH /api/v1/faces/{faceId}

Bu endpoint yalnız mevcut known identity’nin name/metadata’sını optimistic version/lifecycle kurallarıyla günceller. Person aggregate yaratmaz. Önce failing contract test yaz.

3. DATABASE GERÇEĞİ

Current FaceSample schema’yı source ve gerçek PostgreSQL üzerinden doğrula. Audit’e göre face_sample üzerinde model_version veya media_type kolonu yoktur.

Yeni hayali kolon ekleme.

Mixed crop contract:

- online image/video: aligned.webp
- bulk: aligned.jpg
- DB object_key authoritative
- API media type’ı object extension veya gerçek object metadata’dan çözer

Current sample lifecycle:

  pending → active
  pending → failed
  active → inactive

pending_blob/blob_ready/indexed gibi yeni state’ler yaratma ve migration yazma.

4. VIDEO SNAPSHOT / CURRENT PROJECTION

Önce video_track ORM/migration’ını tekrar doğrula. Audit’e göre mevcut processing snapshot alanları zaten vardır:

- status_at_processing
- name_at_processing

Varsayılan karar: migration YOK.

Immutable alanlar:

- statusAtProcessing
- nameAtProcessing

Read-time mutable alanlar:

- currentStatus
- currentName
- currentMetadata

Current projection:

  video_track.face_id
    → batch FaceIdentity repository query
    → VideoResultService/API response enrichment

Bu davranışı VideoTrackPersistenceService’e koyma. Historical track row veya overlay artifact rewrite etme.

N+1 query yapma. `get_many_by_ids` benzeri batch repository metodu kullan.

Overlay artifact teknik faceId/trackId/bbox/snapshot evidence taşıyabilir; current name/status response okunurken PostgreSQL’den enrich edilir.

Enrollment sonrası frontend şu query’leri invalidate/refetch etsin:

- face detail/list/history
- video people
- video appearances
- overlay/timeline frames
- job result

5. MEVCUT 12.578 SAMPLE KARARI

İlk işlem yalnız read-only:

  mv-phase1-bulk audit --run-id ... --dry-run

Audit en az şunları raporlasın:

- distinct discovered/accepted/rejected/failed
- active sample’sız known identities
- duplicate face/sample/object/vector
- missing PG/MinIO/Qdrant records
- wrong face ownership
- JPEG magic/dimensions/content type/SHA
- Qdrant payload/model_version/dimension/finite/L2 norm

Live PG/MinIO/Qdrant üzerinde delete, merge, deactivate veya repair yapma.

Önce forward fixes’i tamamla ve küçük deterministic yeni dataset ile acceptance çalıştır. Eski run remediation ayrı kullanıcı onayı gerektirir.

6. BULK ACCOUNTING

`PersistenceOrchestrator.persist_bundle()` rejected sample’ları zaten failed sonucuna dahil ediyorsa CLI tekrar `+ rejected` yapmayacak.

SampleId bazında distinct accounting:

  discovered = accepted + rejected + failed
  persisted <= accepted

Mevcut sayılar için beklenen gerçek ayrım yaklaşık:

  13,233 discovered
  12,578 accepted/persisted
  655 rejected/quarantine

1.310 failure iddiası double-count ise düzeltilmeli.

Accepted sample bulunmayan subject için known face_identity yaratılmayacak.

7. BULK GERÇEK BATCH

Mevcut subject-by-subject loop’u gerçek batch-256 diye raporlama.

Gerçek flow:

  streaming photo items
    → cross-subject bounded microbatch
    → GPU extract_batch(N)
    → association-preserving result demux
    → subject/face bundle aggregation
    → bounded batched persistence

Batch boyunca şu association korunmalı:

- source namespace
- hashed external subject key
- faceId
- sampleId
- image ordinal
- extraction result

8. NATIVE RESOURCE LIFETIME

Önce failing real-GPU reproducer:

  repeat 50–100 times:
    create
    warmup
    extract
    close

PASS:

- clean exit code 0
- no segfault/abort
- no CUDA #709 context destroyed
- GPU memory run sonunda baseline’a döner
- close ikinci kez çağrıldığında harmless

Official nvImageCodec/CUDA/TensorRT lifetime contract’ını Context7/upstream source ile doğrula.

Explicit olarak yönet:

- correct cudaSetDevice
- outstanding stream work synchronization
- JPEG encoder
- decoder/CV-CUDA objects
- TensorRT execution contexts
- engines/runtime
- arenas/device buffers
- CUDA stream

Dependency-aware teardown uygula; yalnız tahmini destructor sırası yazma.

Detector score async D2H sonucu synchronize edilmeden okunamaz.

Systemic CUDA/context/OOM hatasında per-image fallback yapma. Yalnız doğrulanmış corrupt/unsupported JPEG per-entry rejection olabilir.

9. THRESHOLD / PARITY

Video threshold ile image threshold drift’ini source’tan çıkar. 0.95 ve 0.45/0.55 gibi farklı değerleri keyfi tekleştirme.

Önce ölç:

- bulk vs image embedding cosine parity
- bulk vs video template cosine
- same-person held-out distribution
- different-person distribution
- top1/top2 margin
- alignment/preprocess parity

Threshold yalnız versioned calibration evidence ile değişebilir.

10. WORKER CORRECTNESS

Bu sprintte gerçek worker flow’u düzelt:

- bounded MinIO download
- heartbeat renewal
- lease/fencing token’ın persistence/finalization’a taşınması
- bounded cancellation
- SIGTERM grace + SIGKILL timeout
- finalization öncesi lease/cancel recheck
- strict JSON; NaN yok
- observation/template artifact publication
- original frame index + PTS korunması
- bounded progress updates
- gerçek Range streaming
- stale worker completed/side-effect yazamaz

Existing fake/direct-service integration testini E2E diye sunma.

11. STOP DAVRANIŞI

Sıradan failing testte durma veya kullanıcıya dönme.

Şu sırayı uygula:

  reproduce
  root cause
  minimum fix
  targeted test
  regression
  real runtime validation

Yalnız gerçek blocker’da dur:

- destructive live data operation
- applied migration rewrite ihtiyacı
- secret/model download
- system driver/CUDA change
- unresolved user-owned code conflict
- architecture kararı gerektiren schema expansion

12. UYGULAMA SIRASI

Önce Milestone 0–3:

- baseline/store inventory
- canonical requirements/docs
- face-based frontend recovery
- current identity projection + UI enrollment refresh

Sonra Milestone 4–8:

- native clean shutdown
- lifecycle accounting
- real cross-subject batching
- resume/multi-GPU
- bulk→image→video same-faceId continuity

Sonra Milestone 9–12:

- worker correctness
- real UI/API E2E
- repo hygiene/security
- isolated GPU video lab

Milestone geçişlerinde mikro-onay isteme.

13. İLK BUILD GATE

Şimdi şu sırayla başla:

1. git status ve live store read-only inventory
2. failing frontend contract tests:
   - /people page uses /faces
   - no /api/v1/people request
   - cards open /faces/{faceId}
3. failing video current-projection test
4. failing bulk repeated-close GPU reproducer
5. failing distinct-accounting test

Her birini yeşile çevirmeden sonraki büyük katmana geçme.

Commit/push yapma.

Finalde tek PASS verme; her alanı ayrı raporla:

- Phase 1 image
- Phase 2 video
- bulk runtime
- storage continuity
- UI
- worker orchestration
- full E2E

Şimdi Build Mode’a geç ve Milestone 0’dan başla.
