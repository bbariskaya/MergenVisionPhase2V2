Abi evet, bence doğru yön tam olarak bu. Fakat iki noktayı baştan düzeltelim:

“Phase 1 yapılmış varsayalım” demeyelim. Phase 2’nin ihtiyaç duyduğu kimlik, storage, güvenlik ve image-recognition çekirdeğini gerçekten kuralım.
Requirement dokümanında “UI olmayacak” yazıyor. Senin güncel kararın bunu değiştiriyor. Bunu resmi bir requirement amendment olarak kaydedelim; onun dışında önerdiğin frontend-overlay yaklaşımı requirement’a çok iyi uyuyor.

Çünkü requirement zaten her işlenen frame için orijinal çözünürlükte bbox istiyor ve “istemci videonun üzerine çizsin” diyor. Annotated MP4 üretme zorunluluğu yok.

Yeni sistemin tek cümlelik hedefi

Kullanıcı videoyu bir kez yükler; sistem orijinal videoyu değiştirmeden saklar, GPU worker yüz observation’larını üretir, Python track/identity kararlarını verir, PostgreSQL–MinIO–Qdrant’a güvenli şekilde kaydeder ve UI orijinal videonun üzerine bbox/isim overlay’i çizer.

1. Yeni ana topoloji

```flowchart TD
    UI["React UI"] --> API["FastAPI control plane"]
    API --> PG["PostgreSQL"]
    UI --> MINIO["MinIO original video"]
    PG --> WORKER["GPU worker"]
    MINIO --> WORKER
    WORKER --> EVIDENCE["Detection + embedding evidence"]
    EVIDENCE --> PY["Python tracking + reconciliation"]
    PY --> PG
    PY --> MINIO
    PY --> QD["Qdrant"]
    PG --> UI
    MINIO --> UI
```

Bölümlerin görevleri net olacak:

| Katman | Sorumluluk |
|---|---|
| React UI | Upload, job progress, video playback, bbox overlay, identity düzenleme |
| FastAPI | İş akışı, authorization, job yönetimi, API contract |
| PostgreSQL | Kişi, kimlik, job, track, appearance ve audit source of truth |
| MinIO | Orijinal video, fotoğraf, seçilmiş face crop ve büyük evidence artifact’ları |
| Qdrant | Yeniden üretilebilir 512-D face embedding index’i |
| C++/CUDA/TensorRT | NVDEC, detect, landmarks, align, embedding |
| Python | Tracking, tracklet aggregation, identity matching, lifecycle ve persistence orchestration |

Böylece daha önceki problem ortadan kalkıyor: C++ içinde tracker, gallery lifecycle, rename, persistence, history gibi business mantıkları yazmıyoruz.

C++ yalnız gerçekten GPU’da olması gereken işi yapacak.

2. Upload akışını nasıl yapalım?

Senin “browser byte stream’i hem MinIO’ya hem workera yollasın” düşüncen UX açısından anlaşılır ama production açısından riskli.

İki farklı consumer olduğunda:

- Worker eksik MP4 okumaya başlayabilir.
- MinIO başarılı olurken worker kopabilir.
- Retry sırasında hangi byte dizisinin işlendiği belirsizleşir.
- Browser bandwidth’i iki kat kullanılır.
- Worker’ın hızı browser upload’ına backpressure yapabilir.
- Cancel ve resume karmaşıklaşır.
- Bazı MP4/MOV dosyaları dosyanın sonundaki metadata’yı görmeden açılamaz.

Önerdiğim güvenli akış:

```sequenceDiagram
    participant UI
    participant API
    participant MinIO
    participant PG
    participant Worker

    UI->>API: Upload session oluştur
    API->>PG: videoId ve jobId ayır
    API-->>UI: Presigned multipart URL
    UI->>MinIO: Videoyu tek kez yükle
    UI->>API: Upload complete
    API->>MinIO: Size ve checksum doğrula
    API->>PG: Video ready, job pending
    Worker->>PG: Job claim
    Worker->>MinIO: Tamamlanmış videoyu oku
    Worker->>PG: Progress ve sonuç
```

MinIO’nun presigned upload mekanizması browser’ın storage credential almadan private bucket’a yükleme yapmasını destekliyor: MinIO JavaScript SDK.

Ama kullanıcı upload sürerken videoyu hemen izleyebilir:

```javascript
const previewUrl = URL.createObjectURL(file);
video.src = previewUrl;
```

Yani kullanıcı açısından:

- Dosyayı seçtiği anda local preview başlar.
- Arka planda tek upload MinIO’ya gider.
- Upload finalize olunca worker başlar.
- Overlay sonuçları geldikçe local preview üzerine bile çizilebilir.
- Sayfa yenilenirse MinIO’daki video oynatılır.

API istemcileri için requirement’taki endpoint’i de koruruz:

```
POST /api/v1/videos/recognize
```

Bu endpoint multipart videoyu API üzerinden MinIO’ya stream edip aynı internal upload-finalize-job akışına bağlanabilir. UI ise büyük dosyalarda presigned multipart yolunu kullanır.

3. Videoyu yeniden encode etmeyeceğiz

Production çıktısı annotated MP4 olmayacak.

UI:

```
Original video
+ canvas/SVG overlay
+ identity map
+ timeline
```

kullanacak.

Bunun ciddi avantajları var:

- Görüntü kalitesi kaybolmaz.
- NVENC, nvstreamdemux, OSD ve render backpressure ortadan kalkar.
- Rachel’ın adı sonradan değişince video yeniden render edilmez.
- Aynı video farklı overlay versiyonlarıyla gösterilebilir.
- Bbox ve recognition hataları çok daha kolay debug edilir.
- Storage’a ikinci büyük video yazılmaz.

Playback için MinIO’dan kısa ömürlü presigned GET URL üretilir. Browser seek yapabilsin diye Range ve 206 Partial Content desteklenir.

Overlay çizimi timeupdate ile değil, mümkün olduğunda:

```javascript
video.requestVideoFrameCallback((_, metadata) => {
  drawOverlay(metadata.mediaTime);
});
```

ile yapılır. Bu API gösterilen video frame’iyle senkron callback verir: MDN requestVideoFrameCallback.

BBox backend’den orijinal video koordinatında gelir. UI yalnız CSS ölçüsüne dönüştürür:

```javascript
scale = min(containerWidth / videoWidth,
            containerHeight / videoHeight)

offsetX = (containerWidth  - videoWidth  * scale) / 2
offsetY = (containerHeight - videoHeight * scale) / 2
```

Böylece letterbox/pillarbox olsa bile kutu doğru yerde kalır.

AVI/MOV kabul etmekle browser’ın bunları oynatabilmesi aynı şey değil. Politika şu olur:

- Browser uyumlu MP4 ise orijinal dosya oynatılır.
- Container uyumsuzsa playback için lossless remux üretilir.
- Codec uyumsuzsa yalnız playback derivative oluşturulur.
- Overlay hâlâ frontend tarafındadır; annotated video üretilmez.
- Orijinal video hiçbir zaman değiştirilmez.

4. Tracking ve “Rachel neden sonda yine Rachel?” meselesi

Burada iki kavramı kesin olarak ayırıyoruz:

- `trackletId` = bu video içindeki kesintisiz fiziksel takip parçası
- `faceId` = videolar ve sahneler boyunca kalıcı biyometrik kimlik

Rachel frame 1’de görünüp frame 7345’te tekrar geldiğinde aynı tracker ID’sini korumak zorunda değiliz.

Örneğin:

- frame 1–300 → tracklet T1
- frame 301–7000 → Rachel görünmüyor
- frame 7345–7600 → tracklet T81

Recognition/reconciliation şunu yapar:

- T1 → faceId F_RACHEL
- T81 → faceId F_RACHEL

Yani tracker’ın görevi:

Yakın zamanlı framelerdeki kutular aynı fiziksel yüz mü?

Recognition’ın görevi:

Bu tracklet hangi kalıcı face identity’ye ait?

Reconciliation’ın görevi:

Birbirinden kopmuş T1 ve T81 aynı faceId altında birleşmeli mi?

```flowchart TD
    D["Frame detections"] --> T["Python tracklets"]
    T --> E["Quality-weighted embeddings"]
    E --> S["Qdrant top-K samples"]
    S --> R["PostgreSQL lifecycle validation"]
    R --> C["Canonical faceId"]
```

Batch sıralamayı bozmayacak

Detector batch 8 çalışabilir; tracker batch üzerinde paralel state güncellemez.

Her observation şu bilgiyi taşır:

- sourceId
- frameNumber
- PTS
- bbox
- landmarks
- detectorScore
- embedding
- quality

Python:

- Observation’ları sourceId, PTS, frameNumber ile sıralar.
- Tracker’a frame frame verir.
- Batch bittiğinde tracker state’ini sıfırlamaz.
- Sonraki batch aynı state üzerinden devam eder.

Dolayısıyla batch yalnız GPU inference throughput optimizasyonudur; temporal tracking semantiğini değiştirmez.

İlk sürümde:

- C++ custom tracker yazmayacağız.
- NvDCF kullanmak zorunda olmayacağız.
- Python’da test edilebilir ByteTrack-benzeri tracking kullanacağız.
- Crossing, occlusion, scene cut ve re-entry fixture’larıyla doğrulayacağız.

Tracker hata yaparsa recognition/reconciliation ikinci savunma katmanı olur. Ama “kesinlikle aynı kişi” matematiksel olarak tracker’dan kanıtlanmaz; gallery embedding dağılımı, top-1/top-2 margin, çoklu kaliteli frame consensus’u ve gerektiğinde kullanıcı doğrulamasıyla karar verilir.

5. known / anonymous / new_anonymous döngüsü

En önemli karar: new_anonymous kalıcı database state’i değildir; o job sırasında verilen sonuçtur.

```stateDiagram-v2
    [*] --> NewAnonymous: İlk güvenilir eşleşmeyen yüz
    NewAnonymous --> Anonymous: Persist ve index tamamlandı
    Anonymous --> Known: Kullanıcı kişiye bağladı
    NewAnonymous --> Known: Doğrudan promote
    Known --> Known: İsim veya metadata değişti
```

Video A:

```
faceId = F100
status = new_anonymous
```

Video B:

```
faceId = F100
status = anonymous
```

Kullanıcı:

```
F100 kişisini Rachel'a bağladı
```

Video C:

```
faceId = F100
status = known
name = Rachel
```

faceId hiçbir zaman değişmez.

“faceId’yi Rachel diye değiştirmek” gerçekte:

```
face_identity F100
    person_id -> Rachel person kaydı
```

demektir.

Primary key asla Rachel olmaz.

Rachel zaten sistemde varsa

Elimizde:

- F20 = mevcut Rachel identity
- F100 = yanlışlıkla yeni anonymous oluşmuş identity

varsa, rename değil merge yapılır:

```
F100 → canonical F20
```

F100 silinmez; alias/redirect olarak tutulur. Böylece:

- Eski appearance kayıtları kaybolmaz.
- Eski URL ve faceId sorguları çalışır.
- Qdrant sample’ları çöpe gitmez.
- Audit yapılabilir.
- Canonical sonuç F20/Rachel olur.

Tarihsel sonuç

İki alan saklayacağız:

```json
{
  "decisionStatusAtProcessing": "new_anonymous",
  "currentStatus": "known",
  "currentName": "Rachel",
  "faceId": "F100"
}
```

Böylece:

- Audit: Sistem video işlendiği gün ne karar verdi?
- UI: Bu yüz bugün kim olarak biliniyor?

soruları ayrı cevaplanır.

6. Face crop nasıl saklanmalı?

Crop’un primary key’i faceId olmayacak.

- `face_identity.face_id` = kalıcı biyometrik kimlik
- `face_sample.sample_id` = tek bir crop / embedding örneği

Bir identity’nin birden fazla örneği olur:

```erDiagram
    PERSON o|--o| FACE_IDENTITY : owns
    FACE_IDENTITY ||--o{ FACE_SAMPLE : contains
    VIDEO_JOB ||--o{ VIDEO_PERSON : produces
    VIDEO_PERSON ||--o{ VIDEO_TRACKLET : groups
    VIDEO_TRACKLET ||--o{ VIDEO_DETECTION : contains
```

Her frame’i crop olarak saklamayacağız. Bir track boyunca:

- Yüz yeterince büyük mü?
- Blur düşük mü?
- Pose uygun mu?
- Landmark alignment geçerli mi?
- Detector confidence yüksek mü?
- Önceki örneklerden yeterince farklı mı?
- Occlusion var mı?

bakılacak.

Başlangıç için identity başına en fazla 3–5 temporally diverse best shot saklamak mantıklı.

Known bir kişiye ait her video karesini otomatik gallery’ye eklemeyeceğiz. Yanlış recognition gallery poisoning yaratır. Yeni known crop’lar önce candidate olabilir veya çok sıkı multi-frame consensus sonrası aktif edilir.

İlk implementation için temiz yöntem

İki pass kullanabiliriz:

- Pass A: GPU worker detection + alignment + embedding evidence üretir.
- Python track/reconcile yapar ve saklanacak frame/detection ID’lerini seçer.
- Pass B: GPU crop extractor videoyu hızlıca tekrar decode eder ve yalnız seçilen yüzleri çıkarır.

Decode-only ikinci pass hızlıdır ve sistemi karmaşık bidirectional IPC’den korur. Sistem doğru çalışınca tek-pass crop retention optimizasyonu yapılabilir.

7. Üç storage’ın sahipliği

| Sistem | Sahip olduğu veri |
|---|---|
| PostgreSQL | Business ve relational source of truth |
| MinIO | Binary object’ler |
| Qdrant | Rebuildable embedding index |

PostgreSQL

Phase 1 çekirdeği:

- person
- face_identity
- process_record
- inference_profile
- person_photo
- face_sample
- recognition_result
- process_event

Video genişletmesi için onaylanacak tablolar:

- video_asset
- video_job
- video_person
- video_tracklet
- video_appearance
- video_detection veya detection artifact index’i
- cross-store outbox
- identity merge/redirect kaydı

Embedding veya video binary PostgreSQL’e yazılmaz.

MinIO

Örnek opaque key’ler:

```
videos/{videoId}/source/original
face-samples/{faceId}/{sampleId}/crop.webp
jobs/{jobId}/overlay/chunk-000001.json.gz
jobs/{jobId}/evidence/observations.jsonl
```

Object key’lerde şunlar bulunmaz:

- Rachel adı
- Soyadı
- Ulusal kimlik numarası
- Oracle person ID
- Kullanıcının orijinal dosya adı

Orijinal video retention sonunda silinebilir. Kimlik sample crop’u ayrı retention politikasına sahiptir; video silindi diye otomatik silinmez.

Qdrant

Bir point:

```json
{
  "point_id": "face_sample.sample_id",
  "vector": "512-D normalized embedding",
  "distance": "cosine"
}
```

Minimal payload:

```json
{
  "sample_id": "...",
  "face_id": "...",
  "model_version": "...",
  "preprocess_version": "...",
  "active": true
}
```

Qdrant’a isim, ulusal kimlik, departman veya başka PII yazılmaz.

Bir identity’nin birden fazla sample vector’ü olabilir; sonuçlar faceId bazında gruplanır. Qdrant birden fazla vector/sample temsilini destekler: Qdrant vectors.

8. Cross-store transaction

PostgreSQL, MinIO ve Qdrant tek transaction paylaşamaz. Bu nedenle yeni anonymous oluşturma şu state machine ile ilerlemeli:

```flowchart TD
    A["Tracklet final evidence"] --> B["PG identity/sample reserve"]
    B --> C["MinIO crop upload"]
    C --> D["PG blob_ready + outbox"]
    D --> E["Qdrant idempotent upsert"]
    E --> F["PG sample indexed"]
    F --> G["Job result finalized"]
```

Kurallar:

- ID’ler önceden ve deterministik üretilir.
- origin_job_id + origin_tracklet_id unique constraint olur.
- MinIO object key deterministiktir.
- Qdrant point ID tam olarak sampleId olur.
- Retry duplicate identity/sample oluşturmaz.
- Qdrant tamamlanmadan sample ready görünmez.
- Başarısız outbox işlemleri reconcile edilir.
- Orphan MinIO object’leri cleanup job ile bulunur.

Yani “crop MinIO’ya gitti ama database kayıt olmadı” durumu normal kabul edilmeyecek.

9. Frontend overlay verisi

Uzun videodaki bütün bbox’ları tek JSON response’a koymayacağız.

PostgreSQL’de:

- Job ve progress
- Kişi özeti
- Tracklet
- Appearance interval
- Identity relation

tutulur.

Detaylı overlay timeline:

```
jobs/{jobId}/overlay/manifest.json
jobs/{jobId}/overlay/chunk-000000.json.gz
jobs/{jobId}/overlay/chunk-000001.json.gz
```

gibi 5–15 saniyelik chunk’lar halinde MinIO’da olabilir.

Observation’a isim gömmeyeceğiz:

```json
{
  "ptsNs": 5005000000,
  "trackletId": "T17",
  "bbox": [640, 220, 180, 180],
  "detectorScore": 0.98
}
```

Identity map ayrı olur:

```json
{
  "T17": {
    "faceId": "F100",
    "currentStatus": "known",
    "currentName": "Rachel",
    "identityVersion": 8
  }
}
```

Rachel’ın adı değiştiğinde binlerce bbox record yeniden yazılmaz. UI yalnız küçük identity map’i tekrar çeker.

10. Uygulama sırası

Her şeyi tek dev promptta birden yaptırmak yine skeleton ve false-green test üretir. Ama mikro-sprint de yapmayacağız. Her aşama çalışan bir dikey sonuç üretecek.

Sprint 0 — Contract freeze

Kod yok:

- UI requirement amendment
- ID ve status semantiği
- ERD
- Upload contract
- API/OpenAPI contract
- Storage ownership
- Retention/privacy
- Acceptance fixtures

Sprint 1 — Identity ve storage foundation

- PostgreSQL migrations
- MinIO private buckets
- Qdrant collection/index lifecycle
- UUIDv7 IDs
- National-ID encryption + lookup HMAC + masked display
- Cross-store outbox/reconciliation
- Docker Compose persistence
- Health/readiness

Sprint 2 — Image recognition ve anonymous lifecycle

- Person enrollment
- Person photo/sample
- Image multi-face recognition
- known / new_anonymous / anonymous
- Best-shot crop persistence
- Aynı yüzün ikinci request’te aynı faceId dönmesi
- Anonymous → Rachel promote
- Merge/alias audit

Bu geçmeden video başlamaz.

Sprint 3 — Video upload ve async jobs

- Multipart/presigned upload
- Format/codec/size/duration validation
- MinIO retention
- Job state, progress, cancel, retry
- Playback URL
- No-face completed sonucu

Sprint 4 — GPU observation worker

- NVDEC/GStreamer
- RetinaFace
- Five-landmark align
- GlintR100 embedding
- Original bbox coordinate
- Versioned observation artifact
- Python’a yalnız kompakt embedding/metadata
- Tracker ve render yok

Sprint 5 — Python tracking ve reconciliation

- Temporal ordering
- ByteTrack-benzeri tracklet continuity
- Quality-weighted track embedding
- Qdrant top-K
- Known/anonymous/new_anonymous
- Cross-scene same-face reconciliation
- FirstSeen/lastSeen/appearance/totalDuration
- Best-shot selection ve crop pass

Sprint 6 — Internal UI

- Upload
- Local preview
- Job progress via SSE
- Original video playback
- Canvas bbox/name overlay
- Timeline/person sidebar
- Anonymous → Rachel
- Merge duplicate identity
- Appearance history

Sprint 7 — Full acceptance ve scale

- Restart/retry/cancel/retention
- Cross-store partial failure
- Privacy
- Oracle import/sync boundary
- 10M benchmark harness
- Multi-GPU job ownership
- Throughput/latency/recall ölçümleri

İlk gerçek E2E kabul senaryosu

Bu tek senaryo geçmeden “sistem çalışıyor” demeyelim:

- Rachel’ın fotoğrafı enrollment edilir.
- personId + faceId + sampleId oluşur.
- Friends videosu UI’dan yüklenir.
- UI orijinal videoyu oynatır.
- Worker Rachel ve başka bir kişiyi bulur.
- Rachel known görünür.
- Diğer kişi için tek faceId oluşturulur; 300 frame için 300 identity değil.
- En fazla 3–5 kaliteli crop MinIO’ya yazılır.
- İkinci videoda aynı anonymous aynı faceId ile anonymous döner.
- UI’da bu identity “Gunther” olarak promote edilir.
- faceId değişmez.
- Eski video yeniden encode edilmeden Gunther label’ı gösterir.
- /faces/{faceId}/appearances iki videoyu ve zaman aralıklarını döner.
- Container restart sonrası bütün bilgiler korunur.
- No-face video completed + personCount=0 olur.

Bu geçerse elimizde gerçekten ürün vardır.

Şimdi dondurmamız gereken kararlar

Benim önerdiğim varsayılanlar:

- Internal React UI: evet, requirement amendment ile.
- Product annotated MP4: hayır.
- Original video + frontend overlay: evet.
- Browser → worker canlı çift stream: hayır.
- Worker finalized MinIO object’i okur: evet.
- Python tracking/reconciliation: evet.
- Native katman yalnız GPU hot path: evet.
- faceId immutable UUIDv7: evet.
- new_anonymous job-time result: evet.
- Historical decision + current identity ayrı alanlar: evet.
- Identity başına en fazla 3–5 best shot: evet.
- Detection timeline zaman chunk’ları: evet.
- Tracking trackletId, global kimlik faceId: evet.
- İlk doğruluk sürümünde inference bütün framelerde; sampling sonradan configurable: evet.
- Batch başlangıç default’u 4; benchmark ile 1/2/8 seçilebilir: evet.

Bence bu karar setiyle baştan kurarsak önceki C++/DeepStream bataklığına tekrar düşmeyiz. Mevcut projeden yalnız kanıtlanmış RetinaFace/TensorRT, alignment testleri, engine artifact kontrolleri ve benchmark araçlarını seçerek taşırız; broken tracker, skeleton recognizer ve product render topolojisini taşımayız.

Bir sonraki adım kod yazmak değil: bu mimariyi requirement amendment + ERD + API contract + lifecycle state machine + ilk E2E acceptance olarak dondurmak. Bunlar onaylandıktan sonra implementation agent’a kapsamı değiştirme alanı bırakmadan Sprint 1’den başlayabiliriz.
