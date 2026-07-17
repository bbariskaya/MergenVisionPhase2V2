PROJE BAĞLAMI — ÖNCE BUNU ANLA, SONRA IMPLEMENTASYONA BAŞLA

Bu bölüm teknik görev listesinden üstündür. Buradaki amaç, yalnız dosyaları tamamlaman değil, ortaya çıkarmaya çalıştığımız ürünü anlamandır.

==================================================
1. MERGENVISION NEDİR?
==================================================

MergenVision basit bir “yüz modeli çalıştırma demosu” değildir.

Amacımız fotoğraf ve ileride video üzerinden çalışan, kalıcı kimlik yönetimi bulunan bir yüz tanıma platformu oluşturmaktır.

Kullanıcı açısından ürünün temel sorusu şudur:

“Bu görüntüde kimler var, sistem bunları daha önce gördü mü, gördüyse aynı kimliği koruyabiliyor mu?”

Sistem yalnız cosine score döndürmeyecek. Kalıcı faceId, enrollment, geçmiş, process takibi ve storage lifecycle sağlayacaktır.

Bağlayıcı gereksinimler:

- requirements/ProjectRequirements.md
- requirements/videorequirements.md

İkisini tamamen oku. Bu açıklama onları değiştirmez; ürünün niyetini anlamana yardım eder.

==================================================
2. PHASE 1: FOTOĞRAF TABANLI KİMLİK SİSTEMİ
==================================================

Phase 1’in kullanıcı akışı şöyledir:

1. Kullanıcı bir fotoğraf gönderir.
2. Sistem input’un gerçekten desteklenen ve decode edilebilir bir görüntü olduğunu doğrular.
3. Fotoğraftaki bütün yüzleri bulur.
4. Her yüzü bağımsız olarak tanımaya çalışır.
5. Tek API çağrısı için tek processId üretir.
6. Her yüz için bounding box ve identity sonucu döner.
7. Sonuçlar ve process geçmişi daha sonra sorgulanabilir.

Bir görüntüde üç yüz varsa:

- Üç ayrı process oluşturulmaz.
- Bir process oluşturulur.
- Üç recognition_result kaydı oluşur.
- Her yüzün sonucu bağımsız olabilir.

Örneğin aynı response içinde:

- Bir kişi known
- Bir kişi anonymous
- Bir kişi new_anonymous

olabilir.

No-face bir sistem hatası değildir.

Geçerli fakat yüz bulunmayan görüntü:

- HTTP success döner.
- process completed olur.
- faceCount=0 olur.
- faces=[] olur.

Boş, bozuk veya desteklenmeyen dosya ise structured error döner. Raw CUDA, TensorRT veya decoder exception’ı kullanıcıya gösterilmez.

==================================================
3. IDENTITY SEMANTİĞİ
==================================================

Üç recognition result status’u vardır:

- new_anonymous
- anonymous
- known

Bunları birbirine karıştırma.

new_anonymous:

Sistem bu yüzü daha önce tanımamıştır ve ilk defa kalıcı bir faceId oluşturmuştur.

Bu yalnız ilk recognition sonucunun snapshot status’udur.

Persistent face_identity state’i new_anonymous değildir. Oluşturulan identity PostgreSQL’de anonymous olarak tutulur.

anonymous:

Sistem bu yüzü daha önce görmüştür.

Aynı kalıcı faceId bulunmuştur fakat kullanıcı henüz bu kimliğe isim vermemiştir.

known:

Aynı faceId kullanıcı tarafından enroll edilmiştir. Sonuç name ve metadata içerebilir.

Beklenen lifecycle:

İlk istek:
new_anonymous
faceId = A

Aynı yüz ikinci kez:
anonymous
faceId = A

Kullanıcı enroll eder:
faceId = A korunur
identity known olur

Aynı yüz tekrar:
known
faceId = A
name/metadata döner

Enrollment yeni faceId oluşturmamalıdır.

Eski recognition result’ları daha sonra değiştirilmemelidir. İlk sonuç sonsuza kadar new_anonymous snapshot’ı olarak kalır. Enrollment geçmişi yeniden yazmaz.

==================================================
4. BU SPRINTİN GERÇEK SONUCU
==================================================

Şu anda Sprint 02’de gerçek GPU image identity vertical slice’ını yapıyoruz.

Çalışması gereken zincir:

JPEG
→ gerçek NVIDIA GPU runtime
→ nvJPEG decode
→ CUDA preprocessing
→ TensorRT RetinaFace R50
→ CUDA RetinaFace decode/NMS/landmarks
→ CUDA five-point face alignment
→ TensorRT GlintR100
→ CUDA L2-normalized 512-D embedding
→ Python application service
→ PostgreSQL
→ MinIO
→ Qdrant
→ FastAPI response

Kullanılacak modeller dondurulmuştur:

Detector:
backend/artifacts/models/retinaface_r50_dynamic.onnx

Recognizer:
backend/artifacts/models/glintr100.onnx

Başka modele geçme.

Bu sprintin sonunda senior’a şu gerçek demo gösterilebilmelidir:

1. API çalıştırılır.
2. No-face JPEG gönderilir ve başarılı boş sonuç alınır.
3. Tek yüzlü JPEG gönderilir.
4. Sistem new_anonymous ve bir faceId döndürür.
5. Aynı JPEG tekrar gönderilir.
6. Sistem aynı faceId ile anonymous döndürür.
7. faceId enroll edilir.
8. Aynı JPEG tekrar gönderilir.
9. Sistem aynı faceId ile known döndürür.
10. Multi-face JPEG gönderilir.
11. Bütün yüzler tek processId altında bağımsız sonuçlanır.
12. Process ve face history endpoint’lerinden sonuçlar okunur.
13. PostgreSQL, MinIO ve Qdrant restart sonrasında kayıtlar korunur.

Yalnız engine dosyası üretmek bu sprinti tamamlamaz.

Yalnız detector bounding box çıktısı üretmek tamamlamaz.

Yalnız sentetik embedding testi tamamlamaz.

Gerçek HTTP → gerçek GPU → gerçek storage lifecycle aynı zincirde çalışmalıdır.

==================================================
5. STORAGE’LARIN ROLÜ
==================================================

PostgreSQL business source-of-truth’tür.

Şunları tutar:

- face identity
- current known/anonymous state
- process
- sample lifecycle
- immutable recognition result
- history

PostgreSQL’e yazılmayacak:

- image binary
- crop binary
- embedding vector

MinIO binary owner’dır.

Bu sprintte aligned face crop saklar:

faces/{faceId}/{sampleId}/aligned.webp

Object key içinde isim, metadata veya kişisel bilgi bulunmaz.

Qdrant derived vector index’tir.

Şunları tutar:

- 512-D GlintR100 embedding
- sample_id
- face_id
- active
- model_version

Qdrant’a yazılmayacak:

- name
- identity metadata
- MinIO object key
- history
- raw image

Qdrant point ID tam olarak face_sample.sample_id olmalıdır.

Eski synthetic vector’larla yeni GlintR100 vector’ları karıştırılmamalıdır. Bu nedenle yeni collection kullanılacaktır:

face_samples_retinaface_r50_glintr100_v1

Eski collection silinmeyecek veya resetlenmeyecektir.

==================================================
6. GPU HOT PATH NEDEN BÖYLE TASARLANIYOR?
==================================================

Python business orchestration içindir.

Python’ın görevi:

- API
- process yönetimi
- identity kararı
- PostgreSQL
- MinIO
- Qdrant
- error mapping
- response oluşturma

Native C++/CUDA runtime’ın görevi:

- decode
- preprocess
- detect
- NMS
- landmarks
- alignment
- embedding
- L2 normalization

Production path’te Python/OpenCV/Pillow ile full image decode ve resize istemiyoruz.

Raw TensorRT detector output’unu NumPy’ya taşıyıp CPU postprocess yapmak istemiyoruz.

Full decoded frame’i GPU’dan CPU’ya geri taşımak istemiyoruz.

Python’a yalnız şu compact sonuçlar çıkmalıdır:

- original image dimensions
- bounding boxes
- five landmarks
- detector confidence
- 512-D normalized embedding
- küçük aligned crop
- timing/model metadata

Native runtime identity kararı vermez.

Native taraf:

- known demez
- anonymous demez
- faceId üretmez
- name/metadata bilmez

Bunlar application/business layer kararıdır.

==================================================
7. DYNAMIC BATCH’İN AMACI
==================================================

Current image API bir request’te bir fotoğraf işler.

Buna rağmen engine’leri yalnız batch=1’e kilitlemiyoruz.

RetinaFace profile:

- min=1
- opt=4
- max=8
- spatial size 640×640

GlintR100 profile:

- min=1
- opt=8
- max=32
- crop size 112×112

Şimdiki image API RetinaFace’e batch=1 verir.

Dynamic detector batch’in amacı gelecekte video pipeline’da birden fazla decoded frame’i aynı native core’a verebilmektir.

GlintR100 batch’in amacı bir veya birden fazla görüntüde bulunan bütün yüz crop’larını verimli biçimde embed etmektir.

32’den fazla crop varsa deterministic şekilde chunk edilir; yüzler sessizce atılmaz.

Batch çalışırken detection ile embedding association kaybolmamalıdır.

Her detection şunları korumalıdır:

- hangi source image/frame’den geldiği
- detection index
- bbox
- landmarks
- embedding

==================================================
8. GELECEKTE VİDEO NASIL EKLENECEK?
==================================================

videorequirements.md gelecekteki video ürününü tarif eder.

Video tarafında kullanıcı:

- Video upload edecek.
- Async jobId alacak.
- Job status/progress sorgulayacak.
- Sistem videoyu örnekleyerek işleyecek.
- PTS/time-base doğru tutulacak.
- Aynı kişiye ait ardışık detection’lar track/tracklet olacak.
- Track boyunca en kaliteli yüz örnekleri seçilecek.
- Track template embedding oluşturulacak.
- Known/anonymous identity kararı verilecek.
- Appearance interval/timeline üretilecek.
- İstenirse annotated output oluşturulacak.
- Cancel/retry/restart davranışı olacak.

Fakat bunların hiçbiri Sprint 02’de implement edilmeyecek.

Bu sprintin video açısından görevi, tekrar kullanılabilir bir native çekirdek sağlamaktır:

Image bugün:
nvJPEG → DeviceImageView → FacePipeline

Video gelecekte:
NVDEC/DeepStream → DeviceImageView batch → aynı FacePipeline

Böylece video sprintinde detector, alignment ve recognizer’ı yeniden yazmayacağız. Yalnız video decode, batching, timestamps ve tracking katmanlarını ekleyeceğiz.

TrackId ile faceId aynı şey değildir:

- trackId yalnız bir video/job içindeki yerel hareket kimliğidir.
- faceId global persistent identity’dir.

Native detector trackId veya faceId üretmemelidir.

==================================================
9. LABIN ROLÜ
==================================================

research/video_reference_lab ürün değildir.

Bu sprintte:

- Labı değiştirme.
- Çalıştırma.
- Product dependency yapma.
- Friends sonuçlarını acceptance sayma.
- Ground-truth problemi çözmeye çalışma.
- Labeling UI üzerinde çalışma.

Lab yalnız ileride bir native modülün bbox/alignment/quality davranışı şüpheli olduğunda debugging/reference aracı olabilir.

Bu sprintin ilerlemesi lab doğruluğuna bağlı değildir.

Biz yeni bir computer-vision araştırma problemi çözmeye çalışmıyoruz.

Mevcut RetinaFace ve GlintR100 modellerini doğru preprocess, doğru alignment ve doğru TensorRT runtime ile best-effort ürün sistemine bağlıyoruz.

==================================================
10. MODEL DOĞRULUĞU KONUSUNDA DÜRÜSTLÜK
==================================================

Bu sprint küçük fixture setiyle şunları kanıtlayabilir:

- Model gerçekten GPU’da çalışıyor.
- RetinaFace output’u doğru decode ediliyor.
- Bounding box original coordinate’e doğru dönüyor.
- Landmark order doğru.
- Alignment yönü doğru.
- GlintR100 preprocessing doğru.
- TensorRT embedding reference implementation ile uyumlu.
- Batch 1 ve batch N tutarlı.
- Identity/storage lifecycle çalışıyor.

Şunları kanıtlayamaz:

- Her kamera ortamında doğru yüz tanıma.
- Genel precision/recall.
- Optimal cosine threshold.
- Blur/pose/occlusion eşiklerinin doğruluğu.
- Production-scale throughput.
- 600 FPS.
- Three-GPU scaling.

Bu sınırlamaları saklama fakat bunları bahane ederek implementation’ı durdurma.

İlk acceptance aynı exact yüz görüntüsünün tekrar tanınmasını gösterebilir. Daha geniş accuracy/threshold calibration sonraki ayrı çalışma olacaktır.

==================================================
11. FAILURE DAVRANIŞI
==================================================

Yanlış durumda sahte success dönme.

Örnekler:

- JPEG decode başarısızsa process failed.
- GPU inference başarısızsa process failed.
- Tensor output invalidse process failed.
- MinIO upload başarısızsa sample active olmaz.
- Qdrant upsert başarısızsa sample recognition-ready olmaz.
- Result persistence başarısızsa completed response dönmez.
- No-face ise bu failure değildir; completed faceCount=0 olur.
- GPU request kapasitesi doluysa bounded overload error döner.
- CPU inference fallback yapma.

Raw exception client’a dönmez.

Final raporda gerçek çalışan, çalışmayan ve hiç test edilmeyen davranışları açıkça ayır.

==================================================
12. SPRINT 02’DE YAPILMAYACAKLAR
==================================================

Bu sprintte yapma:

- SCRFD veya başka model
- Video upload/job
- GStreamer/DeepStream
- NVDEC
- Tracking
- Tracklet merge
- Temporal aggregation
- Labeling UI
- Lab düzeltmesi
- React UI
- Blur/pose threshold calibration
- Accuracy benchmark
- 600 FPS optimizasyonu
- Three-GPU
- Bulk enrollment
- National ID
- Oracle
- 10M-person platformu
- Yeni PostgreSQL tablo
- Eski Qdrant collection reset
- Eski plan/doküman silme
- Model/dataset indirme
- Driver/system CUDA değiştirme

==================================================
13. BUILD MODE DAVRANIŞIN
==================================================

Artık Build Mode’dasın.

Onaylanmış Sprint 02 planını uygula.

Tekrar şu kararları sorma:

- Model seçimi
- Build container seçimi
- DeepStream kullanımı
- Dynamic batch profile
- Qdrant collection
- Labı ne yapacağımız
- Eski planı silip silmeyeceğin

Bu kararlar verilmiştir.

Normal implementation/test/debug adımlarında kullanıcıdan mikro-onay isteme.

Yalnız şu gerçek blocker’larda dur:

- Yanlış repository/base.
- Çakışan user changes.
- Pinned NVIDIA container çalışmıyor.
- Host driver container stack’i desteklemiyor.
- Model artifact/contract geçersiz.
- Dynamic batch model tarafından gerçekten desteklenmiyor.
- System dependency/model download gerekiyor.
- Real face fixture yok ve real-face acceptance aşamasına geldin.

Fixture yokluğu bütün source implementation’ı durdurma nedeni değildir. Yapılabilen her şeyi tamamla; yalnız real-face acceptance’ı BLOCKED_REAL_GPU_FIXTURES olarak bırak.

==================================================
14. BİTİRDİĞİNDE NE GETİRECEKSİN?
==================================================

Sprint sonunda yalnız “done” deme.

Şunları getir:

- PASS/PARTIAL/BLOCKED verdict.
- Değişen dosyalar.
- Gerçek GPU identity.
- CUDA/TensorRT versions.
- RetinaFace ve GlintR100 model/engine SHA.
- Dynamic profiles.
- Exact validation commands ve exit results.
- No-face sonucu.
- Single-face lifecycle sonucu.
- Multi-face sonucu.
- PostgreSQL/MinIO/Qdrant evidence.
- API/OpenAPI evidence.
- Docker runtime evidence.
- Batch parity sonuçları.
- Performance p50/p95/p99, yalnız informational.
- CPU fallback kullanılmadığının kanıtı.
- Labın değişmediğinin kanıtı.
- Bilinen sınırlamalar.
- SPRINT-002-CODE-REVIEW-PACKAGE.md yolu.

Kanıt olmadan:

- production-ready
- accuracy verified
- GPU-only E2E
- video-ready
- fully optimized
- 600 FPS

deme.

Şimdi bu ürün bağlamını aklında tutarak onaylanmış Sprint 02 planını Build Mode’da uygula.
