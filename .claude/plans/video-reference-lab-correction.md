# Video Reference Lab — Forensic Correction Plan (Güncel Durum)

## 1. Anlayış özeti (zorunlu)

**Nihai ürün deneyimi:** Kullanıcı video yükler; backend orijinal videoyu bozmadan işler, her karedeki yüzleri bulur, aynı kişinin ardışık karelerdeki görüntülerini tracklet'lere bağlar, parçalanmış tracklet'leri güvenli şekilde aynı canonical track altında birleştirir, gallery/human kanıtı yeterliyse isim (örn. Rachel) gösterir, yetersizse `unknown` kalır. Frontend orijinal videoyu oynatırken zamanlanmış bbox/label overlay çizer; yeniden MP4 render zorunlu değildir.

**Reference Lab vs production:** Bu Python lab, ileride yazılacak GPU/GStreamer/DeepStream/TensorRT hattının **doğruluk oracle'ıdır**. Amaç hız değil, kanıtlanabilir doğru sonuç ve ayrık hata lokalizasyonudur. Production performans iddiası bu lab'dan çıkarılamaz.

**Kimlik namespace'leri:**
- `observation_id`: bir karedeki tek yüz ölçümü (insan kimliği değil).
- `raw_tracklet_id`: tracker'ın ürettiği zamansal segment.
- `canonical_track_id`: bu video içinde reconciliation ile birleştirilmiş iz.
- `face_id`: kalıcı ürün kimliği; **bu lab yaratmaz**.
- `display_label`: gallery/human kararı sonrası gösterilecek isim; `known` olmadan asla top-1 aday ismi yazılmaz.

**Ayrı sorumluluklar:** Tracker "komşu karelerde aynı nesne mi?", recognizer "bu yüz galerideki kime benziyor?", reconciliation "parçalı tracklet'ler aynı gerçek kişiye mi ait?" sorularını yanıtlar. Biri diğerinin yerine kullanılamaz.

**Immutable artifact nedeni:** Detection/embedding pahalıdır; tracker değiştiğinde yeniden infer etmemek için `frames.jsonl`, `observations.jsonl`, `embeddings.npy`, `manifest.json`, `checksums.sha256` dondurulur. Böylece hata katmanı izole edilir.

**Quality gate önceliği:** Düşük cosine görüldüğünde önce model, koordinat dönüşümü, landmark order, BGR/RGB, alignment crop, normalization, embedding finite/unit-normalized, yüz kalitesi ve gallery fingerprint doğrulanır. Threshold düşürerek preprocessing hatası gizlenemez.

**Unknown güvenlik kuralı:** `known=false` ise `display_label` null veya `unknown` olur; asla top-1 aday ismi gösterilmez. Tek kişilik gallery'de sahte 1.0 margin ile PASS yapılmaz.

**Chunk invariance:** Chunk yalnızca I/O partition'ıdır; tracker state chunk sınırında sıfırlanmaz. Farklı chunk boyutları aynı assignment/tracklet sonucunu vermelidir.

**Bu sprintin kanıtlayacağı:** Python referans lab'ının dikey akışı çalışır, InsightFace oracle ile gerçek model çalışır, Friends.mp4 üzerinde frame/PTS, detection, alignment, embedding, tracking, reconciliation, gallery decision ve visual artifact üretilebilir.

**Bu sprintin kanıtlamayacağı:** Production GPU performansı, kalıcı `face_id` persistence, tam ürün API/frontend, end-to-end ürün kabulü.

## 2. Repository baseline (doğrulandı)

- Root: `/home/user/Workspace/MergenVisionPhase2v2`
- Remote: `bbariskaya/MergenVisionPhase2V2`
- HEAD: `352fbec testvideolab`
- Mevcut durum: `research/video_reference_lab/` altında 72 Python dosyası, CLI (cli.py) mevcut, `.venv-cpu` kurulu, `buffalo_l` modelleri `.model_cache/models/buffalo_l/` altında indirilmiş, `Friends.mp4` mevcut.
- Git durumu: `research/friends_characters/**` ve `prompt2.txt` çalışma ağacından silinmiş (D) ama Git history'de hâlâ var (rewrite yetkisi yok). `.gitignore` ve Makefile değiştirilmiş. `test_gallery/` altında hâlâ 180 cast fotoğrafı mevcut; bunlar kullanıcı sağlanmış gallery olarak işlenecek, Git history'de kalmaya devam edecek.

## 3. Ne çalışıyor (kanıtlandı)

- `mv-video-lab --help`, `doctor`, `models acquire`, `extract`, `replay`, `templates`, `reconcile`, `gallery`, `evaluate`, `visualize`, `benchmark`, `run-friends` komutları çalışıyor.
- Unit testler: 164 passed, 1 skipped, 1 xfailed.
- Entegrasyon testleri: 9 passed (synthetic pipeline, artifact resume, real model smoke, Friends 32-frame pipeline).
- 300 karelik Friends `run-friends` başarıyla tamamlandı: 300 decoded frame, 1792 observation, 265 valid embedding, 13 raw tracklet.
- Chunk invariance testi (1/8/17/64) sentetik veride geçiyor.
- Artifact store validate, checksum, resume çalışıyor.
- ByteTrack lifecycle (lost list, reactivation, scene cut reset, timeout) implemente edilmiş görünüyor.
- Complete-link reconciliation ve cannot-link kuralları implemente edilmiş.

## 4. Halen gerçek sorunlar / yanlış kanıtlar

1. **`evaluation.py` namespace hatası:** `evaluate_identity` observation ID'lerini (`obs.observation_id`) doğrudan `clusters` (raw tracklet ID listeleri) içinde arıyor. Bu yanlış; önce observation → raw tracklet → canonical cluster mapping'i yapılmalı. Mevcut unit test (`test_evaluation.py::test_evaluate_identity_pairwise`) bu hatalı semantiği doğruluyor; testler geçtiği için sorun gizleniyor.

2. **`evaluation.py` gallery summary hatası:** `evaluate_gallery` `gallery_top1_label` varlığına bakıyor, `known` kararına değil. `display_label=None` ama `gallery_top1_label` dolu olan bir track `known` sayılabilir; bu güvenlik kuralına aykırı.

3. **Test kalitesi:** Bazı unit testler (özellikle `test_evaluation.py`, `test_aggregation.py` kısmen) implementasyon hatalarını maskeleyecek şekilde yazılmış. "Unit test geçiyor" demek yetmez; testlerin gerçek semantiği doğruladığına dair inceleme gerekir.

4. **Tracklet ID determinizmi:** `Tracklet._id_counter` sınıf değişkeni; farklı tracker instance'ları sıfırdan başlamaz (testler `reset_id_counter()` çağırıyor ama gerçek pipeline çağırmıyor). Bu farklı çalıştırmalarda ID tutarsızlığına yol açabilir.

5. **Benchmark `max_active_tracks_estimate`:** Her zaman 0 döndürüyor; kozmetik ama rapor için yanlış.

6. **Gallery kararının görsel/timeline kanıtı:** 300 karelik çalıştırmada gallery kararları üretildi mi, `known`/`unknown` doğru mu, contact sheet/timeline/overlay üretildi mi henüz detaylı incelenmedi.

7. **Tam Friends çalıştırma:** Henüz sadece 300 kare çalıştırıldı; full every-frame extraction ve visual inspection yapılmadı.

8. **Rapor:** `REPORT.md` önceki ajan tarafından yazılmış ve çok şeyi PASS olarak işaretlemiş; gerçek durum ve yukarıdaki hatalar yansıtılmamış.

## 5. Düzeltme planı

### Aşama A — Gerçek hataları düzelt (test öncelikli)

1. `evaluation.py::evaluate_identity` düzelt:
   - Girdi olarak `assignments` (observation → raw_tracklet) ve `canonical_map` (raw_tracklet → canonical_track_id) al.
   - `resolved_anchors` içindeki observation'ları önce raw tracklet'e, sonra canonical track'e map et.
   - Pairwise precision/recall/F1 canonical cluster üzerinden hesapla, observation ID'leriyle cluster tracklet ID'lerini karşılaştırma.
   - Zero denominator korumasını koru.

2. `evaluation.py::evaluate_gallery` düzelt:
   - `known` sayımı `track.decision_reason == "gallery_match"` ve `track.display_label is not None` üzerinden yap.
   - Sadece `gallery_top1_label` varlığına bakma.

3. Testleri gerçek semantiğe göre yeniden yaz:
   - `test_evaluation.py::test_evaluate_identity_pairwise`: cluster'lar raw tracklet ID'lerden oluşsun, resolved anchors observation ID'leri içersin, beklenen metric'ler bu mapping üzerinden hesaplansın.
   - `test_evaluation_gallery_known_unknown`: `known` track'te `display_label` ve `decision_reason="gallery_match"` olsun; `unknown` track'te `display_label=None` olsun ama `gallery_top1_label` dolu olsun (bu durumda `unknown_count` artsın).

4. `byte_tracker.py` ID determinizmi:
   - `ByteTrackIoUTracker.__init__` içinde `Tracklet.reset_id_counter()` çağrısını ekle, böylece her tracker instance kendi sıfırından başlar.
   - Veya instance-local counter kullan (daha temiz); mevcut kodun minimal değişimi için init içinde reset yeterli.

5. `benchmark.py` `max_active_tracks_estimate`:
   - Her measured run sırasında `len(tracker.active_tracklet_ids())` veya tracked + lost sayısının maksimumunu ölç.

### Aşama B — Full Friends çalıştırma ve görsel doğrulama

6. Full every-frame Friends extraction çalıştır:
   - `mv-video-lab extract --config configs/friends_baseline_cpu.yaml` (max_frames yok).
   - Süreye bağlı olarak uzun sürebilir; önce 300 kare sonuçlarını detaylı incele, sonra full run planla.

7. Replay, templates, reconcile, gallery, evaluate, visualize aşamalarını çalıştır.

8. Üretilen artifact'leri incele:
   - `raw/frames.jsonl`: tüm frame'ler var mı, PTS/timestamp doğru mu?
   - `raw/observations.jsonl`: bbox sınırları içinde mi, landmark'lar anlamlı mı?
   - `visual/contact_sheet.jpg`: alignment crop'lar göz/burun/ağıza oturuyor mu?
   - `tracks/canonical_tracks.json`: `display_label` yalnızca `known` track'lerde mi?
   - `evaluation/metrics.json`: evaluate_identity doğru namespace'lerle mi hesaplanmış?

9. Chunk parity'yi gerçek artifact'ler üzerinde tekrar doğrula:
   - `replay --chunk-size 1/8/17/64` ile farklı chunk boyutlarında aynı assignment/tracklet çıktığını hash'le kontrol et.

### Aşama C — Rapor ve governance

10. `docs/implementation/REFERENCE_DECISION_LOG.md` ve `REPORT.md` güncelle:
    - Mevcut durumu dürüstçe yansıt: hangi gate'ler PASS, hangileri TBD/BLOCKED, hangi hatalar düzeltildi.
    - Public media/biometric exposure notu ekle: `research/friends_characters/` ve `prompt2.txt` ağaçtan silindi, history'de kaldı.
    - `test_gallery/` içindeki görsellerin Git history'de olduğunu belirt.

11. `docs/implementation/CURRENT_SPRINT.md` içine isolated-research-exception paragrafını ekle (Sprint 01 durumunu değiştirmeden).

12. `.gitignore` ve tracked-binary testini gözden geçir; yeni ignore kuralları eklenmişse çalıştır.

### Aşama D — Make/CI doğrulama

13. `make video-reference-ci` çalıştır.
14. `make video-reference-doctor-cpu` ve `make video-reference-real-model-smoke-cpu` çalıştır.
15. `make video-reference-friends-smoke` (32 frame) çalıştır.
16. `make video-reference-friends-acceptance` veya `make video-reference-acceptance` çalıştır (full Friends).

## 6. Çıktı olarak beklenen sonuç

- Düzeltilmiş `evaluation.py` ve testleri.
- Düzeltilmiş `byte_tracker.py` ID determinizmi.
- Düzeltilmiş benchmark aktif track sayımı.
- Full Friends artifact seti (veya makul sürede tamamlanamazsa 300 kare + açık NOT_RUN).
- Güncel `REPORT.md` ve `VIDEO-REFERENCE-LAB-CORRECTION.md`.
- Dürüst verdict: `PASS`, `PARTIAL_NEEDS_HUMAN_LABELS`, veya `BLOCKED_CORRECTION`.

## 7. Riskler

- Full Friends extraction saatler sürebilir; plan bölünebilir.
- `test_gallery/` içindeki 180 cast görseli Git history'de kalacak; bunu raporda açıkça belirtmek gerekir.
- Mevcut değişiklikler commitlenmemiş; herhangi bir hata geri alınabilir.
