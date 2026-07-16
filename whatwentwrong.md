# MergenVision — Eski Repo Forensic İncelemesi ve Ne Yanlış Gitti Raporu

## 1. Yönetici Özeti

Bu rapor, **yalnızca okuma amaçlı** bir forensic inceleme sonucudur. Yeni tek geçerli repo `/home/user/Workspace/MergenVisionFinalVersion`'dur ve bu rapor da yalnızca oraya yazılmıştır. Eski hiçbir repo değiştirilmemiştir.

### En önemli 5–10 bulgu

| ID | Bulgu | Önem | Kanıt sınıfı |
|---|---|---|---|
| GOV-001 | Yeni repoda `requirements/phase1requirements.md` 0 bayt; senior/client Phase 1 gereksinimleri (`REQ-001` Oracle, 10M ölçek vb.) aktif repoda kaybolmuş. | Yüksek | VERIFIED_SOURCE |
| GOV-002 | Yeni repo `backend/`, `frontend/`, `docs/`, `architecture/` dizinleri tamamen boş; sadece gereksinim ve referans dosyaları var. | Yüksek | VERIFIED_SOURCE |
| PROD-001 | Phase 2 DeepStream denemesi (`mergenvisionprod`) temel gereksinimleri karşılamıyor: frame sampling, `GET /faces/{id}/appearances`, track PostgreSQL'e yazma, anonim cross-video persistence, gerçek iptal yok. | Kritik | VERIFIED_SOURCE |
| PROD-002 | Phase 2 video path'leri genelde CPU/OpenCV (`cv2.VideoCapture`) tabanına düşmüş veya GPU worker env haritalaması bozuk. | Kritik | VERIFIED_SOURCE |
| ACC-001 | `MergenVisionProdReal`'deki LFW GPU-only Python hızlı yolu, hizalama/decode/Mode-A kontratlarında ciddi sapmalar barındırıyor; "%99.35" doğruluk tek başına ürün kabulü değil. | Kritik | VERIFIED_SOURCE (eski forensic rapordan) |
| MGPU-001 | Birden fazla repoda GPU worker haritalaması bozuk: `ProjectFaceRecognize/facerecognition` tüm worker'lara `GPU_DEVICES=0` vermiş; `VideoFaceRealtimeLab` worker'lar varsayılan olarak `false`; `Demo12` hardcoded GPU UUID kullanıyor. | Yüksek | VERIFIED_SOURCE |
| SEC-001 | Eski repolarda `docker-compose.yml` ve `.env` dosyalarında düz metin secret/parola bulunması yaygın (değerler rapora yazılmamıştır). | Yüksek | VERIFIED_SOURCE |
| TEST-001 | Test sayısı tek başına acceptance kanıtı değil; birçok "GPU test" gerçekten GPU hot path'i doğrulamıyor, sadece provider listesini kontrol ediyor. | Yüksek | VERIFIED_SOURCE |
| DRIFT-001 | Raporlar ile kaynak kod arasında sürekli sapma: `VideoFaceGpuLab` README "benchmark-only" derken kodda detection/recognition/tracking/API var; `Demo12` raporları TensorFlow/DeepFace derken kod ONNX/InsightFace; `MergenVision` API_CONTRACT `/api/v1` öneki gösterirken `main.py` kökte mount etmiş. | Yüksek | VERIFIED_SOURCE |
| ORACLE-001 | Oracle entegrasyonu gereksiniminde (REQ-001) tüm implementasyonlar PostgreSQL kullanmış; bu kullanıcı tarafından güncellenmiş kararla uyumlu hale getirilmeli. | Orta | VERIFIED_SOURCE / bilinen güncel karar |

### İlk sapma noktaları

1. **Phase 1 ürün hattı:** Persona/face merkezli model, Oracle sınırı, MinIO/Qdrant abstraction, UUIDv7 gibi temel kararlar çoğu repoda uygulanmış, fakat farklı repo isimleri altında tekrar tekrar baştan yazılmış; hiçbiri diğerine merge edilmemiş.
2. **Phase 2 video hattı:** Çalışan CPU/OpenCV pipeline henüz stabilize edilmeden DeepStream/SCRFD/CUDA optimizasyonuna geçilmiş; bu "optimize etmeden önce stabilize et" kuralının ihlali.
3. **Kabul kriteri hattı:** Fake adapter testleri, provider listesi kontrolleri ve rapor claim'leri çoğu zaman runtime acceptance kanıtı yerine kullanılmış.

### Tekrarlayan failure pattern

- Yeni repo açılır → sadece gereksinem/doküman yazılır → implementasyon eski repolardan parça parça taşınmaya çalışılır → GPU haritalama/secret/path/DNS gibi altyapı hataları nedeniyle çalışmaz → tekrar yeni repo açılır.

### Hangi iddiaların kanıtlandığı / kanıtlanmadığı

| İddia | Durum |
|---|---|
| Phase 1 fotoğraf API'si çalışıyor | `Workspace/mergenvision` ile kanıtlanmış (ek aday repo) — fakat aktif listede yok |
| Phase 2 DeepStream pipeline çalışıyor | **Kısmen**; V0–V1 media/detection kanıtları var, ürün özellikleri eksik |
| GPU decode sıfır kopya | **Kanıtlanmadı**; çoğu Python yolu D2H/H2D kopya içeriyor |
| 10M ölçek test edildi | **Kanıtlanmadı**; en fazla ~1M kayıt gösterildi (`MergenVision`) |
| CUDA 700 hatasının kök nedeni bulundu | **Kanıtlanmadı**; spekülatif debugging örnekleri var |

### Yeni repo için en önemli koruma

- `requirements/phase1requirements.md` 0 bayt durumunu hemen düzelt; senior/client Phase 1 gereksinimlerini geri getir.
- Herhangi bir Phase 2 kodu yazmadan önce Phase 1'in gerçekten çalışan, test edilmiş, Docker'dan ayağa kalkan ürün olduğunu kanıtla.
- "Real-path-first": GPU iddiası varsa gerçek GPU decode/preprocess/inference/search hot path'inin kod satırı satırı izlenebildiği kanıtla.

## 2. Kapsam ve Yöntem

### İncelenen yollar (kullanıcı tarafından verilen)

| # | Yol | Git root | HEAD / son commit | Status |
|---|---|---|---|---|
| 1 | `/home/user/MergenVisionProdReal` | Evet | `main` / `72b00ab` | Aktif değil (eski hedef repo) |
| 2 | `/home/user/MergenVision` | Evet | `main` / `f15b945` | Eski tam demo |
| 3 | `/home/user/mergenvisionprod` | Evet | `feature/scrfd-static-b1-deepstream-e2e` / `fdbfaea` | DeepStream denemesi |
| 4 | `/home/user/mergenvisionprod/oldWorking/NVDIAgstreamer` | Evet | `main` / `bc2c5c1` | Çalışan referans/oracle |
| 5 | `/home/user/Demo/Demo12` | Evet | `main` / `60c8afe3` | Foto demo |
| 6 | `/home/user/Demo/Demo12_VGGFace2Lab` | Evet | `main` / `3d815f8a` | Ölçek sandbox |
| 7 | `/home/user/Demo/VideoFaceRealtimeLab` | Evet | `gpu-utilized` / `8bec129` | CPU/GPU video ürünü |
| 8 | `/home/user/Demo/VideoFaceGpuLab` | Evet | `dev` / `0a5dfea` | Deneysel GPU lab |
| 9 | `/home/user/ProjectFaceRecognize/facerecognition` | Evet | `feature/phase4-multigpu-near-realtime-benchmark` / `cfb1789` | InterProbe prototype |
| 10 | `/home/user/Workspace/FaceRecognitionProject` | Evet | `feature/runtime-gpu-readiness` / `1da61df` | Spec-uyumlu backend |

### Eksik/erişilemeyen repo

Kullanıcı tarafından verilen 10 yolun tümü diskte mevcuttur. `PATH_STATUS=missing` yoktur.

### Ek aday keşfedilen repo'lar (kullanıcı listesine dahil değil)

| Yol | Rol | Not |
|---|---|---|
| `/home/user/Workspace/mergenvision` | Phase 1 kaynak kodunun en temiz hali | Kullanıcı listesinde yok; "ek aday" olarak raporlanmıştır |
| `/home/user/Workspace/MergenVisionProd` | DeepStream FastAPI + worker denemesi | Kullanıcı listesinde yok; "ek aday" olarak raporlanmıştır |
| `/home/user/Workspace/MergenVisionCleanVersion` | Mimari dokümantasyon + eski worktree | Kullanıcı listesinde yok |
| `/home/user/Workspace/FaceRecogv1` | Phase 0 iskelet | Kullanıcı listesinde yok |

### Kullanılan yöntem

- Salt okunur `git log`, `git status`, `git diff --stat`, `find`, `ls`, `grep`.
- Kaynak kod ve rapor dosyalarının satır satır okunması.
- `codebase-memory-mcp` üzerinde zaten indekslenmiş projelerin (`home-user-MergenVision`, `home-user-Workspace-mergenvision`, `home-user-Demo-VideoFaceGpuLab`, `home-user-Demo-Demo12_VGGFace2Lab`) mimari/topoloji incelemesi.
- Eski repolara **yeni indeks oluşturulmadı**; `.codebase-memory/` klasörü üretilmedi.
- Hiçbir test, container, migration, model indirme veya engine build çalıştırılmadı.
- Context7, DeepWiki, Exa web search **kullanılmadı** çünkü inceleme yerel kaynak-temelliydi ve kütüphane sorusu içermiyordu.

### Read-only sınır

- Eski repo veya yeni repo içinde `whatwentwrong.md` dışında hiçbir dosya oluşturulmamıştır.
- `git add/commit/push/checkout/reset/stash` yapılmamıştır.
- Container, worker, test, migration, model/engine üretimi yapılmamıştır.

### Kanıt sınıfları

Rapor boyunca: `VERIFIED_SOURCE`, `VERIFIED_GIT`, `VERIFIED_EXISTING_RUNTIME_EVIDENCE`, `REPORTED_ONLY`, `PARTIALLY_VERIFIED`, `CONTRADICTED`, `UNKNOWN`, `NOT_REPRODUCED`, `ROOT_CAUSE_PROVEN`, `ROOT_CAUSE_PLAUSIBLE`, `ROOT_CAUSE_UNPROVEN`.

## 3. Source-of-Truth ve Requirements Reconciliation

### Phase 1 authority

- **En yüksek otorite:** Senior/client tarafından verilen gereksinimler (`REQ-001`–`REQ-007`):
  - `/home/user/MergenVisionProdReal/requirements/phase1requirements.md` (19 satır, 1.284 bayt).
  - `/home/user/MergenVision/requirements/phase1recognitionrequirements.md` (neredeyse aynı içerik).
- **Aktif repoda (`/home/user/Workspace/MergenVisionFinalVersion`)** `requirements/phase1requirements.md` **0 bayt**; bu bir governance hatasıdır.
- Aktif repodaki `requirements/ProjectRequirements.md` daha detaylı bir agent-yazımı Phase 1 spec'tir ve senior/client REQ'lerini genişletir. Fakat `phase1requirements.md` boş kaldığı için iki otorite arasında net hiyerarşi kaybolmuştur.

### Phase 2 authority

- `/home/user/Workspace/MergenVisionFinalVersion/requirements/phase2requirements.md` 11.335 bayt, tam ve okunabilir.
- Eski `MergenVision/requirements/phase2videorequirements.md` ile `mergenvisionprod/requirements/phase2requirements.md` arasında tek satır/boşluk dışında fark yoktur.

### Legacy ProjectRequirements rolü

- `ProjectRequirements.md` sadece çelişmeyen, güncel kullanıcı kararlarıyla uyumlu kısımları için referans olabilir.
- Oracle entegrasyonu (REQ-001) artık kullanıcı kararıyla "MVP zorunlu değil, gelecekte bulk import source" şeklinde güncellenmiştir; bu nedenle eski ProjectRequirements'daki Oracle-online iddiası geçerli değildir.

### UI kararı

- Phase 1 gereksinimleri "API-only" der (`ProjectRequirements.md` §9, `phase2requirements.md` §9).
- `MergenVision` ve `Workspace/mergenvision` gibi repolarda frontend vardır; bu Phase 1'in geçici demo genişlemesi olarak değerlendirilmelidir, yeni repoda zorunlu değildir.

### Oracle kararı

- Kullanıcı kararı: Oracle online recognition bağımlılığı yok; Oracle yalnızca gelecekteki bulk import source.
- Bu, eski repolardaki `REQ-001` ile çelişen PostgreSQL-only implementasyonları meşrulaştırır.

### MinIO kararı

- MinIO aktif object storage'dır ve S3-compatible abstraction korunacaktır.
- `Workspace/mergenvision` ve `Workspace/FaceRecognitionProject` bu kontrata uygundur; `mergenvisionprod` DeepStream denemesi de MinIO kullanır.

### 10M hedefinin doğru yorumu

- Gereksinim "yaklaşık 10.000.000 kişilik kayıt içeren veritabanı üzerinde çalışabilecek kapasitede olmalıdır."
- En yüksek kanıt: `/home/user/MergenVision` canlı istatistiğinde ~1.020.578 kişi, ~1.001.926 fotoğraf, ~1.151.479 face sample.
- **10M kapasitesi kanıtlanmamıştır.**

### UUIDv7 kararı

- Kullanıcı: "UUIDv7 mümkün olan yeni ID'lerde kullanılacaktır."
- Eski repolarda UUIDv7 kullanımı tutarsız; `Workspace/FaceRecognitionProject` UUIDv7 kullanırken diğerleri UUIDv4 veya karışık kullanmıştır.

### Açık çelişkiler

| Konu | Eski rapor/ kod | Güncel karar | Durum |
|---|---|---|---|
| Oracle | REQ-001 zorunlu | MVP'de zorunlu değil | Çözüldü |
| UI | Repolarda frontend var | API-only | Faz 1'de net |
| 10M | 1M'e kadar kanıt | 10M hedefi | Henüz kanıtlanmadı |
| UUID | Karışık | UUIDv7 | Eski repolarda tutarsız |
| Phase 2 öncesi Phase 1 | Çoğu repo atlamış | Phase 1 tamamlanmadan Phase 2 yapılmayacak | Tekrar eden ihlal |

## 4. Coverage Ledger

| Repo | Dosya | Satır sayısı | Okunan aralık | Tamamlandı mı | Rolü |
|---|---|---:|---|---|---|
| Aktif | `requirements/phase1requirements.md` | 0 | tümü | Evet | Boş governance kanıtı |
| Aktif | `requirements/phase2requirements.md` | 159 | 1–159 | Evet | Phase 2 authority |
| Aktif | `requirements/ProjectRequirements.md` | 94 | 1–94 | Evet | Phase 1 spec |
| Aktif | `opensourcereferences/references.md` | 493 | 1–493 | Evet | Referans listesi |
| MergenVisionProdReal | `report.md` | 156 | 1–156 | Evet | LFW B128 rapor |
| MergenVisionProdReal | `documentation/FORENSIC_IMPLEMENTATION_AUDIT.md` | 983 | 1–983 | Evet | Kendi forensic audit'i |
| MergenVisionProdReal | `whatwentwrong.md` | 646 | 1–646 | Evet | Tarihsel özet |
| MergenVisionProdReal | `requirements/phase1requirements.md` | 19 | 1–19 | Evet | Senior/client Phase 1 |
| MergenVision | `requirements/phase1recognitionrequirements.md` | 25 | 1–25 | Evet | Senior/client Phase 1 kopyası |
| mergenvisionprod | `CURRENT_STATE.md` | 97 | 1–97 | Evet | DeepStream port durumu |
| mergenvisionprod | `documentation/v0_evidence.md` | 75 | 1–75 | Evet | V0 media hattı |
| mergenvisionprod | `documentation/v1_evidence.md` | 105 | 1–105 | Evet | V1 detection overlay |
| mergenvisionprod | `documentation/v1_root_cause.md` | 49 | 1–49 | Evet | B/R kanal takası root cause |
| mergenvisionprod/oldWorking | `projectultrareport.md` | ~1.021 / 72.656 bayt | 1–786 | Kısmen | DeepStream mimari tarihçesi |
| mergenvisionprod/oldWorking | `notes/anchored_summary.md` | 64 | 1–64 | Evet | Phase 1 benchmark özeti |
| Demo/Demo12 | `SOURCE_RUNTIME_RECONCILIATION_REPORT.md` | 272 | 1–272 | Evet | ONNX/InsightFace geçişi |
| Demo/Demo12 | `UI_UX_ENDPOINT_AUDIT_REPORT.md` | 170 | 1–170 | Evet | UI/backend contract drift |
| Demo/Demo12 | `Backend/app/infrastructure/vector/qdrant_vector_store.py` | 99 | 1–99 | Evet | `self._collection` bug iddiasını çürütme |
| ProjectFaceRecognize | `PLAN.md` | 415 | 1–415 | Evet | Video extension planı |
| ProjectFaceRecognize | `GPU_PIPELINE_DEBUG_REPORT.md` | 897 | 1–897 | Evet | GPU debug detayları |
| Workspace/FaceRecognitionProject | `README.md` | 408 | 1–408 | Evet | Spec & architecture |
| Workspace/FaceRecognitionProject | `docs/REQUIREMENTS_SOURCE.md` | 14 | 1–14 | Evet | Requirements authority |

Büyük raporlar (`projectbigreport.md`, `bigprojectreport.md`, 700K+ bayt) satır satır okunmamış; kritik bölümleri ve source kod ile çapraz kontrolü yapılmıştır.

## 5. Çapraz-Repo Kronolojisi

| Tarih / dönem | Repo / dal | Olay / karar | Sonuç |
|---|---|---|---|
| 2026-06-23 | `Workspace/FaceRecogv1` | Phase 0 iskelet; Redis Streams, test yok | Saf scaffold |
| 2026-06-26–28 | `ProjectFaceRecognize/facerecognition` | Multi-GPU video denemeleri, CPU+GPU provider | Dirty tree, GPU mapping bozuk |
| 2026-06-30 | `Demo/Demo12`, `Demo12_VGGFace2Lab` | ONNX/InsightFace foto demo + VGGFace2 ölçek sandbox | 1M+ ölçek proof, fakat frontend/parameter drift |
| 2026-07-01 | `Demo/VideoFaceRealtimeLab` | CPU/GPU video ürünü, worker DNS hatası | En tam non-DeepStream video, fakat worker kapalı |
| 2026-07-02 | `Demo/VideoFaceGpuLab` | PyNvVideoCodec/CuPy zero-copy lab | Advanced fakat README/code drift, DB worker aç |
| 2026-07-04 | `MergenVision` | nginx + 3 GPU API replika, ~1M kayıt | Scale proof, fakat Alembik yok, API prefix drift |
| 2026-07-05–06 | `Workspace/mergenvision` | Temiz Phase 1, Alembic, 58 test | En iyi Phase 1 kaynağı (ek aday) |
| 2026-07-07–08 | `Workspace/MergenVisionProd`, `MergenVisionCleanVersion` | DeepStream Phase 2 denemesi | Altyapı etkileyici, ürün özellikleri eksik |
| 2026-07-10–11 | `mergenvisionprod` | OldWorking/NVDIAgstreamer portu | V0/V1 kanıtları var, B/R kanal takası, aspect stretch |
| 2026-07-11–12 | `MergenVisionProdReal` | Temiz hedef repo denemesi | Sadece boş dizin ve raporlar; 0 bayt Phase 2 |
| 2026-07-12 | `Workspace/MergenVisionFinalVersion` | Yeni tek geçerli repo | Sadece gereksinim/referans dosyaları, phase1 boş |

## 6. Repo Bazında Forensic Bulgular

### 6.1 `/home/user/Workspace/MergenVisionFinalVersion` — Yeni Aktif Repo

- **Amacı:** Tek geçerli temiz ürün reposu.
- **Phase iddiası:** Henüz implementasyon yok; sadece Phase 1 ve Phase 2 gereksinemleri ile referans listesi var.
- **Gerçek durum:**
  - `requirements/phase1requirements.md` 0 bayt (`VERIFIED_SOURCE`).
  - `backend/`, `frontend/`, `docs/`, `architecture/` dizinleri boş (`VERIFIED_SOURCE`).
  - `requirements/phase2requirements.md` ve `ProjectRequirements.md` dolu ve okunabilir.
  - `opensourcereferences/references.md` yapılı ve reference-first workflow tanımlı.
- **Maturity verdict:** `REFERENCE_ONLY` / `REQUIREMENTS_MISMATCH` (Phase 1 authority eksik).

### 6.2 `/home/user/MergenVisionProdReal` — Eski Hedef Repo ve Forensic Laboratuvar

- **Amacı:** Daha önceki "temiz hedef repo" denemesi.
- **Gerçek durum:**
  - `backend/`, `frontend/`, `architecture/`, `native/` boş veya yok (`FORENSIC_IMPLEMENTATION_AUDIT.md` §5).
  - LFW B128 GPU-only Python hızlı yolu akademik olarak etkileyici (%99.35 accuracy, ~258 img/s) fakat ürün değil.
  - ALIGN-001, DECODE-001, MODEA-001, PRODUCT-001 gibi kritik bulgular `FORENSIC_IMPLEMENTATION_AUDIT.md` içinde kanıtlanmış.
- **Maturity verdict:** `RESEARCH_ORACLE` / `PHASE1_PRODUCT_NOT_READY`.

### 6.3 `/home/user/MergenVision` — Eski Tam Demo

- **Amacı:** nginx load-balanced 3-GPU API replika ile Phase 1 foto tanıma.
- **Gerçek durum:**
  - Canlı istatistik: ~1.020.578 kişi, ~1.001.926 fotoğraf, ~1.151.479 face sample.
  - `backend/app/main.py:64` route'ları kökte mount eder; `docs/architecture/API_CONTRACT.md` `/api/v1` öneki bekler (`CONTRADICTED`).
  - Alembic yok.
  - `pytest -q`: 107 passed, 3 failed; `mypy`: 77 hata.
- **Maturity verdict:** `FUNCTIONAL_CPU_BASELINE` (foto) / `REQUIREMENTS_MISMATCH` (Oracle).

### 6.4 `/home/user/mergenvisionprod` — DeepStream Phase 2 Denemesi

- **Amacı:** Native C++/CUDA/DeepStream video pipeline.
- **Gerçek durum:**
  - `CURRENT_STATE.md`'ye göre OldWorking/NVDIAgstreamer akışı mevcut repoya port edilmiş; V0–V1 gate'leri PASS.
  - Fakat `v1_root_cause.md` preprocessing'te B/R kanal takasını ortaya koymuş; bu accuracy için kritik.
  - Aspect-preserving letterbox yok; direct-stretch inference modu (`v1_root_cause.md` §3).
  - Phase 2 ürün özellikleri (sampling, `/faces/{id}/appearances`, track persistence, anonim cross-video) henüz yok.
- **Maturity verdict:** `FUNCTIONAL_GPU_EXPERIMENT` / `REQUIREMENTS_MISMATCH`.

### 6.5 `/home/user/mergenvisionprod/oldWorking/NVDIAgstreamer` — Çalışan DeepStream Referansı

- **Amacı:** `output-tensor-meta=1` → tensor-meta probe → SCRFD decode + landmark association → CUDA five-point alignment → ArcFace → gallery → UNKNOWN → stabilization → OSD.
- **Gerçek durum:**
  - `projectultrareport.md` ve `notes/anchored_summary.md`'ye göre pipeline tamamlanmış; Friends full run metrics var.
  - CUDA 11.1 / TensorRT 11.1.0.106 / DeepStream 9.0 gibi eski runtime stack.
  - Production kodu olarak körlemesine kopyalanamaz; runtime stack farklı, aspect-stretch tradeoff var.
- **Maturity verdict:** `REFERENCE_ONLY` / `REUSABLE_WITH_PARITY_TESTS`.

### 6.6 `/home/user/Demo/Demo12` — Foto Demo

- **Amacı:** Foto-only face identification demo.
- **Gerçek durum:**
  - `SOURCE_RUNTIME_RECONCILIATION_REPORT.md`: ONNX/InsightFace stack'e geçiş yapılmış; 177 test geçiyordu.
  - `QdrantVectorStore` `self._collection` kullanımı tutarlı; eski rapordaki `self._collections` bug iddiası **çoğaltılamadı** (`CONTRADICTED` / `NOT_REPRODUCED`).
  - `docker-compose.gpu-api.yml` hardcoded GPU UUID içeriyor.
- **Maturity verdict:** `FUNCTIONAL_CPU_BASELINE` / `WORKING_PROTOTYPE`.

### 6.7 `/home/user/Demo/Demo12_VGGFace2Lab` — Ölçek Sandbox

- **Amacı:** Batched ONNX multi-GPU import ve recognition parity.
- **Gerçek durum:**
  - LFW 13.233 görüntü, VGGFace2 176.398 görüntü import proof.
  - Demo12 kodunu miras alır; benzer GPU UUID/secret riskleri.
- **Maturity verdict:** `WORKING_PROTOTYPE` (ölçek deneyi).

### 6.8 `/home/user/Demo/VideoFaceRealtimeLab` — CPU/GPU Video Ürünü

- **Amacı:** Full face recognition and video analysis platform.
- **Gerçek durum:**
  - `docker-compose.yml:120,151,182` worker'lar varsayılan `VIDEO_WORKER_PROCESSING_ENABLED=false` (`VERIFIED_SOURCE`).
  - `backend/src/core/config.py:25` `GPU_DECODE_BACKEND="ffmpeg_nvdec_cpu_frames"`; bu CPU frame kopyası anlamına gelir.
  - PostgreSQL DNS resolution hatasıyla worker'lar çıkmış (eski rapordan).
  - `GET /faces/{faceId}/appearances` implemente edilmiş.
- **Maturity verdict:** `PARTIAL_IMPLEMENTATION` / `FUNCTIONAL_CPU_BASELINE`.

### 6.9 `/home/user/Demo/VideoFaceGpuLab` — Deneysel GPU Lab

- **Amacı:** README'ye göre "benchmark-only GPU video pipeline lab".
- **Gerçek durum:**
  - `backend/gpu_video_lab/api/routes_videos.py` detection/recognition/tracking/API var; README ile kod arasında `MISLEADING_CLAIM`.
  - `backend/gpu_video_lab/api/config.py:25` `USE_DB_JOB_STORE` varsayılan `false`; DB worker'lar aç kalıyor.
- **Maturity verdict:** `REQUIREMENTS_MISMATCH` / `ADVANCED_BUT_DRIFTED`.

### 6.10 `/home/user/ProjectFaceRecognize/facerecognition` — Multi-GPU Zero-Copy Prototype

- **Amacı:** FastAPI + multi-GPU zero-copy video pipeline.
- **Gerçek durum:**
  - `docker-compose.yml:93-94,141-142,167-168` tüm worker'larda `GPU_DEVICES: "0"`, `CUDA_VISIBLE_DEVICES: "0"` (`VERIFIED_SOURCE`); GPU haritalama bozuk.
  - GPU decode PyNvVideoCodec ile doğrulanmış; fakat CV-CUDA ↔ CuPy bridge, ONNX postprocess, tracker integration, batch edge cases tamamlanmamış (`GPU_PIPELINE_DEBUG_REPORT.md` §7).
- **Maturity verdict:** `PARTIAL_IMPLEMENTATION` / `FUNCTIONAL_GPU_EXPERIMENT`.

### 6.11 `/home/user/Workspace/FaceRecognitionProject` — Spec-uyumlu Backend

- **Amacı:** API-only FastAPI backend for image + video.
- **Gerçek durum:**
  - API contract (`README.md`) Phase 1 + Phase 2 gereksinimlerine uygun.
  - `backend/app/infrastructure/video/video_reader.py:24,46` ile `backend/app/application/services/video_processing_service.py:90` `cv2.VideoCapture` kullanıyor; GPU worker'lar var ama video decode CPU (`CONTRADICTED`).
  - `GET /faces/{faceId}/appearances` implemente edilmiş.
- **Maturity verdict:** `MOST_SPEC_COMPLETE_OUTSIDER` / `PARTIAL_IMPLEMENTATION`.

## 7. Requirements Drift

| Gereksinim | Orijinal | Eski repolarda ne oldu |
|---|---|---|
| Oracle entegrasyonu (REQ-001) | Zorunlu | Tümü PostgreSQL kullandı; kullanıcı kararıyla güncellendi |
| 10M ölçek | Hedef | En fazla ~1M kanıt |
| Foto tabanlı tanıma | Temel | Çoğu repoda var, fakat farklı stack'ler |
| Kişi bilgileri + fotoğraf eşleştirme | Temel | `Person`/`FaceIdentity`/`FaceSample` modeli tekrar ediyor |
| Gizlilik/veri güvenliği | Temel | Düz metin secret'ler compose/env'de yaygın |
| API-only | Phase 1 | Birçok repoda frontend var |
| UUIDv7 | Yeni karar | Eski repolarda tutarsız |
| MinIO object storage | Karar | Kullanılıyor, fakat bucket/key stratejileri farklı |
| Phase 2 video sampling | `phase2requirements.md` §2 | Çoğu repoda fixed veya eksik |
| `GET /faces/{id}/appearances` | `phase2requirements.md` §8 | Sadece VideoFaceRealtimeLab ve FaceRecognitionProject'te var |
| Anonim cross-video persistence | `phase2requirements.md` §4 | Kısmen implemente |

## 8. Phase 1 Tamamlanmadan Phase 2’ye Geçiş

Somut timeline:

- `Workspace/mergenvision` (ek aday) Phase 1 fotoğraf API'sini neredeyse tamamlamışken, Phase 2 video worker/decoder/tracker kodları da eklenmiş ama `TrackIdentityResolver`, `VideoTrackAggregator`, `/videos/*` route'ları ve persistence tamamlanmamış (`projectultrareport.md` §5.3).
- `Workspace/MergenVisionProd` (ek aday) doğrudan DeepStream Phase 2'ye atlamış; Phase 1 ürünü aynı repoda yeterince sertleştirilmemiş.
- `mergenvisionprod` ise native video-first strateji izleyerek Phase 1 backend/API'sini atlamış; bu kullanıcı kararına ("Phase 1 tamamlanmadan Phase 2 yapılmayacak") aykırı.

## 9. Model ve Runtime Drift

| Bileşen | Değişim | Kanıt | Verdict |
|---|---|---|---|
| Detector | RetinaFace → SCRFD 10G | Demo12'de ONNX/InsightFace; `Workspace/mergenvision` SCRFD TRT | SCRFD son seçim |
| Detector input | 640×640 → 320×320 | `Workspace/mergenvision` detector_adapter 320×320 | Hız/dogruluk tradeoff |
| Recognizer | ArcFace R50 / R100 | Çoğu repo ArcFace 512-D | Stabil |
| Runtime | TensorFlow/DeepFace → ONNX Runtime → TensorRT | Demo12 reconciliations; `MergenVisionProdReal` TensorRT B128 | Hızlanma fakat kontrat riskleri |
| Alignment | OpenCV `estimateAffinePartial2D` → GPU `torch.linalg.lstsq` + `grid_sample` | `MergenVisionProdReal/FORENSIC_IMPLEMENTATION_AUDIT.md` §10 | Konvansiyon riski |
| Decode | CPU PIL → GPU JPEG → PyNvVideoCodec → DeepStream NVDEC | Çeşitli repo denemeleri | Sonraki repolarda daha gerçek GPU decode |
| Tracker | IoU+embedding → ByteTrack → DeepStream NvDCF | VideoFaceRealtimeLab, VideoFaceGpuLab, mergenvisionprod | Track ID ≠ face ID kuralı sonradan oturdu |

## 10. Accuracy Hataları

| Bulgu | Repo | Kanıt | Durum |
|---|---|---|---|
| B/R kanal takası preprocessing'de | `mergenvisionprod` | `documentation/v1_root_cause.md:5-11` | `ROOT_CAUSE_PROVEN` |
| Overlay renk takası | `mergenvisionprod` | `documentation/v1_root_cause.md:13-16` | Kozmetik, geometry bozmayan |
| Aspect ratio bozan direct-stretch | `mergenvisionprod` | `documentation/v1_root_cause.md:18-21` | Kabul edilmiş tradeoff, fakat doğruluk etkisi ölçülmemiş |
| GPU alignment `grid_sample` konvansiyonu riski | `MergenVisionProdReal` | `FORENSIC_IMPLEMENTATION_AUDIT.md` §10.2 | `STRONGLY_SUPPORTED` |
| Mode A tam görüntü resize | `MergenVisionProdReal` | `FORENSIC_IMPLEMENTATION_AUDIT.md` §16 | `CONFIRMED` |
| GPU decode piksel toleransı kaldırılmış | `MergenVisionProdReal` | `FORENSIC_IMPLEMENTATION_AUDIT.md` §11 | `CONFIRMED` |
| B8 ile B128 arası ham fark | `MergenVisionProdReal` | `FORENSIC_IMPLEMENTATION_AUDIT.md` §14.2 | `CONFIRMED` |
| `Demo12` `self._collection` bug | `Demo/Demo12` | Okunan kaynak `self._collection` kullanıyor | `NOT_REPRODUCED` / `CONTRADICTED` |

## 11. GPU Hot-Path Gerçeklik Denetimi

| Aşama | Gerçek implementation | Memory location | CPU copy var mı? | Fallback | Evidence | Verdict |
|---|---|---|---|---|---|---|
| Compressed input | file/src/MinIO download | Host disk/RAM | — | — | Tüm repo'lar | — |
| Demux/parser | PyNvVideoCodec demuxer / GStreamer qtdemux | Host | Evet (metadata) | decord / OpenCV | ProjectFaceRecognize, mergenvisionprod | Kısmen GPU |
| Decode | PyNvVideoCodec, nvv4l2decoder, FFmpeg CUVID, cv2.VideoCapture | GPU (NVDEC) / CPU | Çoğunlukla evet (Python repolar) | CPU OpenCV | VideoFaceRealtimeLab `ffmpeg_nvdec_cpu_frames`; FaceRecognitionProject `cv2.VideoCapture` | **Gerçek GPU decode sınırlı** |
| Frame surface | CuPy / NVMM | GPU | Hedef | NumPy | ProjectFaceRecognize, mergenvisionprod | Kısmen |
| Resize/color conversion | CuPy RawKernel / CV-CUDA / torchvision / OpenCV | GPU/CPU | Evet OpenCV yollarında | CPU | FaceRecognitionProject CPU resize; VideoFaceRealtimeLab CPU | **GPU preprocessing sınırlı** |
| Detector preprocessing | NCHW normalize GPU / CPU | GPU/CPU | Evet CPU yollarında | CPU | Genel | Kısmen |
| Detector inference | TensorRT / ONNX Runtime CUDA | GPU | Hayır | CPUExecutionProvider | ProjectFaceRecognize IOBinding | GPU proven |
| Detector postprocess/NMS | batched_nms GPU / CPU NMS / custom C++ parser | GPU/CPU | Evet CPU parser'lerde | CPU | mergenvisionprod C++ parser; Python repo CPU/GPU karışık | Kısmen |
| Landmark transport | bounded map / tensor meta | Host/GPU | Evet | global dict | mergenvisionprod bounded map | Kısmen güvenli |
| Tracker | ByteTrack/IoU+embedding/NvDCF | CPU/GPU (NvDCF) | Evet | — | VideoFaceGpuLab ByteTrack CPU; mergenvisionprod NvDCF GPU | Kısmen |
| Crop/alignment | CUDA kernel / F.grid_sample / OpenCV warpAffine | GPU/CPU | Evet | cv2 | mergenvisionprod CUDA alignment; ProjectFaceRecognize OpenCV | Kısmen|
| ArcFace inference | TensorRT / ONNX Runtime CUDA | GPU | Hayır | CPU | Proven | GPU proven |
| L2 normalization | F.normalize / GPU kernel | GPU | Hayır | CPU | Proven | GPU proven |
| Embedding D2H | `.cpu().numpy()` / `.get()` | Host | Evet | — | Çoğu repo | **D2H kopya var** |
| Qdrant search | `query_points` async | Network/CPU | Evet | — | Tekil arama | **Batch değil** |
| PostgreSQL persistence | SQLAlchemy async | Host | — | — | Tüm repo'lar | CPU |
| MinIO artefacts | MinIO client | Network | — | — | Tüm repo'lar | IO |
| Overlay/encode | NVENC / CPU encoder | GPU/CPU | Evet CPU output'ta | CPU | mergenvisionprod NVENC | Kısmen |

**Özet:** "Full GPU hot path" iddiası sadece bazı alt aşamalar için kanıtlanmıştır; decode sonrası CPU kopya, D2H embedding transferi, tekli Qdrant araması ve CPU tracker yaygındır.

## 12. CUDA 700 ve TensorRT/DeepStream Forensics

| Soru | Değerlendirme | Kanıt |
|---|---|---|
| CUDA 700 hatası gerçekten görüldü mü? | `REPORTED_ONLY` / `HISTORICAL_CUDA700_CONFIRMED` — eski raporlarda geçiyor; mevcut kanıt dosyalarında doğrudan log bulunamadı. | Eski raporlar |
| Şu an reproducible mi? | `CURRENT_CUDA700_REPRODUCIBLE = no` (test çalıştırılmadı) | — |
| Detector pipeline stabil mi? | `CURRENT_DETECTOR_PIPELINE_STABLE = unknown` | — |
| Root cause proven? | `ROOT_CAUSE_PROVEN = no` | Spekülatif debugging var |
| Environment compatibility? | Farklı repo'lar farklı TensorRT/CUDA/DeepStream versiyonları yüklemiş (örn. TRT 10.x, 11.x, CUDA 12.1, 12.4, 13.0) | `GPU_PIPELINE_DEBUG_REPORT.md`, `v0_evidence.md` |
| External venv TensorRT dependency? | `EXTERNAL_VENV_TENSORRT_DEPENDENCY = unknown` | Kodda `.venv` TensorRT referansları var; runtime kanıtı yok |

**Spekülatif debugging örnekleri:**
- `mergenvisionprod` V1 root cause B/R kanal takasını doğruladı; fakat bunun CUDA 700 ile doğrudan ilişkisi kanıtlanmadı.
- `MergenVisionProdReal` cross-stream senkronizasyon eksikliği (`ZEROCOPY-002`) nondeterministik hatalara açık fakat CUDA 700'ün tek nedeni olarak kanıtlanmadı.
- Engine silinmesi / provenance kaybı: `mergenvisionprod/CURRENT_STATE.md` static-B1 engine yerine DeepStream-compatible dynamic engine'e geçiş yaptığını belirtiyor; bu bir tür "provenance reset"dir.

## 13. Test ve Acceptance Yanılgıları

| Tip | Örnek | Sorun |
|---|---|---|
| Fake-adapter integration | `Demo/Demo12` 177 testin bazıları `FakeFacePipeline` üzerinden | Üretim hot path'i doğrulamaz |
| Provider listesi kontrolü | `ProjectFaceRecognize` ONNX CUDA provider check | Provider var diye inference GPU'da çalışıyor varsayımı |
| Container GPU görme | `nvidia-smi` çıktıları | Container GPU görmesi hot path'in GPU olduğu anlamına gelmez |
| Output file oluşması | `.mp4` çıktısı | Encode doğru anlamına gelmez |
| Test sayısı | 58 passed, 107 passed | Sayı tek başına acceptance değil |
| Report claim = acceptance | "GPU verified" with pixel diff | Kanıt standardı düşük |

## 14. API, UI ve Documentation Drift

| Çelişki | Konum | Kanıt |
|---|---|---|
| API prefix | `MergenVision` | `API_CONTRACT.md` `/api/v1`; `main.py` `/` |
| UI search parametre | `Demo/Demo12` | Frontend `search`, API `query` |
| README vs code | `Demo/VideoFaceGpuLab` | README "benchmark-only"; kodda detection/recognition/tracking/API |
| Report stack vs code | `Demo/Demo12` | Raporda TensorFlow/DeepFace; kodda ONNX/InsightFace |
| Health shape | `Demo/VideoFaceGpuLab` | `docs/API_CONTRACT.md` ile `/health` response farklı |
| Cancellation API | `Workspace/MergenVisionProd` | DELETE var fakat çalışan subprocess öldürülmüyor |
| Video result endpoint | `Workspace/MergenVisionProd` | Ayrı `GET .../result` yok; sonuç status içinde |
| Phase 2 sampling | `mergenvisionprod` | `nvinfer interval=0`; configurable sampling yok |

## 15. Veri Tutarlılığı ve Persistence Sorunları

| Konu | Durum |
|---|---|
| `Workspace/MergenVisionProd` track tabloları | `track_observations` model ve migration var, fakat worker sadece `result_json` yazar |
| `Demo/VideoFaceGpuLab` job store | Default `USE_DB_JOB_STORE=false`; DB worker'lar aç kalır |
| Anonim cross-video persistence | Kısmen implemente; `VideoFaceRealtimeLab` ve `FaceRecognitionProject`'te daha tam |
| Video raw retention | Gereksinimde env-configurable; implementasyonlar farklı |
| Process/job/track ID ayrımı | Son repo'larda oturmuş; erken repolarda `faceId == trackId` hatası |

## 16. Security ve Privacy Bulguları

- **PII / national ID riski:** Eski repolarda `nationalIdHash`/`nationalIdMasked` deseni var; bu iyi practice. Fakat `Qdrant` payload'larda national ID bulunmamalıdır.
- **Secret riski:** `docker-compose.yml` ve `.env` dosyalarında düz metin PostgreSQL/MinIO/Qdrant parola ve credential'ları yaygın. Değerler rapora yazılmamıştır; varlıkları doğrulandı.
- **MinIO object key riski:** object key'lerde `job_id` / `face_id` gibi UUID'ler kullanılıyor; national ID object key'e sızmamalı.
- **Hardcoded secret şüphesi:** `Demo/Demo12/SOURCE_RUNTIME_RECONCILIATION_REPORT.md` örnek komutlarında plaintext `DATABASE_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` gösteriliyor; bunlar rapor örneği ve varsayılan değerler.

## 17. İlk Sapma Noktaları

### Phase 1 ürün çizgisi
- **İlk sapma:** `MergenVision` ~1M kayıt elde ettikten sonra Alembic, test temizliği ve API prefix kontratı ihmal edildi; sonrasında `Workspace/mergenvision` açılarak "temiz repo" stratejisine geçildi. Bu, tek repo yerine sıfırdan başlama pattern'inin başlangıcıdır.

### Phase 2 CPU video çizgisi
- **İlk sapma:** `ProjectFaceRecognize/facerecognition` ve `Workspace/FaceRecognitionProject` CPU/OpenCV video pipeline'ını tamamlamadan, GPU zero-copy optimizasyonuna ve DeepStream'a kaydı.

### GPU video / DeepStream çizgisi
- **İlk sapma:** `Workspace/MergenVisionProd` / `mergenvisionprod` tam bir Phase 1 ürünü stabilize etmeden C++/CUDA/DeepStream native core inşasına başladı.

### Yeni clean-repo denemeleri
- **İlk sapma:** Her seferinde yeni repo açılıp yalnızca gereksinem/doküman yazıldı; implementasyon ve acceptance kanıtları taşınmadı. Aktif `MergenVisionFinalVersion` da aynı hataya düşmüş: `phase1requirements.md` boş, `backend/frontend` boş.

## 18. Yeniden Kullanılabilir Parçalar

| Repo | Dosya/symbol | Ne işe yarıyor? | Kanıtlanan durum | Öneri |
|---|---|---|---|---|
| Aktif repo | `opensourcereferences/references.md` | Reference-first workflow ve lisans kapısı | Dolu ve anlamlı | `reuse_concept` |
| Aktif repo | `requirements/phase2requirements.md`, `ProjectRequirements.md` | Phase 1/2 spec şablonu | Dolu | `reuse_concept` |
| `Workspace/mergenvision` (ek aday) | `backend/app/domain/models.py`, `backend/app/infrastructure/adapters/pipelines.py` | Phase 1 domain + TRT pipeline | 58 test, LFW benchmark | `reuse_with_parity_tests` |
| `mergenvisionprod/oldWorking` | C++ SCRFD parser, bounded landmark store, CUDA alignment | DeepStream reference path | V1 PASS | `reuse_with_parity_tests` |
| `Demo/VideoFaceRealtimeLab` | video identity resolution, `face_video_appearance` | Cross-video appearance persistence | En tam non-DS video | `reuse_with_parity_tests` |
| `Workspace/FaceRecognitionProject` | API contracts, worker SKIP LOCKED, `appearances` | Spec-complete reference | Readme/code uyumlu | `reuse_concept` |
| `Demo/Demo12` | `OnnxFacePipeline` + concurrency semaphore | ONNX/InsightFace adaptasyonu | 177 test geçmiş | `reuse_with_parity_tests` |

## 19. Kesinlikle Taşınmaması Gereken Yaklaşımlar

| Yaklaşım | Neden |
|---|---|
| Yeni repo açıp sadece doküman bırakma | Implementasyon ve kanıt aktarılmıyor |
| GPU zero-copy iddiası olmadan DeepStream'a atlamak | Phase 1 stabilizasyonunu atlar |
| Hardcoded GPU UUID | Donanım değişikliğinde kırılır |
| `CUDA_VISIBLE_DEVICES=0` tüm worker'lara | Multi-GPU haritalama bozuk |
| Plaintext secret'ler compose/env'de | Güvenlik ihlali |
| Provider listesini acceptance kanıtı saymak | Inference hot path'i doğrulamaz |
| `faceId == trackId` | Video-local track ile global identity karışır |
| Aspect-stretch inference üretimde | Doğruluk riski; parity test şart |
| B/R kanal takası veya renk takası | Accuracy bozar |
| Rapor claim'ini code/runtime proof yerine geçirmek | False confidence |

## 20. Yeni Repo İçin Zorunlu Koruma Kuralları

Henüz mimari veya implementation planı üretmeden process ve acceptance guardrail'leri:

1. **real-path-first:** Her GPU iddiası için ilgili kod satırını göster; "provider var", "GPU görünüyor" veya "container ayağa kalktı" kabul değil.
2. **reference-first:** DeepStream, SCRFD, ArcFace, InsightFace, RetinaFace referansları seçilirken exact upstream commit/tag/versiyon ve lisans kaydedilecek.
3. **phase gates:** Phase 2 video endpoint'ine / DeepStream'a başlamadan önce Phase 1'in Docker'dan clean-start yapabilen, test edilmiş, acceptance kanıtlı ürün olduğunu kanıtla.
4. **parity tests:** Detector/recognizer/alignment değişikliklerinde eski çalışan davranışla eşleşme testi; LFW veya benzeri golden set.
5. **provenance:** Tüm TensorRT engine, ONNX model, model config için manifest (versiyon, commit, GPU, driver, build flags).
6. **one-variable debugging:** Aynı anda birden fazla değişken (engine, model, preprocessing, config) değiştirme.
7. **no hidden fallback:** CPU fallback varsa açık ve loglanmalı; sessiz `CPUExecutionProvider` geçişi yasak.
8. **no fake acceptance:** Testlerde fake provider kullanımı açıkça etiketlenecek; fake testler acceptance yerine geçemez.
9. **evidence before completion:** "production ready", "GPU-only", "code complete" gibi ifadeler için ilgili source + runtime kanıtı zorunlu.
10. **no Phase 2 before Phase 1 acceptance:** Phase 1 fotoğraf API'si, enrollment, identify, history, Docker, test, MinIO/Qdrant/PostgreSQL entegrasyonu çalışana kadar video/DeepStream yok.

## 21. Çözülmemiş Sorular ve Kanıt Eksikleri

| ID | Soru | Sınıflandırma |
|---|---|---|
| Q-001 | CUDA 700'ün asıl kök nedeni nedir? | `ROOT_CAUSE_UNPROVEN` |
| Q-002 | Mevcut donanımda gerçek 10M kişi ölçeği test edilebilir mi? | `UNKNOWN` |
| Q-003 | `mergenvisionprod` V1 B/R kanal takası correction sonrası accuracy değeri nedir? | `NOT_REPRODUCED` |
| Q-004 | DeepStream pipeline'ında ArcFace alignment'in gerçekten GPU'da ve doğru konvansiyonda çalıştığına dair piksel-parity kanıtı var mı? | `UNKNOWN` |
| Q-005 | `Demo/VideoFaceRealtimeLab` worker DNS hatası fix sonrası tam video işleme kanıtı var mı? | `UNKNOWN` |
| Q-006 | Hangi detector input size (320/640) üretim hedefi için doğru? | `UNKNOWN` |
| Q-007 | Threshold değerleri (`matched_threshold`, `possible_match_threshold`) hangi ayrılmış video benchmark seti ile tune edildi? | `UNKNOWN` |

## 22. Tool, MCP ve Skill Accountability

| Capability | Tür | Kullanıldı mı? | Amaç | Sonuç | Kullanılmadıysa neden |
|---|---|---|---|---|---|
| `read` / `bash` / `grep` | Tool | Evet | Dosya okuma, git durumu, arama | Gereksinim/rapor/kod incelemesi | — |
| `codebase-memory-mcp` | MCP | Evet | Zaten indekslenmiş projelerin mimari/topoloji incelemesi | `home-user-MergenVision`, `home-user-Workspace-mergenvision`, `home-user-Demo-VideoFaceGpuLab`, `home-user-Demo-Demo12_VGGFace2Lab` kullanıldı | — |
| `codebase-memory-mcp index_repository` | MCP | Hayır | Yeni indeks oluşturmak yasaklandı | — | Kullanıcı yasakladı; `.codebase-memory/` üretilmemeli |
| Context7 | MCP | Hayır | Kütüphane dokümanı sorusu yoktu | — | İnceleme yerel kaynak-temelliydi |
| DeepWiki | MCP | Hayır | Açık kaynak repo sorusu yoktu | — | İnceleme yerel kaynak-temelliydi |
| Exa | MCP | Hayır | Web search gerekmedi | — | Yerel kanıt yeterli |
| Postman | MCP | Hayır | Runtime acceptance görevi değil | — | Forensic audit |
| Playwright | MCP | Hayır | UI runtime görevi değil | — | Forensic audit |
| Ruflo | Yasak | **Kullanılmadı** | — | — | `FORBIDDEN_NOT_USED` |
| 21st | Yasak | **Kullanılmadı** | — | — | `FORBIDDEN_NOT_USED` |
| `brainstorming` | Skill | Hayır | Yaratıcı iş değil; forensic audit | — | Uygunsuz |
| `systematic-debugging` | Skill | Kısmen | Yapısal hata analizi için referans | — | Audit görevi için `codebase-memory` yeterli |
| `verification-before-completion` | Skill | Evet | Raporu yazmadan önce doğrulama adımları | Kanıtlar kontrol edildi | — |
| `codebase-memory` | Skill | Evet | Knowledge graph ile keşif | Kullanıldı | — |

## 23. Sonuç

### En temel süreç hatası
Yeni repo açma alışkanlığı: her döngüde "temiz başlangıç" yapılıp implementasyon ve acceptance kanıtları aktarılmadı. Aktif `MergenVisionFinalVersion` bile sadece boş dizin ve boş `phase1requirements.md` ile başlıyor.

### En temel teknik hata
Phase 1 ürünü stabilize edilmeden Phase 2 video/GPU optimizasyonuna atlamak. Bu, hem DeepStream denemesinde (`mergenvisionprod`) hem de Python zero-copy prototiplerde (`ProjectFaceRecognize/facerecognition`, `VideoFaceGpuLab`) tekrarlandı.

### En temel kanıt hatası
Rapor claim'lerini, provider listelerini, test sayılarını ve container GPU görme kanıtlarını gerçek runtime acceptance kanıtı yerine kullanmak. "GPU verified" damgası piksel farkı varken basıldı; "58 passed" Phase 2'nin eksik olduğunu gizledi.

### Yeni repo başlamadan önce yapılacak tek sonraki şey
`requirements/phase1requirements.md`'yi senior/client Phase 1 gereksinimleriyle doldurmak ve "Phase 1 acceptance" için somut, ölçülebilir kriterleri tanımlamak. Implementation başlamadan önce kullanıcı bu kriterleri onaylamalı.

---

**FORENSIC_AUDIT_COMPLETE:** yes

**Aktif yeni repo:** `/home/user/Workspace/MergenVisionFinalVersion`

**Üretilen dosya:** `/home/user/Workspace/MergenVisionFinalVersion/whatwentwrong.md`

**İncelenen repo sayısı:** 10 (kullanıcı listesi) + 4 ek aday keşfedilen repo

**Eksik/erişilemeyen repo:** Yok

**Eski repolarda değişiklik:** Hayır

**Yeni repoda değişen dosyalar:** Yalnızca `whatwentwrong.md`

**Codebase-memory-mcp:** Başlangıçta var olan indeksler read-only kullanıldı; yeni indeks oluşturulmadı.
