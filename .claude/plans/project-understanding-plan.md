# MergenVision Derin Anlama ve Hafıza Planı

## Amaç

MergenVisionPhase2v2 projesini requirement → mimari → kod → test → sprint durumu düzeyinde derinlemesine anlamak; önemli bilgileri `prompt-memory-mcp`'ye `Memory` / `Decision` node'ları olarak kaydetmek. Bu aşamada **kaynak kod değiştirilmeyecek**, sadece okunacak ve not alınacak.

## Kapsam

### Dokümanlar
- `AGENTS.md` — Engineering Constitution
- `ProjectGoalandContext.md` — ürün bağlamı, Sprint 02 hedefi
- `requirements/ProjectRequirements.md` — image/identity requirements
- `requirements/videorequirements.md` — video requirements
- `docs/implementation/CURRENT_SPRINT.md` — sprint durumu ve milestone ledger
- `architectureplan.md` — onaylanmış mimari kararları
- `PLAN.md` — Phase 1 Sprint 01 planı (geçmiş, referans)
- `opensourcereferences/references.md` — official/upstream referans kataloğu
- `prompt.txt` … `prompt11.txt` — geçmiş sprint instruction'ları

### Kod
- `backend/app/api/*` — FastAPI routes, controllers, schemas
- `backend/app/application/*` — ports, services, orchestration
- `backend/app/domain/*` — domain entities, value objects, lifecycle
- `backend/app/infrastructure/*` — SQLAlchemy, MinIO, Qdrant adapters
- `backend/native/*` — C++/CUDA/TensorRT video worker, FacePipeline
- `backend/app/worker/*` — Python job worker, tracking, reconciliation
- `backend/tests/*` — test piramidi ve acceptance hedefleri
- `Makefile`, `docker-compose*.yml`, `frontend/src/api/videos.ts`, `frontend/src/pages/VideoPage.tsx`

## Yaklaşım

### Phase A — Doküman yükleme
- Halihazırda line-by-line yüklenmiş 3 dosya (`ProjectRequirements.md`, `videorequirements.md`, `ProjectGoalandContext.md`) atlanır; sadece eksik/uzun özet memory'ler tamamlanır.
- `AGENTS.md`, `CURRENT_SPRINT.md`, `architectureplan.md` line-by-line `Memory` node olarak yüklenir (boş satırlar hariç).
- `PLAN.md`, `references.md`, `prompt*.txt` için paragraf/bölüm düzeyinde özet `Memory` node'ları oluşturulur; çok uzun oldukları için satır satır değil.
- Yükleme, daha önce kullanılan toplu Python script ile doğrudan SQLite üzerinden yapılır; `store_memory` MCP çağrısı binlerce satır için kullanılmaz.

### Phase B — Codebase keşfi
1. `codebase-memory-mcp__get_architecture` ve `search_graph` ile entry point'ler, hotspot'lar ve katmanlar haritalanır.
2. Paralel sub-agent'larla her büyük modülün kaynağı okunur:
   - API katmanı (routes, controllers, schemas)
   - Application katmanı (ports, services)
   - Domain katmanı (entities, value objects, state machine)
   - Persistence katmanı (SQLAlchemy models, repositories, migrations)
   - Infrastructure adapters (MinIO, Qdrant)
   - Native worker / C++ GPU pipeline
   - Python tracking & reconciliation
   - Tests / Makefile acceptance hedefleri
3. Her sub-agent structured output döner: modül amacı, ana sınıflar/fonksiyonlar, kritik kararlar, test kapsamı, açık sorular.

### Phase C — Özet ve hafızaya alma
- Her modül için 1-3 kısa `Memory` node'u oluşturulur.
- Mimari kararlar `Decision` label ile kaydedilir (örn. "Original video + frontend overlay", "Python tracker first").
- Tag'ler: `MergenVision`, `phase2`, `sprint-02`, `architecture`, `api`, `native-gpu`, `tracking`, `data-model`, `tests`, `important`, `compaction-retrieve`.
- DB'de zaten var olan özetleri güncelle; duplicate oluşturma.

### Phase D — Son rapor
- Kaç yeni memory eklendi / güncellendi.
- Ana başlık indexi (isim + tag listesi).
- Derinlemesine bakılması gereken açık noktalar (varsa).

## Kısıtlar ve Kurallar
- Sadece okuma; hiçbir dosya değiştirilmeyecek.
- AGENTS.md Section 36 (açık talimat gerektirir) bu görev için açık kullanıcı talimatıyla geçersiz sayılır; memory storage izinlidir.
- `prompt-memory-mcp` / doğrudan SQLite kullanılabilir.
- Sub-agent sayısı 6-8 arasında, paralel çalışacak.
- Ruflo / 21st kullanılmayacak.

## Onay

Plan onaylanırsa Phase A ile başlanır.
