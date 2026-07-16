# Phase 1 Sprint 01 — Minimal Identity Storage Foundation
## Revised Binding Plan

> Plan Mode çıktısıdır. Implementation yapılmamıştır. Human review sonrası Build Mode'a geçilecektir.
> Aktif repository: `/home/user/Workspace/MergenVisionPhase2v2`.

---

## 1. Verdict

**READY_FOR_HUMAN_REVIEW**

Gerekçe:
- Aktif repository `/home/user/Workspace/MergenVisionPhase2v2` olarak doğrulandı.
- Requirement dosyaları kullanıcı tarafından human-approved kabul edildi; hash gate yok.
- `architectureplan.md` eski taslak olarak işaretlendi; mevcut gereksinimler ve bu talimat önceliklidir.
- National-ID, Oracle, 10M-person kapsamı yasak.
- Sprint 01 MVP storage foundation olarak sınırlandırıldı.
- Dört tabloluk minimal schema, `IdentityStorageLifecycleService`, gerçek PG/MinIO/Qdrant bağlantıları ve lifecycle testleri için uygulanabilir plan hazır.

---

## 2. Repository Baseline

```text
root:        /home/user/Workspace/MergenVisionPhase2v2
branch:      main
HEAD:        fd74a51d0f2d25d45316f17d312deeb8b90a66be
HEAD message: requirement files changed
dirty:       clean (git status --short boş)
```

Mevcut source:
- `agents.md` (Engineering Constitution)
- `requirements/ProjectRequirements.md` (image/identity source-of-truth)
- `requirements/videorequirements.md` (additive video source-of-truth)
- `architectureplan.md` (eski taslak; çelişki durumunda göz ardı edilecek)
- `opensourcereferences/references.md`
- `whatwentwrong.md`
- `frontend/` (Sprint 01 scope dışında)

Backend, migration, domain, test, Docker veya API source'u henüz yoktur.

---

## 3. Requirement Approval and Hash Gate

`requirements/ProjectRequirements.md` ve `requirements/videorequirements.md` kullanıcı tarafından human-approved olarak kabul edilmiştir. Bu planda **requirement SHA-256 hash karşılaştırması yapılmayacak**, hash gate acceptance'a dahil edilmeyecek ve kaynak kodda hash kontrolü olmayacaktır.

---

## 4. Four-Table Compliance Matrix

| Tablo | Durum | Eksik / Uygulanacak |
|---|---|---|
| `face_identity` | MISSING | Migration, ORM mapping, domain entity, repository |
| `face_sample` | MISSING | Migration, ORM mapping, domain entity, repository |
| `process_record` | MISSING | Migration, ORM mapping, domain entity, repository |
| `recognition_result` | MISSING | Migration, ORM mapping, domain entity, repository |

**Sprint 01'de başka tablo yok.** Video tabloları, `inference_profile`, `idempotency_record`, `face_observation`, `process_event`, `outbox_event` ve diğer yardımcı tablolar ertelenmiştir.

### State Machine Uyumu

| Varlık | State'ler | Transition'lar |
|---|---|---|
| `FaceIdentity` | `anonymous`, `known` | `promote_to_known(name, metadata)`, `deactivate()` |
| `FaceSample` | `pending`, `active`, `failed`, `inactive` | `mark_active(bucket, key)`, `mark_failed(code)`, `mark_inactive()` |
| `ProcessRecord` | `processing`, `completed`, `failed` | `complete(face_count, details)`, `fail(error_code, details)` |
| `RecognitionResult` | immutable snapshot | insert-only; `known`, `anonymous`, `new_anonymous` |

---

## 5. Exact Changed-File Map

### 5.1 Proje yapılandırması

| # | Path | Amaç |
|---|---|---|
| 1 | `backend/pyproject.toml` | Python 3.12, SQLAlchemy 2.x, Alembic 1.x, asyncpg, qdrant-client, minio, pydantic-settings, pytest, pytest-asyncio, ruff, mypy. Kesin patch sürümleri official verification sonrası belirlenecek; şimdilik major/minor range yazılacak. |
| 2 | `backend/Makefile` | Target'lar: `phase1-sprint-01-static`, `phase1-sprint-01-postgres`, `phase1-sprint-01-minio`, `phase1-sprint-01-qdrant`, `phase1-sprint-01-lifecycle`, `phase1-sprint-01-failure`, `phase1-sprint-01-restart`, `phase1-sprint-01-acceptance`. |
| 3 | `backend/app/infrastructure/config.py` | `PydanticSettings`; PG URL, MinIO endpoint/credentials, Qdrant URL, bucket/collection sabitleri. |
| 4 | `backend/app/infrastructure/uuid7.py` | `generate_uuid7() -> uuid.UUID`. |
| 5 | `backend/app/infrastructure/clock.py` | `utc_now()` portu. |

### 5.2 Migration

| # | Path | Amaç |
|---|---|---|
| 6 | `backend/alembic.ini` | Alembic yapılandırması. |
| 7 | `backend/app/infrastructure/persistence/alembic/env.py` | Alembic environment. |
| 8 | `backend/app/infrastructure/persistence/alembic/versions/0001_identity_storage_foundation.py` | Tek migration: `face_identity`, `process_record`, `face_sample`, `recognition_result`. |

### 5.3 SQLAlchemy ORM Mappings

| # | Path | Sınıf |
|---|---|---|
| 9 | `backend/app/infrastructure/persistence/sqlalchemy/base.py` | `Base = declarative_base()` |
| 10 | `backend/app/infrastructure/persistence/sqlalchemy/models/face_identity.py` | `FaceIdentityOrm` |
| 11 | `backend/app/infrastructure/persistence/sqlalchemy/models/face_sample.py` | `FaceSampleOrm` |
| 12 | `backend/app/infrastructure/persistence/sqlalchemy/models/process_record.py` | `ProcessRecordOrm` |
| 13 | `backend/app/infrastructure/persistence/sqlalchemy/models/recognition_result.py` | `RecognitionResultOrm` |

### 5.4 Domain (Pure Python)

| # | Path | Sınıf / Dosya |
|---|---|---|
| 14 | `backend/app/domain/value_objects.py` | `FaceId`, `SampleId`, `ProcessId`, `ResultId`, `BoundingBox` |
| 15 | `backend/app/domain/errors.py` | `DomainError`, `InvalidTransitionError`, `ValidationError` |
| 16 | `backend/app/domain/entities/face_identity.py` | `FaceIdentity` |
| 17 | `backend/app/domain/entities/face_sample.py` | `FaceSample` |
| 18 | `backend/app/domain/entities/process_record.py` | `ProcessRecord` |
| 19 | `backend/app/domain/entities/recognition_result.py` | `RecognitionResult` |

### 5.5 Application Ports

| # | Path | Arayüz |
|---|---|---|
| 20 | `backend/app/application/ports/unit_of_work.py` | `UnitOfWork` |
| 21 | `backend/app/application/ports/repositories.py` | `FaceIdentityRepository`, `FaceSampleRepository`, `ProcessRepository`, `RecognitionResultRepository` |
| 22 | `backend/app/application/ports/object_store.py` | `ObjectStore` |
| 23 | `backend/app/application/ports/vector_store.py` | `VectorStore` |

### 5.6 Application Service

| # | Path | Sınıf |
|---|---|---|
| 24 | `backend/app/application/services/identity_storage_lifecycle_service.py` | `IdentityStorageLifecycleService` |

### 5.7 Infrastructure Adapters

| # | Path | Sınıf |
|---|---|---|
| 25 | `backend/app/infrastructure/persistence/sqlalchemy/session.py` | `create_async_engine`, `async_session_maker` |
| 26 | `backend/app/infrastructure/persistence/sqlalchemy/unit_of_work.py` | `SqlAlchemyUnitOfWork` |
| 27 | `backend/app/infrastructure/persistence/sqlalchemy/repositories/` | `SqlAlchemyFaceIdentityRepository`, `SqlAlchemyFaceSampleRepository`, `SqlAlchemyProcessRepository`, `SqlAlchemyRecognitionResultRepository` |
| 28 | `backend/app/infrastructure/storage/minio_adapter.py` | `MinIOObjectStore` |
| 29 | `backend/app/infrastructure/vectors/qdrant_adapter.py` | `QdrantVectorStore` |

### 5.8 Testler

| # | Path | Kapsam |
|---|---|---|
| 30 | `backend/tests/unit/domain/test_face_identity.py` | Domain transition tests |
| 31 | `backend/tests/unit/domain/test_face_sample.py` | Sample transition tests |
| 32 | `backend/tests/unit/domain/test_process_record.py` | Process transition tests |
| 33 | `backend/tests/unit/test_domain_dependency_boundary.py` | Domain dosyalarının SQLAlchemy/minio/qdrant/fastapi import etmediği doğrulanır |
| 34 | `backend/tests/integration/persistence/test_migrations.py` | Tek migration upgrade + schema introspection |
| 35 | `backend/tests/integration/storage/test_minio_adapter.py` | Gerçek MinIO'ya valid WebP upload/stat |
| 36 | `backend/tests/integration/vectors/test_qdrant_adapter.py` | Gerçek Qdrant collection/upsert/search/filter |
| 37 | `backend/tests/integration/lifecycle/test_identity_storage_lifecycle.py` | `new_anonymous -> anonymous -> known` |
| 38 | `backend/tests/integration/lifecycle/test_multiple_samples.py` | Birden fazla sample tek faceId altında |
| 39 | `backend/tests/integration/lifecycle/test_inactive_rejection.py` | Inaktif identity/sample Qdrant candidate olarak kabul edilmez |
| 40 | `backend/tests/integration/lifecycle/test_failure_paths.py` | MinIO failure, Qdrant failure |
| 41 | `backend/tests/integration/lifecycle/test_restart_persistence.py` | `docker compose restart postgres minio qdrant` sonrası veri kalıcılığı |

### 5.9 Docker ve Compose

| # | Path | Amaç |
|---|---|---|
| 42 | `docker-compose.yml` | `postgres:16-alpine`, `minio/minio:latest` (veya official verification sonrası sabit minor), `qdrant/qdrant:latest` (veya official verification sonrası sabit minor); named volumes; healthchecks. |
| 43 | `backend/Dockerfile` | `python:3.12-slim` multi-stage build. |

### 5.10 Fixtures

| # | Path | Amaç |
|---|---|---|
| 44 | `backend/tests/fixtures/valid_crop.webp` | Küçük geçerli WebP görüntü fixture'ı. |
| 45 | `backend/tests/fixtures/embedding_fixtures.py` | Deterministik 512-D L2-normalized test vektörleri. |

### 5.11 Dokümantasyon

| # | Path | Amaç |
|---|---|---|
| 46 | `docs/implementation/CURRENT_SPRINT.md` | Sprint 01 hedef, sınır ve durum. Build Mode onayı sonrası oluşturulacak. |
| 47 | `docs/implementation/IMPLEMENTATION_DETAILS.md` | Kod kararları. Sprint bitiminde güncellenecek. |
| 48 | `docs/implementation/review_packages/SPRINT-001-CODE-REVIEW-PACKAGE.md` | Review package. Sprint bitiminde oluşturulacak. |

---

## 6. One-Migration Plan

### 6.1 Tek revision

```text
backend/app/infrastructure/persistence/alembic/versions/0001_identity_storage_foundation.py
```

### 6.2 Oluşturma sırası

1. `face_identity`
2. `process_record`
3. `face_sample`
4. `recognition_result`

### 6.3 Constraints ve indexes

Promptta belirtilen her CHECK constraint, FK `ON DELETE RESTRICT`, explicit index ve composite unique index migration'da tanımlanır.

### 6.4 Downgrade

Downgrade Sprint 01 acceptance'a dahil değildir. Eğer eklenecekse:

```text
DROP INDEX recognition_result_process_id_result_index_idx
DROP TABLE recognition_result
DROP INDEX face_sample_face_id_sample_state_idx
DROP INDEX face_sample_bucket_key_unique_idx
DROP TABLE face_sample
DROP INDEX process_record_process_type_created_at_idx
DROP INDEX process_record_status_created_at_idx
DROP TABLE process_record
DROP INDEX face_identity_status_is_active_idx
DROP INDEX face_identity_created_at_idx
DROP TABLE face_identity
```

`DROP ... CASCADE` kullanılmaz.

---

## 7. Domain / Infrastructure Boundary Plan

### 7.1 Domain kuralları

Domain dosyaları pure Python'dur. Şu import'ları içermez:

```text
sqlalchemy
asyncpg
minio
qdrant_client
fastapi
```

Domain transition örnekleri:

```python
# backend/app/domain/entities/face_identity.py
class FaceIdentity:
    def promote_to_known(self, display_name: str, metadata: dict) -> None:
        ...

    def deactivate(self) -> None:
        ...

# backend/app/domain/entities/face_sample.py
class FaceSample:
    def mark_active(self, bucket: str, key: str) -> None:
        ...

    def mark_failed(self, failure_code: str) -> None:
        ...

    def mark_inactive(self) -> None:
        ...

# backend/app/domain/entities/process_record.py
class ProcessRecord:
    def complete(self, face_count: int, details_json: dict) -> None:
        ...

    def fail(self, error_code: str, details_json: dict) -> None:
        ...
```

### 7.2 Infrastructure mapping

SQLAlchemy declarative modelleri yalnızca `backend/app/infrastructure/persistence/sqlalchemy/models/` altındadır. ORM modelleri data mapping yapar; business rule'ları tekrar etmez. Repository adapter'ları domain entity ↔ ORM row dönüşümünü üstlenir.

### 7.3 Unit of Work

```python
# backend/app/infrastructure/persistence/sqlalchemy/unit_of_work.py
class SqlAlchemyUnitOfWork:
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, ...): ...
    async def commit(self): ...
    async def rollback(self): ...
```

SQLAlchemy 2.0 async pattern doğrulandı: `create_async_engine("postgresql+asyncpg://...")` ve `async_sessionmaker.begin()` transaction context manager.

---

## 8. MinIO Adapter Plan

### 8.1 Bucket

Sprint 01'de yalnızca:

```text
mergenvision-face-samples
```

Bucket adı environment variable ile yapılandırılabilir.

### 8.2 Object key

```text
faces/{faceId}/{sampleId}/aligned.webp
```

Key yalnızca UUID ve teknik segment içerir; isim/metadata içermez.

### 8.3 Adapter

```python
# backend/app/infrastructure/storage/minio_adapter.py
class MinIOObjectStore(ObjectStore):
    def __init__(self, client, bucket: str): ...

    async def upload(self, key: str, data: bytes, content_type: str) -> ObjectStat:
        # MinIO Python SDK synchronous; isolate with asyncio.to_thread
        ...

    async def stat(self, key: str) -> ObjectStat | None:
        ...
```

### 8.4 Kurallar

- Geçerli küçük WebP fixture kullanılır (`b"fake-webp"` değil).
- Upload sonrası `stat` ile varlık ve boyut doğrulanır.
- Custom SHA metadata/checksum comparison Sprint 01'de yok.
- Blocking SDK çağrıları `asyncio.to_thread` ile isolate edilir.

---

## 9. Qdrant Adapter Plan

### 9.1 Collection

```text
collection: face_samples_v1
vector dimension: 512
distance: cosine
```

### 9.2 Upsert

```python
await self.client.upsert(
    collection_name="face_samples_v1",
    points=[
        models.PointStruct(
            id=str(sample_id),
            vector=embedding,
            payload={"face_id": str(face_id), "active": True},
        )
    ],
    wait=True,
)
```

`upsert` ve `PointStruct` Qdrant Python client official docs ile doğrulandı.

### 9.3 Search

Mevcut non-deprecated method `query_points`:

```python
result = await self.client.query_points(
    collection_name="face_samples_v1",
    query=embedding,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="active", match=models.MatchValue(value=True)
            )
        ]
    ),
    limit=top_k,
    with_payload=True,
)
```

### 9.4 Kurallar

- Vektör uzunluğu 512 ve tüm değerler finite olmalı; adapter validate eder.
- Payload sadece `face_id` ve `active` içerir.
- Name, metadata, MinIO key, process data payload'a yazılmaz.
- Search sonucu PG'de `face_identity` ve `face_sample` active doğrulamasından geçer.

---

## 10. IdentityStorageLifecycleService Plan

### 10.1 Kontrat

```python
# backend/app/application/services/identity_storage_lifecycle_service.py
class IdentityStorageLifecycleService:
    async def store_new_identity(
        self,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        process_type: str = "image_recognize",
    ) -> ProcessRecord:
        """Yeni anonymous identity, sample, MinIO crop, Qdrant vector ve new_anonymous result oluşturur."""

    async def recognize_existing(
        self,
        embedding: Sequence[float],
        bbox: BoundingBox,
        process_type: str = "image_recognize",
    ) -> ProcessRecord:
        """Mevcut active sample/identity arar ve anonymous/known result oluşturur."""

    async def enroll_identity(
        self,
        face_id: FaceId,
        display_name: str,
        metadata: dict,
    ) -> FaceIdentity:
        """Anonymous identity'yi known yapar; faceId korur."""
```

### 10.2 Yeni identity akışı

```text
1. PG transaction:
   - process_record(status=processing)
   - face_identity(status=anonymous, is_active=True)
   - face_sample(state=pending)
   COMMIT

2. MinIO:
   - upload faces/{faceId}/{sampleId}/aligned.webp

3. Qdrant:
   - upsert point_id=sampleId, payload={face_id, active=True}

4. PG transaction:
   - face_sample.mark_active(bucket, key)
   - recognition_result(status=new_anonymous, bbox, match_confidence)
   - process_record.complete(face_count=1)
   COMMIT
```

### 10.3 Mevcut eşleşme akışı

```text
1. PG transaction:
   - process_record(status=processing)
   COMMIT

2. Qdrant:
   - query_points active=True, limit=1 (veya k > 1)

3. PG:
   - candidate face_identity + face_sample active doğrula
   - recognition_result(status=anonymous veya known, bbox, match_confidence)
   - process_record.complete(face_count=1)
```

### 10.4 Enrollment akışı

```text
1. PG transaction:
   - SELECT face_identity FOR UPDATE
   - require active + anonymous + expected version
   - promote_to_known(display_name, metadata)
   - version += 1
   COMMIT
```

### 10.5 Failure behavior

- MinIO upload fails: `face_sample.mark_failed(...)`, `process_record.fail(...)`, recognition result oluşturulmaz.
- Qdrant upsert fails: `face_sample.mark_failed(...)`, `process_record.fail(...)`, recognition result oluşturulmaz.
- Otomatik background recovery, outbox, reconciliation, saga yok.
- Retry yeni process + sample oluşturabilir.

---

## 11. Step-by-Step TDD Plan

Her adım için failing test → command → expected failure → minimum source change → pass command → kanıtlanan davranış.

### Step 1 — Project skeleton + dependency boundary

- **Failing test:** `backend/tests/unit/test_domain_dependency_boundary.py` — domain dosyalarının SQLAlchemy/minio/qdrant/fastapi import etmediği doğrulanır.
- **Command:** `cd backend && python -m pytest tests/unit/test_domain_dependency_boundary.py -v`
- **Expected failure:** `ModuleNotFoundError` / domain dosyaları yok.
- **Minimum change:** `pyproject.toml` + domain `__init__.py` dosyaları + boş entity stub'ları.
- **Pass command:** `python -m pytest tests/unit/test_domain_dependency_boundary.py -v`
- **Kanıtlar:** Domain pure Python sınırı.

### Step 2 — UUIDv7 generator

- **Failing test:** `tests/unit/infrastructure/test_uuid7.py`
- **Command:** `pytest tests/unit/infrastructure/test_uuid7.py -v`
- **Expected failure:** `ImportError`
- **Minimum change:** `backend/app/infrastructure/uuid7.py`
- **Pass command:** `pytest tests/unit/infrastructure/test_uuid7.py -v`
- **Kanıtlar:** UUIDv7 generation.

### Step 3 — Domain transition tests

- **Failing test:** `tests/unit/domain/test_face_identity.py`, `test_face_sample.py`, `test_process_record.py`
- **Command:** `pytest tests/unit/domain/ -v`
- **Expected failure:** `NotImplementedError` / assertion
- **Minimum change:** `FaceIdentity`, `FaceSample`, `ProcessRecord` domain entities + transition metotları.
- **Pass command:** `pytest tests/unit/domain/ -v`
- **Kanıtlar:** State machine contract.

### Step 4 — Single migration upgrade

- **Failing test:** `tests/integration/persistence/test_migrations.py::test_upgrade_head`
- **Command:** `cd backend && alembic upgrade head && pytest tests/integration/persistence/test_migrations.py -v`
- **Expected failure:** `ProgrammingError` relation does not exist.
- **Minimum change:** `0001_identity_storage_foundation.py` + Alembic env.
- **Pass command:** `alembic upgrade head && pytest tests/integration/persistence/test_migrations.py -v`
- **Kanıtlar:** Real PostgreSQL'de 4 tablo oluşur.

### Step 5 — Schema introspection

- **Failing test:** `test_migrations.py::test_required_constraints_and_indexes`
- **Command:** `pytest tests/integration/persistence/test_migrations.py::test_required_constraints_and_indexes -v`
- **Expected failure:** AssertionError
- **Minimum change:** Migration dosyasına named CHECK, FK index, partial unique index ekle.
- **Pass command:** `pytest tests/integration/persistence/test_migrations.py -v`
- **Kanıtlar:** Prompttaki constraint/index contract'ı.

### Step 6 — Repository adapters

- **Failing test:** `tests/integration/persistence/test_repositories.py`
- **Command:** `pytest tests/integration/persistence/test_repositories.py -v`
- **Expected failure:** Adapter missing.
- **Minimum change:** SQLAlchemy ORM models + `SqlAlchemyUnitOfWork` + repository adapter'ları.
- **Pass command:** `pytest tests/integration/persistence/test_repositories.py -v`
- **Kanıtlar:** PG read/write, UoW commit/rollback.

### Step 7 — MinIO valid WebP upload

- **Failing test:** `tests/integration/storage/test_minio_adapter.py::test_upload_and_stat_valid_webp`
- **Command:** `docker compose up -d minio && pytest tests/integration/storage/test_minio_adapter.py -v`
- **Expected failure:** ConnectionRefused / adapter missing.
- **Minimum change:** `MinIOObjectStore` + `docker-compose.yml` MinIO service.
- **Pass command:** `docker compose up -d minio && pytest tests/integration/storage/test_minio_adapter.py -v`
- **Kanıtlar:** Gerçek MinIO'ya valid WebP upload ve stat.

### Step 8 — Qdrant upsert/search/filter

- **Failing test:** `tests/integration/vectors/test_qdrant_adapter.py::test_query_points_with_active_filter`
- **Command:** `docker compose up -d qdrant && pytest tests/integration/vectors/test_qdrant_adapter.py -v`
- **Expected failure:** Collection does not exist / adapter missing.
- **Minimum change:** `QdrantVectorStore` + `docker-compose.yml` Qdrant service.
- **Pass command:** `docker compose up -d qdrant && pytest tests/integration/vectors/test_qdrant_adapter.py -v`
- **Kanıtlar:** 512-D cosine collection, UUID point ID, `query_points` active filter.

### Step 9 — First fixture -> new_anonymous

- **Failing test:** `tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_store_new_identity_returns_new_anonymous`
- **Command:** `docker compose up -d postgres minio qdrant && alembic upgrade head && pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_store_new_identity_returns_new_anonymous -v`
- **Expected failure:** Service missing.
- **Minimum change:** `IdentityStorageLifecycleService.store_new_identity`
- **Pass command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_store_new_identity_returns_new_anonymous -v`
- **Kanıtlar:** Cross-store yeni anonymous lifecycle.

### Step 10 — Same vector -> same faceId, anonymous

- **Failing test:** `test_identity_storage_lifecycle.py::test_same_vector_returns_anonymous`
- **Command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_same_vector_returns_anonymous -v`
- **Expected failure:** Search/PG validation eksik.
- **Minimum change:** `recognize_existing` implementasyonu.
- **Pass command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_same_vector_returns_anonymous -v`
- **Kanıtlar:** Aynı faceId ile anonymous sonuç.

### Step 11 — Enrollment preserves faceId, later known

- **Failing test:** `test_identity_storage_lifecycle.py::test_enroll_preserves_face_id`
- **Command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_enroll_preserves_face_id -v`
- **Expected failure:** Enrollment service eksik.
- **Minimum change:** `enroll_identity` + optimistic locking.
- **Pass command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_enroll_preserves_face_id -v`
- **Kanıtlar:** faceId korunur, identity known olur.

### Step 12 — Old result snapshot unchanged

- **Failing test:** `test_identity_storage_lifecycle.py::test_old_result_snapshot_remains_new_anonymous`
- **Command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_old_result_snapshot_remains_new_anonymous -v`
- **Expected failure:** Result update ediliyor.
- **Minimum change:** Repository'de result update metodu kaldırılır/engellenir.
- **Pass command:** `pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py::test_old_result_snapshot_remains_new_anonymous -v`
- **Kanıtlar:** Immutable snapshot.

### Step 13 — Multiple samples per faceId

- **Failing test:** `tests/integration/lifecycle/test_multiple_samples.py::test_multiple_samples_same_face`
- **Command:** `pytest tests/integration/lifecycle/test_multiple_samples.py -v`
- **Expected failure:** Constraint / repository eksik.
- **Minimum change:** Sample repo + unique bucket/key index.
- **Pass command:** `pytest tests/integration/lifecycle/test_multiple_samples.py -v`
- **Kanıtlar:** Bir faceId altında birden fazla sample.

### Step 14 — Inactive rejection

- **Failing test:** `tests/integration/lifecycle/test_inactive_rejection.py::test_inactive_identity_not_returned`
- **Command:** `pytest tests/integration/lifecycle/test_inactive_rejection.py -v`
- **Expected failure:** Inaktif candidate kabul ediliyor.
- **Minimum change:** `recognize_existing` PG active check.
- **Pass command:** `pytest tests/integration/lifecycle/test_inactive_rejection.py -v`
- **Kanıtlar:** Inaktif identity/sample search sonucuna dahil edilmez.

### Step 15 — Failure paths

- **Failing test:** `tests/integration/lifecycle/test_failure_paths.py::test_minio_failure_no_completed_result`, `test_qdrant_failure_no_completed_result`
- **Command:** `pytest tests/integration/lifecycle/test_failure_paths.py -v`
- **Expected failure:** Failure durumunda completed result oluşuyor.
- **Minimum change:** Service error handling + sample fail + process fail.
- **Pass command:** `pytest tests/integration/lifecycle/test_failure_paths.py -v`
- **Kanıtlar:** MinIO/Qdrant failure sahte başarı üretmez.

### Step 16 — Restart persistence

- **Failing test:** `tests/integration/lifecycle/test_restart_persistence.py::test_data_survives_restart`
- **Command:** `docker compose restart postgres minio qdrant && pytest tests/integration/lifecycle/test_restart_persistence.py -v`
- **Expected failure:** Data lost.
- **Minimum change:** Named volumes in `docker-compose.yml`.
- **Pass command:** `docker compose restart postgres minio qdrant && pytest tests/integration/lifecycle/test_restart_persistence.py -v`
- **Kanıtlar:** Restart sonrası veri korunur.

### Step 17 — Acceptance

- **Command:** `make phase1-sprint-01-acceptance`
- **Expected:** Tüm target'lar PASS.

---

## 12. Acceptance Matrix

### Makefile targets

```makefile
phase1-sprint-01-static:
	cd backend && ruff check .
	cd backend && ruff format --check .
	cd backend && mypy .

phase1-sprint-01-postgres:
	docker compose up -d postgres --wait
	cd backend && alembic upgrade head
	cd backend && python -m pytest tests/integration/persistence/test_migrations.py -v

phase1-sprint-01-minio:
	docker compose up -d minio --wait
	cd backend && python -m pytest tests/integration/storage/test_minio_adapter.py -v

phase1-sprint-01-qdrant:
	docker compose up -d qdrant --wait
	cd backend && python -m pytest tests/integration/vectors/test_qdrant_adapter.py -v

phase1-sprint-01-lifecycle:
	docker compose up -d postgres minio qdrant --wait
	cd backend && alembic upgrade head
	cd backend && python -m pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py tests/integration/lifecycle/test_multiple_samples.py tests/integration/lifecycle/test_inactive_rejection.py -v

phase1-sprint-01-failure:
	docker compose up -d postgres minio qdrant --wait
	cd backend && alembic upgrade head
	cd backend && python -m pytest tests/integration/lifecycle/test_failure_paths.py -v

phase1-sprint-01-restart:
	docker compose up -d postgres minio qdrant --wait
	cd backend && alembic upgrade head
	cd backend && python -m pytest tests/integration/lifecycle/test_restart_persistence.py -v

phase1-sprint-01-acceptance: phase1-sprint-01-static
	cd backend && python -m pytest tests/unit/test_domain_dependency_boundary.py -v
	$(MAKE) phase1-sprint-01-postgres
	$(MAKE) phase1-sprint-01-minio
	$(MAKE) phase1-sprint-01-qdrant
	$(MAKE) phase1-sprint-01-lifecycle
	$(MAKE) phase1-sprint-01-failure
	$(MAKE) phase1-sprint-01-restart
	cd backend && ruff check . && mypy .
	git diff --check
```

Acceptance içermez:
- requirement SHA komutları
- migration downgrade-to-base
- outbox/reconciliation testleri
- model SHA kontrolleri
- git commit/push
- `docker compose down`, `down -v`, volume silme

---

## 13. Basic Failure Behavior

| Kırılma noktası | Persisted state | Davranış |
|---|---|---|
| MinIO upload fails | `face_sample.state=failed`, `process_record.status=failed` | Recognition result oluşturulmaz. Retry yeni process/sample oluşturabilir. |
| Qdrant upsert fails | `face_sample.state=failed`, `process_record.status=failed` | Recognition result oluşturulmaz. |
| PG transaction fails | Rollback; hiçbir şey persist olmaz | Yeni deneme baştan başlar. |
| Search candidate inactive | PG active check reddeder | Sonuçta o candidate kullanılmaz. |

Otomatik recovery, outbox worker, reconciliation, saga, dead-letter Sprint 01'de yok.

---

## 14. Deferred Features List

Sprint 01'de **yapılmayacak** özellikler:

- `inference_profile` tablosu ve model profilleri
- `idempotency_record` ve raw idempotency-key hashing platformu
- `face_observation` tablosu ve video observation detayları
- `process_event` append-only audit event stream
- `outbox_event`, outbox worker, dead-letter queue
- reconciliation scanner/platform
- distributed saga abstraction
- model artifact SHA fields
- requirement SHA checks
- MinIO custom SHA metadata/checksum machinery
- merge/canonical-redirect implementation
- concurrent same-unknown race solution
- migration downgrade-to-base acceptance
- API endpoint'leri (FastAPI router/controller)
- UI
- Gerçek face detection / alignment / embedding
- GPU inference
- Video processing / tracking
- Production threshold kalibrasyonu
- Orphan cleanup
- National-ID, Oracle, 10M-person kapsamı

---

## 15. Proposed `docs/implementation/CURRENT_SPRINT.md` Content

```markdown
# Current Sprint: Phase 1 Sprint 01 — Minimal Identity Storage Foundation

## Objective

Establish real PostgreSQL, MinIO and Qdrant connections and prove the persistent identity lifecycle using a deterministic test vector and a tiny valid image fixture.

## In Scope

- Four PostgreSQL tables: `face_identity`, `face_sample`, `process_record`, `recognition_result`.
- One initial Alembic migration.
- Pure Python domain layer with explicit state transitions.
- SQLAlchemy 2.0 async repository adapters.
- MinIO adapter for `mergenvision-face-samples` bucket.
- Qdrant adapter for `face_samples_v1` cosine 512-D collection.
- `IdentityStorageLifecycleService`:
  - store_new_identity
  - recognize_existing
  - enroll_identity
- Integration tests on real services.

## Out of Scope

- API endpoints / FastAPI
- UI
- Real detection / recognition / alignment / GPU inference
- Video / tracking
- `inference_profile`, `idempotency_record`, `face_observation`, `process_event`, `outbox_event`
- Outbox, saga, reconciliation, dead-letter
- Model artifact SHA, requirement SHA checks
- National-ID, Oracle, 10M-person

## Acceptance

Run:

```bash
make phase1-sprint-01-acceptance
```

Expected: all targets PASS.

## Status

Planned — awaiting Build Mode start after human review.
```

---

## 16. Build Mode First Action

### First failing test

`backend/tests/unit/test_domain_dependency_boundary.py`

### First production files

1. `backend/pyproject.toml`
2. `backend/app/domain/__init__.py`
3. `backend/app/domain/entities/__init__.py`
4. `backend/app/domain/entities/face_identity.py` (stub)
5. `backend/app/domain/entities/face_sample.py` (stub)
6. `backend/app/domain/entities/process_record.py` (stub)
7. `backend/app/domain/entities/recognition_result.py` (stub)
8. `backend/app/domain/value_objects.py`
9. `backend/app/domain/errors.py`

### First command

```bash
cd backend && python -m pytest tests/unit/test_domain_dependency_boundary.py -v
```

Expected initial result: `FAILED` (domain files missing or forbidden imports present).

---

## 17. MCP / Skill Accountability

| Tool/Skill | Kullanım | Sonuç |
|---|---|---|
| `Read` | `agents.md`, requirement'ler, `architectureplan.md`, `references.md`, `whatwentwrong.md` | ✅ Kullanıldı |
| `Bash` | Repo baseline doğrulama | ✅ Kullanıldı |
| `AskUserQuestion` | Repo yolu doğrulama | ✅ Kullanıldı |
| `codebase-memory-mcp` | Aktif repo'da backend source yok; eski repo indeksleri kullanılmadı | ⚠️ SKIPPED — backend boş; Build Mode'da yeni source oluşturuldukça gerekirse kullanılacak |
| `context7` | Qdrant Python client `query_points`/`upsert` API ve SQLAlchemy 2.0 async session doğrulaması | ✅ Kullanıldı (`/qdrant/qdrant-client`, `/websites/sqlalchemy_en_20`) |
| `deepwiki` | Official docs yeterli | ⚠️ SKIPPED_NOT_NEEDED |
| `exa` | Official docs yeterli | ⚠️ SKIPPED_NOT_NEEDED |
| `postman` | API endpoint Sprint 01 scope dışında | ⚠️ SKIPPED_NOT_RELEVANT |
| `playwright` | UI Sprint 01 scope dışında | ⚠️ SKIPPED_NOT_RELEVANT |
| `21st` | Yasak | 🚫 FORBIDDEN_NOT_USED |
| Ruflo | Yasak | 🚫 FORBIDDEN_NOT_USED |
| `using-superpowers` | Workflow governance ve tool seçimi | ✅ Kullanıldı |
| `brainstorming` | Eski plandan rejected feature listesi çıkarımı | ✅ Kullanıldı |
| `writing-plans` | Bu revised plan | ✅ Kullanıldı |

---

## Sonuç

Bu revised plan, `/home/user/Workspace/MergenVisionPhase2v2` reposunda Phase 1 Sprint 01 — Minimal Identity Storage Foundation için uygulanabilir, test-driven bir yol haritasıdır. Dört tablo, `IdentityStorageLifecycleService`, gerçek PG/MinIO/Qdrant adaptörleri ve temel failure path'leri içerir. Obsolete kapsam (outbox, saga, reconciliation, inference profile, SHA machinery, video, API, UI) çıkarılmıştır.

**READY_FOR_HUMAN_REVIEW**
