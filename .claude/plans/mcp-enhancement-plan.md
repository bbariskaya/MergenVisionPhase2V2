# MCP Enhancement Plan — MergenVision Phase2v2

## 1. Mevcut Durum Analizi

### 1.1 codebase-memory-mcp
- **Durum:** Kurulu, bağlı, indexlenmiş.
- **Proje adı:** `home-user-Workspace-MergenVisionPhase2v2`
- **İstatistikler:** 2.022 node, 8.285 edge, 105 dosya, 13 farklı node label.
- **Mevcut araçlar:** `get_architecture`, `search_graph`, `search_code`, `trace_path`, `query_graph`, `get_code_snippet`, `detect_changes`, `index_repository`, `manage_adr`, `ingest_traces`.
- **Gözlemlenen eksiklikler:**
  - Her kod değişikliğinden sonra graph otomatik güncellenmiyor.
  - ADR henüz kullanılmamış (`adr_present: false`).
  - Doğal dil sorguları için tek bir orchestrator yok; birden fazla tool elle çağrılıyor.
  - Cross-repo intelligence açık değil; `MergenVisionDemo` ile pattern karşılaştırması yapılamıyor.
  - Index persistence artifact (`.codebase-memory/graph.db.zst`) oluşturulmamış.

### 1.2 prompt-memory-mcp
- **Durum:** 1.893 Memory node, 13 Decision node.
- **Eksiklikler:**
  - Embeddings kapalı (`use_embeddings: false`), semantic recall sınırlı.
  - Codebase-memory ile senkronizasyon yok.

### 1.3 Ruflo Entegrasyonu
- `CLAUDE.md` içinde `ToolSearch` ile ruflo MCP tool'larının bulunması öneriliyor.
- Ruflo'nun `memory_store`, `memory_search`, `hooks_route`, `swarm_init`, `agent_spawn` gibi yetenekleri mevcut.
- Bu plan ruflo'nun var olan yeteneklerini de kullanarak, özellikle `hooks_route` ile otomatik refresh ve `swarm_init`/`agent_spawn` ile parallel analysis workflow'ları güçlendirmeyi hedefliyor.

## 2. Hedefler

1. **Akıllı Context Retrieval:** Kullanıcı sorusuna göre en alakalı kod parçalarını otomatik bulan ve özetleyen bir orchestrator.
2. **Otomatik Graph Refresh:** Kod değişiklikleri sonrası graph'in sürekli güncel kalmasını sağlayan mekanizma.
3. **Impact Analysis Workflow:** Bir değişiklik önerisinin etki alanını otomatik haritalayan rapor.
4. **ADR & Karar Yönetimi:** Önemli mimari kararların codebase-memory ADR'lerine ve prompt-memory Decision'larına kaydedilmesi.
5. **Cross-Repo Intelligence:** `MergenVisionDemo` ile karşılaştırmalı pattern retrieval ve kod önerisi.

## 3. Detaylı Plan

### 3.1 Akıllı Context Retrieval Orchestrator

**Dosyalar:**
- `scripts/mcp_context.py`: Ana Python script (stdio MCP client).
- `.claude/skills/context-orchestrator/SKILL.md`: Claude skill tanımı.

**Akış:**
1. Kullanıcı sorusunu al.
2. `search_graph(query=..., limit=20)` ile semantik/graph araması yap.
3. `search_code(pattern=..., limit=20)` ile tam metin desteği ekle.
4. Bulunan top semboller için `trace_path(function_name=..., mode="calls", depth=2)` çalıştır.
5. Sonuçları dosya/sembol bazında deduplicate et ve struct öneme göre sırala (definition > popular function > test).
6. En alakalı 10–15 dosya/sembolü döndür; istersen `get_code_snippet` ile içerikleri de getir.
7. Çıktı: markdown tablo veya JSON.

**Başarı kriteri:** Kullanıcı "şu fonksiyonu bul" veya "bu feature nerede implemente" dediğinde tek tool çağrısıyla yapılandırılmış sonuç dönmeli.

**Kullanım:**
- `make mcp-context QUERY="bulk enrollment"`
- `/context-orchestrator "bulk enrollment"`

### 3.2 Otomatik Graph Refresh Hook

**Dosyalar:**
- `scripts/mcp_refresh.py`: Değişiklikleri algılayıp index'leyen script.
- `scripts/mcp_status.py`: Graph index status checker.
- `Makefile` hedefleri: `make mcp-refresh`, `make mcp-status`.
- Opsiyonel: `.git/hooks/pre-commit` veya Claude hook entegrasyonu.

**Akış:**
1. `git diff --name-only` ve `git ls-files --others` ile değişen ve untracked dosyaları bul.
2. Eğer çok fazla dosya değişmişse (`>50`) `index_repository(mode="moderate")` çalıştır.
3. Az sayıda dosya için `index_repository(mode="fast")` çalıştır.
4. Eğer mimari dosyalar (domain, ports, api/routes) değiştiyse ADR kontrolü tetikle.
5. `.codebase-memory/graph.db.zst` artifact'ını `persistence=True` ile yeniden yaz.

**Başarı kriteri:** Her büyük değişiklik seti sonrası graph'in güncel olduğu doğrulanabilmeli.

**Kullanım:**
- `make mcp-refresh`
- `make mcp-refresh MCP_REFRESH_SINCE=HEAD~3`
- `make mcp-status`

### 3.3 Impact Analysis Workflow

**Dosyalar:**
- `scripts/mcp_impact.py`: stdio MCP client ile impact raporu üreten script.
- `.claude/skills/impact-analysis/SKILL.md`: Claude skill tanımı.

**Akış:**
1. Girdi: dosya adı, fonksiyon adı veya sembol adı.
2. `search_graph` ile hedefi çöz, `qualified_name` al.
3. `trace_path(function_name=..., mode="calls", direction="both", depth=3, include_tests=True)` ile etki alanını bul.
4. `query_graph` ile hedef ve ilgili fonksiyonların complexity, loop_depth, hotspot metriklerini çek.
5. Entry point'leri (route, worker entry) ve testleri listele.
6. Markdown rapor üret: etkilenen dosyalar, risk skoru, test coverage önerileri.

**Başarı kriteri:** Bir refactor/değişiklik önerildiğinde 60 saniye içinde etki raporu sunulabilmeli.

**Kullanım:**
- `make mcp-impact TARGET="enroll_batch"`
- `/impact-analysis enroll_batch`

### 3.4 ADR & Karar Yönetimi

**Kararlar:**
- Person aggregate root ve redirect/alias tasarımı.
- Bulk enrollment: önce Python service scaffold, sonra native GPU batch.
- Phase1 → Phase2 identity continuity: shared Qdrant collection + canonical resolution.
- Smart context retrieval orchestrator tasarımı.

**Araç:** `manage_adr(mode="update")`.

**Dosyalar:**
- `scripts/mcp_adr.py`: ADR oluşturma/güncelleme/okuma helper'ı.

**Akış:**
1. Her büyük karar için ADR oluştur.
2. Plan moduna başlamadan önce ilgili ADR'leri `manage_adr(mode="get")` ile retrieve et.
3. Karar değişikliğinde ADR'yi güncelle.

**Başarı kriteri:** ADR codebase-memory'de `manage_adr(mode="get")` ile görüntülenebilmeli.

**Kullanım:**
- `python scripts/mcp_adr.py --get`
- `python scripts/mcp_adr.py --init`
- `python scripts/mcp_adr.py --update docs/adr.md`

### 3.5 Cross-Repo Intelligence

**Adımlar:**
1. `MergenVisionDemo` reposunu `index_repository(mode="moderate", persistence=True)` ile indexle.
2. Her iki proje için `index_repository(mode="cross-repo-intelligence", target_projects=["*"])` çalıştır.
3. `query_graph` ile `CROSS_HTTP_CALLS`, `CROSS_ASYNC_CALLS`, `CROSS_CHANNEL` edge'lerini sorgula.
4. Context orchestrator'a cross-repo arama seçeneği ekle.

**Dosyalar:**
- `scripts/mcp_cross_repo.py`: Cross-repo indexing helper.

**Başarı kriteri:** Örneğin "MergenVisionDemo'daki bulk enrollment pattern'ini getir" sorusuyla ilgili dosyalar bulunabilmeli.

**Kullanım:**
- `python scripts/mcp_cross_repo.py --repo /home/user/MergenVisionDemo --name MergenVisionDemo --mode moderate`
- `python scripts/mcp_cross_repo.py --cross-repo-only`

## 4. Fazlar

### Faz A: Temel Altyapı (öncelik: yüksek)
- A1: `scripts/mcp_context.py` yaz.
- A2: Context orchestrator skill tanımını oluştur.
- A3: `scripts/mcp_refresh.py` yaz.
- A4: Makefile hedefleri ekle.

### Faz B: Analiz & Karar Yönetimi (öncelik: orta)
- B1: `scripts/mcp_impact.py` yaz.
- B2: Mevcut önemli kararlar için ADR oluştur.
- B3: Plan modu workflow'una ADR retrieve adımı ekle.

### Faz C: İleri Seviye (öncelik: düşük/ileri tarihli)
- C1: MergenVisionDemo'yu indexle.
- C2: Cross-repo intelligence aç.
- C3: Context orchestrator'a cross-repo modülü ekle.
- C4: prompt-memory embeddings'i aktif et (eğer performans kabul edilebilirse).

## 5. Riskler ve Alınacak Kararlar

1. **Performans:** `index_repository(mode="full")` her değişiklikte çok yavaş olabilir. Çözüm: fast/moderate modları ve sadece değişen dosyaları indexleme.
2. **Gizlilik/Güvenlik:** `persistence=True` ile `.codebase-memory/graph.db.zst` dosyası oluşur; repo içinde kalabilir veya `.gitignore` eklenebilir.
3. **prompt-memory explicit-memory-only kuralı:** Kararları ADR'ye kaydederken prompt-memory'e otomatik yazmaktan kaçınılmalı; sadece kullanıcı "remember" dediğinde yazılmalı.
4. **Skill vs Script:** Skill kullanıcı deneyimi daha iyi ama test edilmesi zor. İlk aşamada Python script'leri tercih edilecek, sonrasında skill'lere dönüştürülecek.
5. **Cross-repo indexing:** `MergenVisionDemo` farklı bir yerde (`/home/user/MergenVisionDemo`); indexleme uzun sürebilir.

## 6. Başarı Kriterleri (Genel)

- `make mcp-context QUERY="..."` tek komutla yapılandırılmış sonuç döndürmeli.
- `make mcp-refresh` her büyük değişiklik seti sonrası graph'i güncellemeli.
- `make mcp-impact TARGET=...` etki raporu üretmeli.
- `get_architecture(aspects=["all"])` ADR'leri göstermeli.
- Kullanıcı "tüm contexti alamıyorsun" hissiyatı azalmalı.
