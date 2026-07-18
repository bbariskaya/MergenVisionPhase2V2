---
name: context-orchestrator
description: Akıllı codebase context retrieval. Kullanıcı "şu fonksiyonu bul", "bu feature nerede implemente", "context topla", "kodu keşfet", "explore the codebase", "who calls X" dediğinde çalışır. Graph araması, tam metin araması ve call-chain tracing'i birleştirerek en alakalı dosya ve sembolleri sıralar.
when_to_use: "Trigger phrases: context topla, şuraya bak, bu ne yapıyor, nerede tanımlı, kim çağırıyor, etkisini göster, kodu keşfet, explore. Kullanıcı doğal dilde bir konu sorduğunda veya bir sembol/kavram hakkında bilgi istediğinde bu skill'i kullan."
allowed-tools: mcp__codebase-memory-mcp__search_graph mcp__codebase-memory-mcp__search_code mcp__codebase-memory-mcp__trace_path mcp__codebase-memory-mcp__get_code_snippet mcp__codebase-memory-mcp__get_architecture
---

# /context-orchestrator — Akıllı Context Retrieval

Amaç: Kullanıcının sorusuna göre codebase'teki en alakalı kod parçalarını otomatik bul ve özetle.

## Sabitler

- Proje adı: `home-user-Workspace-MergenVisionPhase2v2`
- Varsayılan sonuç limiti: 15 dosya/sembol
- Trace derinliği: 2

## Akış

1. **Soruyu analiz et.**
   - Eğer kullanıcı bir sembol/kavram adı verdiyse önce `search_graph` ile bul.
   - Eğer genel bir feature/concept sorusuysa önce `search_graph(query=...)` ile semantik/graph araması yap.
   - Eğer tam metin (string literal, hata mesajı, endpoint path) varsa `search_code(pattern=...)` ile destekle.

2. **Graph araması.**
   - `mcp__codebase-memory-mcp__search_graph(project="home-user-Workspace-MergenVisionPhase2v2", query="<kullanıcı sorusu>", limit=20)`
   - Sonuçlardan fonksiyon, sınıf, route, interface isimlerini çıkar.

3. **Tam metin desteği.**
   - `mcp__codebase-memory-mcp__search_code(project="home-user-Workspace-MergenVisionPhase2v2", pattern="<anahtar terim>", limit=20, mode="compact")`
   - Tekrarlayan dosyaları ve test dosyalarını sonlara at.

4. **Call-chain genişletme.**
   - Bulunan önemli fonksiyonlar için `trace_path(function_name="<isim>", mode="calls", direction="both", depth=2, include_tests=False)` çalıştır.
   - Bu adımı her sembol için yapma; sadece top 5 sembol için yap.

5. **Deduplicate ve sırala.**
   - Aynı dosya/sembol birden fazla kaynaktan geliyorsa birleştir.
   - Sıralama: tanım noktaları > yüksek dereceli (degree) fonksiyonlar > route'lar > testler.
   - En fazla 15 sonuç tut.

6. **İsteğe bağlı snippet.**
   - Kullanıcı "kodu göster" veya "implementation'ını getir" derse `get_code_snippet(qualified_name="...")` ile kaynak getir.

## Çıktı formatı

Markdown tablo:

```markdown
| # | Tür | Sembol | Dosya | Not |
|---|-----|--------|-------|-----|
| 1 | Function | create_person | app/api/controllers/person_controller.py:42 | Route tarafından çağrılır |
```

Ardından kısa bir özet: "Bu sonuçlar X feature'ı için Y controller ve Z service katmanında toplanıyor."

## Kurallar

- Sadece `home-user-Workspace-MergenVisionPhase2v2` projesini kullan.
- Test dosyalarını sonlara at; eğer kullanıcı özellikle test istemediyse öne çıkarma.
- `trace_path` çağrılarından önce sembolün tam adını `search_graph(name_pattern=...)` ile doğrula.
- Çıktıyı abartma; en alakalı 15 sonuç ve 3-4 cümle özet yeterli.
