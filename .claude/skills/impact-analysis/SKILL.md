---
name: impact-analysis
description: "Bir fonksiyon, sınıf veya dosyanın etki alanını codebase-memory graph üzerinden haritalar. Trigger: etki analizi, impact analysis, kim çağırıyor, bu değişiklik neyi etkiler, refactor riski, who calls this."
when_to_use: "Trigger phrases: impact analysis, etki analizi, kim çağırıyor, neyi etkiler, refactor riski, dependency map. Kullanıcı bir sembol veya dosya üzerinde değişiklik düşünüyor ve etki alanını görmek istiyorsa bu skill'i kullan."
allowed-tools: mcp__codebase-memory-mcp__search_graph mcp__codebase-memory-mcp__trace_path mcp__codebase-memory-mcp__query_graph mcp__codebase-memory-mcp__get_code_snippet
---

# /impact-analysis — Etki Analizi

Amaç: Bir sembol/dosya değişikliğinin codebase'teki etki alanını hızlıca haritalamak.

## Sabitler

- Proje adı: `home-user-Workspace-MergenVisionPhase2v2`
- Varsayılan trace derinliği: 3
- Risk skoru: 0–10 arası; 7+ yüksek, 4–6 orta, 0–3 düşük

## Akış

1. **Hedefi çöz.**
   - `mcp__codebase-memory-mcp__search_graph(project="...", name_pattern=".*<hedef>.*", limit=10)`
   - Eğer tam eşleşme varsa onu kullan; yoksa ilk suggestion'ı al.
   - Dosya yolu verildiyse `search_code(pattern="<hedef>", limit=5)` ile bul.

2. **Call-chain çek.**
   - `trace_path(function_name="<qualified_name>", mode="calls", direction="both", depth=3, include_tests=True)`
   - Eğer `status: ambiguous` dönerse ilk suggestion'ın `qualified_name` ile tekrar çağır.

3. **Complexity metrikleri çek.**
   - `query_graph` ile hedef ve ilgili fonksiyonların complexity değerlerini al:
     ```cypher
     MATCH (f:Function|Method) WHERE f.name IN [...]
     RETURN f.name, f.file_path, f.start_line, f.complexity, f.cognitive,
            f.transitive_loop_depth, f.linear_scan_in_loop, f.recursion_in_loop,
            f.is_entry_point, f.is_test, f.route_path
     ```

4. **Raporla.**
   - Hedef: tür, konum, tam ad, risk skoru.
   - Entry point'leri (Route / is_entry_point) listele.
   - Testleri listele.
   - İlgili sembolleri ve karmaşıklıklarını listele.
   - Riskli pattern'leri vurgula: yüksek transitive_loop_depth, linear_scan_in_loop, recursion_in_loop.

## Çıktı formatı

Markdown başlıkları:

```markdown
# Impact Analysis: `enroll_batch`
- **Tür:** Method
- **Konum:** backend/app/api/controllers/bulk_enrollment_controller.py:17
- **Tam ad:** `...`
- **Risk:** 🟢 LOW (skor: 0)

## Complexity metrikleri
...

## Entry points (N)
| Sembol | Tür | Dosya | Karmaşıklık |
...

## Tests (N)
...

## Related symbols (N)
...
```

## Kurallar

- Sadece `home-user-Workspace-MergenVisionPhase2v2` projesini kullan.
- `trace_path` öncesinde hedefin `qualified_name`'ini `search_graph` ile doğrula.
- `query_graph` sonuçları list of lists olarak gelir; `columns` ile `zip` et.
- Sayısal değerleri karşılaştırmadan önce int'e çevir.
- Etki raporunu 60 saniye içinde tamamla; gerekirse derinliği ve limitleri kıs.
