---
name: adr
description: "Codebase-memory ADR yönetimi. Kullanim: /adr get, /adr init, /adr update docs/adr.md"
disable-model-invocation: true
allowed-tools: "Bash(python scripts/mcp_adr.py *)"
argument-hint: "[get|init|update] [file]"
---

# /adr — Architecture Decision Record yönetimi

Codebase-memory-mcp'ye kayitli ADR'yi getir, olustur veya guncelle.

## Komutlar

- `/adr get` — Mevcut ADR icerigini goster.
- `/adr init` — Varsayilan ADR sablonunu codebase-memory'ye yazar.
- `/adr update <dosya>` — Belirtilen markdown dosyasini yeni ADR olarak kaydeder.

## Kurallar

1. Ilk argumani komut olarak yorumla: `$0`.
2. `get` icin calistir:
   ```bash
   python scripts/mcp_adr.py --get
   ```
3. `init` icin calistir:
   ```bash
   python scripts/mcp_adr.py --init
   ```
4. `update` icin ikinci arguman `$1` gerekli; calistir:
   ```bash
   python scripts/mcp_adr.py --update "$1"
   ```
5. Bilinmeyen veya eksik komut varsa kisa yardim mesaji goster.
6. Bu skill sadece kullanici tarafindan `/adr` ile cagrildiginda calisir; Claude otomatik cagiramaz.
