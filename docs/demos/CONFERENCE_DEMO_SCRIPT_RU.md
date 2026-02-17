# NeuroniumAgent — сценарий демо для конференции (RU)

Цель демо: показать, что это **не чат-бот**, а **архитектура**: Plan→Execute→Control→Adapt, граф-план (DAG), trace/audit, strict replay, policy-gated tools и critics.

## 0) Что подготовить заранее (чтобы демо не сорвалось)

### Вариант A (самый надёжный): офлайн replay
- Никаких ключей, сети и Docker не нужно.
- Показываем: `replay` + trace + детерминизм.

### Вариант B (live): настоящий прогон
- Нужны:
  - `NEURONIUM_OPENAI_API_KEY` в `.env` или env,
  - Docker Desktop (для `CodeNode`).

## 1) 60-секундный pitch (можно проговорить поверх слайдов)

- “Линейные планы у prompt-агентов ломаются из‑за каскада ошибок и отсутствия явного состояния.”
- “Мы делаем контрольный контур: **Commit → Execute → Control → Adapt**.”
- “План — это **Action Graph (DAG)**, а не цепочка: его проще исполнять параллельно и точечно чинить.”
- “Качество подтверждается **critic’ом**: PASS возможен только при наличии evidence.”
- “Любой прогон пишет **audit trace**, а затем может быть **строго воспроизведён офлайн** (strict replay).”

## 2) Команды демо (PowerShell)

Важно: запускать из корня репозитория, чтобы FS allowlist (CWD) разрешал чтение документов.

### 2.1 Прогон 1: “autofix_demo” (код → запуск → критик → автофикс)

```powershell
neuronium-agent run --objective "Write a fibonacci function in Python" --trace-export .\trace.jsonl
```

Что показать:
- в трейсе есть `decision`, `node_start/node_end`, `critic_verdict`, `checkpoint`
- при ошибке происходит `replan` и вторая итерация fix-пайплайна

### 2.2 Прогон 2: “docs_report_v1” (чтение локальных документов → отчёт)

```powershell
neuronium-agent run --runbook docs_report_v1 --objective "Сделай краткий статус проекта: что готово, что не готово, риски" --trace-export .\trace_docs.jsonl
```

Что показать:
- `stage_start/stage_end` (runbook как последовательность стадий)
- что отчёт требует ссылок вида `[doc_000]` (evidence discipline)

### 2.3 Прогон 3 (план B): офлайн replay (самый стабильный для сцены)

```powershell
neuronium-agent replay --trace-id <TRACE_ID_ИЗ_ПРЕДЫДУЩЕГО_ПРОГОНА>
```

Что сказать:
- “Мы не дергаем внешние сервисы: ответы узлов уже записаны как `replay_data`.”
- “Strict replay падает, если данных для воспроизведения не хватает — это важная гарантия аудита.”

## 3) На что в trace смотреть (минимальный чек‑лист)

- `decision`: выбор runbook, создание плана, strict_fail и т.д.
- `node_start/node_end`: входы/выходы узла, статус
- `critic_verdict`: PASS/FAIL + evidence/gaps
- `replay_data`: записанные ответы для воспроизведения
- `checkpoint`: phase-boundary снимки для resume
- `stage_start/stage_end`: границы стадий runbook’а

## 4) Честные ограничения (важно проговорить, чтобы не “перепродать”)

- Planner сейчас: **детерминированные шаблоны** (для демо и воспроизводимости), не “универсальный HTN”.
- MCP: пока “local transport” (ин‑процесс инструменты), не внешний протокол интеграций.
- Memory: “GraphRAG-lite” на чанках + provenance; **semantic embeddings/graph traversal/iterative retrieval loop** — следующий этап.

