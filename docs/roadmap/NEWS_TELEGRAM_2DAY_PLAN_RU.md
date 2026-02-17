# План на 2 дня: `ai_news_telegram_v1` (NeurоniumAgent)

Дата фиксации: 2026-02-16  
Цель: сделать флагманский сценарий **“новости → саммари → Telegram”** так, чтобы он демонстрировал идеи презентации (DAG + control loop + verification + trace/replay + tools/MCP) и был основой для множества похожих сценариев.

---

## 0) Принципы (чтобы не превратиться в “копию n8n”)

- **Не добавляем новые типы узлов** под интеграции (Telegram/HTTP/RSS). Интеграции = **tools**, вызываемые через `node_type="mcp"` (`McpToolNode`).
- **Секреты не попадают в trace**: токены/ключи не передаются в `tool_args`, инструменты читают их из env/credentials.
- **Воспроизводимость и аудит** — часть продукта: trace + `replay_data` + `replay`.
- Планировщик/таймер **снаружи** (на 2 дня): Windows Task Scheduler вызывает CLI. Внутренние “scheduler nodes” — не приоритет.

---

## 1) Definition of Done (DoD)

Считаем задачу выполненной, если:

1) Есть runbook `ai_news_telegram_v1`, запускаемый через CLI:
   - `neuronium-agent run --runbook ai_news_telegram_v1 --objective "..."`
2) Он выполняет pipeline:
   - получает новости из источников
   - нормализует items (title/url/source/date)
   - делает дедуп “не отправлять повторно”
   - делает краткое саммари (1–3 предложения) + ссылка
   - отправляет результат в Telegram ботом
3) Ключ Telegram **не оказывается** в `tool_args` / `replay_data` / export trace.
4) Trace содержит понятные события (`stage_start/stage_end`, `node_start/node_end`, `critic_verdict`, `replay_data`).
5) Есть “план B” для сцены: `replay` работает офлайн (без сети/ключей) на seeded trace.
6) `pytest` зелёный.

---

## 2) Решения (зафиксировать перед реализацией)

### 2.1 Формат отправки
Выбрать один (можно сделать флагом позже, но сейчас зафиксировать default):
- **A: 1 сообщение = 1 новость** (может спамить, но проще)
- **B: 1 сообщение = дневной дайджест** (рекомендуется default для v1)

### 2.2 Источники (v1)
Источники задаются **вручную** (список RSS/Atom URLs), авто-поиск источников — отдельный будущий runbook.

### 2.3 Хранилище “seen”
Минимум для v1:
- key-value (например файл/SQLite таблица/использование уже существующей SQLite) с ключом типа `seen:<hash(url)> -> ts`.

---

## 3) План работ: День 1 (P0 — платформа tools)

### P0.1 Добавить инструменты (tools) под кейс
Цель: собрать набор “операторов”, из которых строятся сценарии.

- **`rss.fetch`** (или `http.get` + `rss.parse`, но лучше RSS-first)
  - Input: `{ "urls": [..], "max_items_per_source": N, "timeout_seconds": T }`
  - Output: `{ "items": [ {title, url, published_at, source} ... ], "warnings": [...] }`

- **`telegram.send_message`**
  - Input: `{ "chat_id": "...", "text": "...", "parse_mode": "Markdown" | null }`
  - Secrets: `NEURONIUM_TELEGRAM_BOT_TOKEN` читается **из env**, не из args.
  - Output: `{ "ok": true, "message_id": "..."}`

- **`state.kv_get` / `state.kv_set`**
  - Input: `{ "key": "...", "value": "...", "ttl_seconds": ... }`
  - Output: `{ "found": bool, "value": ... }`
  - Назначение: дедуп, курсоры, idempotency.

Примечание: реализация может начать как “local tools” (in-process) и позже стать реальным MCP transport.

### P0.2 Policy gates и безопасные дефолты
- Для сетевых инструментов:
  - allowlist доменов (минимально) или хотя бы запрет на `file://` и локальные адреса
  - таймауты по умолчанию
  - лимит размера ответа
- Для Telegram:
  - защита от слишком длинного сообщения (обрезать/пакетировать)

### P0.3 Тесты tools
Минимум:
- unit tests для сериализации/контрактов
- тест, что `telegram.send_message` **не требует** токен в args (берёт из env) и не пишет его в outputs

---

## 4) План работ: День 1 (вечер) — Runbook + DAG

### P0.4 Runbook `ai_news_telegram_v1`
Собрать DAG стадийно, в стиле уже существующих runbook’ов:

**Stage 1: Fetch + Normalize + Dedupe**
- `mcp:rss.fetch`
- `code:normalize_items` (превратить в список item DTO)
- `code:dedupe_items` (с помощью `state.kv_get/kv_set` пометить отправленные)

**Stage 2: Summarize + Verify + Send**
- `model:summarize_batch`
- `model:critic_digest` (PASS только при evidence: ссылки/ID items)
- `mcp:telegram.send_message` (или send_many)

### P1.1 “План B”: режим офлайн демо
- seeded trace с `replay_data` для всех узлов (включая tools), чтобы `replay` работал без сети.

---

## 5) План работ: День 2 — надежность, replay, демо, docs

### P0.5 E2E тест runbook’а (seeded replay)
- прогон “успех” (PASS с evidence)
- прогон “нет новостей” (корректное поведение: пустое сообщение или “нет новостей”)
- прогон “missing replay_data” → strict replay падает ожидаемо

### P0.6 Документация для пользователя (минимум)
- `docs/ops/ENV_VARS_RU.md` (или аналог) со списком:
  - `NEURONIUM_OPENAI_API_KEY`
  - `NEURONIUM_TELEGRAM_BOT_TOKEN`
  - `NEURONIUM_TELEGRAM_CHAT_ID`
- `.env.example` (без секретов)
- `docs/demos/` обновить: добавить команды для `ai_news_telegram_v1`

### P1.2 Инструкция по расписанию (runtime)
- кратко: Windows Task Scheduler → запуск `neuronium-agent run --runbook ai_news_telegram_v1 ...`

---

## 6) Риски и как их контролировать

- **Сеть/HTML парсинг ломает сроки** → RSS/Atom first; HTML extraction только если останется время.
- **Секреты утекут в trace** → запретить токен в args, токен только через env.
- **Демо зависит от внешних сервисов** → обязательный seeded `replay` путь.
- **Слишком “узкий кейс”** → подчеркнуть, что это шаблон “input tools → code transforms → model reasoning → output tools”.

---

## 7) Следующие этапы (после 2 дней, не в scope)

- Реальный MCP transport (не local-only).
- HTN-планировщик (универсальная декомпозиция objective → subgoals → tool-level operators).
- Credentials store (не только env), UI/“supervised mode”.
- Memory v1: semantic retrieval (embeddings) + iterative retrieval loop state machine.

