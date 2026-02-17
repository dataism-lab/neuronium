# Мастер‑план демо: Dynamic Pipeline Engine (capabilities‑first)

Дата фиксации: 2026-02-16 (живой документ)  
Назначение: **зафиксировать главные мысли и абстрактные шаги**, чтобы дальше каждую подзадачу детализировать отдельными агентами.

---

## 0) О чём демо
Пользователь даёт **цель на естественном языке** + минимальные входные параметры → система **сама планирует и исполняет DAG** → на диске появляются **артефакты результата** и простой **локальный индекс** для просмотра/переиспользования.

---

## 1) Инварианты (нельзя сломать ради скорости)
- **Динамичность**: runtime‑DAG не является заранее прошитым “шаблоном пайплайна” в коде.
- **Три типа действий**: используем существующие узлы (`model` / `code` / `mcp` + служебные `decision`/`aggregate`), без введения доменных “telegram node” и т.п.
- **Детерминизм/воспроизводимость**: всё, что недетерминировано (LLM/внешние инструменты), должно поддерживать запись `replay_data`, а строгий replay должен работать офлайн.
- **Secrets discipline**: секреты читаются из env внутри tool‑реализаций и **не попадают** в trace/args/replay_data.

---

## 2) Что уже сделано (зафиксировано статусом `done`)
- **T0.1 Dynamic planner stage**: планировщик‑`ModelNode` возвращает `ActionGraph` по JSON‑схеме; оркестратор валидирует и исполняет.
- **T0.2 Web tools**: есть минимальные `web.fetch_html` + `web.extract_article` (best‑effort).
- **T0.3 File write**: есть `fs.write_text` через `McpToolNode` с политикой allowlist.

Этого достаточно, чтобы система **уже умела**: “спланировать” DAG под objective и сделать реальные web/fs шаги.

---

## 3) Целевой продукт демо (минимальный результат)
**Одна “единица результата”** = (а) один рендеренный файл (по умолчанию HTML), (б) запись в локальном индексе (`index.html` или JSONL).

Минимальная структура данных “Artifact Bundle” (то, что проходит по узлам и пишется на диск):
- **source**: входные параметры (например, URL), timestamp
- **raw**: сырьё/заготовка (например, HTML, текст)
- **extracted**: нормализованные данные (например, основной текст, список изображений/вложений)
- **synthesized** (опционально): структурированный продукт после model‑узлов — когда планировщик начнёт такие шаги выдавать
- **rendered**: финальный файл(ы) + пути

---

## 4) Следующие абстрактные шаги (оставшееся “ядро”)

Специально вводить договор “synthesized” и учить текущий планировщик его выдавать **не приоритет**: новый/HTN‑планировщик сам будет собирать из model/code/mcp любые пайплайны и форматы. Сейчас фокус — на рендере и индексе, которые могут опираться на то, что DAG уже выдаёт.

### 4.1 T0.5 Deterministic renderer
Задача: из выходов DAG (например, отчёт/контент model‑узлов + extracted) собрать финальный рендер (по умолчанию HTML) **детерминированно** (предпочтительно `code`‑узел).

Критерий готовности:
- есть артефакт на диске (HTML или иной формат), содержащий результат пайплайна и ссылку на источник.

### 4.2 T0.6 Local index (gallery) generator
Задача: поддерживать `index.html` (или JSONL‑реестр), который показывает список результатов и ссылки на файлы.

Критерий готовности:
- после двух запусков в галерее две записи, без ручного редактирования.

### 4.3 T0.7 Seeded replay demo + тесты
Задача: обеспечить офлайн‑replay на “эталонном” URL (или фикстуре) и тестами зафиксировать контракт.

Критерий готовности:
- один “live” прогон записывает всё нужное;
- “strict replay” повторяет run без сети/LLM и даёт идентичные результаты по контракту.

---

## 5) Доска задач (коротко, без деталей реализации)
Легенда: `todo` / `in_progress` / `done` / `blocked`

### P0 — обязательно для демо
- [x] **T0.1 Dynamic planner stage** — `done`
- [x] **T0.2 Web tools: fetch+extract** — `done`
- [x] **T0.3 File tools: write_text** — `done`
- [ ] **T0.5 Deterministic renderer (HTML)** — `todo`
- [ ] **T0.6 Local index (gallery)** — `todo`
- [ ] **T0.7 Tests + seeded replay demo** — `todo`

### P1 — желательно (если останется время)
- [ ] **T1.2 CLI UX (`--url`)** — `todo`
- [ ] **T1.3 “ENV doctor”** — `todo`
- [ ] **T1.1 Real image captioning** — `todo`

### Отложено (по мере появления нового планировщика)
- **T0.4** Договор “synthesized” / model‑шаг под домен — не тратим время сейчас; будущий планировщик сам будет собирать любые пайплайны из model/code/mcp.

---

## 6) После демо: шаг к “рекурсивному HTN” (HTN v0)
Сейчас планирование “в один шаг”: objective → сразу DAG. Следующий прагматичный шаг к твоему идеалу:
- планировщик сначала выдаёт **крупные подцели** (абстрактные подзадачи с DoD/входами/выходами),
- затем система **итеративно** разворачивает каждую подцель до leaf‑узлов (model/code/mcp),
- хорошие развёртки кэшируются как “methods/templates”.

Это и будет мостом от “доменного dynamic planner” к универсальному рекурсивному планированию без переписывания исполнителя DAG.

---

## 7) Связь с прошлым кейсом (telegram/news)
`docs/roadmap/NEWS_TELEGRAM_2DAY_PLAN_RU.md` остаётся как интеграционный use‑case. Но в этой итерации приоритет: **планирование + web/fs + галерея + replay**; Telegram — следующий коннектор после стабилизации базовой механики.

---

## 8) Пример демо‑сценария (не цель, а “проверочный кейс”)
Для демонстрации удобен сценарий “URL → извлечь → синтезировать → отрендерить → добавить в индекс”. Содержательно это может быть “обзор статьи”, но движок должен оставаться сценарий‑независимым.

---

## 9) Readiness for next planner iteration (2026-02-17)
- Вынесены planner-контракты и backend boundary: текущий dynamic planner работает через адаптер (`legacy_dynamic_v1`), без жёсткой сцепки с orchestrator.
- Введён машиночитаемый `OperatorCatalog` (контракты, политики, deterministic/replay требования), и runtime DAG проходит валидацию по каталогу.
- Planning trace усилен planning envelopes (`Planner request envelope`, `Planner result envelope`) с фиксацией `operator_catalog_hash`.
- Strict replay получил gate на совместимость каталога операторов (hash mismatch => явный fail).
- Добавлен materialized artifact layer: детерминированный HTML рендер результата + локальный индекс запусков.
- Текущий executor core остаётся неизменным; в следующей итерации можно добавлять новый HTN-lite/recursive planner backend как новый backend-провайдер.

## 10) HTN recursive backend iteration update (2026-02-17)
- Реализован и зарегистрирован backend `htn_recursive_v0` как второй planner backend рядом с `legacy_dynamic_v1`.
- HTN-lite декомпозиция вынесена в отдельный planning-модуль; `planner_backend.py` используется как thin registry/adapter слой.
- Добавлен demo runbook `htn_recursive_demo_v0` для интеграционного smoke-сценария через динамический COMMIT.
- В trace события `Plan created (dynamic)` добавлен `planner_decision_trace`, чтобы видеть путь декомпозиции и выбранные методы.
- Для replay подготовлен multi-step сценарий на уровне провайдера: multiple `replay_data` entries по одному `node_id` теперь накапливаются.
- Добавлены/обновлены тесты readiness для нового backend (`planner backend contract`, `integration stage`, `replay strict provider path`, `operator validation`).

## 11) Supervised clarification in-graph update (2026-02-17)
- HTN backend now runs extraction and parameter resolution as planner-graphs (`artifact.put_json`, `text.extract_entities`, `model.extract_envelope`) instead of out-of-band preprocessing.
- Added strict extraction/clarification contracts (`ExtractionEnvelope`, `ClarificationRequest`, `ClarificationResponse`) and schema registry export for structured model IO.
- Introduced `PlannerEscalation` outcome and orchestrator handling for `Commit -> Adapt -> Escalate -> PAUSED`, including escalation decisions and paused checkpoint context.
- Clarification request/response are persisted as content-addressed artifacts with lineage (`request -> response`) for auditability and pattern analysis.
- Extended `control revise` semantics to ADAPT flow with response artifact persistence and resume-time binding restoration.
- CLI supervised mode now includes an interactive clarification loop; default runbook switched to `super_agent_v0`.

