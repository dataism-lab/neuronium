# Абстрактное задание: Dynamic Pipeline Engine (в духе Super Agent)

Дата: 2026-02-16  
Назначение: короткий бриф/ТЗ для запуска планирования в отдельном чате (Plan mode).

---

## Контекст (смысл из презентации)
Система должна решать “длинные” задачи не как линейный chain, а как **Action Graph (DAG)** в контуре **Plan → Execute → Control → Adapt**:
- **Planning = HTN → DAG**: objective → subgoals → tool-level operators; допускается пошаговая/итеративная декомпозиция; методы можно кэшировать как шаблоны.
- **Узлы**: `model` (типизированная генерация/решения), `code` (детерминированные трансформации), `mcp` (типизированные инструменты с политиками).
- **Надёжность**: trace/audit + строгий replay; верификация через simulated critics + evidence.
- **Безопасность**: sandbox/allowlist, least privilege, секреты не попадают в trace.

Источник: `docs/architecture/Super_Agent_presentation.md`.

---

## Цель (в 1 фразе)
По запросу пользователя на естественном языке система **сама строит и исполняет** runtime‑DAG из доступных операторов, сохраняя результат как артефакты и оставляя полный проверяемый trace/replay.

---

## Инварианты (нельзя нарушать)
- **Динамичность**: runtime‑DAG не является заранее прошитым “шаблоном пайплайна” в коде (допускаются только доменные ограничения вида “allowed tools/types”).
- **Три типа действий**: используем существующие узлы `model` / `code` / `mcp` (и служебные `decision`/`aggregate`).
- **Контракты**: входы/выходы узлов — типизированные JSON (схемы). Никаких “объяснений вместо JSON”.
- **Воспроизводимость**: всё недетерминированное обязано поддерживать запись `replay_data`; strict replay выполняется офлайн.
- **Secrets discipline**: секреты читаются только из env внутри tool‑реализаций; секретов нет в args/trace/replay_data.

---

## Минимальный “скелет” capability‑стека (что должно быть в системе)
1) **Tool/Operator library**: набор `mcp` tools с контрактами (например web/fs/…); у каждого — политика доступа (allowlist/лимиты).
2) **Planner → ActionGraph**: планировщик (model) возвращает валидный `ActionGraph` по JSON‑схеме; есть валидация графа (allowed types/tools, DAG acyclic).
3) **Executor**: исполняет DAG, пишет trace событий по узлам + решения.
4) **Artifacts**: результаты (файлы/тексты/индексы) пишутся детерминированно; есть понятные пути/ID.
5) **Control/Verification**: хотя бы один “critic” шаг и критерии успеха stage/run.
6) **Replay**: запись ответов для LLM/tools и режим strict replay без внешних вызовов.

---

## Определение готовности (DoD) для “демо‑уровня”
- **Вход**: 1 текстовый objective + небольшой metadata JSON (например `{ "url": "..." }` или иной параметр).
- **Планирование**: появляется событие/решение “Plan created (dynamic)” с перечислением узлов/рёбер planned DAG.
- **Исполнение**: отрабатывают реальные `mcp`/`code`/`model` узлы; нет скрытых “магических” шагов вне DAG.
- **Результат**: на диске появляется как минимум 1 финальный артефакт + локальный индекс/реестр запусков.
- **Верификация**: есть critic‑вердикт и evidence‑ссылки (на входные источники/артефакты).
- **Replay**: один раз записали run → второй раз strict replay проходит офлайн и даёт эквивалентный результат по контракту.

---

## Что НЕ делать в этой итерации
- Не строить полноценный универсальный HTN “на всё” сразу.
- Не добавлять таймеры/триггеры как узлы внутри DAG (это внешний слой запуска).
- Не добавлять новые доменные типы узлов (TelegramNode и т.п.) — только `mcp` tools.

---

## Обязательное “перед новым (рекурсивным) планировщиком”
Если цель — HTN‑подобная рекурсивная декомпозиция, то до неё критично иметь:
- **Стабильный каталог операторов** (tools) с контрактами и понятными ограничениями.
- **Единый формат “контрактов”** (JSON schemas) и дисциплину structured output.
- **Trace + strict replay** как базовую гарантию отладки/демо.
- **Детерминированный слой артефактов** (рендер/индекс), чтобы результаты были “материальны”.

Без этого даже “умный” планировщик будет генерировать красивые, но неисполняемые планы.

---

## Пример проверочного кейса (не цель, а smoke‑test)
“Взять входной параметр (например URL), выполнить 2–3 tool‑операции, получить промежуточные данные, сгенерировать результат моделью/кодом, сохранить артефакт и обновить индекс.”

---

## Readiness update (2026-02-17)
- Введена граница `planner backend` (контракты + адаптер), чтобы следующий планировщик подключался без переписывания executor core.
- Добавлен `OperatorCatalog` с контрактами операторов и обязательной проверкой planned DAG на соответствие каталогу.
- Усилен planning trace/replay: пишутся planning envelope события и hash каталога операторов; strict replay валидирует совместимость по hash.
- Добавлен детерминированный слой materialized output: рендер HTML-артефакта запуска + локальный индекс (`index.jsonl`/`index.html`).
- Добавлены readiness тесты на backend-контракт, catalog validation, strict replay hash-gate и artifact index flow.

## HTN recursive backend update (2026-02-17)
- Подключён новый backend-планировщик `htn_recursive_v0` как отдельный провайдер, без изменения COMMIT->EXECUTE->CONTROL->ADAPT цикла в orchestrator.
- HTN-lite backend выполняет рекурсивную декомпозицию objective -> subgoals -> leaf operators -> `ActionGraph` и возвращает расширенный `PlannerDecisionTrace` (шаги декомпозиции, method path, leaf operators, notes).
- Добавлен отдельный demo-runbook `htn_recursive_demo_v0`, который активирует `htn_recursive_v0` через `DynamicPlannerSpec.backend_name`.
- Planning trace события дополнены полезной нагрузкой `planner_decision_trace`, чтобы декомпозиция была видна в audit/trace.
- Усилен replay provider: `replay_data` для одного node_id теперь аккумулируется (а не перезаписывается), что готовит базу для многошагового planning-path.
- Добавлены readiness-тесты для `htn_recursive_v0` (backend contract, integration stage, strict replay provider path, operator validation).

## Supervised clarification in-graph update (2026-02-17)
- В HTN backend встроен extraction/resolution pipeline как planner-graphs: `artifact.put_json` (user request evidence), `text.extract_entities`, `model.extract_envelope`.
- Добавлены строгие контракты `ExtractionEnvelope`, `ClarificationRequest`, `ClarificationResponse` и экспорт их JSON Schema через registry.
- Реализован outcome `PlannerEscalation` и обработка в orchestrator по семантике `Commit -> Adapt -> Escalate -> PAUSED` с phase-boundary checkpoint.
- Уточнения теперь фиксируются как content-addressed artifacts с lineage: request artifact -> response artifact.
- `control revise` расширен под ADAPT semantics: принимает `answers`/`answer_text`, сохраняет `clarification_response` artifact и восстанавливает bindings при resume.
- В CLI `--mode supervised` добавлен интерактивный цикл вопросов/ответов; default runbook переключён на `super_agent_v0`.

