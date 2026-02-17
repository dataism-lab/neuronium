# DEMO_WALKTHROUGH_RU — как рассказать, что делает NeuroniumAgent

Этот файл — шпаргалка для демонстрации: **какой путь проходит задача** от CLI до результата, какие компоненты участвуют, и что смотреть в trace.

## 1) Вход и запуск

### CLI

- Пользователь запускает:
  - `neuronium-agent run --objective "..." --trace-export .\trace.jsonl`
- CLI загружает конфигурацию и стартует run.

### Конфиг: TOML + env + `.env`

- `neuronium.toml` (или `--config`) задаёт параметры проекта/хранилищ/LLM/исполнения.
- Переменные окружения `NEURONIUM_*` имеют приоритет выше TOML.
- `.env` (в корне проекта или рядом с файлом `--config`) автоматически подхватывается (без override уже заданных env).

**Почему это важно на демо:** можно показать “конфиг — это система, а не промпт”, секреты не хардкодятся.

## 2) Cognitive Core: фиксированный цикл (2 итерации)

Оркестратор выполняет демонстрационный цикл:

### Iteration 1 (iter1)

- **Plan**: `plan_iter1()` строит DAG из 3 узлов:
  - `generate` (ModelNode) → `execute` (CodeNode) → `critic` (ModelNode)
  - у `critic` есть входы и от `generate` (код), и от `execute` (результат выполнения)
- **Execute**: DAGExecutor исполняет узлы в топологическом порядке и пишет события в trace
- **Control**: критик возвращает минимальный JSON-вердикт
- **Adapt**:
  - если `execute` успешен **и** `critic` вернул `PASS` **с непустым evidence** → `COMPLETED`
  - иначе → `replan` и переход к iter2

### Iteration 2 (iter2 fix-pipeline)

- **Plan**: `plan_iter2_fix()` строит DAG:
  - `fix` (ModelNode) → `execute_fix` (CodeNode) → `critic_fix` (ModelNode)
- `fix` получает контекст провала iter1: предыдущий код, stdout/stderr/exit_code, gaps критика
- **Условие успеха**: `execute_fix` успешен и `critic_fix` = `PASS` с evidence
- **Иначе**: `FAILED` с сообщением “Auto-fix exhausted after 2 iterations”

**Почему это важно на демо:** демонстрируется архитектурный принцип “Plan → Execute → Control → Adapt”, а не “одна большая генерация”.

## 3) Planning: Action Graph (DAG), а не линейный план

В демо-пайплайне планирование реализовано как **детерминированные шаблоны DAG**:

- плюс: воспроизводимость и понятность для аудитора
- минус: это именно демо/каркас, не “универсальный HTN”

На демо можно говорить:
- “В этой версии мы показываем closed-loop и инфраструктуру (trace/replay/verification).”
- “Дальше планировщик можно расширять до полноценной декомпозиции.”

## 4) Execution: DAGExecutor

DAGExecutor:

- собирает входы узла из outputs предшественников + initial_inputs (`objective`, `constraints`)
- исполняет узлы (параллельно там, где возможно)
- пишет в trace:
  - `node_start` (inputs, node_id, node_type, node_ref)
  - `node_end` (status, outputs, error)

## 5) Типы узлов (Node types) в демо

### ModelNode (LLM inference)

- получает system prompt + user prompt (+ контекст)
- возвращает `outputs.content` (и иногда `outputs.parsed` если structured output)
- записывает quality_signals (например tokens_used)
- поддерживает запись/воспроизведение (replay_data)

### CodeNode (sandboxed execution)

- исполняет Python код в Docker контейнере (по умолчанию `python:3.11-slim`)
- сохраняет `stdout`, `stderr`, `exit_code`
- важный момент: если модель оборачивает код в Markdown fences (```python ...```), перед исполнением fences снимаются

### Critic (LLM-based verification)

- это отдельный ModelNode с минимальным контрактом:
  - `verdict`: PASS/FAIL/UNCERTAIN
  - `confidence`
  - `evidence` (обязательно непустой для PASS)
  - `gaps`
- structured output работает через OpenAI `response_format: json_schema`

## 6) Trace / artifacts / replay (что показать на демо)

### Trace (audit trail)

В `trace.jsonl` (или в export) вы увидите:

- `decision`: коммит намерения, создание плана, инъекция replay и т.п.
- `node_start` / `node_end`: начало/конец каждого узла
- `critic_verdict`: вынесенный вердикт критика
- `replan`: переход iter1 → iter2 с причиной и добавленными constraints
- `checkpoint`: состояние run (run_state/message/progress)
- `replay_data`: записанные ответы для офлайн воспроизведения

### Artifacts (immutability + lineage)

Результаты узлов могут сохраняться как immutable артефакты (CAS) с `artifact_id` и метаданными в индексе.

### Replay (офлайн воспроизводимость)

- Можно seed’ить trace (см. `examples/seed_autofix_demo.py`) и делать `neuronium-agent replay --trace-id ...`
- Цель: запуск без внешних сервисов, но со строгой воспроизводимостью (включая critic)

## 7) Короткий “скрипт речи” (30–60 секунд)

- “Мы не делаем чат-бота. Мы делаем агента с явным состоянием и контрольным контуром.”
- “Задача → Action Graph (DAG): generate → execute → critic.”
- “Каждый шаг пишет trace: что сделали, что получили, почему решили дальше.”
- “Успех — это не ‘модель сказала’, а: execution ok + critic PASS + evidence.”
- “Если нет — replan и fix-пайплайн (2 итерации максимум).”
- “Выходы версионируются как артефакты, а run можно replay-нуть офлайн.”

## 8) Быстрый чек-лист перед показом

- `.env` в корне: `NEURONIUM_OPENAI_API_KEY=...`
- Docker Desktop запущен; `docker ps` работает
- Команда:
  - `neuronium-agent run --objective "Print hello" --trace-export .\trace.jsonl`
- В `trace.jsonl` должны быть:
  - `node_start/node_end` хотя бы для `generate/execute/critic`
  - `critic_verdict`
  - финальный `checkpoint` с `run_state` = `COMPLETED` (если всё прошло)

