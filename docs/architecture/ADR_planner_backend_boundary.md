# ADR: Planner Backend Boundary

Дата: 2026-02-17  
Статус: accepted

## Контекст
Текущая реализация dynamic planner была встроена в оркестратор напрямую. Это мешало безопасно добавить новый HTN-lite/recursive planner в следующей итерации без рефакторинга `Orchestrator` и без риска регрессий в execution loop.

## Решение
- Ввести отдельные planner-контракты:
  - `PlannerRequest`
  - `PlannerResult`
  - `PlannerDecisionTrace`
  - `DynamicPlannerSpec`
- Ввести backend-границу `PlannerBackend` и реестр `get_planner_backend(...)`.
- Текущий dynamic planner упаковать как backend `legacy_dynamic_v1`.
- Вызов планирования из `Orchestrator` переводится на backend API.
- В planning trace фиксируются:
  - backend name/version
  - planner request/result envelope
  - `operator_catalog_hash`

## Последствия
### Плюсы
- Следующий planner добавляется как новый backend без изменения цикла COMMIT->EXECUTE->CONTROL->ADAPT.
- Прозрачная совместимость live/replay через hash каталога операторов.
- Валидация planned DAG становится capability-driven через `OperatorCatalog`.

### Минусы
- Появляется дополнительный абстрактный слой (backend/adapters), который нужно поддерживать тестами.
- Нужно синхронизировать контракты planner backend и operator catalog при расширении tool library.

## Что дальше
- Добавить backend `htn_recursive_v0` в следующей итерации и использовать тот же контракт.
- Расширить `PlannerDecisionTrace` для многослойной декомпозиции (subgoals/method expansion path).

## Implementation update (2026-02-17)
- В рамках следующей итерации backend `htn_recursive_v0` реализован и подключён через тот же `PlannerBackend` контракт.
- Оркестратор не менялся по API: выбор backend остаётся через `DynamicPlannerSpec.backend_name`.
- `PlannerDecisionTrace` расширен полями декомпозиции (`decomposition_steps`, `method_expansion_path`, `leaf_operators`) для audit/replay диагностики.
- Добавлен отдельный demo-runbook для `htn_recursive_v0`, подтверждающий, что расширение работает как plug-in backend без миграции executor core.
