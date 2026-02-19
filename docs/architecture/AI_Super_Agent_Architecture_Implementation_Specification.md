

# AI Super Agent Architecture Implementation Specification

## 1. System Overview

### 1.1 Purpose and Scope

The **AI Super Agent** system, codenamed **NEURONIUM**, is a self-organizing multi-agent architecture designed for **reliable long-horizon, high-entropy task solving** in real-world environments. The system addresses critical failures of conventional prompted agents—linear plan fragility, error propagation cascades, implicit state opacity, weak self-checking, and tool format mismatches—by embedding large language models within a rigorous **commitment-aware control architecture** rather than treating them as standalone reasoning engines.

The specification defines a **core library** with a **reference CLI runtime**, engineered to support **staged LLM-driven code generation** where implementation proceeds deterministically without architectural decision-making during coding phases. The architecture ensures that fundamental invariants—**deterministic execution**, **artifact lineage immutability**, and **replay reproducibility**—are preserved across all implementation stages.

#### 1.1.1 Core Library Design Principles

The core library adheres to five foundational design principles that distinguish it from conventional agent frameworks:

| Principle | Description | Architectural Manifestation |
|-----------|-------------|----------------------------|
| **Goal-Directed Execution** | Replaces chat-completion with structured commitment protocols | Intention lifecycle state machine with formal semantics |
| **Graph-Native Planning** | DAG-based plans enabling parallelism and surgical recovery | HTN decomposition → Action Graph with dependency edges |
| **Verification-Integrated Control** | Critics embedded at decision points, not post-hoc | Control state with mandatory critic evaluation before progression |
| **Memory-as-System-Component** | Retrieval and synthesis as first-class architectural elements | Hybrid memory with GraphRAG + agentic retrieval |
| **Audit-by-Construction** | Complete traceability without additional instrumentation | Immutable artifact lineage with content-addressed storage |

These principles enforce explicit constraints: **all node outputs are typed JSON with schema validation**; **all state transitions are logged for deterministic replay**; **all failures trigger classified recovery with defined escalation paths**; and **all memory operations maintain provenance chains**.

#### 1.1.2 Reference CLI Runtime Architecture

The CLI runtime provides three primary interaction modes: **batch execution** for fully automated task completion; **supervised execution** with step-by-step user confirmation; and **interactive execution** enabling real-time control protocol engagement. The runtime maintains **session persistence** across interruptions, supports **trace export** in multiple formats, and implements **granular progress indication**.

The architecture separates concerns through distinct layers: the **command layer** handles argument parsing and configuration; the **session layer** manages authentication, state persistence, and recovery; the **orchestration layer** translates user commands into cognitive core invocations; and the **presentation layer** renders execution progress and diagnostic information. Each layer communicates through defined interfaces, enabling alternative frontends to reuse core orchestration logic.

#### 1.1.3 Staged LLM-Driven Code Generation Support

The specification supports implementation through **six explicit stages**, each producing verifiable artifacts:

| Stage | Focus | Deliverable | Validation Criteria |
|-------|-------|-------------|---------------------|
| 1 | Domain models and interface contracts | JSON Schemas, TypeScript interfaces | Schema validation against reference cases |
| 2 | State machines and lifecycle management | Intention, node, retrieval state machines | Exhaustive transition coverage |
| 3 | Node execution infrastructure | Model, MCP, Code node implementations | Contract compliance, quality signal generation |
| 4 | Planning and verification systems | HTN engine, critic framework | End-to-end task completion with validation |
| 5 | Memory and persistence layers | GraphRAG, agentic retrieval, storage interfaces | Query correctness, replay fidelity |
| 6 | CLI runtime and testing | Command interface, test suites | Workflow execution, failure simulation |

Each stage includes **acceptance criteria based on deterministic behavior verification**, enabling automated validation of LLM-generated implementations.

### 1.2 Architectural Invariants

Three **non-negotiable invariants** govern all system behavior. Violation constitutes specification non-compliance.

#### 1.2.1 Deterministic Execution Guarantee

Given **identical initial state**, **identical inputs**, and **identical available tools**, the system must produce **identical execution traces** and **identical outputs**. This requires:

- **Seeded random number generation** for all stochastic operations (temperature=0 for LLM calls, explicit seeds for sampling)
- **Deterministic ordering** of parallel execution results (lexicographic node ID tie-breaking)
- **Stable serialization formats** for all inter-node communication (canonical JSON with sorted keys)
- **Rejection of non-deterministic tool implementations** at registration time

The guarantee enables **reproducible debugging**, **regression testing**, and **forensic analysis** of production failures. Non-determinism in external tool behavior is captured through **response recording and replay**, making the trace itself the source of truth.

#### 1.2.2 Artifact Lineage Immutability

Once created, **no artifact may be modified**; versioning creates new artifacts with extended lineage. This invariant ensures:

- **Complete audit trails** with tamper-evident provenance
- **Safe caching** with integrity verification through content addressing
- **Time-travel queries** determining what was known when

Implementation requires **content-addressed storage** (SHA-256 of canonical serialization), **cryptographic hash verification**, and **explicit prohibition of in-place updates**. Lineage forms a **directed acyclic graph** where nodes are artifacts and edges represent derivation relationships.

#### 1.2.3 Replay Reproducibility Requirement

Any completed execution must be **reproducible from its audit trace alone**, without access to external systems or time-dependent state. The trace must capture:

| Category | Required Content |
|----------|---------------|
| Non-deterministic inputs | LLM responses, tool outputs, with source identifiers |
| Timing information | Monotonic clock readings for ordering |
| Concurrency decisions | Resolution criteria for parallel execution |
| External state | Caching metadata for tool responses |

Replay infrastructure **validates trace completeness** before execution and **detects any divergence** from recorded behavior. Divergence triggers investigation: incomplete trace, implementation bug, or trace corruption.

### 1.3 System Boundaries

The v1 scope defines **three implementation tiers** with distinct completeness requirements.

#### 1.3.1 Fully Implemented Components (v1)

| Component | Implementation Requirement | Validation Criteria |
|-----------|---------------------------|---------------------|
| **Action Graph Planner** | Complete HTN decomposition with LLM-enhanced method generation | All planning scenarios produce valid DAGs with dependency edges |
| **DAG Executor** | Deterministic topological ordering with parallel execution | All valid DAGs execute to completion with correct results |
| **Typed Node System** | Model, MCP, Code nodes with unified contracts | All node types satisfy interface contracts under test |
| **Audit Trace Pipeline** | Complete decision/evidence/outcome capture with replay support | Traces can be serialized, retrieved, and consumed for replay |
| **CLI Runtime** | All control protocol commands and trace export capabilities | Interactive and batch modes function correctly |

These components form the **minimal viable system** capable of executing complex workflows with full observability.

#### 1.3.2 Minimally Functional Components (v1)

| Component | Minimum Functionality | Known Limitations |
|-----------|----------------------|-------------------|
| **Verification Layer** | Basic critic with pass/fail verdicts | No multi-critic ensembles, limited uncertainty quantification |
| **Intention Lifecycle** | Four-state machine with primary transitions | Simple rollback, basic escalation heuristics |
| **GraphRAG Core** | Entity/relation storage with single-hop retrieval | Limited scale, basic contradiction detection |
| **Agentic Retrieval Loop** | Iterative retrieval with stopping criteria | Simple gap detection, basic synthesis rules |

These components demonstrate **architectural integration** with clear extension paths for v2 enhancement.

#### 1.3.3 Stub and Extension Points

| Component | Interface Status | Stub Behavior |
|-----------|---------------|---------------|
| **Learning Subsystem** | Dataset collection, training pair generation, fine-tuning integration | No-op with logging, data export for external training |
| **Distributed/Scaling Runtime** | Remote execution interface, coordination protocol | Local-only execution with interface validation |

These stubs establish **contracts for future evolution** without compromising v1 deliverability.

---

## 2. Domain Model

### 2.1 Core Entity Definitions

The domain model establishes **foundational entities** with precise semantics, relationships, and lifecycle constraints.

#### 2.1.1 Agent State Composition

**Agent State** represents the complete volatile context of task execution, comprising six interconnected elements:

| Element | Contents | Update Frequency | Persistence |
|---------|----------|-----------------|-------------|
| **Goals** | Hierarchical objectives with success criteria, priorities, dependencies | Per intention lifecycle | Full trace retention |
| **Constraints** | Hard requirements (inviolable) and soft preferences (optimizable) | Per planning/replanning | Full trace retention |
| **Intentions** | Active commitments with lifecycle state, bound parameters, execution context | Per state transition | Full trace retention |
| **Evidence** | Verified facts, retrieved documents, computed results with provenance | Per node execution | Full trace retention |
| **Errors** | Categorized failures with recovery status, retry counts, escalation history | Per failure event | Full trace retention |
| **Working Memory** | Temporary structures for current reasoning, subject to capacity limits | Continuous | Checkpoint-based |

State modifications follow **transactional semantics** with **optimistic concurrency control**: modifications are staged, validated against invariants, committed to durable storage with append-only logging, and only then applied to active execution. This enables **recovery to any historical point** and **branching exploration of alternatives**.

#### 2.1.2 Intention as Commitment Abstraction

The **Intention** abstraction transforms LLM outputs from **suggestions to binding commitments**, preventing goal drift in long-horizon executions. An Intention encapsulates:

| Component | Description |
|-----------|-------------|
| **Objective** | The specific goal being pursued, with formal success criteria |
| **Plan Fragment** | The Action Graph subgraph committed for execution |
| **Preconditions** | Requirements that must hold for commitment validity |
| **Postconditions** | Expected state changes upon successful completion |
| **Verification Criteria** | How completion is validated (critics, tests, human approval) |
| **Rollback Procedures** | Operations to restore consistent state upon failure |
| **Resource Allocation** | Reserved compute, API budget, time limits |

The **commitment semantics** enforce: **resource reservation** at commitment time; **state mutation constraints** through defined transition rules; **complete observability** via audit logging; and **accountability** with traceable justification. Intentions are **immutable once committed**; modification requires explicit **revocation and re-commitment** with audit trail.

#### 2.1.3 Artifact and Lineage Model

**Artifacts** are **immutable, content-addressed data objects** flowing through the Action Graph:

```json
{
  "artifact": {
    "id": "sha256:canonical-content-hash",
    "type": "schema-reference",
    "createdAt": "ISO-8601-nanosecond-timestamp",
    "producedBy": "node-execution-reference",
    "inputs": ["artifact-id-array"],
    "content": "any-typed-per-schema",
    "metadata": {
      "sizeBytes": "integer",
      "encoding": "string",
      "compression": "string|null"
    },
    "qualitySignals": {
      "confidence": "float-0-1",
      "uncertainty": "float",
      "verificationStatus": "enum"
    }
  }
}
```

**Lineage** forms a **directed acyclic graph** enabling: **provenance queries** (all inputs contributing to output); **impact analysis** (all outputs derived from input); **recomputation** (reproducing any artifact from ancestors); and **verification** (checking recorded lineage matches actual computation). The DAG property is **enforced at creation**: cycle detection rejects any artifact that would transitively depend upon itself.

### 2.2 Data Type System

#### 2.2.1 Typed JSON Contracts

All inter-node communication follows **JSON Schema contracts** with three validation levels:

| Level | Enforcement | Failure Handling |
|-------|-------------|----------------|
| **Syntactic** | Valid JSON conforming to schema structure | Rejection at boundary with error classification |
| **Semantic** | Value ranges, cross-field constraints, referential integrity | Retry with modified inputs or escalation |
| **Pragmatic** | Contextual appropriateness, task-specific validity | Critic evaluation, potential replanning |

Contracts specify: **structure** (required/optional fields, nesting depth); **constraints** (ranges, patterns, enums); **semantics** (field purpose, valid interpretations); and **versioning** (schema evolution with compatibility rules). The type system distinguishes **domain types** (problem entities), **control types** (execution metadata), and **system types** (infrastructure references).

#### 2.2.2 Single Item vs Collection Semantics

| Aspect | Single Item | Collection |
|--------|-------------|------------|
| **Schema** | `object` with defined properties | `array` with homogeneous element schema |
| **Dependency** | Single predecessor node | Multiple predecessors with merge semantics |
| **Parallelism** | Sequential consumption | Stream processing, batch operations |
| **Error handling** | Per-item failure | Partial failure with success/failure partition |
| **Lineage** | Direct parent reference | Multiple parent references with aggregation |

Collections carry **explicit metadata**: item count (actual/expected bounds), homogeneity guarantee, and ordering semantics. The system provides **automatic collection handling**: mapping single-item nodes over collections; filtering with predicate nodes; reducing through aggregation functions; and zipping aligned collections for multi-input nodes.

#### 2.2.3 Quality Signal Taxonomy

| Signal | Type | Interpretation | Propagation |
|--------|------|---------------|-------------|
| **confidence** | float [0,1] | Estimated probability of correctness | Aggregated through lineage with combination rules |
| **uncertainty** | float [0,∞) | Entropy or variance measure | Combined as variance for independent sources |
| **completeness** | float [0,1] | Coverage of required information | Minimum across dependent outputs |
| **freshness** | timestamp | Currency of information | Explicit expiration with staleness detection |
| **consistency** | float [0,1] | Agreement across multiple paths | Conflict detection with contradiction flagging |
| **sourceReliability** | float [0,1] | Assessed trustworthiness of origin | Weighted by source authority and track record |

Quality signals are **themselves artifacts with lineage**, enabling meta-quality tracking and calibration assessment.

### 2.3 Identity and Reference Model

#### 2.3.1 Artifact ID Generation

Artifact identifiers use **content-addressing with temporal context**:

```
artifact_id = multibase_base58btc(sha256(canonical_json(content) || creation_context))
```

Where **creation context** includes: creating node ID, execution timestamp, input artifact IDs. This ensures **global uniqueness without coordination**, **automatic deduplication**, and **integrity verification**. The **48-bit timestamp prefix** (milliseconds since Unix epoch) enables efficient time-range queries.

#### 2.3.2 Node Reference Scheme

Node references use **hierarchical paths** for unambiguous identification:

```
node_ref = execution_id ":" plan_id "/" phase "/" node_id ["[" instance_index "]"]
```

Components enable: **plan versioning** (graph_id with version); **lifecycle phase tracking** (commit/execute/control/adapt); **conditional instance identification** (runtime expansion indexing); and **stable comparison** across replanning (semantic equivalence detection).

#### 2.3.3 Trace Correlation Structure

Execution traces use **hierarchical correlation IDs**:

| ID Type | Scope | Purpose |
|---------|-------|---------|
| `trace_id` | Complete execution run | Top-level grouping for all related events |
| `intention_id` | Single intention lifecycle | Commit-to-completion tracking with state transitions |
| `node_execution_id` | Single node attempt | Retry and failure analysis with attempt sequencing |
| `critic_evaluation_id` | Single critic assessment | Verification audit with evidence linking |
| `span_id` | Distributed operation | Cross-component timing and causality |

All events include **monotonic timestamps** with microsecond precision, **parent span references** for tree reconstruction, and **baggage** for context propagation.

---

## 3. Core Architecture

### 3.1 Cognitive Core

The **Cognitive Core** is the **central coordination component** unifying perception, reasoning, memory, and goal management through a structured control loop.

#### 3.1.1 Agent State Management

State management provides **transactional operations with ACID semantics** at the intention boundary:

| Operation | Semantics | Isolation Level |
|-----------|-----------|---------------|
| `getSnapshot()` | Consistent point-in-time read | Snapshot isolation |
| `proposeTransition(delta)` | Validation without commitment | Read-committed |
| `commitTransition(proposal)` | Atomic apply with logging | Serializable |
| `rollback(checkpoint)` | Restore to prior consistent state | — |
| `checkpoint()` | Capture recoverable state | — |

All modifications **append to immutable log** with: before/after values, transition justification, and correlation IDs. State snapshots are **computed on demand** for checkpointing, with **incremental delta encoding** for efficiency.

#### 3.1.2 Perception-Reasoning-Memory Unification

The Core integrates three functions through **shared artifact representation**:

| Function | Input | Output | Integration Point |
|----------|-------|--------|-----------------|
| **Perception** | User commands, tool outputs, environmental changes | Structured evidence with quality assessment | Evidence domain in Agent State |
| **Reasoning** | Evidence, goals, constraints | Plans, decisions, conclusions | Intention lifecycle transitions |
| **Memory** | Queries from reasoning, execution context | Retrieved context with provenance | Working memory with relevance scoring |

**Unification eliminates "prompt as memory" anti-pattern**: all information has typed representation, explicit provenance, and quality metadata. The **closed loop** of Plan → Execute → Control → Adapt ensures reasoning is grounded in executed actions with observed outcomes.

#### 3.1.3 Goal and Constraint Handling

**Goals** are structured with **explicit satisfaction criteria**:

```json
{
  "goal": {
    "id": "uuid",
    "description": "natural-language-and-formal-specification",
    "type": "achievement|maintenance|avoidance|optimization",
    "metric": "quantitative-evaluation-function",
    "threshold": "satisfaction-boundary",
    "deadline": "timestamp|null",
    "priority": "integer-1-10",
    "dependencies": ["goal-id-array"],
    "decompositionHint": "preferred-htn-method"
  }
}
```

**Constraints** bound the solution space:

| Type | Enforcement | Violation Response |
|------|-------------|-------------------|
| **Hard** | Pre-action validation, runtime monitoring | Immediate block, escalation if unresolvable |
| **Soft** | Preference weighting in optimization | Relaxation with explicit cost logging |
| **Temporal** | Deadline tracking, progress estimation | Warning, then throttle, then escalation |
| **Derived** | Inferred from goals and evidence | Dynamic tightening as information accumulates |

### 3.2 Intention Lifecycle State Machine

The **four-state state machine** governs commitment progression with **explicit transitions, triggers, and failure modes**.

#### 3.2.1 Commit State Semantics

**Entry conditions**: goal selected with satisfiable criteria; feasibility assessed; resources allocated.

**In-state activities**: validate preconditions against current state; select or generate plan via HTN; allocate and lock resources; record commitment with timestamp and bounds.

**Exit transitions**:
- → **Execute**: validation passed, plan ready, no interrupts
- → **Adapt**: precondition failure detected, replanning possible
- → **Cancel**: explicit revocation, resources released

#### 3.2.2 Execute State Semantics

**Entry actions**: initialize execution context; prepare input artifacts; start progress monitoring.

**In-state tracking**: completed nodes with outputs; active nodes with resource consumption; pending nodes blocked on dependencies or resources; failure detection and classification.

**Exit transitions**:
- → **Control**: all nodes completed, results assembled
- → **Adapt**: execution failure with recovery options
- → **Pause**: external interruption, state checkpointed

#### 3.2.3 Control State Semantics

**Entry triggers**: execution completion, timeout, or interrupt.

**In-state activities**: assess results against success criteria; invoke configured critics; evaluate quality signals; determine continuation, revision, or escalation.

**Exit transitions**:
- → **Execute**: critics pass, continue with current plan
- → **Adapt**: critics reject, revision required
- → **Commit**: replanning needed, new intention formulation
- → **Escalate**: unresolvable conflict, human judgment required

#### 3.2.4 Adapt State Semantics

**Entry triggers**: failure detection, critic rejection, or environmental change.

**In-state activities**: analyze root cause; generate adaptation options (retry, revise, replan); evaluate feasibility; implement selected adaptation; validate resulting structure.

**Exit transitions**:
- → **Commit**: new or modified intention ready
- → **Execute**: minor adjustment, resume with modifications
- → **Control**: adaptation complete, evaluation needed
- → **Escalate**: adaptation exhausted, no viable alternatives

#### 3.2.5 State Transition Triggers and Guards

| From | To | Trigger | Guard | Action |
|------|----|---------|-------|--------|
| — | Commit | Goal selected, feasibility OK | Resources available | Reserve resources, record commitment |
| Commit | Execute | Plan ready, prerequisites done | No interrupts, valid plan | Initialize context, start execution |
| Commit | Adapt | Precondition failure | Adaptation possible | Analyze, generate options |
| Commit | Cancel | Explicit revocation | — | Log reason, release resources |
| Execute | Control | Completion/timeout/interrupt | Checkpoint valid | Assemble results, initialize critics |
| Execute | Adapt | Execution failure | Recoverable | Capture context, classify |
| Execute | Pause | User pause command | — | Checkpoint, suspend |
| Control | Execute | Continue decision | Critics pass, resources remain | Resume/continue execution |
| Control | Adapt | Revise/replan decision | Modification scope defined | Initiate adaptation |
| Control | Escalate | Unresolvable conflict | User involvement needed | Prepare context, notify |
| Adapt | Commit | Replan success | New intention valid | Form new commitment |
| Adapt | Execute | Retry approved | Retry budget remains | Reset, re-execute |
| Adapt | Escalate | Exhaustion | No alternatives | Prepare full context |

### 3.3 Meta-Control Actions

Meta-control actions enable **user intervention at any execution point** with defined semantics.

#### 3.3.1 Continue Action Specification

Resumes execution from current state **without modification**. Parameters: optional speed modifier (normal/accelerated/single-step), breakpoint specification, notification preference. **Idempotent**—multiple continues without intervening changes have no additional effect. Does not override safety assessments; escalation requirements remain enforced.

#### 3.3.2 Revise Action Specification

Modifies current intention **while preserving commitment to underlying goal**. Parameters: revision scope (plan, constraints, parameters, or combination); revision specification (natural language or structured); preservation preferences (which results to retain).

Triggers **Adapt state** with specific focus. Appropriate for: parameter tuning, constraint adjustment, local plan repair. **Preserves completed node outputs** where valid, enabling incremental recomputation.

#### 3.3.3 Replan Action Specification

**Abandons current plan** and generates new plan from current state. Parameters: replan trigger classification; revised objective or constraints; preservation preferences (evidence, learned patterns).

Triggers **Adapt → Commit** transition. Preserves: achieved subgoals with valid results; accumulated evidence and working memory; user preference history. Discards: specific node sequences; cached results dependent on invalidated plan; quality signals from obsolete execution.

#### 3.3.4 Escalate Action Specification

**Transfers control to human oversight** with complete context. Parameters: escalation reason (classification and description); urgency level (informational/time-sensitive/critical); context packaging preference (summary/full/interactive).

Creates **suspension point**: execution paused, state preserved, notification triggered. Resolution options: **Continue** (current direction acceptable), **Revise/Replan** (specific changes directed), **Stop** (intention abandoned), or **information provision** (clarification, additional constraints).

### 3.4 Rollback and Escalation Rules

#### 3.4.1 Rollback Scope Determination

| Failure Type | Rollback Scope | Preservation |
|-------------|--------------|------------|
| Node execution error | Affected node + transitive dependents | Independent branches, upstream results |
| Critic rejection | Rejected output + dependent subgraph | Passed critic outputs, alternative paths |
| Constraint violation | All nodes since constraint binding | Evidence, constraint history for analysis |
| Plan invalidation | Entire Action Graph | Artifacts for potential reuse, failure patterns |

Rollback is **logical, not physical**: artifacts remain stored, marked deprecated for lineage purposes. New artifacts created on re-execution receive fresh IDs with extended lineage.

#### 3.4.2 Escalation Condition Detection

| Condition | Detection Method | Automatic Response |
|-----------|---------------|------------------|
| Repeated rollback | Same node fails 3+ times after adaptation | Escalate with failure pattern analysis |
| Resource exhaustion | Budget/time exceeded with no progress | Escalate with consumption report |
| Constraint unsatisfiability | No feasible solution with relaxation | Escalate with constraint analysis |
| Confidence collapse | Aggregate confidence below threshold | Escalate with uncertainty attribution |
| Safety-critical uncertainty | High stakes + low confidence | Immediate pause, urgent escalation |
| User request | Explicit escalation command | Immediate with full context |

#### 3.4.3 User Escalation Protocol

| Phase | Activity | Output |
|-------|----------|--------|
| **Context Assembly** | Summarize state, decisions, obstacles | Structured package with key information |
| **Notification** | Deliver through configured channels | Urgency-classified alert with acknowledgment |
| **Response Collection** | Parse structured or natural language guidance | Validated command or information request |
| **Resolution Integration** | Apply user decision to system state | Updated intention, revised plan, or termination |
| **Outcome Recording** | Log escalation and resolution for learning | Decision-outcome pair for policy improvement |

Supports **asynchronous escalation** with state checkpointing for extended deliberation.

### 3.5 Decision Audit Flow

#### 3.5.1 Decision Point Capture

Every significant decision is captured with:

| Element | Content |
|---------|---------|
| **Decision ID** | Unique identifier for reference |
| **Timestamp** | Microsecond-precision with monotonic ordering |
| **Decision type** | Planning, execution, control, adaptation, escalation |
| **Decision maker** | Component, critic, or user identifier |
| **Input state** | Relevant Agent State snapshot |
| **Options considered** | Alternatives with evaluation scores |
| **Selected option** | With explicit justification |
| **Expected outcome** | Predicted results with confidence |

#### 3.5.2 Evidence Attachment Requirements

| Decision Type | Required Evidence |
|-------------|-----------------|
| Planning | Goal decomposition, method comparison, selection criteria |
| Execution | Node readiness, resource availability, dependency satisfaction |
| Control | Quality signals, critic assessments, threshold comparisons |
| Adaptation | Failure analysis, option evaluation, predicted outcomes |
| Escalation | Unresolvable conflict documentation, recommended options |

Evidence links use **artifact IDs** enabling trace traversal without data duplication.

#### 3.5.3 Outcome Correlation

Post-decision, **actual outcomes are correlated**:

| Aspect | Comparison |
|--------|-----------|
| Success/failure | Against decision-time prediction |
| Metric achievement | Quantified objective satisfaction |
| Side effects | Unintended consequences inventory |
| Decision quality | Retrospective appropriateness assessment |

Correlation feeds: **critic training** (improving evaluation accuracy); **planning optimization** (better method selection); and **predictive decision quality** (confidence calibration).

---

## 4. Planning System

### 4.1 HTN Decomposition Engine

#### 4.1.1 Objective to Subgoal Transformation

The decomposition process:

```
function decompose(objective, context):
    if isPrimitive(objective):
        return primitiveNode(objective, context)
    
    applicableMethods = selectMethods(objective, context)
    if applicableMethods.empty():
        if context.allowLLMGeneration:
            generatedMethod = generateMethod(objective, context)
            cacheMethod(generatedMethod)
            applicableMethods = [generatedMethod]
        else:
            raise DecompositionFailure
    
    for method in rankMethods(applicableMethods, context):
        try:
            subtasks = instantiateMethod(method, objective, context)
            subgraph = merge([decompose(st, context) for st in subtasks])
            if validateSubgraph(subgraph, context):
                return subgraph
        except DecompositionFailure:
            continue
    
    raise DecompositionFailure
```

**Method selection criteria**: precondition satisfaction; estimated efficiency (historical performance); resource fit; and preference alignment with user constraints.

#### 4.1.2 LLM-Enhanced Method Generation

When no applicable method exists:

| Step | Activity | Output |
|------|----------|--------|
| Problem analysis | Identify structure, required capabilities | Capability requirements list |
| Prompt construction | Objective, context, examples, constraints | Structured generation prompt |
| LLM invocation | Constrained output schema | Method sketch with subgoals |
| Validation | Syntactic, semantic, safety checks | Validated method or rejection |
| Integration | Registration with provenance | Cached template for reuse |

Generated methods are **marked provisional** with elevated monitoring; **promotion to confirmed** requires successful execution across diverse cases.

#### 4.1.3 Method Template Caching

| Cache Component | Content | Management |
|-----------------|---------|------------|
| Structural pattern | Abstract objective matching criteria | Similarity-based retrieval |
| Instantiation history | Success rates, contexts, adaptations | Success-weighted retention |
| Version control | Evolution tracking, deprecation | Migration paths for updates |
| Refinement log | Adaptations made from base template | Learning for generation improvement |

### 4.2 Action Graph (DAG) Structure

#### 4.2.1 Node Types and Roles

| Type | Role | Execution Semantics |
|------|------|---------------------|
| **Model Node** | LLM inference | Synchronous, stochastic (seeded), quality signals required |
| **MCP Node** | External tool call | Synchronous with timeout, sandboxed, audit logged |
| **Code Node** | Deterministic computation | Synchronous with resource limits, sandboxed, reproducible |
| **Decision Node** | Runtime conditional expansion | Evaluated during execution, branches activated dynamically |
| **Aggregate Node** | Multi-input synchronization | Executes when all inputs ready, with merge semantics |

#### 4.2.2 Edge Dependency Semantics

| Edge Type | Semantics | Scheduling Constraint |
|-----------|-----------|----------------------|
| **Data dependency** | Output → input flow | Producer must complete before consumer starts |
| **Control dependency** | Sequencing without data flow | Source completion enables target start |
| **Resource dependency** | Shared limited resource | Coordinated access, potential queuing |
| **Conditional dependency** | Runtime-determined activation | Evaluated at expansion time |

#### 4.2.3 Conditional Branching Representation

**Decision Nodes** enable runtime adaptation:

```
[Decision Node: condition_evaluation]
    ├── condition: expression → branch_selector
    ├── branches: {value: subgraph}
    ├── default: subgraph (if no match)
    └── merge: convergence_point (optional)
```

Expansion occurs **during execution** when condition evaluation is possible, with **unselected branches pruned** from active graph. Preserves DAG property: no cycles even with dynamic activation.

### 4.3 Dependency Scheduling

#### 4.3.1 Least-Commitment Scheduling Algorithm

```
function schedule(graph, resources):
    ready = graph.sources()  // nodes with no unexecuted predecessors
    executing = {}
    completed = {}
    
    while ready.notEmpty() or executing.notEmpty():
        // Start ready nodes within resource constraints
        while ready.notEmpty() and resources.available():
            node = selectNext(ready, resources.strategy)  // priority, critical path, then lexicographic
            executing.add(node)
            resources.allocate(node)
            startExecution(node)
        
        // Wait for completion event
        completedNode = awaitAnyCompletion(executing)
        executing.remove(completedNode)
        completed.add(completedNode)
        resources.release(completedNode)
        
        // Update ready queue
        newReady = graph.successors(completedNode)
            .filter(s → s.dependencies.allIn(completed))
        ready.addAll(newReady)
    
    return completed
```

**"Least-commitment"** refers to delaying binding decisions until necessary: resource allocation deferred until execution start; result routing determined by actual output availability; and dynamic adaptation to runtime conditions.

#### 4.3.2 Parallel Execution Identification

| Parallelism Pattern | Identification | Exploitation |
|--------------------|---------------|------------|
| Independent branches | No path between nodes in either direction | Concurrent execution up to resource limits |
| Pipeline stages | Streaming dataflow with bounded buffers | Overlapped execution with backpressure |
| Speculative branches | Condition evaluation before known need | Parallel evaluation with discard of unneeded |
| Collection elements | Homogeneous processing of collection items | Data-parallel map operations |

#### 4.3.3 Critical Path Determination

Critical path analysis computes:

| Metric | Definition | Use |
|--------|-----------|-----|
| Earliest start/finish | Longest path from sources | Schedule feasibility, deadline checking |
| Latest start/finish | Reverse pass from sinks with deadline | Slack identification, flexibility quantification |
| Slack | Latest − earliest (zero = critical) | Priority assignment, optimization focus |
| Critical nodes | All paths with zero slack | Resource prioritization, monitoring intensity |

### 4.4 Backtracking and Replanning

#### 4.4.1 Failure Point Detection

| Detection Source | Failure Mode | Classification |
|-----------------|------------|--------------|
| Node execution | Error, timeout, validation failure | Transient / Persistent / Systemic |
| Dependency tracking | Required input unavailable | Structural / Environmental |
| Critic evaluation | Quality threshold breach | Correctable / Fundamental |
| Progress monitoring | Stall, resource exhaustion | Recoverable / Escalation-required |

#### 4.4.2 Partial Plan Invalidation

Invalidation scope determination:

| Scope | Trigger | Preservation |
|-------|---------|------------|
| Node-level | Isolated failure with retry | Independent branches, completed upstream |
| Subgraph-level | Dependency failure with alternatives | Valid alternative paths, shared inputs |
| Intention-level | Fundamental approach flaw | Evidence, learned patterns, user guidance |
| Cascade-level | Multiple intention failure | Minimal state for fresh start |

#### 4.4.3 Replanning Trigger Conditions

| Condition | Assessment | Response |
|-----------|-----------|----------|
| Tool failure | Alternative tool available? | Substitute, or replan with different approach |
| Missing inputs | Inputs obtainable via alternative retrieval? | Alternative retrieval, or scope reduction |
| Critic rejection | Local fix possible? | Parameter adjustment, or structural replan |
| New constraints | Current plan violates with no relaxation? | Constraint-aware replan, or escalation |
| Optimization opportunity | Significantly better plan discovered? | Cost-benefit analysis, optional replan |

### 4.5 Inspectable Plan Representation

#### 4.5.1 Serialization Format

```json
{
  "actionGraph": {
    "metadata": {
      "id": "uuid",
      "createdAt": "timestamp",
      "objective": "goal-reference",
      "provenance": "htn-decomposition-chain"
    },
    "nodes": [
      {
        "id": "node-id",
        "type": "model|mcp|code|decision|aggregate",
        "configuration": "type-specific",
        "inputContract": "schema-ref",
        "outputContract": "schema-ref",
        "estimatedResources": {"time": "ms", "cost": "currency", "tokens": "integer"}
      }
    ],
    "edges": [
      {
        "source": "node-id",
        "target": "node-id",
        "type": "data|control|resource|conditional",
        "transformation": "optional-schema",
        "condition": "optional-predicate"
      }
    ],
    "conditionalBranches": {
      "decisionNodeId": {
        "branches": {"value": "subgraph-ref"},
        "mergePoint": "node-id|null"
      }
    }
  }
}
```

#### 4.5.2 Human-Readable Rendering

| View | Purpose | Content |
|------|---------|---------|
| **Structural** | Topology understanding | Node-link diagram with type icons, dependency highlighting |
| **Temporal** | Execution planning | Gantt-style schedule with parallelism, critical path |
| **Dataflow** | Information tracking | Artifact movement, transformation annotations |
| **Rationale** | Decision explanation | HTN decomposition trace, method selection justification |

#### 4.5.3 Plan Reuse and Template Extraction

Extraction process: **identify parameterizable elements** (specific values → typed variables); **validate generalization** (applicability across diverse scenarios); **index by pattern** (objective structure, capability requirements); and **retrieve by similarity** (embedding-based or rule-based matching).

---

## 5. Node Contracts

### 5.1 Unified Node Interface

#### 5.1.1 Typed Input Contract Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "inputs": {
      "type": "object",
      "additionalProperties": {
        "oneOf": [
          {"$ref": "#/definitions/singleItemInput"},
          {"$ref": "#/definitions/collectionInput"}
        ]
      }
    },
    "parameters": {
      "type": "object",
      "description": "Node-specific configuration override"
    },
    "context": {
      "type": "object",
      "properties": {
        "executionId": {"type": "string"},
        "traceId": {"type": "string"},
        "retryCount": {"type": "integer", "minimum": 0},
        "randomSeed": {"type": "integer"}
      }
    }
  },
  "definitions": {
    "singleItemInput": {
      "type": "object",
      "properties": {
        "artifactId": {"type": "string"},
        "artifactType": {"type": "string"},
        "qualitySignals": {"type": "object"}
      },
      "required": ["artifactId"]
    },
    "collectionInput": {
      "type": "array",
      "items": {"$ref": "#/definitions/singleItemInput"}
    }
  }
}
```

#### 5.1.2 Typed Output Contract Schema

```json
{
  "type": "object",
  "properties": {
    "outputs": {
      "type": "object",
      "additionalProperties": {
        "oneOf": [
          {"type": "string"},  // artifact ID
          {"type": "array", "items": {"type": "string"}}
        ]
      }
    },
    "qualitySignals": {
      "type": "object",
      "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "number", "minimum": 0},
        "executionTimeMs": {"type": "integer"},
        "resourceConsumption": {"type": "object"}
      }
    },
    "status": {
      "type": "string",
      "enum": ["success", "partial", "failure"]
    }
  },
  "required": ["status", "qualitySignals"]
}
```

#### 5.1.3 Metadata Requirements

| Category | Content | Purpose |
|----------|---------|---------|
| Descriptive | Name, description, documentation reference | User interface, help generation |
| Capability | Required tools, models, resources | Planning-time feasibility checking |
| Execution | Determinism, idempotency, side effects | Scheduling optimization, recovery planning |
| Provenance | Version, author, verification status | Trust assessment, update management |

#### 5.1.4 Execution Status Enumeration

| Status | Meaning | Valid Transitions |
|--------|---------|-----------------|
| `PENDING` | Awaiting dependency satisfaction | `RUNNING`, `CANCELLED` |
| `READY` | Dependencies satisfied, awaiting scheduling | `RUNNING` |
| `RUNNING` | Actively executing | `COMPLETED`, `FAILED`, `TIMEOUT`, `CANCELLED` |
| `COMPLETED` | Successful finish, outputs validated | (terminal) |
| `FAILED` | Execution error or validation failure | `RETRYING`, `ADAPTING`, (terminal if exhausted) |
| `TIMEOUT` | Exceeded execution limit | `RETRYING`, `ADAPTING`, (terminal if exhausted) |
| `RETRYING` | Scheduled for retry with modified parameters | `RUNNING` |
| `CANCELLED` | Explicitly aborted | (terminal) |

### 5.2 Model Node Specification

#### 5.2.1 Model Binding and Resolution

| Aspect | Specification | Resolution |
|--------|-------------|------------|
| Model type | text, vision, multimodal, embedding, image-gen | Capability-based matching |
| Model identifier | provider/model-name with version constraint | Registry lookup with fallback chain |
| Provider configuration | endpoint, authentication, rate limits | Runtime selection based on availability |
| Capability requirements | context window, tool use, structured output | Filter available models, select optimal |

#### 5.2.2 Prompt Composition Rules

Composition order (fixed, no insertion):

1. **System prompt** — fixed behavioral instructions, persona definition
2. **Task prompt** — specific objective with parameter substitution
3. **Context injection** — retrieved evidence, prior outputs, working memory (relevance-ranked, truncated to fit)
4. **Few-shot examples** — format demonstration (optional, configurable)
5. **Output schema** — structured generation specification

**Escaping rules** prevent prompt injection; **truncation priority** ranks context elements by relevance score.

#### 5.2.3 Output Validation and Format Constraints

| Validation Level | Check | Failure Response |
|-----------------|-------|----------------|
| Structural | Valid JSON, required fields present | Retry with strengthened format instruction |
| Semantic | Value ranges, cross-field consistency | Retry with constraint clarification, or escalate |
| Format-specific | Code syntax, image dimensions, etc. | Domain-specific regeneration, or escalate |
| Critic assessment | Quality threshold, semantic appropriateness | Adaptation or escalation per critic verdict |

#### 5.2.4 Control Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | [0.0, 2.0] | 0.7 | Sampling randomness; 0.0 for deterministic |
| `maxTokens` | [1, model_limit] | model-dependent | Output length bound |
| `topP` | [0.0, 1.0] | 1.0 | Nucleus sampling threshold |
| `topK` | [1, ∞) | model-dependent | Top-k sampling limit |
| `stopSequences` | string[] | [] | Early termination triggers |
| `presencePenalty` | [-2.0, 2.0] | 0.0 | Token repetition discouragement |
| `frequencyPenalty` | [-2.0, 2.0] | 0.0 | Token frequency discouragement |

#### 5.2.5 Quality Signal Generation

| Signal | Generation Method | Calibration |
|--------|-----------------|-------------|
| `confidence` | Mean token logprob, or explicit self-assessment | Historical accuracy correlation |
| `uncertainty` | Token probability variance, or entropy estimate | Reliability-weighted aggregation |
| `consistency` | Multiple sample agreement (if configured) | Sample size, agreement threshold |
| `calibrationError` | |Brier score on held-out predictions| Ongoing monitoring, threshold alerts |

### 5.3 MCP Tool Node Specification

#### 5.3.1 Capability Discovery Integration

| Phase | Activity | Output |
|-------|----------|--------|
| Server enumeration | Query configured MCP servers | Available server list with health status |
| Tool listing | Per-server tool enumeration | Tool catalog with schemas, constraints |
| Schema retrieval | Input/output contract fetching | Validated JSON Schemas for type checking |
| Capability negotiation | Version, extension agreement | Agreed capability profile |
| Change monitoring | Subscription or polling for updates | Cache invalidation, dynamic adaptation |

#### 5.3.2 Sandbox and Access Rule Enforcement

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| Filesystem | Root restriction, path allowlisting | Read/write access limited to declared roots |
| Network | Destination allowlist, protocol filtering | Outbound connections to approved hosts only |
| Process | Resource limits, syscall filtering | CPU, memory, execution time boundaries |
| Capability | Tool-level permission grants | Specific operations authorized per tool |

**Policy gates** require additional approval for: data exfiltration-risk operations; destructive operations; and high-cost operations.

#### 5.3.3 Execution Controls

| Control | Configuration | Behavior |
|---------|-------------|----------|
| Timeout | Soft limit (warning) + hard limit (termination) | Graceful degradation, then forced stop |
| Retries | Max attempts, backoff strategy, retryable errors | Automatic recovery from transient failures |
| Rate limiting | Token bucket or leaky bucket per server/tool | Throttling with queue or error response |
| Circuit breaker | Failure threshold, recovery timeout | Fail-fast after repeated failures, automatic retry |

#### 5.3.4 Audit Logging Requirements

| Log Entry | Content | Sensitivity Handling |
|-----------|---------|-------------------|
| Invocation request | Tool, parameters, timestamp | Parameter redaction for PII/secrets |
| Execution outcome | Status, output summary, duration | Output truncation, content hashing |
| Resource consumption | CPU time, memory, I/O | Aggregate metrics, no fine-grained tracing |
| Policy decisions | Authorization results, overrides | Full logging for compliance review |

### 5.4 Code Node Specification

#### 5.4.1 Runtime Sandbox Configuration

| Aspect | Specification | Validation |
|--------|-------------|------------|
| Language runtime | Python 3.11+, Node.js 20+, SQL dialects | Version pinning, security patch level |
| Dependencies | Explicit list with version constraints | Lockfile hash verification, vulnerability scan |
| Environment variables | Declared names, secret classification | Value injection at runtime, no hardcoding |
| Import restrictions | Allowlist/blocklist of modules | Static analysis, runtime enforcement |

#### 5.4.2 Resource Limit Enforcement

| Limit | Enforcement | Action on Exceedance |
|-------|-------------|---------------------|
| Execution time (CPU) | cgroup CPU quota, watchdog timer | SIGTERM, then SIGKILL |
| Execution time (wall) | Monotonic timer, async cancellation | Cooperative termination, then forced |
| Memory | cgroup memory limit, OOM killer | Termination with partial output preservation |
| Disk | Temporary directory quota | Write failure, execution continuation or termination |
| Output size | Stream size monitoring | Truncation with warning, or termination |

#### 5.4.3 Dependency Control and Isolation

| Mechanism | Implementation | Guarantee |
|-----------|---------------|-----------|
| Reproducible environments | Locked dependencies, pinned versions | Identical resolution across executions |
| Host isolation | Container or VM sandbox | No system modification, no host data access |
| Network isolation | Disabled by default, explicit opt-in | No unexpected external communication |
| Cache efficiency | Shared read-only base layers, copy-on-write | Fast startup, space efficiency |

#### 5.4.4 Artifact and Log Production

| Output Type | Destination | Format |
|-------------|-------------|--------|
| Primary return value | Declared output artifact | JSON per output schema |
| Secondary files | Designated output directory | Any, with type annotation |
| Execution logs | stdout/stderr capture | Structured with severity levels |
| Performance metrics | Side channel to execution engine | Timing, resource profiles |

### 5.5 Common Node Semantics

#### 5.5.1 Retry Policy Configuration

```json
{
  "retryPolicy": {
    "maxAttempts": {"type": "integer", "minimum": 1, "default": 3},
    "backoffStrategy": {
      "type": "string",
      "enum": ["fixed", "linear", "exponential"],
      "default": "exponential"
    },
    "baseDelayMs": {"type": "integer", "minimum": 0, "default": 1000},
    "maxDelayMs": {"type": "integer", "minimum": 0, "default": 60000},
    "jitterFactor": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.1},
    "retryableErrors": {
      "type": "array",
      "items": {"type": "string"},
      "default": ["timeout", "transient_failure", "rate_limited"]
    }
  }
}
```

#### 5.5.2 Timeout Policy Specification

| Timeout Type | Trigger | Handling |
|-------------|---------|----------|
| Execution timeout | Total wall time exceeded | Graceful cancellation request, then forced termination |
| Stall timeout | No progress indication for interval | Health check, potential termination |
| Progress timeout | Incremental output expected but not received | Warning, then escalation |
| Stage timeout | Specific execution phase duration limit | Phase-specific handling (extend, terminate, escalate) |

#### 5.5.3 Failure Classification and Handling

| Class | Characteristics | Examples | Handling |
|-------|---------------|----------|----------|
| **TRANSIENT** | Likely success on retry | Network timeout, rate limit, temporary unavailability | Automatic retry with backoff |
| **PERSISTENT** | Consistent failure with same inputs | Invalid parameters, missing dependencies, logic errors | Escalate to adaptation or user |
| **SYSTEMIC** | Indicates fundamental approach flaw | Method inapplicable, constraint unsatisfiability | Trigger replanning |
| **CRITICAL** | Safety violation or resource exhaustion | Security policy breach, unrecoverable resource depletion | Immediate escalation, potential halt |

---

## 6. Execution Semantics

### 6.1 Deterministic DAG Executor

#### 6.1.1 Topological Execution Order

The executor implements **Kahn's algorithm variant** with deterministic tie-breaking:

```
READY_SET = {nodes with no unexecuted predecessors}
while READY_SET not empty or EXECUTING not empty:
    // Selection: priority score, then critical path, then lexicographic node ID
    node = selectHighestPriority(READY_SET)
    READY_SET.remove(node)
    EXECUTING.add(node)
    startExecution(node)
    
    // Completion handling
    completed = awaitAnyCompletion(EXECUTING)
    EXECUTING.remove(completed)
    COMPLETED.add(completed)
    
    // Ready update
    for successor in completed.successors:
        if successor.predecessors ⊆ COMPLETED:
            READY_SET.add(successor)
```

#### 6.1.2 Determinism Guarantees

| Aspect | Guarantee Mechanism |
|--------|---------------------|
| Execution order | Topological sort with deterministic tie-breaking |
| Parallel result ordering | Commit in node ID order, regardless of completion order |
| Randomness | Explicit seeds, reproducible generators |
| Serialization | Canonical JSON, sorted keys, normalized numbers |
| External interactions | Recorded in trace, replayed during re-execution |

#### 6.1.3 Non-Linear Processing Preservation

The executor **explicitly maintains DAG structure**:

| Anti-Pattern | Prevention |
|-------------|-----------|
| Linearization | Ready set maintains all available nodes; no sequential queue |
| False dependencies | Only explicit edges constrain ordering; no artificial sequencing |
| Early binding | Resource allocation deferred until execution start |
| Result ordering | Parallel results committed in deterministic order, not completion order |

Verification: execution trace analysis confirms parallel node execution where valid; structural comparison of planned vs. executed graph.

### 6.2 Node Lifecycle Management

#### 6.2.1 Pending to Running Transition

| Check | Validation | Failure Action |
|-------|-----------|--------------|
| Dependency satisfaction | All predecessors in `COMPLETED` or `SKIPPED` | Remain in `PENDING` |
| Input availability | All required artifacts exist and valid | Escalate for missing inputs |
| Schema compatibility | Actual input types match contract | Type error, potential adaptation |
| Resource availability | Budget, concurrency slots available | Queue, or escalate if persistent |
| Configuration validity | Parameters in allowed ranges | Parameter error, retry with defaults |

#### 6.2.2 Running to Completed/Success Transition

| Validation | Success Action | Failure Action |
|-----------|--------------|--------------|
| Execution completion | Capture outputs, compute quality signals | — |
| Output schema conformance | Register artifacts, update lineage | Validation failure, retry or escalate |
| Quality threshold | Mark `COMPLETED`, notify dependents | Quality failure, adaptation or escalation |
| Side effect verification | Confirm expected external state | Inconsistency detection, rollback initiation |

#### 6.2.3 Running to Failed/Error Transition

| Failure Source | Classification | Immediate Action |
|--------------|---------------|----------------|
| Uncaught exception | Implementation error | Capture stack trace, classify severity |
| Timeout expiration | Resource limit or stall | Terminate, capture partial state |
| Output validation failure | Contract violation | Log discrepancy, trigger retry or adaptation |
| Quality signal breach | Insufficient reliability | Flag for critic evaluation, potential rejection |
| External dependency failure | Tool, service, or resource unavailable | Classify transient vs. persistent, apply retry policy |

#### 6.2.4 Cancellation and Abort Handling

| Mechanism | Trigger | Grace Period | Cleanup |
|-----------|---------|------------|---------|
| Cooperative cancellation | User command, system shutdown signal | Configurable (default 5s) | Save checkpoint, release resources, log state |
| Forced termination | Non-cooperative node, critical error | None | Minimal state preservation, immediate resource reclamation |
| Emergency halt | Safety violation, security breach | None | Audit log entry, notify operators, preserve evidence |

### 6.3 Artifact Lineage Tracking

#### 6.3.1 Input-to-Output Provenance Recording

For each artifact production, lineage records:

| Element | Content | Purpose |
|---------|---------|---------|
| Direct inputs | Artifact IDs consumed as inputs | Backward trace, dependency analysis |
| Transformation | Node execution reference, configuration hash | Reproducibility, debugging |
| Execution context | Timestamp, environment fingerprint, random seed | Replay fidelity, environment debugging |
| Derivation type | Direct computation, aggregation, filtering, expansion | Appropriate aggregation semantics |

#### 6.3.2 Cross-Node Lineage Propagation

| Operation | Lineage Handling |
|-----------|---------------|
| Direct pass-through | Input lineage preserved, node reference added |
| Transformation | New lineage node with input artifacts as ancestors |
| Aggregation | Multiple input lineages merged, all ancestors preserved |
| Filtering | Subset of input lineage, filter predicate recorded |
| Expansion | Single input to multiple outputs, shared ancestry noted |

#### 6.3.3 Lineage Query Interface

```typescript
interface LineageQuery {
  // Ancestry: what contributed to this artifact?
  getAncestors(artifactId: ArtifactId, options?: {depth?: number, includeMetadata?: boolean}): LineageGraph;
  
  // Descendants: what was derived from this?
  getDescendants(artifactId: ArtifactId, options?: {depth?: number, transitive?: boolean}): LineageGraph;
  
  // Path: specific derivation chain between artifacts
  findPath(sourceId: ArtifactId, targetId: ArtifactId): ArtifactPath | null;
  
  // Impact: what would be affected by changing this?
  getImpactSet(artifactId: ArtifactId): {direct: ArtifactId[], transitive: ArtifactId[]};
  
  // Verification: check lineage integrity
  verifyIntegrity(artifactId: ArtifactId): {valid: boolean, issues: IntegrityIssue[]};
  
  // Recomputation: reproduce artifact from lineage
  recompute(artifactId: ArtifactId, context: ExecutionContext): Promise<Artifact>;
}
```

### 6.4 Execution State Persistence

#### 6.4.1 Checkpoint Creation Rules

| Trigger | Content | Storage |
|---------|---------|---------|
| Intention state transition | Complete Agent State, pending queue, in-progress status | Durable, replicated |
| Pre-action verification | Current state, next action context, rollback target | Durable |
| Periodic (time-based) | Working memory truncation, incremental delta | Durable, with retention policy |
| Pre-escalation | Full context for user resumption | Durable, with extended retention |
| Node boundary (configurable) | Minimal state for fast resume | Local, ephemeral |

#### 6.4.2 Recovery and Resume Semantics

| Scenario | Recovery Action | Resume Point |
|----------|---------------|--------------|
| Clean shutdown | Load latest checkpoint, replay from log | Last committed state |
| Unexpected termination | Load latest valid checkpoint, verify consistency | Checkpoint state with log replay |
| User pause | Preserve checkpoint, release resources | Exact pause point |
| Distributed failure | Coordinate via consensus, select consistent cut | Agreed checkpoint across components |

**Replay verification**: compare recomputed artifact IDs with recorded; divergence triggers investigation or fresh execution.

---

## 7. Verification Layer

### 7.1 Simulated Critic Architecture

#### 7.1.1 Critic Role and Responsibility Definition

| Responsibility | Description | Boundary |
|--------------|-------------|----------|
| Output assessment | Evaluate against quality criteria | No modification of evaluated output |
| Uncertainty quantification | Calibrate confidence in assessment | Explicit uncertainty, not false precision |
| Failure explanation | Identify specific deficiencies | Constructive, actionable diagnosis |
| Improvement guidance | Suggest recovery approaches | Suggestion, not mandatory prescription |

Critics **do not execute actions**—they assess and report, with system decisions based on critic verdicts.

#### 7.1.2 Critic Instantiation and Configuration

| Configuration Element | Specification | Inheritance |
|----------------------|-------------|-------------|
| Evaluation criteria | Rubric, schema, examples, thresholds | Task-type defaults, user override |
| Input scope | What context to consider | Objective-driven, user-expandable |
| Output format | Verdict structure, confidence calibration | Standard template, domain customization |
| Calibration data | Historical performance, bias correction | Organization-wide, task-specific |

### 7.2 Critic Contracts

#### 7.2.1 Evaluation Input Schema

```json
{
  "type": "object",
  "required": ["outputToEvaluate", "evaluationCriteria", "evaluationContext"],
  "properties": {
    "outputToEvaluate": {
      "type": "object",
      "properties": {
        "artifactId": {"type": "string"},
        "artifactType": {"type": "string"},
        "contentPreview": {"type": "string"}  // truncated for large artifacts
      }
    },
    "evaluationCriteria": {
      "type": "object",
      "properties": {
        "rubric": {"type": "string", "description": "Natural language evaluation instructions"},
        "schema": {"$ref": "json-schema"},  // Structural requirements
        "rules": {"type": "array", "items": {"type": "string"}},  // Specific constraints
        "examples": {
          "type": "object",
          "properties": {
            "positive": {"type": "array"},
            "negative": {"type": "array"}
          }
        },
        "threshold": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "evaluationContext": {
      "type": "object",
      "properties": {
        "objective": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "downstreamUse": {"type": "string"},  // How output will be used
        "evidence": {"type": "array", "items": {"type": "string"}}  // Supporting artifact IDs
      }
    }
  }
}
```

#### 7.2.2 Verdict Output Schema

```json
{
  "type": "object",
  "required": ["verdict", "confidence", "rationale"],
  "properties": {
    "verdict": {
      "type": "string",
      "enum": ["PASS", "CONDITIONAL_PASS", "FAIL", "UNCERTAIN"]
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "rationale": {"type": "string", "description": "Explanation of verdict reasoning"},
    "assessment": {
      "type": "object",
      "properties": {
        "validity": {"type": "number", "minimum": 0, "maximum": 1},
        "completeness": {"type": "number", "minimum": 0, "maximum": 1},
        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
        "appropriateness": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "deficiencies": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "aspect": {"type": "string"},
          "severity": {"enum": ["MINOR", "MAJOR", "CRITICAL"]},
          "description": {"type": "string"},
          "location": {"type": "string"}  // Path to problematic element
        }
      }
    },
    "suggestions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "action": {"type": "string"},
          "expectedImprovement": {"type": "string"},
          "effortEstimate": {"enum": ["LOW", "MEDIUM", "HIGH"]}
        }
      }
    },
    "evidence": {
      "type": "array",
      "items": {"type": "string"}  // Artifact IDs supporting assessment
    }
  }
}
```

#### 7.2.3 Evidence Requirement Specification

| Criterion Criticality | Required Evidence | Optional Evidence |
|----------------------|-------------------|-------------------|
| Critical | Direct citations, provenance chains, verification records | Cross-reference consistency |
| Major | Source identification, extraction method, confidence | Comparative analysis |
| Minor | General provenance indication | Detailed derivation |

Insufficient evidence triggers: `UNCERTAIN` verdict with explicit gap identification; or escalation for evidence collection guidance.

### 7.3 Acceptance and Rejection Semantics

#### 7.3.1 Pass Criteria Definition

| Verdict | Conditions | Handling |
|---------|-----------|----------|
| `PASS` | All critical criteria satisfied; aggregate score ≥ threshold; confidence ≥ minimum | Proceed to dependent execution |
| `CONDITIONAL_PASS` | Minor deficiencies with explicit waiver; monitoring commitment; fallback identified | Proceed with annotation, enhanced monitoring |
| `FAIL` | Any critical criterion failed; or aggregate score < threshold; or confidence too low for reliable verdict | Trigger adaptation or escalation |
| `UNCERTAIN` | Insufficient evidence for reliable assessment; or criterion conflict without resolution | Evidence collection, alternative critic, or escalation |

#### 7.3.2 Rejection Severity Classification

| Severity | Deficiency Characteristic | System Response |
|----------|--------------------------| ---------------|
| **MINOR** | Cosmetic, optimization, alternative acceptable | Log, annotate, proceed |
| **MAJOR** | Functional deficiency, recovery possible | Require revision, retry with fix |
| **CRITICAL** | Safety violation, correctness failure, fundamental flaw | Halt, escalate, or emergency rollback |

#### 7.3.3 Partial Acceptance Handling

| Scenario | Handling |
|----------|----------|
| Collection with mixed quality | Decompose: accept passing elements, re-execute or escalate failing elements |
| Composite output with component failures | Isolate failed components, preserve valid structure, targeted repair |
| Conditional acceptance with monitoring | Proceed with explicit tracking, re-verification at downstream checkpoint |

### 7.4 Uncertainty Handling

#### 7.4.1 Confidence Threshold Management

| Factor | Threshold Adjustment | Example |
|--------|---------------------|---------|
| Task criticality | Higher for safety-sensitive | Medical diagnosis > email drafting |
| Recovery cost | Lower when cheap retry available | API call retry vs. multi-day computation |
| Information value | Lower for learning opportunities | Exploratory research vs. production deployment |
| Historical calibration | Adjust based on observed accuracy | Under-confident critic → lower threshold |

Dynamic adjustment: increase after repeated failures; decrease when progress stalled; user override with explicit acknowledgment.

#### 7.4.2 Ambiguity Detection and Flagging

| Ambiguity Type | Detection | Response |
|---------------|-----------|----------|
| Criterion conflict | Multiple criteria suggest different verdicts | Additional evidence, alternative critic, or escalation |
| Evidence insufficiency | Required information unavailable | Targeted retrieval, or explicit uncertainty acknowledgment |
| Model uncertainty | Low confidence in evaluation itself | Ensemble evaluation, or conservative default |
| Temporal ambiguity | Information may change validity | Expiration timestamp, re-verification trigger |

### 7.5 Replanning Triggers

#### 7.5.1 Critic Rejection Response

```
onCriticRejection(rejection):
    classifyDeficiencyType(rejection.deficiencies)
    
    if localFixAvailable(rejection):
        attemptLocalFix(rejection.suggestions)
        if fixSucceeds:
            reEvaluateWithSameCritics()
        else:
            escalateOrReplan()
    
    else if structuralChangeRequired(rejection):
        triggerReplanning(
            scope = determineInvalidationScope(rejection),
            preserve = identifyValidPortions(rejection),
            focus = rejection.deficiencies.map(d → d.aspect)
        )
    
    else:
        escalateToUser(rejection)
```

#### 7.5.2 Verification Failure Escalation

Escalation occurs when: **critics disagree irreconcilably** (no consensus with confidence); **confidence too low** for reliable verdict with high stakes; or **critical criterion failed** with no recovery path. Escalation package includes: all critic evaluations with disagreement analysis; relevant evidence and context; options (override, revise criteria, replan, abort); and system recommendation with confidence.

---

## 8. Memory Architecture

### 8.1 Hybrid Memory Design

#### 8.1.1 GraphRAG and Agentic RAG Integration

| Capability | GraphRAG Contribution | Agentic RAG Contribution | Integration Point |
|-----------|----------------------|-------------------------|-------------------|
| Structured queries | Entity-relation traversal | — | Unified query planner selects optimal path |
| Semantic search | Vector index on entities | Iterative query refinement | Hybrid scoring combines similarity and relevance |
| Multi-hop reasoning | Explicit path following | Dynamic sub-question generation | Path evaluation with gap detection |
| Synthesis | Evidence-linked facts | Incremental draft refinement | Combined provenance tracking |
| Provenance | Source entity/edge IDs | Retrieval step traceability | Unified evidence graph |

#### 8.1.2 Unified Query Interface

```json
{
  "memoryQuery": {
    "query": {"type": "string", "description": "Natural language or structured query"},
    "mode": {
      "type": "string",
      "enum": ["semantic", "structured", "hybrid", "iterative"],
      "default": "hybrid"
    },
    "constraints": {
      "entityTypes": {"type": "array", "items": {"type": "string"}},
      "relationTypes": {"type": "array", "items": {"type": "string"}},
      "timeRange": {"start": "ISO-8601", "end": "ISO-8601"},
      "sourceFilter": {"type": "array", "items": {"type": "string"}},
      "confidenceThreshold": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "iteration": {
      "maxIterations": {"type": "integer", "default": 5, "maximum": 20},
      "stoppingCriteria": {
        "type": "array",
        "items": {"enum": ["coverage", "confidence", "timeBudget", "userSatisfaction"]}
      }
    },
    "synthesis": {
      "required": {"type": "boolean", "default": false},
      "format": {"type": "string", "enum": ["facts", "summary", "argument", "answer"]}
    }
  }
}
```

### 8.2 GraphRAG Subsystem

#### 8.2.1 Entity and Relation Model

| Element | Properties | Constraints |
|---------|-----------|-------------|
| **Entity** | `id`, `type`, `name`, `attributes`, `vector`, `validFrom`, `validTo` | Unique ID, typed attributes, versioned validity |
| **Relation** | `id`, `type`, `source`, `target`, `properties`, `confidence`, `provenance` | Directed, typed, attributed, confidence-weighted |

Entity types: `Concept`, `Instance`, `Event`, `Document`, `Agent`, `Artifact`, `Claim`. Relation types: `instanceOf`, `partOf`, `causes`, `supports`, `contradicts`, `mentions`, `derivedFrom`.

#### 8.2.2 Vector Search Integration

| Component | Implementation | Configuration |
|-----------|---------------|-------------|
| Embedding model | Configurable (sentence-transformers, OpenAI, etc.) | Dimension, context length, fine-tuning |
| Index structure | HNSW or IVF for approximate nearest neighbors | Recall@K, build time, memory trade-off |
| Hybrid scoring | `score = α·vector_similarity + β·graph_relevance + γ·recency` | Tunable weights per query type |
| Deduplication | Entity resolution via clustering and linking | Threshold, blocking strategy |

#### 8.2.3 Multi-Hop Traversal Semantics

| Operation | Parameters | Result |
|-----------|-----------|--------|
| `follow` | entity, relationType, direction | Adjacent entities with edge properties |
| `path` | source, target, maxLength, constraints | Shortest or all paths meeting criteria |
| `expand` | entities, relationTypes, depth | Ego network with hop-limited traversal |
| `subgraph` | query pattern, bindings | Induced subgraph matching pattern |

Traversal is **lazy and composable**: intermediate results are views, materialized only when needed.

#### 8.2.4 Evidence Linking and Provenance

| Provenance Element | Recording | Query |
|-------------------|-----------|-------|
| Source document | Document ID, retrieval timestamp, excerpt location | Source retrieval, citation generation |
| Extraction method | Model/tool used, prompt/version, confidence | Method reliability assessment |
| Temporal validity | Observation time, expiration, update frequency | Currency checking, staleness detection |
| Derivation chain | Transformations applied, intermediate artifacts | Reconstruction, debugging, audit |

### 8.3 Agentic Retrieval Loop

#### 8.3.1 Iterative Retrieval Process State Machine

| State | Activity | Exit Condition |
|-------|----------|---------------|
| **PLAN** | Formulate information needs, generate sub-questions, prioritize | Query plan ready, coverage targets set |
| **RETRIEVE** | Execute queries against GraphRAG, tools, external sources | Results received or timeout |
| **VALIDATE** | Assess coverage, detect gaps, identify contradictions, check source reliability | Validation complete, gaps identified |
| **SYNTHESIZE** | Integrate findings, draft conclusion, identify remaining uncertainty | Draft produced, synthesis quality assessed |
| **DECIDE** | Evaluate stopping criteria, determine need for iteration | Continue (to PLAN with refined needs), or Stop |

#### 8.3.2 Gap Detection Algorithm

```
function detectGaps(currentEvidence, informationNeeds):
    gaps = []
    for need in informationNeeds:
        coverage = assessCoverage(currentEvidence, need)
        if coverage < need.threshold:
            gaps.append({
                need: need,
                coverage: coverage,
                missingAspects: identifyMissingAspects(currentEvidence, need),
                suggestedQueries: generateRefinementQueries(need, currentEvidence)
            })
    
    # Detect contradictions
    contradictions = findContradictoryClaims(currentEvidence)
    for contradiction in contradictions:
        gaps.append({
            type: "CONTRADICTION",
            claims: contradiction.claims,
            resolutionStrategies: suggestResolution(contradiction)
        })
    
    return prioritizeGaps(gaps)
```

#### 8.3.3 Synthesis Rule Application

| Synthesis Type | Rule | Output |
|--------------|------|--------|
| Fact aggregation | Merge consistent claims, flag conflicts | Unified fact set with confidence |
| Summary generation | Extractive or abstractive, length-constrained | Condensed representation |
| Argument construction | Claim-premise-evidence structure | Structured reasoning chain |
| Answer formulation | Direct response to query with justification | Final deliverable with provenance |

#### 8.3.4 Stopping Criteria Evaluation

| Criterion | Evaluation | Default Threshold |
|-----------|-----------|-------------------|
| Coverage | Fraction of information needs with sufficient evidence | 0.9 |
| Confidence | Aggregate confidence in synthesized conclusions | 0.85 |
| Stability | Minimal change between synthesis iterations | <5% content change |
| Time budget | Cumulative retrieval and synthesis time | 5 minutes |
| Cost budget | API calls, compute, external queries | Configurable per task |
| User satisfaction | Explicit feedback or implicit signals (dwell time, follow-up) | N/A (explicit override) |

### 8.4 Execution Trace Storage

#### 8.4.1 Input/Output Recording

| Element | Content | Compression |
|---------|---------|-------------|
| Node inputs | Artifact IDs, parameter values, context snapshot | Reference-based, no content duplication |
| Node outputs | Artifact IDs, quality signals, execution metadata | Content-addressed storage |
| Intermediate values | Large outputs, streaming results | Chunking with incremental hashing |

#### 8.4.2 Error and Decision Persistence

| Record Type | Content | Indexing |
|------------|---------|----------|
| Error record | Classification, context, stack trace, recovery attempts | By type, node, time, intention |
| Decision record | Options, selection, rationale, expected outcome | By type, objective, outcome correlation |
| State transition | Before/after states, triggering event, guard evaluation | By intention, chronological |

#### 8.4.3 Outcome Correlation

Correlation links: **decision → expected outcome prediction → actual outcome observation → quality assessment**. Enables: decision policy improvement; critic calibration; and predictive decision quality estimation.

---

## 9. Control Protocol

### 9.1 User Command Interface

#### 9.1.1 Continue Command Semantics

| Aspect | Specification |
|--------|-------------|
| Preconditions | Execution paused, checkpoint valid, no blocking safety conditions |
| Effects | Clear pause flag, resume scheduling, reset timeout monitors |
| Parameters | `speed` (normal/accelerated/single-step), `breakpoint` (optional node/condition), `notification` (silent/progress/verbose) |
| Idempotency | Multiple continues without intervening changes: no additional effect |

#### 9.1.2 Pause Command Semantics

| Aspect | Specification |
|--------|-------------|
| Triggers | User request, automatic at verification points, pre-escalation |
| Effects | Halt new node scheduling, allow active nodes to reach checkpoint, preserve state |
| Grace period | Configurable (default: 30s for cooperative completion) |
| State | `PAUSED` with full context for resume |

#### 9.1.3 Revise Command Semantics

| Aspect | Specification |
|--------|-------------|
| Scope | Current intention parameters, constraints, or plan fragment |
| Input | Natural language description or structured modification request |
| Effects | Transition to `ADAPT` with revision focus, preserve valid results |
| Preservation | User-specified: all results, none, or selective by artifact type |

#### 9.1.4 Replan Command Semantics

| Aspect | Specification |
|--------|-------------|
| Scope | Discard current plan, regenerate from current state |
| Triggers | User request, automatic on structural failure, optimization opportunity |
| Preservation | Evidence, learned patterns, user preferences; discard plan-specific results |
| Effects | Transition to `ADAPT` → `COMMIT` with new intention formulation |

#### 9.1.5 Stop Command Semantics

| Aspect | Specification |
|--------|-------------|
| Scope | Current intention, or entire session |
| Effects | Graceful termination with checkpoint, resource release, final trace export |
| Options | `immediate` (abort active nodes), `graceful` (wait for checkpoints), `export` (trace destination) |

### 9.2 Natural Language Feedback Processing

#### 9.2.1 Feedback to Control Signal Conversion

```
function convertFeedback(naturalLanguage, context):
    intent = classifyIntent(naturalLanguage)  // continue, revise, replan, stop, info-request
    
    if intent == "revise":
        scope = identifyRevisionScope(naturalLanguage, context)
        specification = extractModification(naturalLanguage, scope)
        return ReviseCommand(scope, specification)
    
    if intent == "replan":
        trigger = classifyReplanTrigger(naturalLanguage)
        constraints = extractConstraintChanges(naturalLanguage)
        return ReplanCommand(trigger, constraints)
    
    # ... similar for other intents
    
    if confidence < threshold:
        return ClarificationRequest(parsedInterpretation, alternatives)
```

#### 9.2.2 Preference Extraction

| Preference Type | Extraction Method | Application |
|-----------------|-------------------|-------------|
| Output format | Explicit request, pattern in feedback | Node output contract adjustment |
| Verbosity level | Direct statement, implicit in questions | Progress indication, explanation detail |
| Risk tolerance | Hedging language, explicit safety requests | Confidence thresholds, escalation triggers |
| Prioritization | Comparative statements, emphasis | Scheduling priority, resource allocation |

#### 9.2.3 Acceptance Criteria Derivation

| Source | Derivation | Formalization |
|--------|-----------|-------------|
| Explicit success statement | Direct parsing | Success criteria predicate |
| Comparative feedback ("better than X") | Benchmark identification, gap analysis | Quantitative metric with threshold |
| Negative feedback ("avoid Y") | Constraint inversion | Hard or soft constraint addition |
| Iterative refinement | Change tracking between versions | Convergence detection, stability metric |

### 9.3 Critic Configuration Propagation

#### 9.3.1 Rule and Checklist Generation

| Feedback Type | Critic Enhancement | Generation Method |
|-------------|-------------------|-------------------|
| Specific deficiency mention | Add explicit check for that aspect | Template instantiation with context |
| Comparative assessment ("X was better") | Add comparative evaluation dimension | Benchmark-based criteria |
| Process complaint ("took too long") | Add efficiency or cost criterion | Resource consumption tracking |
| Safety concern | Elevate to critical criterion, add policy gate | Safety rule database lookup |

#### 9.3.2 Auto-Critique Policy Learning

| Learning Target | Data Source | Update Mechanism |
|-----------------|-------------|----------------|
| Critic weight calibration | Historical accuracy of predictions | Bayesian update, moving average |
| Threshold adjustment | User override patterns | Conservative shift with feedback integration |
| Rule prioritization | Resolution path success rates | Multi-armed bandit or gradient-based |
| Example selection | Similar task performance | Embedding-based retrieval, case-based reasoning |

### 9.4 Command Impact Scope

| Command | Execution System | Planning System | Verification Layer |
|---------|---------------|---------------|-------------------|
| `continue` | Resume scheduling, reset timeouts | No direct effect | May trigger pending critic evaluations |
| `pause` | Halt scheduling, preserve checkpoints | No new planning initiated | Pause critic pipelines, preserve state |
| `revise` | Adaptation of active/queued nodes | Local replanning, method substitution | Critic criteria update, re-evaluation queue |
| `replan` | Subgraph invalidation, result preservation | Full or partial regeneration | New plan validation, critic configuration |
| `stop` | Graceful termination, resource release | No effect (already committed) | Final verification, trace completion |

---

## 10. Audit and Artifacts

### 10.1 Formal Trace Artifact Schema

#### 10.1.1 Decision Record Structure

```json
{
  "decisionRecord": {
    "id": "uuid",
    "timestamp": "ISO-8601-nanoseconds",
    "traceId": "correlation-id",
    "intentionId": "intention-reference",
    "decisionType": "planning|execution|control|adaptation|escalation|meta-control",
    
    "context": {
      "agentStateSnapshot": "state-version-reference",
      "relevantGoals": ["goal-id-array"],
      "activeConstraints": ["constraint-summary"],
      "availableEvidence": ["artifact-id-array"]
    },
    
    "optionsConsidered": [
      {
        "optionId": "identifier",
        "description": "natural-language-or-structured",
        "predictedOutcome": "expected-result",
        "score": "evaluation-metric",
        "confidence": "probability-estimate"
      }
    ],
    
    "selectedOption": {
      "optionId": "reference-to-above",
      "selectionRationale": "justification-text",
      "decisionAuthority": "component|user|hybrid",
      "expectedOutcome": "prediction-for-correlation"
    },
    
    "outcomeCorrelation": {
      "actualOutcome": "observed-result",
      "correlationTimestamp": "when-determined",
      "qualityAssessment": "success|partial|failure",
      "learningSignal": "improvement-opportunity"
    }
  }
}
```

#### 10.1.2 Evidence Attachment Format

| Attachment Type | Structure | Validation |
|-----------------|-----------|------------|
| Direct artifact reference | `{"artifactId": "...", "relevanceScore": 0.0-1.0}` | Artifact existence, type compatibility |
| Retrieved context | `{"query": "...", "results": ["artifact-id"], "retrievalMetadata": {...}}` | Query provenance, result freshness |
| Computed derivation | `{"operation": "...", "inputs": ["..."], "output": "...", "verification": "..."}` | Reproducibility check |
| External source | `{"source": "...", "accessTimestamp": "...", "contentHash": "..."}` | Source reliability assessment |

#### 10.1.3 Outcome Correlation Format

| Field | Description | Use |
|-------|-------------|-----|
| `predictedOutcome` | Structured prediction at decision time | Bias detection, calibration assessment |
| `actualOutcome` | Observed result with measurement | Ground truth for learning |
| `attributionScore` | Degree outcome caused by decision vs. external factors | Policy improvement targeting |
| `latency` | Time from decision to outcome observation | Timeliness assessment |
| `surprises` | Unexpected aspects, model failure modes | Error analysis, robustness improvement |

### 10.2 Artifact ID and Lineage System

#### 10.2.1 ID Generation Algorithm

```
function generateArtifactId(content, creationContext):
    canonicalContent = canonicalizeJson(content)  // sorted keys, normalized numbers
    contextSerialization = serialize(creationContext)  // timestamp, nodeRef, inputs
    
    hashInput = concat(
        multicodecPrefix("sha2-256"),
        canonicalContent,
        contextSerialization
    )
    
    rawHash = sha256(hashInput)
    return multibaseEncode(rawHash, "base58btc")
```

#### 10.2.2 Lineage Graph Structure

| Element | Representation | Operations |
|---------|---------------|------------|
| Nodes | Artifact productions with metadata | CRUD (Create, Read only; no Update, Delete) |
| Edges | `producedFrom` relationships with transformation type | Append-only, cycle detection |
| Annotations