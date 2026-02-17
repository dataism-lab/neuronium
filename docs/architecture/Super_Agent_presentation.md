1AI Super Agent

NEURONIUM

Self-Organizing Multi-Agent
System for Complex Task Solving

ABSTRACT

AI Super Agent is a self-organizing multi-agent system for reliable
long-horizon, high-entropy task solving in real-world environments.

At its core, a Cognitive Core unifies perception, reasoning, memory, and
goal management via a structured control loop. Planning is represented as an
Action Graph (DAG) generated via LLM-enhanced HTN.

The system combines hybrid memory (GraphRAG + agentic retrieval), while
reliability and safety are enforced through simulated critics.

Tool interoperability and security are enabled via MCP for typed, sandboxed
enterprise integration, evaluated on business automation and research
intelligence workflows.

Long-horizon & high-entropy tasks

● Dozens of steps, latency, tool failures
● Uncertain data and volatile environments
● Hallucinations, goal drift, looping
● Reliability, auditability, predictability

WHY PROMPTED AGENTS FAIL?
PROBLEM

Linear plans (CoT/ReAct): single
point of failure

Error propagation: one bad step
collapses the whole trajectory

No explicit state: hard to debug,
verify, reproduce

Weak self-checking without
a control loop

Tool mismatch: formats errors
→ wrong outputs

● LLMs become reliable when embedded into a control architecture.
● Replace “chat completion” with goal-directed execution
● Represent plans as graphs, not chains
● Add verification loops before actions
● Handle failures via backtracking + replanning
● Treat memory as a system component, not a prompt

SOLUTION
Commitment-aware Cognitive Core

An explicit Intention lifecycle
(commit → execute → control → adapt)
for stable long-horizon behavior

KEY CONTRIBUTIONS

Action-Graph Planning
HTN-generated DAG execution
with backtracking +
replanning instead of
brittle linear plans

Verification Control
Uncertainty-driven
verification that uses
simulated critique
to reduce hallucinations
and false success

SYSTEM OVERVIEW

● Cognitive Core: control loop + commitment-aware decision making
● Action Graph Planning: HTN-driven decomposition + dependency-aware scheduling
for parallelism
● Robust Execution: backtracking + replanning
● Hybrid Memory: GraphRAG + agentic retrieval for iterative research and synthesis
● Reliability Layer: simulated critics for verification
● MCP Tool Layer: typed, sandboxed enterprise interoperability

COGNITIVE CORE

● Maintains explicit Agent State: goals, constraints, intentions, evidence, errors
● Runs a closed loop: Plan → Execute → Control → Adapt
● Treats intentions as commitments (prevents goal drift in long runs)
● Unifies perception, reasoning, memory, and goal management
● Meta-control actions: continue, revise, replan, escalate
● Outputs an audit trace (decisions + evidence + outcomes)

PLANNING = HTN → ACTION GRAPH (DAG)

● HTN decomposition: objective → subgoals → tool-level operators
● LLM-enhanced methods: generate missing methods on demand, cache as templates
● Output is a Action Graph: nodes = actions, edges = dependencies
● Conditional branching: decision nodes expanded at runtime
● Least-commitment scheduling enables parallel execution where possible
● Plan representation is inspectable, reusable, and auditable

MODEL NODE

● Receives a task request already bound to a specific model (LLM / Vision / Image Gen / etc.)
● Executes inference with a fixed system prompt + task prompt + context
● Produces a typed output (JSON / text / code / image) according to a contract
● Validates output against format constraints
● Exposes controls: temperature, max tokens, stop rules
● Returns quality signals: confidence/uncertainty, error codes, trace metadata

MCP NODES

● Represent a single MCP tool call inside the Action Graph
● Consume typed inputs, execute the tool, return typed outputs (JSON contracts)
● Use capability discovery (available tools + constraints)
● Enforce sandbox & access rules: roots, least privilege, policy gates
● Provide execution controls: timeouts, retries, rate limits, audit logging
● Serve as the typed interface boundary between the agent and enterprise systems

CODE NODES

● Execute deterministic code inside the graph (Python/JS/SQL snippets)
● Used for parsing, validation, data transforms, formatting, calculations
● Inputs/outputs follow typed contracts (schemas, artifact handles)
● Sandboxed runtime with timeouts, resource limits, dependency controls
● Produces artifacts + logs for reproducibility and debugging
● Offload deterministic steps from LLMs to reduce hallucinations and variance

EXECUTION

● Every node consumes/produces typed JSON (contracted inputs/outputs)
● Single item: one JSON object
● Collection: a JSON list of objects
● All artifacts are versioned with IDs + lineage
● Node execution is tracked with status + retries
● Full trace logs per-node + outcomes for audit and debugging

CONTROL

● User can control the run at any step: continue, pause, revise, replan, stop
● User provides feedback in natural language
● Feedback is converted into control signals: preferences, acceptance criteria
● The same feedback is used to configure simulated critics (rules + checklists)
● Replan triggers: tool failure, missing inputs, critic rejection, new constraints
● Escalates to the user when it can’t proceed safely
● Over time, feedback becomes default auto-critique policies for similar tasks

HYBRID MEMORY

● GraphRAG + Agentic RAG
● Stores facts, entities, relations, and constraints as a semantic graph
● Stores execution traces: inputs/outputs, errors, decisions, outcomes
● Enables iterative retrieval and evidence-linked synthesis (deep research)
● Retrieves evidence from internal docs, databases, and tools via MCP
● Provides provenance: what was used, why, and where it came from

GRAPHRAG (GRAPH-BASED RETRIEVAL-AUGMENTED GENERATION)

● GraphRAG combines vector search (semantic similarity) with graph traversal (explicit relations)
● Supports multi-hop reasoning across entities, documents, and dependencies
● Enables structured navigation by following entity, relation, and dependency paths
● Retrieves connected evidence, not just top-k chunks
● Enables contradiction-aware synthesis (compare sources, track provenance)
● Outputs are evidence-linked (node/edge IDs + source references)

AGENTIC RAG (ITERATIVE RETRIEVAL AS A PROCESS)

● Retrieval is a loop, not a single top-k query
● Plans retrieval steps: query expansion, sub-questions, targeted tool calls
● Validates evidence: detect gaps, contradictions, missing sources
● Synthesizes incrementally: drafts, critique, refinement with new retrieval
● Uses stopping criteria: coverage, confidence, budget/time limits
● Produces an evidence graph + final answer with traceability

LEARNING

● Maintain a dataset of solved tasks with labeled “good” outputs
● Convert cases into training pairs: instruction + context → target output
● Fine-tune OpenAI models via SFT to reproduce high-quality solutions
● Curate data using critic pass + human approval
● Validate with held-out tests before deployment
● Deploy as a dedicated Model Node per task domain

DEMO: Report preparation

● Baseline vs AI Super Agent
● Baseline: workflow engineering — 60–90 min to set up a new reporting flow
● Treatment: 1 instruction → DAG in 10–30 sec
● Workload: weekly management report — 1 run / week
● Evaluation: end-to-end time human 30 min → agent 3 min
● Auditability & Deliverables: 0% trace → 100% trace
(final report + DAG + execution trace + evidence lineage)

CONCLUSION

● Long-horizon, high-entropy work requires architecture, not just prompting
● AI Super Agent: Cognitive Core with Action Graph and Hybrid Memory
● Produces auditable artifacts with traceable execution, not just chat outputs
● Enables enterprise workflows: faster setup, lower human effort, higher reliability
● Next step: scale a library of company skills and tuned models per task family
● Open challenge: standardized evaluation for long-horizon, tool-based workflows