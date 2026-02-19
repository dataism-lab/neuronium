"""HTN-lite recursive planner backend (`htn_recursive_v0`)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from pydantic import ValidationError

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphEdge, GraphMetadata, GraphNode
from neuronium_agent.planning.extraction_contract import (
    ClarificationQuestion,
    ClarificationRequest,
    ExtractionEnvelope,
    MissingField,
    extraction_envelope_json_schema,
)
from neuronium_agent.planning.htn_contract import (
    HtnDecompositionStep,
    HtnLeafOperator,
    HtnMethodChoice,
    HtnSubgoal,
)
from neuronium_agent.planning.planner_contract import (
    PlannerDecisionTrace,
    PlannerEscalation,
    PlannerOutcome,
    PlannerRequest,
    PlannerResult,
)
from neuronium_agent.verification.business_critic import (
    BUSINESS_CRITIC_SYSTEM_PROMPT,
    WEB_CRITIC_SYSTEM_PROMPT,
)
from neuronium_agent.verification.generic_critic import GENERIC_TASK_CRITIC_SYSTEM_PROMPT
from neuronium_agent.verification.demo_critic import critic_json_schema


ExecutePlannerGraphFn = Callable[
    [ActionGraph, dict[str, object], bool],
    dict[str, NodeOutput],
]


@dataclass(frozen=True)
class _BackendOptions:
    max_depth: int = 4
    max_frontier: int = 64
    max_total_nodes: int = 64
    model_assisted_method_selection: bool = False
    model_assisted_max_calls: int = 2
    planner_node_prefix: str = "htn_method_select"


@dataclass(frozen=True)
class _ExtractionArtifacts:
    envelope: ExtractionEnvelope
    user_request_artifact_id: str
    candidate_urls: list[str]
    candidate_paths: list[str]
    candidate_basenames: list[str]


@dataclass(frozen=True)
class HtnRecursivePlannerBackend:
    """HTN-lite backend: recursively decompose objective into a DAG."""

    @property
    def backend_name(self) -> str:
        return "htn_recursive_v0"

    @property
    def backend_version(self) -> str:
        return "0"

    def plan(
        self,
        *,
        request: PlannerRequest,
        execute_graph: ExecutePlannerGraphFn,
    ) -> PlannerOutcome:
        opts = self._options_from_request(request)

        extraction = self._run_extraction_pipeline(
            request=request,
            execute_graph=execute_graph,
            options=opts,
        )
        effective_metadata, missing_fields = self._resolve_inputs_tool_first(
            request=request,
            extraction=extraction,
            execute_graph=execute_graph,
        )

        if missing_fields:
            clarification = self._build_clarification_request(
                request=request,
                missing_fields=missing_fields,
                extraction=extraction,
                effective_metadata=effective_metadata,
                execute_graph=execute_graph,
                options=opts,
            )
            clarification_request_artifact_id = self._persist_clarification_request_artifact(
                request=request,
                clarification=clarification,
                evidence_artifact_ids=[extraction.user_request_artifact_id],
                execute_graph=execute_graph,
            )
            decision_trace = PlannerDecisionTrace(
                subgoals=["extract_inputs", "resolve_inputs"],
                selected_methods=["tool_first_resolution", "escalate_for_clarification"],
                justification_keys=["missing_critical_parameters"],
                notes={
                    "context_kind": self._context_kind(effective_metadata),
                    "missing_fields": [m.model_dump(mode="json") for m in missing_fields],
                    "clarification_request_artifact_id": clarification_request_artifact_id,
                },
            )
            return PlannerEscalation(
                reason="missing_critical_parameters",
                backend_name=self.backend_name,
                backend_version=self.backend_version,
                clarification_request_artifact_id=clarification_request_artifact_id,
                missing_fields=[m.model_dump(mode="json") for m in missing_fields],
                evidence_artifact_ids=[extraction.user_request_artifact_id],
                operator_catalog_hash=request.operator_catalog_hash,
                decision_trace=decision_trace,
            )

        context_kind = self._context_kind(effective_metadata)

        subgoals: list[HtnSubgoal] = []
        method_choices: list[HtnMethodChoice] = []
        leaves: list[HtnLeafOperator] = []
        steps: list[HtnDecompositionStep] = []
        method_path: list[str] = []

        planner_calls = 0
        step_idx = 0

        root = HtnSubgoal(
            subgoal_id="sg_root",
            title=request.objective,
            depth=0,
            kind="root",
            payload={"context_kind": context_kind},
        )
        queue: list[HtnSubgoal] = [root]
        subgoals.append(root)

        while queue:
            if len(queue) > opts.max_frontier:
                raise ValueError(
                    "HTN planner frontier exceeded max_frontier "
                    f"({opts.max_frontier})"
                )
            current = queue.pop(0)
            if current.depth > opts.max_depth:
                raise ValueError(
                    "HTN planner exceeded max_depth "
                    f"({opts.max_depth}) at subgoal {current.subgoal_id}"
                )

            method_id, rationale = self._select_method(
                request=request,
                subgoal=current,
                context_kind=context_kind,
                options=opts,
                execute_graph=execute_graph,
                planner_calls=planner_calls,
            )
            if rationale == "model_assisted":
                planner_calls += 1
            method_path.append(method_id)

            produced_subgoals, produced_leaves = self._expand_subgoal(
                request=request,
                metadata=effective_metadata,
                subgoal=current,
                method_id=method_id,
                context_kind=context_kind,
            )
            method_choices.append(
                HtnMethodChoice(
                    subgoal_id=current.subgoal_id,
                    method_id=method_id,
                    rationale_key=rationale,
                    produced_subgoal_ids=[s.subgoal_id for s in produced_subgoals],
                )
            )
            step_idx += 1
            steps.append(
                HtnDecompositionStep(
                    step_index=step_idx,
                    subgoal_id=current.subgoal_id,
                    depth=current.depth,
                    action="expand_subgoal",
                    details={
                        "method_id": method_id,
                        "produced_subgoals": [s.subgoal_id for s in produced_subgoals],
                        "produced_leaf_nodes": [l.node_id for l in produced_leaves],
                    },
                )
            )

            if produced_subgoals:
                subgoals.extend(produced_subgoals)
                queue.extend(produced_subgoals)
            if produced_leaves:
                leaves.extend(produced_leaves)

            if len({leaf.node_id for leaf in leaves}) > opts.max_total_nodes:
                raise ValueError(
                    "HTN planner exceeded max_total_nodes "
                    f"({opts.max_total_nodes})"
                )

        action_graph = self._build_action_graph(
            request=request,
            leaves=leaves,
            context_kind=context_kind,
        )
        decision_trace = PlannerDecisionTrace(
            subgoals=[s.subgoal_id for s in subgoals],
            selected_methods=[m.method_id for m in method_choices],
            justification_keys=[m.rationale_key for m in method_choices],
            decomposition_steps=[asdict(s) for s in steps],
            method_expansion_path=list(method_path),
            leaf_operators=[asdict(l) for l in leaves],
            notes={
                "context_kind": context_kind,
                "planner_calls": planner_calls,
                "subgoal_count": len(subgoals),
                "leaf_count": len(leaves),
                "max_depth_observed": max((s.depth for s in subgoals), default=0),
                "backend_options": {
                    "max_depth": opts.max_depth,
                    "max_frontier": opts.max_frontier,
                    "max_total_nodes": opts.max_total_nodes,
                    "model_assisted_method_selection": opts.model_assisted_method_selection,
                    "model_assisted_max_calls": opts.model_assisted_max_calls,
                },
                "extraction": {
                    "task_type": extraction.envelope.intent.task_type,
                    "inputs_keys": sorted(extraction.envelope.inputs.keys()),
                },
                "effective_inputs": {
                    "urls": list(effective_metadata.get("urls", [])) if isinstance(effective_metadata.get("urls"), list) else [],
                    "doc_paths": list(effective_metadata.get("doc_paths", [])) if isinstance(effective_metadata.get("doc_paths"), list) else [],
                },
                "user_request_artifact_id": extraction.user_request_artifact_id,
            },
        )
        return PlannerResult(
            action_graph=action_graph,
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            operator_catalog_hash=request.operator_catalog_hash,
            decision_trace=decision_trace,
        )

    @staticmethod
    def _options_from_request(request: PlannerRequest) -> _BackendOptions:
        raw = request.spec.backend_options or {}
        return _BackendOptions(
            max_depth=int(raw.get("max_depth", 4)),
            max_frontier=int(raw.get("max_frontier", 64)),
            max_total_nodes=int(raw.get("max_total_nodes", 64)),
            model_assisted_method_selection=bool(
                raw.get("model_assisted_method_selection", False)
            ),
            model_assisted_max_calls=int(raw.get("model_assisted_max_calls", 2)),
            planner_node_prefix=str(raw.get("planner_node_prefix", "htn_method_select")),
        )

    def _run_extraction_pipeline(
        self,
        *,
        request: PlannerRequest,
        execute_graph: ExecutePlannerGraphFn,
        options: _BackendOptions,
    ) -> _ExtractionArtifacts:
        extraction_graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"htn-extract-{request.execution_id[:12]}",
                description=f"Extract user inputs for {request.stage_id}",
            ),
            nodes=[
                GraphNode(
                    node_id="persist_user_request",
                    node_type="mcp",
                    label="Persist user request artifact",
                    parameters={
                        "tool_name": "artifact.put_json",
                        "tool_args": {
                            "artifact_type": "planner.user_request",
                            "json": {
                                "objective": request.objective,
                                "constraints": list(request.constraints),
                                "metadata": request.metadata,
                                "runbook_id": request.runbook_id,
                                "stage_id": request.stage_id,
                                "execution_id": request.execution_id,
                            },
                            "produced_by_node_ref": (
                                f"{request.execution_id}:{request.runbook_id}:{request.stage_id}/commit/persist_user_request"
                            ),
                            "parent_artifact_ids": [],
                            "media_type": "application/json",
                        },
                    },
                    priority=0,
                ),
                GraphNode(
                    node_id="extract_entities",
                    node_type="mcp",
                    label="Extract deterministic entities from objective",
                    parameters={
                        "tool_name": "text.extract_entities",
                        "tool_args": {"text": request.objective},
                    },
                    priority=1,
                ),
            ],
            edges=[],
        )
        extraction_outputs = execute_graph(extraction_graph, {}, True)

        request_artifact_id = ""
        request_artifact_output = extraction_outputs.get("persist_user_request")
        if request_artifact_output is not None:
            request_artifact_id = str(request_artifact_output.outputs.get("artifact_id", "")).strip()

        entities_output = extraction_outputs.get("extract_entities")
        candidate_urls: list[str] = []
        candidate_paths: list[str] = []
        candidate_basenames: list[str] = []
        if entities_output is not None:
            raw_urls = entities_output.outputs.get("urls", [])
            raw_paths = entities_output.outputs.get("file_paths", [])
            raw_basenames = entities_output.outputs.get("basenames", [])
            if isinstance(raw_urls, list):
                candidate_urls = [str(x) for x in raw_urls if str(x).strip()]
            if isinstance(raw_paths, list):
                candidate_paths = [str(x) for x in raw_paths if str(x).strip()]
            if isinstance(raw_basenames, list):
                candidate_basenames = [str(x) for x in raw_basenames if str(x).strip()]

        model_graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"htn-extract-envelope-{request.execution_id[:12]}",
                description=f"Build extraction envelope for {request.stage_id}",
            ),
            nodes=[
                GraphNode(
                    node_id=f"{options.planner_node_prefix}_extract_envelope",
                    node_type="model",
                    label="Extract intent and input envelope",
                    parameters={
                        "system_prompt": (
                            "You extract planning inputs from user objective. "
                            "Return strict JSON matching schema."
                        ),
                        "json_schema": extraction_envelope_json_schema(),
                    },
                )
            ],
            edges=[],
        )
        model_prompt = self._build_extraction_prompt(
            request=request,
            candidate_urls=candidate_urls,
            candidate_paths=candidate_paths,
            candidate_basenames=candidate_basenames,
        )
        model_outputs = execute_graph(model_graph, {"prompt": model_prompt}, True)
        model_node_id = model_graph.nodes[0].node_id
        model_payload = self._parse_json_payload(model_outputs.get(model_node_id))

        envelope = self._fallback_envelope(
            request=request,
            candidate_urls=candidate_urls,
            candidate_paths=candidate_paths,
        )
        if model_payload is not None:
            try:
                envelope = ExtractionEnvelope.model_validate(model_payload)
            except ValidationError:
                envelope = self._fallback_envelope(
                    request=request,
                    candidate_urls=candidate_urls,
                    candidate_paths=candidate_paths,
                )

        return _ExtractionArtifacts(
            envelope=envelope,
            user_request_artifact_id=request_artifact_id,
            candidate_urls=candidate_urls,
            candidate_paths=candidate_paths,
            candidate_basenames=candidate_basenames,
        )

    @staticmethod
    def _build_extraction_prompt(
        *,
        request: PlannerRequest,
        candidate_urls: list[str],
        candidate_paths: list[str],
        candidate_basenames: list[str],
    ) -> str:
        metadata_json = json.dumps(request.metadata, ensure_ascii=False, sort_keys=True)
        return (
            "Extract a structured envelope for planning.\n"
            "Use candidates as hints and fill missing_fields for critical gaps.\n"
            "For file tasks, set intent.task_type='write_file' and provide inputs.output_filename "
            "and inputs.output_text.\n"
            "For optional slots that are unknown, use JSON null.\n"
            "Never return placeholder strings like 'None', 'null', 'n/a', or empty wrappers.\n"
            "If you set inputs.output_filename, it must be a bare filename with extension "
            "(example: 'summary.html') and must not contain paths, prefixes, or suffixes.\n"
            "Return only JSON.\n\n"
            f"Objective: {request.objective}\n"
            f"Constraints: {request.constraints}\n"
            f"Raw metadata: {metadata_json}\n"
            f"Candidate URLs: {candidate_urls}\n"
            f"Candidate file paths: {candidate_paths}\n"
            f"Candidate basenames: {candidate_basenames}\n"
        )

    def _fallback_envelope(
        self,
        *,
        request: PlannerRequest,
        candidate_urls: list[str],
        candidate_paths: list[str],
    ) -> ExtractionEnvelope:
        inputs: dict[str, Any] = {}
        if isinstance(request.metadata.get("url"), str) and str(request.metadata.get("url", "")).strip():
            inputs["url"] = str(request.metadata["url"])
            inputs["urls"] = [str(request.metadata["url"])]
        if isinstance(request.metadata.get("doc_paths"), list):
            doc_paths = [str(x) for x in request.metadata.get("doc_paths", []) if str(x).strip()]
            if doc_paths:
                inputs["doc_paths"] = doc_paths
        if isinstance(request.metadata.get("output_filename"), str):
            output_filename = str(request.metadata.get("output_filename", "")).strip()
            if output_filename:
                inputs["output_filename"] = output_filename
        if isinstance(request.metadata.get("output_text"), str):
            output_text = str(request.metadata.get("output_text", "")).strip()
            if output_text:
                inputs["output_text"] = output_text
        if isinstance(request.metadata.get("output_format"), str):
            output_format = str(request.metadata.get("output_format", "")).strip()
            if output_format:
                inputs["output_format"] = output_format

        missing_fields: list[MissingField] = []
        if not inputs.get("urls") and not inputs.get("doc_paths"):
            if candidate_urls:
                inputs["urls"] = list(candidate_urls)
            elif candidate_paths:
                inputs["doc_paths"] = list(candidate_paths)
            elif not inputs.get("output_text"):
                missing_fields.append(MissingField(
                    field="source",
                    reason="Provide URL or local file path to continue",
                    critical=True,
                ))

        # Best-effort filename extraction by shape, not language keywords.
        if not inputs.get("output_filename"):
            m = re.search(r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})\b", request.objective)
            if m:
                inputs["output_filename"] = m.group(1)

        task_type = self._effective_task_type(
            requested_task_type=str(request.metadata.get("task_type", "")).strip(),
            metadata=inputs,
        )

        return ExtractionEnvelope(
            intent={"task_type": task_type, "confidence": 0.4},
            inputs=inputs,
            missing_fields=missing_fields,
            extras={},
        )

    @staticmethod
    def _effective_task_type(
        *,
        requested_task_type: str,
        metadata: dict[str, Any],
    ) -> str:
        """Resolve task type from structured signals, not keyword heuristics."""
        urls = metadata.get("urls")
        if isinstance(urls, list) and any(str(x).strip() for x in urls):
            return "web_summary"
        url = metadata.get("url")
        if isinstance(url, str) and url.strip():
            return "web_summary"
        doc_paths = metadata.get("doc_paths")
        if isinstance(doc_paths, list) and any(str(x).strip() for x in doc_paths):
            return "docs_summary"
        output_text = metadata.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return "write_file"
        if requested_task_type in {"web_summary", "news_summary", "docs_summary", "write_file", "generic_task"}:
            return requested_task_type
        return "generic_task"

    def _resolve_inputs_tool_first(
        self,
        *,
        request: PlannerRequest,
        extraction: _ExtractionArtifacts,
        execute_graph: ExecutePlannerGraphFn,
    ) -> tuple[dict[str, Any], list[MissingField]]:
        metadata = dict(request.metadata)
        metadata.update(dict(extraction.envelope.inputs))

        urls = metadata.get("urls")
        if not isinstance(urls, list):
            urls = []
        urls = [str(x) for x in urls if str(x).strip()]

        if not urls:
            raw_url = metadata.get("url")
            if isinstance(raw_url, str) and raw_url.strip():
                urls = [raw_url.strip()]
            elif len(extraction.candidate_urls) == 1:
                urls = [extraction.candidate_urls[0]]

        if urls:
            metadata["urls"] = urls
            metadata["url"] = urls[0]

        doc_paths = metadata.get("doc_paths")
        if not isinstance(doc_paths, list):
            doc_paths = []
        doc_paths = [str(x) for x in doc_paths if str(x).strip()]

        # If URL is available, treat objective as web-source request.
        # Do not mix URL with docs branch unless URL is absent.
        if urls:
            doc_paths = []
        else:
            if not doc_paths and len(extraction.candidate_paths) == 1:
                doc_paths = [extraction.candidate_paths[0]]

            resolved_doc_paths: list[str] = []
            unresolved_doc_paths: list[str] = []
            for path in doc_paths:
                if self._is_path_like(path):
                    resolved_doc_paths.append(path)
                    continue
                resolved = self._resolve_single_basename_with_glob(
                    request=request,
                    basename=path,
                    execute_graph=execute_graph,
                )
                if resolved is None:
                    unresolved_doc_paths.append(path)
                else:
                    resolved_doc_paths.append(resolved)
            doc_paths = resolved_doc_paths
            if unresolved_doc_paths:
                metadata["unresolved_doc_paths"] = unresolved_doc_paths

        if doc_paths:
            metadata["doc_paths"] = doc_paths
        elif "doc_paths" in metadata:
            metadata.pop("doc_paths", None)

        missing_fields = self._compute_missing_fields(
            request=request,
            envelope=extraction.envelope,
            metadata=metadata,
        )
        return metadata, missing_fields

    @staticmethod
    def _resolve_single_basename_with_glob(
        *,
        request: PlannerRequest,
        basename: str,
        execute_graph: ExecutePlannerGraphFn,
    ) -> str | None:
        root = str(request.metadata.get("search_root", os.getcwd()))
        graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"htn-resolve-glob-{request.execution_id[:12]}",
                description=f"Resolve basename {basename}",
            ),
            nodes=[
                GraphNode(
                    node_id="resolve_path_glob",
                    node_type="mcp",
                    parameters={
                        "tool_name": "fs.glob",
                        "tool_args": {
                            "root": root,
                            "pattern": f"**/{basename}",
                        },
                    },
                )
            ],
            edges=[],
        )
        outputs = execute_graph(graph, {}, True)
        node_output = outputs.get("resolve_path_glob")
        if node_output is None or node_output.status != "COMPLETED":
            return None
        raw_paths = node_output.outputs.get("paths", [])
        if not isinstance(raw_paths, list):
            return None
        paths = [str(x) for x in raw_paths if str(x).strip()]
        if len(paths) != 1:
            return None
        return paths[0]

    @staticmethod
    def _is_path_like(value: str) -> bool:
        if not value:
            return False
        if value.startswith("./") or value.startswith("../"):
            return True
        if value.startswith("/"):
            return True
        if re.match(r"^[A-Za-z]:\\", value):
            return True
        return False

    def _compute_missing_fields(
        self,
        *,
        request: PlannerRequest,
        envelope: ExtractionEnvelope,
        metadata: dict[str, Any],
    ) -> list[MissingField]:
        task_type = self._effective_task_type(
            requested_task_type=envelope.intent.task_type or "generic_task",
            metadata=metadata,
        )
        urls = metadata.get("urls") if isinstance(metadata.get("urls"), list) else []
        doc_paths = metadata.get("doc_paths") if isinstance(metadata.get("doc_paths"), list) else []
        output_filename = metadata.get("output_filename")
        output_text = metadata.get("output_text")
        unresolved_doc_paths = (
            metadata.get("unresolved_doc_paths")
            if isinstance(metadata.get("unresolved_doc_paths"), list)
            else []
        )

        missing: list[MissingField] = []
        for mf in envelope.missing_fields:
            if mf.field == "url" and urls:
                continue
            if mf.field == "doc_paths" and doc_paths:
                continue
            if mf.field == "output_filename" and isinstance(output_filename, str) and output_filename.strip():
                continue
            if mf.field == "output_text" and isinstance(output_text, str) and output_text.strip():
                continue
            if task_type != "write_file" and mf.field in {"output_filename", "output_text"}:
                continue
            if mf.field == "source" and (urls or doc_paths):
                continue
            if mf.field == "doc_paths" and urls:
                continue
            if mf.critical:
                missing.append(mf)

        if unresolved_doc_paths and not urls:
            missing.append(MissingField(
                field="doc_paths",
                reason=(
                    "Could not resolve doc_paths uniquely: "
                    + ", ".join(str(x) for x in unresolved_doc_paths)
                ),
                critical=True,
            ))

        if task_type in {"news_summary", "web_summary"} and not urls:
            missing.append(MissingField(
                field="url",
                reason="URL is required for web summary",
                critical=True,
            ))
        if task_type in {"docs_summary"} and not doc_paths and not urls:
            missing.append(MissingField(
                field="doc_paths",
                reason="Document paths are required for docs summary",
                critical=True,
            ))
        if task_type == "write_file":
            if not (isinstance(output_filename, str) and output_filename.strip()):
                missing.append(MissingField(
                    field="output_filename",
                    reason="Filename is required for file task",
                    critical=True,
                ))
            if not (isinstance(output_text, str) and output_text.strip()):
                missing.append(MissingField(
                    field="output_text",
                    reason="Text content is required for file task",
                    critical=True,
                ))
        if task_type == "generic_task" and not urls and not doc_paths:
            missing.append(MissingField(
                field="source",
                reason="Need at least one source: URL or doc path",
                critical=True,
            ))

        # If we already have specific actionable missing fields, avoid a vague
        # umbrella "source" question to reduce redundant clarifications.
        # This keeps the UI/CLI flow clean (e.g. only ask for url if web summary).
        if any(m.field in {"url", "doc_paths"} for m in missing):
            missing = [m for m in missing if m.field != "source"]

        # Stable dedupe by field+reason
        dedup: dict[tuple[str, str], MissingField] = {}
        for mf in missing:
            dedup[(mf.field, mf.reason)] = mf
        return [dedup[k] for k in sorted(dedup)]

    def _build_clarification_request(
        self,
        *,
        request: PlannerRequest,
        missing_fields: list[MissingField],
        extraction: _ExtractionArtifacts,
        effective_metadata: dict[str, Any],
        execute_graph: ExecutePlannerGraphFn,
        options: _BackendOptions,
    ) -> ClarificationRequest:
        # Prefer model-generated questions (more natural UX),
        # but always fall back deterministically if the model path fails.
        questions = self._build_clarification_questions_with_model(
            request=request,
            missing_fields=missing_fields,
            execute_graph=execute_graph,
            options=options,
        )
        if not questions:
            questions = self._build_clarification_questions_fallback(missing_fields)

        return ClarificationRequest(
            request_id=f"clarify-{request.execution_id[:10]}-{request.stage_id.replace(':', '_')}",
            missing_fields=missing_fields,
            questions=questions,
            candidate_evidence=[
                {"kind": "candidate_urls", "value": extraction.candidate_urls},
                {"kind": "candidate_paths", "value": extraction.candidate_paths},
                {"kind": "candidate_basenames", "value": extraction.candidate_basenames},
            ],
            context={
                "objective": request.objective,
                "runbook_id": request.runbook_id,
                "stage_id": request.stage_id,
                "metadata_keys": sorted(effective_metadata.keys()),
            },
        )

    @staticmethod
    def _build_clarification_questions_fallback(
        missing_fields: list[MissingField],
    ) -> list[ClarificationQuestion]:
        """Universal deterministic fallback without scenario templates."""
        questions: list[ClarificationQuestion] = []
        for mf in missing_fields:
            key = str(mf.field or "").strip()
            if not key:
                continue
            reason = str(mf.reason or "").strip()
            prompt = f"Уточни значение поля '{key}'."
            if reason:
                prompt += f" Причина: {reason}"
            expected_type = "string"
            low = key.lower()
            if "url" in low:
                expected_type = "url"
            elif low.endswith("s") or "paths" in low:
                expected_type = "string_list"
            questions.append(
                ClarificationQuestion(
                    key=key,
                    prompt=prompt,
                    expected_type=expected_type,
                    required=bool(mf.critical),
                    examples=[],
                )
            )
        return questions

    def _build_clarification_questions_with_model(
        self,
        *,
        request: PlannerRequest,
        missing_fields: list[MissingField],
        execute_graph: ExecutePlannerGraphFn,
        options: _BackendOptions,
    ) -> list[ClarificationQuestion]:
        """Generate clarification questions via model node (with strict schema)."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["key", "prompt", "expected_type", "required", "examples"],
                        "properties": {
                            "key": {"type": "string"},
                            "prompt": {"type": "string"},
                            "expected_type": {"type": "string"},
                            "required": {"type": "boolean"},
                            "examples": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                }
            },
        }
        node_id = f"{options.planner_node_prefix}_clarification_questions"
        graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"htn-clarification-{request.execution_id[:12]}",
                description="Generate clarification questions",
            ),
            nodes=[
                GraphNode(
                    node_id=node_id,
                    node_type="model",
                    label="Generate clarification questions",
                    parameters={
                        "system_prompt": (
                            "You generate clarification questions for a user in Russian. "
                            "Use only provided missing field keys. Return strict JSON."
                        ),
                        "timeout_seconds": 60,
                        "max_retries": 1,
                        "json_schema": schema,
                    },
                )
            ],
            edges=[],
        )
        prompt = (
            "Сформулируй краткие вопросы для уточнения недостающих параметров.\n"
            "Не добавляй новые ключи, используй только перечисленные.\n\n"
            f"Objective: {request.objective}\n"
            "Missing fields:\n"
            + "\n".join(
                f"- key={mf.field}; reason={mf.reason}; critical={mf.critical}"
                for mf in missing_fields
            )
        )
        outputs = execute_graph(graph, {"prompt": prompt}, True)
        payload = self._parse_json_payload(outputs.get(node_id))
        if payload is None:
            return []
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            return []
        allowed_keys = {str(mf.field) for mf in missing_fields}
        out: list[ClarificationQuestion] = []
        for item in raw_questions:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if key not in allowed_keys:
                continue
            try:
                out.append(ClarificationQuestion.model_validate(item))
            except ValidationError:
                continue
        return out

    @staticmethod
    def _persist_clarification_request_artifact(
        *,
        request: PlannerRequest,
        clarification: ClarificationRequest,
        evidence_artifact_ids: list[str],
        execute_graph: ExecutePlannerGraphFn,
    ) -> str:
        graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"htn-clarify-{request.execution_id[:12]}",
                description=f"Persist clarification request for {request.stage_id}",
            ),
            nodes=[
                GraphNode(
                    node_id="persist_clarification_request",
                    node_type="mcp",
                    parameters={
                        "tool_name": "artifact.put_json",
                        "tool_args": {
                            "artifact_type": "planner.clarification_request",
                            "json": clarification.model_dump(mode="json"),
                            "parent_artifact_ids": evidence_artifact_ids,
                            "produced_by_node_ref": (
                                f"{request.execution_id}:{request.runbook_id}:{request.stage_id}/commit/clarification_request"
                            ),
                            "media_type": "application/json",
                        },
                    },
                )
            ],
            edges=[],
        )
        outputs = execute_graph(graph, {}, True)
        node_output = outputs.get("persist_clarification_request")
        if node_output is None or node_output.status != "COMPLETED":
            raise ValueError("Failed to persist clarification request artifact")
        aid = str(node_output.outputs.get("artifact_id", "")).strip()
        if not aid:
            raise ValueError("Clarification request artifact has empty artifact_id")
        return aid

    @staticmethod
    def _context_kind(metadata: dict[str, Any]) -> str:
        urls = metadata.get("urls")
        if isinstance(urls, list) and urls:
            return "web"
        url = metadata.get("url")
        if isinstance(url, str) and url.strip():
            return "web"
        doc_paths = metadata.get("doc_paths")
        if isinstance(doc_paths, list) and doc_paths:
            return "docs"
        out_fn = metadata.get("output_filename")
        if isinstance(out_fn, str) and out_fn.strip():
            return "file"
        return "generic"

    def _select_method(
        self,
        *,
        request: PlannerRequest,
        subgoal: HtnSubgoal,
        context_kind: str,
        options: _BackendOptions,
        execute_graph: ExecutePlannerGraphFn,
        planner_calls: int,
    ) -> tuple[str, str]:
        if (
            options.model_assisted_method_selection
            and planner_calls < options.model_assisted_max_calls
        ):
            assisted = self._model_assisted_method(
                request=request,
                subgoal=subgoal,
                context_kind=context_kind,
                options=options,
                execute_graph=execute_graph,
                call_index=planner_calls,
            )
            if assisted is not None:
                return assisted, "model_assisted"
        return self._rule_based_method(subgoal=subgoal, context_kind=context_kind), "rule_based"

    @staticmethod
    def _rule_based_method(*, subgoal: HtnSubgoal, context_kind: str) -> str:
        if subgoal.kind == "root":
            if context_kind == "docs":
                return "root_docs_pipeline"
            if context_kind == "web":
                return "root_web_pipeline"
            if context_kind == "file":
                return "root_generic_pipeline"
            return "root_generic_pipeline"
        if subgoal.kind == "collect_inputs":
            if context_kind == "docs":
                return "collect_docs"
            if context_kind == "web":
                return "collect_web"
            if context_kind == "file":
                return "collect_generic"
            return "collect_generic"
        if subgoal.kind == "read_docs":
            return "expand_read_docs"
        if subgoal.kind == "synthesize":
            return "draft_direct"
        if subgoal.kind == "verify":
            return "critic_standard"
        return "noop"

    def _model_assisted_method(
        self,
        *,
        request: PlannerRequest,
        subgoal: HtnSubgoal,
        context_kind: str,
        options: _BackendOptions,
        execute_graph: ExecutePlannerGraphFn,
        call_index: int,
    ) -> str | None:
        if subgoal.kind not in {"root", "synthesize"}:
            return None

        if subgoal.kind == "root":
            candidates = [
                "root_docs_pipeline",
                "root_web_pipeline",
                "root_generic_pipeline",
            ]
        else:
            candidates = ["draft_direct"]

        planner_node_id = f"{options.planner_node_prefix}_{call_index:02d}_{subgoal.kind}"
        schema = {
            "type": "object",
            "required": ["method_id"],
            "properties": {
                "method_id": {"type": "string"},
                "justification_key": {"type": "string"},
            },
            "additionalProperties": False,
        }
        planner_graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"htn-method-select-{request.execution_id[:12]}-{call_index:02d}",
                description=f"HTN method selection for {subgoal.subgoal_id}",
            ),
            nodes=[
                GraphNode(
                    node_id=planner_node_id,
                    node_type="model",
                    label="Select HTN decomposition method",
                    parameters={
                        "system_prompt": (
                            "You select one decomposition method id from the allowed list. "
                            "Return only valid JSON."
                        ),
                        "json_schema": schema,
                    },
                )
            ],
            edges=[],
        )
        prompt = (
            f"Objective: {request.objective}\n"
            f"Subgoal: {subgoal.subgoal_id}\n"
            f"Subgoal title: {subgoal.title}\n"
            f"Context kind: {context_kind}\n"
            f"Allowed methods: {', '.join(candidates)}\n"
            "Return JSON with method_id and optional justification_key."
        )
        outputs = execute_graph(planner_graph, {"prompt": prompt}, True)
        model_output = outputs.get(planner_node_id)
        payload = self._parse_json_payload(model_output)
        if payload is None:
            return None
        method_id = str(payload.get("method_id", "")).strip()
        if method_id in candidates:
            return method_id
        return None

    @staticmethod
    def _parse_json_payload(output: NodeOutput | None) -> dict[str, Any] | None:
        if output is None or output.status != "COMPLETED":
            return None
        parsed = output.outputs.get("parsed")
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                value = json.loads(parsed)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
        content = output.outputs.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _expand_subgoal(
        self,
        *,
        request: PlannerRequest,
        metadata: dict[str, Any],
        subgoal: HtnSubgoal,
        method_id: str,
        context_kind: str,
    ) -> tuple[list[HtnSubgoal], list[HtnLeafOperator]]:
        next_depth = subgoal.depth + 1
        if subgoal.kind == "root":
            children = [
                HtnSubgoal(
                    subgoal_id=f"{subgoal.subgoal_id}.collect",
                    title="Collect input evidence",
                    depth=next_depth,
                    parent_subgoal_id=subgoal.subgoal_id,
                    kind="collect_inputs",
                ),
                HtnSubgoal(
                    subgoal_id=f"{subgoal.subgoal_id}.synthesize",
                    title="Draft output from collected evidence",
                    depth=next_depth,
                    parent_subgoal_id=subgoal.subgoal_id,
                    kind="synthesize",
                ),
                HtnSubgoal(
                    subgoal_id=f"{subgoal.subgoal_id}.verify",
                    title="Verify output quality",
                    depth=next_depth,
                    parent_subgoal_id=subgoal.subgoal_id,
                    kind="verify",
                ),
            ]
            return children, []

        if subgoal.kind == "collect_inputs":
            if method_id == "collect_docs":
                children = [
                    HtnSubgoal(
                        subgoal_id=f"{subgoal.subgoal_id}.read_docs",
                        title="Read document set",
                        depth=next_depth,
                        parent_subgoal_id=subgoal.subgoal_id,
                        kind="read_docs",
                    )
                ]
                return children, []
            if method_id == "collect_web":
                url = str(metadata.get("url", "")).strip()
                leaves = [
                    HtnLeafOperator(
                        subgoal_id=subgoal.subgoal_id,
                        node_id="fetch_html",
                        node_type="mcp",
                        tool_name="web.fetch_html",
                        parameters={"tool_name": "web.fetch_html", "tool_args": {"url": url}},
                    ),
                    HtnLeafOperator(
                        subgoal_id=subgoal.subgoal_id,
                        node_id="extract_article",
                        node_type="mcp",
                        tool_name="web.extract_article",
                        parameters={
                            "tool_name": "web.extract_article",
                            "tool_args": {"url": url},
                        },
                        depends_on=["fetch_html"],
                    ),
                ]
                return [], leaves
            return [], []

        if subgoal.kind == "read_docs":
            doc_paths = metadata.get("doc_paths")
            if not isinstance(doc_paths, list) or not doc_paths:
                return [], []
            leaves: list[HtnLeafOperator] = []
            read_node_ids: list[str] = []
            for idx, raw_path in enumerate(sorted(str(p) for p in doc_paths)):
                node_id = f"read_{idx:03d}"
                out_key = f"doc_{idx:03d}"
                leaves.append(
                    HtnLeafOperator(
                        subgoal_id=subgoal.subgoal_id,
                        node_id=node_id,
                        node_type="mcp",
                        tool_name="fs.read_text",
                        parameters={
                            "tool_name": "fs.read_text",
                            "tool_args": {"path": raw_path, "out_key": out_key},
                        },
                    )
                )
                read_node_ids.append(node_id)
            leaves.append(
                HtnLeafOperator(
                    subgoal_id=subgoal.subgoal_id,
                    node_id="merge_docs",
                    node_type="aggregate",
                    parameters={},
                    depends_on=read_node_ids,
                )
            )
            return [], leaves

        if subgoal.kind == "synthesize":
            if context_kind == "file":
                filename = str(metadata.get("output_filename", "")).strip()
                text = metadata.get("output_text")
                text = text if isinstance(text, str) else ""
                leaves = [
                    HtnLeafOperator(
                        subgoal_id=subgoal.subgoal_id,
                        node_id="write_file",
                        node_type="mcp",
                        tool_name="export.write_text",
                        parameters={
                            "tool_name": "export.write_text",
                            "tool_args": {
                                "run_id": request.execution_id[:12],
                                "filename": filename or "output.txt",
                                "text": text,
                                "encoding": "utf-8",
                                "overwrite": True,
                            },
                        },
                    ),
                    HtnLeafOperator(
                        subgoal_id=subgoal.subgoal_id,
                        node_id="draft_report",
                        node_type="model",
                        parameters={
                            "system_prompt": (
                                "You are a helpful assistant.\n"
                                "Confirm the file action performed and briefly summarize what was written.\n"
                                "Be concise."
                            )
                        },
                        depends_on=["write_file"],
                    ),
                ]
                return [], leaves

            deps = (
                ["merge_docs"]
                if context_kind == "docs"
                else ["extract_article"]
                if context_kind == "web"
                else []
            )
            leaves = [
                HtnLeafOperator(
                    subgoal_id=subgoal.subgoal_id,
                    node_id="draft_report",
                    node_type="model",
                    parameters={
                        "system_prompt": (
                            "You are an operations analyst. "
                            "Produce a concise report using the provided inputs."
                        )
                    },
                    depends_on=deps,
                )
            ]
            # Optional: export user-facing summary to deterministic local file.
            if context_kind == "web":
                output_format = str(metadata.get("output_format", "")).strip().lower()
                ext = "md"
                if output_format in {"html", "htm"}:
                    ext = "html"
                elif output_format in {"txt", "text"}:
                    ext = "txt"
                elif output_format in {"md", "markdown"}:
                    ext = "md"
                output_filename = str(metadata.get("output_filename", "")).strip()
                if not output_filename:
                    output_filename = f"summary.{ext}"
                leaves.append(
                    HtnLeafOperator(
                        subgoal_id=subgoal.subgoal_id,
                        node_id="export_user_output",
                        node_type="mcp",
                        tool_name="export.write_text",
                        parameters={
                            "tool_name": "export.write_text",
                            "tool_args": {
                                "run_id": request.execution_id[:12],
                                "kind": "news_summary",
                                "filename": output_filename,
                                "overwrite": True,
                            },
                        },
                        # Needs both extracted article fields and draft_report content.
                        depends_on=["draft_report"] + list(deps),
                    )
                )
            return [], leaves

        if subgoal.kind == "verify":
            deps = ["draft_report"]
            if context_kind == "docs":
                deps.append("merge_docs")
            elif context_kind == "web":
                deps.append("extract_article")
                deps.append("fetch_html")
            critic_prompt = (
                WEB_CRITIC_SYSTEM_PROMPT
                if context_kind == "web"
                else BUSINESS_CRITIC_SYSTEM_PROMPT
                if context_kind == "docs"
                else GENERIC_TASK_CRITIC_SYSTEM_PROMPT
            )
            leaves = [
                HtnLeafOperator(
                    subgoal_id=subgoal.subgoal_id,
                    node_id="critic_report",
                    node_type="model",
                    parameters={
                        "system_prompt": critic_prompt,
                        "json_schema": critic_json_schema(),
                    },
                    depends_on=deps,
                )
            ]
            return [], leaves

        return [], []

    @staticmethod
    def _build_action_graph(
        *,
        request: PlannerRequest,
        leaves: list[HtnLeafOperator],
        context_kind: str,
    ) -> ActionGraph:
        if not leaves:
            raise ValueError("HTN planner produced no leaf operators")

        node_map: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        for leaf in leaves:
            if leaf.node_id in node_map:
                continue
            params = dict(leaf.parameters)
            if leaf.node_type == "mcp" and leaf.tool_name:
                params.setdefault("tool_name", leaf.tool_name)
            node_map[leaf.node_id] = GraphNode(
                node_id=leaf.node_id,
                node_type=leaf.node_type,  # type: ignore[arg-type]
                parameters=params,
            )

        for leaf in leaves:
            for dep in sorted(set(leaf.depends_on)):
                if dep not in node_map:
                    continue
                edges.append(
                    GraphEdge(
                        source=dep,
                        target=leaf.node_id,
                        edge_type="data",
                    )
                )

        for idx, node_id in enumerate(sorted(node_map)):
            node_map[node_id].priority = idx

        return ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"plan-htn-recursive-v0-{request.execution_id[:12]}",
                description=(
                    f"HTN recursive plan ({context_kind}) for {request.runbook_id}:{request.stage_id}"
                ),
            ),
            nodes=[node_map[nid] for nid in sorted(node_map)],
            edges=sorted(edges, key=lambda e: (e.source, e.target, e.edge_type, e.label)),
        )
