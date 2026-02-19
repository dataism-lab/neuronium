"""Formal Decision Record schema (IBS §10.1.1).

Structured decision events for audit, learning and outcome correlation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Decision classification per §10.1.1."""

    PLANNING = "planning"
    EXECUTION = "execution"
    CONTROL = "control"
    ADAPTATION = "adaptation"
    ESCALATION = "escalation"
    META_CONTROL = "meta-control"


class DecisionAuthority(str, Enum):
    """Who made the decision."""

    COMPONENT = "component"
    USER = "user"
    HYBRID = "hybrid"


class OptionConsidered(BaseModel):
    """One option in optionsConsidered."""

    option_id: str = Field(..., description="Identifier for this option")
    description: str | None = None
    predicted_outcome: str | Any | None = None
    score: float | None = None
    confidence: float | None = None


class SelectedOption(BaseModel):
    """The chosen option and rationale."""

    option_id: str = Field(..., description="Reference to optionsConsidered or free-form id")
    selection_rationale: str = Field(..., description="Justification text")
    decision_authority: DecisionAuthority = DecisionAuthority.COMPONENT
    expected_outcome: str | Any | None = None


class DecisionContext(BaseModel):
    """Minimal context snapshot (refs only, no heavy state)."""

    agent_state_snapshot: str | None = None
    relevant_goals: list[str] = Field(default_factory=list)
    active_constraints: list[str] = Field(default_factory=list)
    available_evidence: list[str] = Field(default_factory=list)


class OutcomeCorrelation(BaseModel):
    """Filled later: actual outcome vs expected."""

    actual_outcome: str | Any = Field(..., description="Observed result")
    correlation_timestamp: str | None = None  # ISO-8601
    quality_assessment: str | None = None  # success | partial | failure
    learning_signal: str | None = None


class DecisionRecord(BaseModel):
    """Full decision record per §10.1.1. All fields except id/timestamp/decision_type/selected_option are optional."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique decision id")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601",
    )
    trace_id: str | None = None
    intention_id: str | None = None
    decision_type: DecisionType = Field(..., description="planning|execution|control|adaptation|escalation|meta-control")
    context: DecisionContext | None = None
    options_considered: list[OptionConsidered] = Field(default_factory=list)
    selected_option: SelectedOption = Field(...)
    outcome_correlation: OutcomeCorrelation | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize for trace payload; keys match §10.1.1 (camelCase for spec alignment in payload)."""
        out: dict[str, Any] = {
            "decisionRecord": {
                "id": self.id,
                "timestamp": self.timestamp,
                "decisionType": self.decision_type.value,
                "selectedOption": {
                    "optionId": self.selected_option.option_id,
                    "selectionRationale": self.selected_option.selection_rationale,
                    "decisionAuthority": self.selected_option.decision_authority.value,
                    "expectedOutcome": self.selected_option.expected_outcome,
                },
            }
        }
        dr = out["decisionRecord"]
        if self.trace_id is not None:
            dr["traceId"] = self.trace_id
        if self.intention_id is not None:
            dr["intentionId"] = self.intention_id
        if self.context is not None:
            dr["context"] = {
                "agentStateSnapshot": self.context.agent_state_snapshot,
                "relevantGoals": self.context.relevant_goals,
                "activeConstraints": self.context.active_constraints,
                "availableEvidence": self.context.available_evidence,
            }
        if self.options_considered:
            dr["optionsConsidered"] = [
                {
                    "optionId": o.option_id,
                    "description": o.description,
                    "predictedOutcome": o.predicted_outcome,
                    "score": o.score,
                    "confidence": o.confidence,
                }
                for o in self.options_considered
            ]
        if self.outcome_correlation is not None:
            dr["outcomeCorrelation"] = {
                "actualOutcome": self.outcome_correlation.actual_outcome,
                "correlationTimestamp": self.outcome_correlation.correlation_timestamp,
                "qualityAssessment": self.outcome_correlation.quality_assessment,
                "learningSignal": self.outcome_correlation.learning_signal,
            }
        return out

    @classmethod
    def from_legacy(cls, description: str, details: dict[str, Any], decision_type: DecisionType = DecisionType.CONTROL) -> DecisionRecord:
        """Build a minimal record from old record_decision(description, details) for backward-compatible wrapper."""
        option_id = details.get("runbook_id") or details.get("command") or details.get("reason") or "legacy"
        if isinstance(option_id, dict):
            option_id = str(option_id)[:64]
        return cls(
            decision_type=decision_type,
            selected_option=SelectedOption(
                option_id=str(option_id)[:128],
                selection_rationale=description,
                decision_authority=DecisionAuthority.COMPONENT,
                expected_outcome=None,
            ),
        )
