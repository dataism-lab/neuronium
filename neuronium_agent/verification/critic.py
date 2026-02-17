"""Simulated Critic (IBS §10).

Critic evaluates node outputs against criteria and produces a verdict.
v1: minimal implementation — always PASS (placeholder for real evaluation).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A piece of evidence supporting the verdict."""

    source: str = ""
    description: str = ""
    confidence: float = 1.0


class CriticVerdict(BaseModel):
    """Critic evaluation result (IBS §10.2)."""

    verdict: Literal["PASS", "CONDITIONAL_PASS", "FAIL", "UNCERTAIN"] = "PASS"
    confidence: float = 1.0
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class CriticInput(BaseModel):
    """Input to the critic evaluation."""

    node_id: str
    node_type: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    quality_signals: dict[str, Any] = Field(default_factory=dict)
    criteria: dict[str, Any] = Field(default_factory=dict)


class SimulatedCritic:
    """Simulated critic for verifying node outputs.

    v1 implementation is a stub that always returns PASS.
    Future: LLM-based evaluation, rule-based checks, etc.
    """

    def evaluate(self, critic_input: CriticInput) -> CriticVerdict:
        """Evaluate node output and return a verdict."""
        # v1 stub: always passes
        return CriticVerdict(
            verdict="PASS",
            confidence=1.0,
            reasoning="v1 stub critic — auto-pass",
            evidence=[
                Evidence(
                    source=critic_input.node_id,
                    description="Output produced successfully",
                )
            ],
        )
