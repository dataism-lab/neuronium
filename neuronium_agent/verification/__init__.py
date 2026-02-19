"""Verification layer — simulated critics (IBS §10)."""

from neuronium_agent.verification.critic import SimulatedCritic, CriticVerdict
from neuronium_agent.verification.demo_critic import (
    DemoCriticVerdict,
    parse_critic_verdict,
    CRITIC_SYSTEM_PROMPT,
)

__all__ = [
    "SimulatedCritic",
    "CriticVerdict",
    "DemoCriticVerdict",
    "parse_critic_verdict",
    "CRITIC_SYSTEM_PROMPT",
]
