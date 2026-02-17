"""Core subsystem — agent state machine and orchestration loop."""

from neuronium_agent.core.state import AgentState, IntentionPhase

__all__ = ["AgentState", "IntentionPhase", "Orchestrator"]


def __getattr__(name: str):
    if name == "Orchestrator":
        from neuronium_agent.core.orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(name)
