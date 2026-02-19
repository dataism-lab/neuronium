"""Recovery: failure classification and recovery policy (B1 Part 1, §3.4).

Classify node failures (TRANSIENT/PERSISTENT/SYSTEMIC/CRITICAL) and decide
RETRY_STAGE / ESCALATE / FAIL after stage gate failure.
"""

from neuronium_agent.recovery.classifier import classify_failure
from neuronium_agent.recovery.models import (
    RecoveryAction,
    RecoveryDecision,
    RollbackScope,
    RollbackScopeType,
)
from neuronium_agent.recovery.policy import decide_recovery
from neuronium_agent.recovery.scope import compute_rollback_scope
from neuronium_agent.types import FailureClass

__all__ = [
    "classify_failure",
    "compute_rollback_scope",
    "decide_recovery",
    "FailureClass",
    "RecoveryAction",
    "RecoveryDecision",
    "RollbackScope",
    "RollbackScopeType",
]
