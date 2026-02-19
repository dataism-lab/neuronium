"""Node system — unified contracts and concrete implementations."""

from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput, NodeContext
from neuronium_agent.nodes.model_node import ModelNode
from neuronium_agent.nodes.code_node import CodeNode
from neuronium_agent.nodes.mcp_node import McpToolNode
from neuronium_agent.nodes.decision_node import DecisionNode
from neuronium_agent.nodes.aggregate_node import AggregateNode

__all__ = [
    "BaseNode",
    "NodeInput",
    "NodeOutput",
    "NodeContext",
    "ModelNode",
    "CodeNode",
    "McpToolNode",
    "DecisionNode",
    "AggregateNode",
]
