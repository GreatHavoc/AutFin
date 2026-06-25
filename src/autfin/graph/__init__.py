"""
AutFin Graph Workflow Definitions

v3.0: Primary export = build_supervisor_graph / compile_supervisor_graph.

v3.0 graph usage:
    from src.autfin.graph import compile_supervisor_graph
    graph = compile_supervisor_graph(checkpointer=checkpointer)
    state = {"context": "morning_routine", ...}
    result = await graph.ainvoke(state, config)
"""

from .supervisor import (
    build_supervisor_graph,
    compile_supervisor_graph,
    supervisor_node,
)

from .hitl import (
    trade_gate_node,
    human_review_node,
    _route_after_trade_gate,
    _needs_human_approval,
)

__all__ = [
    "build_supervisor_graph",
    "compile_supervisor_graph",
    "supervisor_node",
    "trade_gate_node",
    "human_review_node",
    "_route_after_trade_gate",
    "_needs_human_approval",
]
