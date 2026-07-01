"""LangGraph Agent Server entry point for AutFin.

This module exposes the compiled supervisor graph for `langgraph dev`.
"""
from src.autfin.graph.supervisor import compile_supervisor_graph

graph = compile_supervisor_graph()
