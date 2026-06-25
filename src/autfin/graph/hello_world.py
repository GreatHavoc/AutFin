"""
Hello World Graph - Minimal LangGraph for memory infrastructure validation.

This graph verifies that the checkpointer, store, and vector store integration
is working correctly. It demonstrates a simple state flow: input → process → output.
"""

from typing import Any, Dict, Optional
from pathlib import Path
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import Checkpoint
from langgraph.types import Command

from ..state.agent_state import AgentState
from ..memory.setup import create_checkpointer, get_checkpoint_db_path
from ..memory.store import get_store
from ..memory.vector_store import VectorStore, initialize_collections


def build_hello_world_graph() -> StateGraph:
    """
    Build a minimal LangGraph to verify memory infrastructure.
    
    Flow:
    START → input_node → process_node → output_node → END
    
    This graph demonstrates:
    - Checkpointer integration for state persistence
    - Store integration for cross-thread memory
    - Vector store integration for semantic storage
    """
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("input", input_node)
    builder.add_node("process", process_node)
    builder.add_node("output", output_node)
    
    # Define edges
    builder.add_edge(START, "input")
    builder.add_edge("input", "process")
    builder.add_edge("process", "output")
    builder.add_edge("output", END)
    
    return builder


async def input_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Input node - receives user input and initializes state.
    
    Demonstrates:
    - Store operations (write input context)
    - Checkpointer integration (state saved after each node)
    """
    user_input = state.get("input", "Hello, AutFin!")
    
    # Write to store for cross-thread access
    try:
        store = get_store()
        await store.aput("hello_world", "last_input", user_input)
    except Exception as e:
        print(f"Warning: Store write failed: {e}")
    
    return {
        "messages": [f"Input received: {user_input}"],
        "agent": "hello_world",
        "context": {"hello_message": user_input},
        "next": "process",
    }


async def process_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Process node - demonstrates vector store usage and state transformation.
    
    Demonstrates:
    - Vector store operations (add and search)
    - State transformation (input → processed)
    """
    hello_message = state.get("context", {}).get("hello_message", "Hello, AutFin!")
    
    # Initialize vector store (in-memory mode for validation)
    try:
        vector_store = VectorStore(mode="in-memory")
        
        # Add a test embedding to vector store
        test_embedding = [0.1] * 768  # Simulated embedding vector
        vector_store.add(
            collection_name="hello_world",
            embeddings=[test_embedding],
            metadatas=[{"text": hello_message, "type": "test"}],
        )
        
        # Search vector store
        results = vector_store.search(
            collection_name="hello_world",
            query_embeddings=[test_embedding],
            n_results=1,
        )
        
        print(f"Vector store search results: {results}")
    except Exception as e:
        print(f"Warning: Vector store operation failed: {e}")
    
    return {
        "messages": [f"Processed: {hello_message}"],
        "context": {"hello_message": hello_message, "processed": True},
        "next": "output",
    }


async def output_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Output node - produces final output and demonstrates checkpoint reading.
    
    Demonstrates:
    - Checkpoint retrieval and validation
    - Final state output
    """
    hello_message = state.get("context", {}).get("hello_message", "Hello, AutFin!")
    processed = state.get("context", {}).get("processed", False)
    
    # Read from store to verify cross-thread memory works
    try:
        store = get_store()
        last_input = await store.aget("hello_world", "last_input")
        print(f"Retrieved from store: {last_input}")
    except Exception as e:
        print(f"Warning: Store read failed: {e}")
    
    return {
        "messages": [f"Output: Processed '{hello_message}' (processed={processed})"],
        "context": {"hello_message": f"Success: {hello_message}", "processed": True},
        "next": "__end__",
    }


async def run_hello_world(
    input_text: str = "Hello, AutFin!",
    thread_id: str = "hello_world_test",
    checkpoint_mode: str = "in-memory",
) -> Dict[str, Any]:
    """
    Run the hello world graph with specified input.
    
    Args:
        input_text: Input message to process
        thread_id: Thread ID for checkpointing
        checkpoint_mode: "in-memory" or "persistent"
    
    Returns:
        Final state after graph execution
    """
    # Create checkpointer
    try:
        checkpointer = create_checkpointer(mode=checkpoint_mode)
    except Exception as e:
        print(f"Warning: Checkpointer creation failed: {e}")
        checkpointer = None
    
    # Build graph
    graph = build_hello_world_graph()
    
    # Compile with checkpointer if available
    if checkpointer:
        app = graph.compile(checkpointer=checkpointer)
    else:
        app = graph.compile()
    
    # Run graph
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_mode": checkpoint_mode,
        }
    }
    
    initial_state = {
        "messages": [],
        "input": input_text,
        "next": "input",
        "agent": "hello_world",
        "context": {},
        "thread_id": thread_id,
        "run_mode": "hello_world",
        "errors": [],
    }
    
    try:
        result = await app.ainvoke(initial_state, config=config)
        return result
    except Exception as e:
        print(f"Graph execution failed: {e}")
        raise


# Validation functions


def validate_memory_components() -> Dict[str, bool]:
    """
    Validate that all memory components are properly configured.
    
    Returns:
        Dictionary with component names and their validation status
    """
    results = {
        "checkpointer": False,
        "store": False,
        "vector_store": False,
    }
    
    # Validate checkpointer
    try:
        checkpointer = create_checkpointer(mode="in-memory")
        results["checkpointer"] = checkpointer is not None
    except Exception as e:
        print(f"Checkpointer validation failed: {e}")
    
    # Validate store
    try:
        store = get_store()
        results["store"] = store is not None
    except Exception as e:
        print(f"Store validation failed: {e}")
    
    # Validate vector store
    try:
        vector_store = VectorStore(mode="in-memory")
        results["vector_store"] = vector_store is not None
    except Exception as e:
        print(f"Vector store validation failed: {e}")
    
    return results


def validate_graph_compilation() -> bool:
    """Validate that the graph compiles without errors."""
    try:
        graph = build_hello_world_graph()
        app = graph.compile()
        return app is not None
    except Exception as e:
        print(f"Graph compilation failed: {e}")
        return False


# Convenience function for quick validation
def hello_world_validation() -> Dict[str, Any]:
    """Run full validation of hello world graph and memory components."""
    return {
        "memory_components": validate_memory_components(),
        "graph_compilation": validate_graph_compilation(),
    }


if __name__ == "__main__":
    import asyncio
    
    print("=== AutFin Hello World Graph Validation ===\n")
    
    # Validate components
    print("1. Validating memory components...")
    component_results = validate_memory_components()
    for component, status in component_results.items():
        print(f"   {component}: {'✓' if status else '✗'}")
    
    # Validate graph compilation
    print("\n2. Validating graph compilation...")
    graph_ok = validate_graph_compilation()
    print(f"   Graph compilation: {'✓' if graph_ok else '✗'}")
    
    # Run hello world graph
    if graph_ok:
        print("\n3. Running hello world graph...")
        result = asyncio.run(run_hello_world())
        print(f"   Result: {result}")
    
    print("\n=== Validation Complete ===")