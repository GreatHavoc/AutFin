"""
AutFin Memory Layer
"""
from .database import (
    get_connection,
    init_database,
)

# Lazy imports to avoid optional dependencies (chromadb, orjson)
# Use direct imports: from autfin.memory.vector_store import ...
# Use direct imports: from autfin.memory.retrieval import ...
# Use direct imports: from autfin.memory.performance import ...

__all__ = [
    # Database
    "get_connection",
    "init_database",
]
