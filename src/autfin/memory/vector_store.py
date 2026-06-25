"""ChromaDB vector store for AutFin embeddings and semantic search."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from ..config.settings import AutFinSettings


# Required collections for AutFin
REQUIRED_COLLECTIONS = [
    "market_data",
    "research_notes",
    "trade_history",
    "learning_journal",
]


def get_chroma_path() -> Path:
    """Get the ChromaDB persistence path from config or default."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    try:
        settings = AutFinSettings()
        return data_dir / "chroma"
    except Exception:
        return data_dir / "chroma"


def create_chroma_client(
    mode: str = "persistent", persist_path: Optional[Path] = None
) -> chromadb.Client:
    """
    Create a ChromaDB client.
    
    Args:
        mode: Either "persistent" (file-based) or "in-memory"
        persist_path: Path for persistent storage (defaults to data/chroma/)
    
    Returns:
        ChromaDB Client instance
    """
    if mode == "in-memory":
        return chromadb.EphemeralClient()
    
    # Persistent mode - use file-based storage
    if persist_path is None:
        persist_path = get_chroma_path()
    
    # Ensure directory exists
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    
    return chromadb.PersistentClient(path=str(persist_path))


def get_or_create_collection(
    client: chromadb.Client,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.
    
    Args:
        client: ChromaDB client instance
        name: Collection name
        metadata: Optional collection metadata (e.g., "hnsw:config")
    
    Returns:
        ChromaDB Collection instance
    """
    return client.get_or_create_collection(name=name, metadata=metadata)


def initialize_collections(
    client: chromadb.Client,
    collection_names: Optional[List[str]] = None,
) -> Dict[str, chromadb.Collection]:
    """
    Initialize all required or specified collections.
    
    Args:
        client: ChromaDB client instance
        collection_names: Optional list of collection names (defaults to all required)
    
    Returns:
        Dictionary mapping collection names to Collection instances
    """
    if collection_names is None:
        collection_names = REQUIRED_COLLECTIONS
    
    collections = {}
    for name in collection_names:
        collections[name] = get_or_create_collection(client, name)
    
    return collections


class VectorStore:
    """
    ChromaDB-based vector store for AutFin.
    
    Provides semantic search capabilities for:
    - Market data vectors (price movements, technical indicators)
    - Research notes (analyst reports, documents)
    - Trade history (execution vectors)
    - Learning journal (strategy adjustments, lessons)
    """
    
    def __init__(
        self,
        mode: str = "persistent",
        persist_path: Optional[Path] = None,
        collection_names: Optional[List[str]] = None,
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            mode: "persistent" (file-based) or "in-memory"
            persist_path: Path for persistent storage
            collection_names: Optional specific collections to initialize
        """
        self.mode = mode
        self.persist_path = persist_path or get_chroma_path()
        self.client = create_chroma_client(mode=mode, persist_path=self.persist_path)
        self.collections = initialize_collections(
            self.client, collection_names=collection_names
        )
    
    def add(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to a collection.
        
        Args:
            collection_name: Target collection name
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of item IDs (auto-generated if not provided)
        
        Returns:
            List of created item IDs
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = get_or_create_collection(
                self.client, collection_name
            )
        
        collection = self.collections[collection_name]
        
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        
        return ids
    
    def search(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Target collection name
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            where: Metadata filter (e.g., {"date": {"$gte": "2024-01-01"}})
            where_document: Document text filter
        
        Returns:
            Dictionary with results (ids, distances, metadatas, documents)
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
        )
        
        return results
    
    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get vectors from a collection without embedding comparison.
        
        Args:
            collection_name: Target collection name
            ids: Optional list of specific IDs to retrieve
            where: Metadata filter
            limit: Maximum number of results
            offset: Number of results to skip
        
        Returns:
            Dictionary with results
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        results = collection.get(
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
        )
        
        return results
    
    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Delete vectors from a collection.
        
        Args:
            collection_name: Target collection name
            ids: Optional list of IDs to delete
            where: Metadata filter for bulk deletion
        
        Returns:
            List of deleted IDs
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        if ids is None and where is None:
            raise ValueError("Either 'ids' or 'where' must be provided")
        
        if ids:
            deleted = collection.delete(ids=ids)
        else:
            deleted = collection.delete(where=where)
        
        return deleted
    
    def update(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Update existing vectors in a collection.
        
        Args:
            collection_name: Target collection name
            ids: List of IDs to update
            embeddings: Optional new embeddings
            metadatas: Optional new metadata
            documents: Optional new documents
        
        Returns:
            List of updated IDs
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        updated = collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        
        return updated
    
    def get_collection(self, collection_name: str) -> Optional[chromadb.Collection]:
        """Get a specific collection by name."""
        return self.collections.get(collection_name)
    
    def list_collections(self) -> List[str]:
        """List all available collection names."""
        collections = self.client.list_collections()
        # ChromaDB 0.5+ returns list[str], older versions return list[Collection]
        if collections and isinstance(collections[0], str):
            return collections
        return [col.name for col in collections]
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entirely."""
        if collection_name in self.collections:
            self.client.delete_collection(name=collection_name)
            del self.collections[collection_name]
    
    def persist(self) -> None:
        """Persist changes to disk (PersistentClient handles this automatically)."""
        # PersistentClient automatically persists to disk
        # This method exists for API compatibility but is a no-op
        pass
