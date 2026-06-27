"""Shared embedding utilities for AutFin.

Centralizes sentence-transformers / hash fallback so every module
uses the same model instance and cosine similarity logic.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, shared across all callers
_model = None
_model_loaded = False


def _get_model():
    """Lazy-load sentence-transformers model (cached at module level)."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — using hash embeddings. "
            "Install with: pip install sentence-transformers"
        )
        _model = None
    return _model


def generate_embedding(text: str) -> List[float]:
    """Generate semantic embedding for text.

    Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) if available.
    Falls back to hash-based embedding for backward compatibility.
    """
    model = _get_model()
    if model is None:
        h = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in h] + [0.0] * (384 - len(h))

    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import numpy as np
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
