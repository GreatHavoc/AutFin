"""Shared embedding utilities for AutFin.

Centralizes sentence-transformers / hash fallback so every module
uses the same model instance and cosine similarity logic.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Suppress HuggingFace warnings — model is cached locally, no download needed
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Module-level singleton — loaded once per process
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
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=True,  # Don't check hub, use local cache only
        )
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — using hash embeddings. "
            "Install with: pip install sentence-transformers"
        )
        _model = None
    except Exception as e:
        logger.debug("Failed to load sentence-transformers: %s — using hash fallback", e)
        _model = None
    return _model


def generate_embedding(text: str) -> List[float]:
    """Generate a semantic embedding for text.

    Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim). When the model is not
    installed this raises instead of returning a sha256 vector: two unrelated
    sentences hash to uncorrelated vectors, so every similarity score built on
    the fallback is noise that looks like signal. Callers that want ranking
    without a model should use ConversationSearch over stored history instead.
    """
    model = _get_model()
    if model is None:
        raise RuntimeError(
            "No embedding model available. sentence-transformers is not a "
            "dependency, and the previous sha256 fallback produced meaningless "
            "similarity scores. Install sentence-transformers, or rank with "
            "pydantic_ai_harness.conversation_search over stored history."
        )

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
