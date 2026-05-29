"""
Stage 4a Utility: Cross-Encoder Re-ranking for RAG Retrieval
Improves Context Precision by re-scoring raw ChromaDB candidates with a
cross-encoder model before returning them to the LLM.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Size: ~66 MB (CPU-friendly, no GPU required)
  - Designed for passage re-ranking (query + passage scored jointly)
  - Significantly outperforms bi-encoder cosine similarity for relevance ranking

Usage:
    from pipeline.stage4a_rerank import rerank_passages

    raw_passages = [...]  # list of (text, metadata) tuples from ChromaDB
    top_k = rerank_passages(query, raw_passages, top_k=8)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Singleton cross-encoder (loaded once, reused across all queries)
_cross_encoder = None
_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    """Lazy-load the cross-encoder. Falls back to None if unavailable."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder  # noqa: available via sentence-transformers
        _cross_encoder = CrossEncoder(_CE_MODEL, max_length=512)
        print(f"  [Rerank] Cross-encoder loaded: {_CE_MODEL}")
    except Exception as exc:
        logger.warning(
            "[Rerank] CrossEncoder unavailable (%s). "
            "Falling back to original retrieval order.", exc
        )
        _cross_encoder = None
    return _cross_encoder


def rerank_passages(query: str,
                    passages: list[str],
                    top_k: int | None = None) -> list[str]:
    """
    Re-rank a list of text passages by relevance to `query` using a cross-encoder.

    Args:
        query:    The retrieval query string (anomaly context + topic keywords).
        passages: List of raw text passages from ChromaDB (one per retrieved doc).
        top_k:    Number of passages to return after re-ranking.
                  Pass None or len(passages) to reorder ALL passages without truncating.
                  This is the preferred mode for RAGAS evaluation, as context recall
                  benefits from maximum coverage while the LLM still sees the best
                  passages first (position bias in attention).

    Returns:
        List of passages ordered by descending relevance score.
        If top_k is None, all passages are returned (just reordered).
        If the cross-encoder is unavailable, returns passages in original order.
    """
    if not passages:
        return passages

    ce = _get_cross_encoder()

    # Resolve top_k: None or values >= len(passages) mean "keep all"
    keep = len(passages) if (top_k is None or top_k >= len(passages)) else top_k

    if ce is None:
        # Graceful fallback: no re-ranking, return in original order
        return passages[:keep]

    try:
        # Cross-encoder scores (query, passage) pairs jointly
        pairs  = [(query, p) for p in passages]
        scores = ce.predict(pairs)  # numpy array of floats

        # Sort by descending score
        ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        return [p for _, p in ranked[:keep]]

    except Exception as exc:
        logger.warning("[Rerank] Cross-encoder prediction failed (%s). "
                       "Using original order.", exc)
        return passages[:keep]

