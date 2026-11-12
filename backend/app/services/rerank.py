import logging
import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from backend.app.services.embeddings import get_device
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.rerank")

_reranker_instance: CrossEncoder = None

def get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        device = get_device()
        logger.info(f"Loading CrossEncoder model '{settings.RERANKER_MODEL_NAME}' on device: {device}...")
        _reranker_instance = CrossEncoder(settings.RERANKER_MODEL_NAME, device=device)
        logger.info("Loaded CrossEncoder model successfully.")
    return _reranker_instance

def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = settings.RERANK_TOP_K
) -> List[Dict[str, Any]]:
    """
    Score every unique chunk against query using BAAI/bge-reranker-base.
    Returns the top_k scored chunks with rerank_score attached.
    """
    if not chunks:
        return []

    start_time = time.time()
    reranker = get_reranker()

    pairs = []
    for c in chunks:
        text = c.get("metadata", {}).get("text", "") or ""
        pairs.append([query, text])

    scores = reranker.predict(pairs)
    # If single item, scores might be float/scalar
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    elif not isinstance(scores, list):
        scores = [float(scores)]

    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        chunk_copy = dict(chunk)
        chunk_copy["rerank_score"] = float(score)
        scored_chunks.append(chunk_copy)

    # Sort descending by rerank_score
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = scored_chunks[:top_k]

    elapsed = time.time() - start_time
    score_summary = [f"{c['id']}: {c['rerank_score']:.4f}" for c in top_chunks]
    logger.info(
        f"Reranked {len(chunks)} chunks in {elapsed:.3f}s. Top {len(top_chunks)} scores: {score_summary}"
    )

    return top_chunks
