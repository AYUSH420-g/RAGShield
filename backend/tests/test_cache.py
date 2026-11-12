import pytest
import numpy as np
from backend.app.services.cache import SemanticCache

def test_semantic_cache_hit_and_miss():
    cache = SemanticCache()
    # Force in-memory fallback for deterministic test
    cache.is_connected = False
    cache.clear()

    # Create dummy 768-dim vector
    v1 = [0.1] * 768
    v1_norm = (np.array(v1) / np.linalg.norm(v1)).tolist()

    cache.set(
        question="What is the revenue for Q3?",
        query_vector=v1_norm,
        answer="The revenue was $5.2 million.",
        sources=[{"file_name": "financials.pdf", "page": 4}]
    )

    # 1. Exact match (cosine similarity = 1.0 >= 0.95) -> HIT
    res_exact = cache.get(v1_norm)
    assert res_exact is not None
    assert res_exact["hit"] is True
    assert res_exact["similarity"] >= 0.99
    assert res_exact["answer"] == "The revenue was $5.2 million."
    assert res_exact["sources"][0]["file_name"] == "financials.pdf"

    # 2. Similar vector (high cosine similarity >= 0.95) -> HIT
    v_similar = (np.array(v1_norm) + np.random.normal(0, 0.005, 768)).tolist()
    res_sim = cache.get(v_similar)
    assert res_sim is not None
    assert res_sim["hit"] is True

    # 3. Completely different orthogonal vector -> MISS
    v_diff = ([1.0] * 384 + [-1.0] * 384)
    res_diff = cache.get(v_diff)
    assert res_diff is None

    # 4. Clear cache
    cache.clear()
    assert cache.get(v1_norm) is None
