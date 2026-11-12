import json
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List
import redis
from redis.commands.search.field import TextField, VectorField, NumericField
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.cache")

class SemanticCache:
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self.index_name = settings.CACHE_INDEX_NAME
        self.dim = settings.EMBEDDING_DIM
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        self._in_memory_fallback: Dict[str, Dict[str, Any]] = {}
        self._init_redis()

    def _init_redis(self):
        try:
            self.client = redis.Redis.from_url(self.redis_url, decode_responses=False)
            self.client.ping()
            self.is_connected = True
            logger.info("Connected to Redis Stack successfully.")
            self._ensure_index()
        except Exception as e:
            logger.warning(f"Could not connect to Redis ({e}). Running in-memory semantic cache fallback.")
            self.is_connected = False

    def _ensure_index(self):
        if not self.is_connected:
            return
        try:
            self.client.ft(self.index_name).info()
            logger.info(f"Redis vector index '{self.index_name}' already exists.")
        except Exception:
            logger.info(f"Creating Redis vector index '{self.index_name}' with COSINE metric...")
            try:
                schema = (
                    TextField("question"),
                    TextField("answer"),
                    TextField("sources_json"),
                    NumericField("timestamp"),
                    VectorField(
                        "vector",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.dim,
                            "DISTANCE_METRIC": "COSINE",
                        }
                    )
                )
                definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
                self.client.ft(self.index_name).create_index(schema, definition=definition)
                logger.info(f"Created Redis vector index '{self.index_name}'.")
            except Exception as e:
                logger.warning(f"Failed to create Redis vector index: {e}")

    def get(self, query_vector: List[float]) -> Optional[Dict[str, Any]]:
        """
        Search for cached answer with cosine similarity >= threshold (default 0.95).
        Returns Dict with 'answer', 'sources', 'similarity', 'question' if hit, else None.
        """
        if not query_vector:
            return None

        norm_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(norm_vec)
        if norm > 0:
            norm_vec = norm_vec / norm

        if self.is_connected:
            try:
                query_bytes = norm_vec.tobytes()
                # Redis COSINE distance metric: score = 1 - cosine_similarity
                # So cosine_similarity = 1 - distance
                q = (
                    Query("*=>[KNN 1 @vector $vec AS dist]")
                    .sort_by("dist")
                    .return_fields("dist", "question", "answer", "sources_json")
                    .dialect(2)
                )
                params = {"vec": query_bytes}
                res = self.client.ft(self.index_name).search(q, query_params=params)

                if res and res.total > 0:
                    doc = res.docs[0]
                    dist = float(getattr(doc, "dist", 1.0))
                    similarity = 1.0 - dist
                    logger.debug(f"Cache check: closest distance={dist:.4f}, similarity={similarity:.4f}")

                    if similarity >= self.threshold:
                        answer = getattr(doc, "answer", "")
                        sources_json = getattr(doc, "sources_json", "[]")
                        sources = json.loads(sources_json) if isinstance(sources_json, str) else sources_json
                        question = getattr(doc, "question", "")
                        return {
                            "hit": True,
                            "similarity": similarity,
                            "answer": answer,
                            "sources": sources,
                            "question": question
                        }
            except Exception as e:
                logger.warning(f"Redis cache search error: {e}. Checking in-memory fallback.")

        # In-memory fallback
        best_sim = -1.0
        best_match = None
        for key, entry in self._in_memory_fallback.items():
            stored_vec = entry["vector"]
            # Cosine similarity for normalized vectors
            sim = float(np.dot(norm_vec, stored_vec))
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match and best_sim >= self.threshold:
            return {
                "hit": True,
                "similarity": best_sim,
                "answer": best_match["answer"],
                "sources": best_match["sources"],
                "question": best_match["question"]
            }

        return None

    def set(self, question: str, query_vector: List[float], answer: str, sources: List[Dict[str, Any]]):
        """
        Store question, embedding, answer, and sources in semantic cache.
        Never call this on errors or timeouts.
        """
        if not answer or not query_vector:
            return

        norm_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(norm_vec)
        if norm > 0:
            norm_vec = norm_vec / norm

        key = f"cache:{abs(hash(question))}_{int(time.time()*1000)}"

        if self.is_connected:
            try:
                mapping = {
                    "question": question,
                    "answer": answer,
                    "sources_json": json.dumps(sources),
                    "timestamp": time.time(),
                    "vector": norm_vec.tobytes()
                }
                self.client.hset(key, mapping=mapping)
                logger.info(f"Stored question in Redis semantic cache: '{question[:40]}...'")
                return
            except Exception as e:
                logger.warning(f"Failed to write to Redis semantic cache: {e}")

        # Store in in-memory fallback
        self._in_memory_fallback[key] = {
            "question": question,
            "vector": norm_vec,
            "answer": answer,
            "sources": sources,
            "timestamp": time.time()
        }
        logger.info(f"Stored question in in-memory semantic cache: '{question[:40]}...'")

    def clear(self):
        """
        Clear all cache keys. Called after any upload or delete.
        """
        logger.info("Clearing semantic cache...")
        if self.is_connected:
            try:
                # Find all cache keys and delete them
                cursor = 0
                while True:
                    cursor, keys = self.client.scan(cursor=cursor, match="cache:*", count=500)
                    if keys:
                        self.client.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("Redis semantic cache cleared successfully.")
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")

        self._in_memory_fallback.clear()
        logger.info("In-memory semantic cache cleared.")

_cache_instance: SemanticCache = None

def get_semantic_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
