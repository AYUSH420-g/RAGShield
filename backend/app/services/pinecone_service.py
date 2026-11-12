import logging
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.pinecone")

class PineconeService:
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.metric = settings.PINECONE_METRIC
        self.dimension = settings.EMBEDDING_DIM
        self.cloud = settings.PINECONE_CLOUD
        self.region = settings.PINECONE_REGION
        self.client: Optional[Pinecone] = None
        self.index = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or self.api_key == "mock-pinecone-key" or "your_" in self.api_key:
            logger.warning("Pinecone API key not set or mock key provided. Operating in local in-memory fallback mode.")
            self._init_local_mock()
            return

        try:
            self.client = Pinecone(api_key=self.api_key)
            existing_indexes = [idx["name"] for idx in self.client.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone serverless index '{self.index_name}' with metric '{self.metric}'...")
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                # Wait for index to be ready
                while not self.client.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                logger.info(f"Pinecone index '{self.index_name}' is ready.")

            self.index = self.client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index '{self.index_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}. Falling back to in-memory store.")
            self._init_local_mock()

    def _init_local_mock(self):
        """In-memory mock store for testing or offline environments."""
        self.is_mock = True
        self._mock_vectors: Dict[str, Dict[str, Any]] = {}

    def is_using_mock(self) -> bool:
        return getattr(self, "is_mock", False)

    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert vectors to Pinecone.
        Format of each item:
        {
            "id": "{doc_id}#chunk-{n}",
            "values": List[float], (dense)
            "sparse_values": {"indices": List[int], "values": List[float]},
            "metadata": {"doc_id": str, "file_name": str, "page": int, "type": str, "text": str}
        }
        """
        if self.is_using_mock():
            for v in vectors:
                self._mock_vectors[v["id"]] = v
            logger.info(f"[Mock Pinecone] Upserted {len(vectors)} vectors. Total store: {len(self._mock_vectors)}")
            return

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                logger.debug(f"Upserted batch of {len(batch)} vectors to Pinecone")
            except Exception as e:
                logger.error(f"Error upserting vectors to Pinecone: {e}")
                raise e

    def hybrid_query(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        top_k: int = 15,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search query using alpha weighting.
        alpha = 1.0 -> purely dense
        alpha = 0.0 -> purely sparse
        """
        scaled_dense = [v * alpha for v in dense_vector]
        scaled_sparse = {
            "indices": sparse_vector.get("indices", []),
            "values": [v * (1.0 - alpha) for v in sparse_vector.get("values", [])]
        }

        if self.is_using_mock():
            # In-memory hybrid scoring for local testing
            results = []
            for vid, item in self._mock_vectors.items():
                # Dense dotproduct
                d_val = item["values"]
                dense_score = sum(a * b for a, b in zip(scaled_dense, d_val))
                
                # Sparse dotproduct
                sparse_score = 0.0
                item_sparse = item.get("sparse_values", {})
                item_indices = dict(zip(item_sparse.get("indices", []), item_sparse.get("values", [])))
                for idx, val in zip(scaled_sparse["indices"], scaled_sparse["values"]):
                    if idx in item_indices:
                        sparse_score += val * item_indices[idx]

                score = dense_score + sparse_score
                results.append({
                    "id": vid,
                    "score": score,
                    "metadata": item.get("metadata", {})
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        try:
            # Query Pinecone with hybrid search
            response = self.index.query(
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=top_k,
                include_metadata=True
            )
            matches = []
            for match in response.get("matches", []):
                matches.append({
                    "id": match["id"],
                    "score": match.get("score", 0.0),
                    "metadata": match.get("metadata", {})
                })
            return matches
        except Exception as e:
            logger.error(f"Pinecone hybrid_query failed: {e}")
            return []

    def delete_by_doc_id(self, doc_id: str, chunk_ids: Optional[List[str]] = None):
        """
        Delete vectors belonging to a doc_id.
        """
        if self.is_using_mock():
            to_delete = [vid for vid in self._mock_vectors if vid.startswith(f"{doc_id}#") or vid in (chunk_ids or [])]
            for vid in to_delete:
                del self._mock_vectors[vid]
            logger.info(f"[Mock Pinecone] Deleted {len(to_delete)} vectors for doc {doc_id}")
            return

        try:
            if chunk_ids:
                # Delete by explicit list of IDs
                self.index.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunk IDs for doc {doc_id}")
            else:
                # Pinecone filter delete
                self.index.delete(filter={"doc_id": {"$eq": doc_id}})
                logger.info(f"Deleted vectors for doc_id {doc_id} by metadata filter")
        except Exception as e:
            logger.warning(f"Error during Pinecone delete for doc_id {doc_id}: {e}")
            if chunk_ids:
                try:
                    self.index.delete(ids=chunk_ids)
                except Exception as ex:
                    logger.error(f"Failed deleting chunk IDs fallback: {ex}")

_pinecone_instance: PineconeService = None

def get_pinecone_service() -> PineconeService:
    global _pinecone_instance
    if _pinecone_instance is None:
        _pinecone_instance = PineconeService()
    return _pinecone_instance
