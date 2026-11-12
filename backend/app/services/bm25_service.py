import os
import json
import logging
from typing import List, Dict, Any
from pinecone_text.sparse import BM25Encoder
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.bm25")

class BM25Service:
    def __init__(self, params_path: str = settings.BM25_PARAMS_PATH):
        self.params_path = params_path
        self.encoder = BM25Encoder()
        self._load_or_default()

    def _load_or_default(self):
        if os.path.exists(self.params_path) and os.path.getsize(self.params_path) > 0:
            try:
                self.encoder.load(self.params_path)
                logger.info(f"Loaded existing BM25 params from {self.params_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 params from {self.params_path}: {e}")
        
        # Initialize with default parameters if file doesn't exist yet
        try:
            self.encoder = BM25Encoder.default()
            logger.info("Initialized default BM25Encoder.")
        except Exception as e:
            logger.warning(f"Could not load BM25Encoder.default(): {e}")
            self.encoder = BM25Encoder()

    def fit_and_save(self, corpus: List[str]):
        """
        Fit BM25 encoder on all text chunks and save parameters to disk.
        """
        if not corpus:
            logger.warning("Empty corpus provided for BM25 fit. Retaining current encoder.")
            return

        logger.info(f"Fitting BM25Encoder on {len(corpus)} document chunks...")
        self.encoder.fit(corpus)
        
        os.makedirs(os.path.dirname(self.params_path), exist_ok=True)
        self.encoder.dump(self.params_path)
        logger.info(f"Saved fitted BM25 params to {self.params_path}")

    def encode_query(self, query: str) -> Dict[str, Any]:
        """
        Encode query to sparse vector: {'indices': [...], 'values': [...]}
        """
        try:
            return self.encoder.encode_queries(query)
        except Exception as e:
            logger.warning(f"Error encoding query for BM25: {e}. Falling back to default encode.")
            try:
                default_enc = BM25Encoder.default()
                return default_enc.encode_queries(query)
            except Exception:
                return {"indices": [], "values": []}

    def encode_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Encode list of document chunks to sparse vectors.
        """
        results = []
        for doc in documents:
            try:
                res = self.encoder.encode_documents(doc)
            except Exception as e:
                logger.warning(f"Error encoding document for BM25: {e}")
                res = {"indices": [], "values": []}
            results.append(res)
        return results

_bm25_instance: BM25Service = None

def get_bm25_service() -> BM25Service:
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Service()
    return _bm25_instance
