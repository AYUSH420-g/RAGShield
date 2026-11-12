import logging
import torch
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.embeddings")

_embedding_instance: HuggingFaceEmbeddings = None

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_embeddings_model() -> HuggingFaceEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        device = get_device()
        logger.info(f"Loading HuggingFaceEmbeddings model '{settings.EMBEDDING_MODEL_NAME}' on device: {device}...")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Loaded embeddings model successfully.")
    return _embedding_instance

def embed_query(text: str) -> List[float]:
    model = get_embeddings_model()
    return model.embed_query(text)

def embed_documents(texts: List[str]) -> List[List[float]]:
    model = get_embeddings_model()
    return model.embed_documents(texts)
