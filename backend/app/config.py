import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Admin Credentials & Security
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "adminsecretpassword"
    ADMIN_SECRET_KEY: str = "change-this-to-a-super-secret-key-for-jwt-signing"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # LLM Providers
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Pinecone Vector DB
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "pdf-qa-index"
    PINECONE_METRIC: str = "dotproduct"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Redis Stack (Semantic Cache)
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_INDEX_NAME: str = "semantic_cache_idx"

    # Local Models (HuggingFace)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # RAG & Ingestion Parameters
    MAX_CHUNK_SIZE: int = 1000
    HYBRID_ALPHA: float = 0.5
    RETRIEVAL_TOP_K: int = 15
    RERANK_TOP_K: int = 3
    ANSWER_TIMEOUT_SECONDS: float = 60.0
    QUERY_REWRITE_TIMEOUT_SECONDS: float = 5.0

    # Database & Storage
    SQLITE_DB_PATH: str = str(BASE_DIR / "backend" / "data" / "pdf_qa.db")
    UPLOAD_DIR: str = str(BASE_DIR / "backend" / "data" / "uploads")
    BM25_PARAMS_PATH: str = str(BASE_DIR / "backend" / "data" / "bm25" / "bm25_params.json")

    # OCR Configuration
    TESSERACT_CMD: str = "tesseract"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(Path(settings.SQLITE_DB_PATH).parent, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(Path(settings.BM25_PARAMS_PATH).parent, exist_ok=True)

# Register workspace nltk_data path
import nltk
nltk_data_dir = str(BASE_DIR / "backend" / "data" / "nltk_data")
if os.path.exists(nltk_data_dir) and nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)
