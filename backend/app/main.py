import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database import init_db
from backend.app.api import admin, chat

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pdf_qa")

# Ensure DB tables are initialized
init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing PDF Q&A Assistant backend...")
    init_db()
    logger.info("Database tables initialized.")
    yield
    logger.info("Shutting down PDF Q&A Assistant backend.")

app = FastAPI(
    title="PDF Q&A Assistant API",
    description="Full-stack Advanced RAG Assistant with Gemini + Groq Fallback, Redis Semantic Cache, Pinecone Hybrid Search, and Local Cross-Encoder Reranking.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(admin.router)
app.include_router(chat.router)

@app.get("/api/health")
async def health_check():
    """Health check endpoint reporting on backend subsystem readiness."""
    return {
        "status": "online",
        "app_name": "PDF Q&A Assistant",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "gemini_model": settings.GEMINI_MODEL,
        "groq_model": settings.GROQ_MODEL,
        "cache_threshold": settings.CACHE_SIMILARITY_THRESHOLD,
        "hybrid_alpha": settings.HYBRID_ALPHA,
        "answer_timeout_sec": settings.ANSWER_TIMEOUT_SECONDS
    }

# If frontend build exists, mount static assets
frontend_dist = os.path.join(os.path.dirname(settings.UPLOAD_DIR), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
