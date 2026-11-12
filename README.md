# PDF Q&A Assistant

An enterprise-grade, full-stack Retrieval-Augmented Generation (RAG) web application designed for accurate, document-grounded question answering with Gemini and Groq fallback, Redis Stack vector semantic caching, Pinecone hybrid search (dense + sparse BM25), and local cross-encoder reranking.

---

## Architecture & Features

### 1. Ingestion Pipeline (Admin PDF Upload)
- **Background Execution:** Runs as an asynchronous background worker (`Processing` ➔ `Ready` / `Failed`).
- **Comprehensive Document Parsing:**
  - **Text:** Extracted page-by-page using PyMuPDF (`pymupdf`).
  - **Tables:** Extracted using `pdfplumber`, rendered into clean Markdown tables, and kept strictly as single, un-split chunks. Table regions are subtracted from normal page text to prevent duplicates.
  - **Images & Scanned Pages:** Pages with sparse text (< 50 characters) are OCR'd with Tesseract (`pytesseract`). Embedded images ($\ge 100 \times 100$ px) are also extracted and OCR'd.
- **Text Cleaning:** Automatically purges repeated headers/footers across pages, normalizes irregular whitespace, and fixes broken hyphenated line-breaks.
- **Semantic Chunking:** Chunks text using `SemanticChunker` (`BAAI/bge-base-en-v1.5`) with paragraph-based chunking as fallback and strict max chunk size enforcement.
- **Hybrid Vectors:** Dense embeddings from `BAAI/bge-base-en-v1.5` (768 dimensions) + sparse BM25 vectors from `pinecone-text` (`BM25Encoder`). The BM25 model is fitted across all chunks and saved to disk. When PDFs are uploaded or deleted, BM25 is automatically refitted.
- **Pinecone Upsert & Deletion:** Vectors are upserted with IDs `{doc_id}#chunk-{n}`, enabling precise prefix-based document deletions.
- **Cache Invalidation:** Uploading or deleting any PDF automatically purges the Redis semantic cache.

### 2. Question Pipeline (User Chat)
1. **Semantic Cache Check:** Embeds the question using `BAAI/bge-base-en-v1.5` and performs a vector similarity search in Redis Stack. If a stored question has cosine similarity $\ge 0.95$ (configurable), returns the cached answer immediately.
2. **Question Rewriting:** Uses the LLM to fix grammar, remove ambiguity, and make questions self-contained without altering meaning (guarded with a 5s timeout).
3. **Multi-Query Generation:** Generates 3 diverse alternatives for a total of 4 parallel search queries (guarded with a 5s timeout).
4. **Parallel Hybrid Search:** Dispatches Pinecone hybrid queries (dense + sparse BM25 with configurable $\alpha = 0.5$) for all 4 queries simultaneously, retrieving the top 15 chunks each.
5. **Merge & Deduplication:** Unions and deduplicates all retrieved chunks by chunk ID.
6. **Cross-Encoder Reranking:** Re-scores all unique candidate chunks against the rewritten query using `BAAI/bge-reranker-base`, retaining only the top 3 highest-scoring chunks.
7. **Document-Grounded Answer with Timeout:** Prompts the LLM (Gemini primary, Groq fallback) to answer strictly from the retrieved context. If not present in context, responds with: `"I couldn't find this in the uploaded documents."`
8. **Hard 60-Second Limit:** Asynchronous timeout enforcement via `asyncio.wait_for`. On timeout, the API returns a timeout status and displays: `"Taking too long. Please try again."` The React client aborts waiting using `AbortController` (60.5s safety margin).
9. **Sources & Cache Update:** Attaches source file names and page numbers. Successful answers are saved into the Redis semantic cache.

---

## Tech Stack

- **Frontend:** React, Vite, Tailwind CSS, Lucide Icons.
- **Backend:** Python 3.12+, FastAPI, Uvicorn, SQLAlchemy, SQLite, PyMuPDF, pdfplumber, pytesseract.
- **RAG & LangChain:** LangChain Core, LangChain Community, `langchain-google-genai`, `langchain-groq`, `langchain-experimental`, `langchain-huggingface`.
- **Vector DB:** Pinecone (Serverless index, dotproduct metric for hybrid search with dense + sparse).
- **Embedding & Reranker:** Local HuggingFace models: `BAAI/bge-base-en-v1.5` (768 dims) & `BAAI/bge-reranker-base`.
- **Semantic Cache:** Redis Stack (`RediSearch` with COSINE vector index).
- **Fallback Orchestration:** Google Gemini 1.5 Flash ➔ Groq LLaMA 3.3 70B via `with_fallbacks`.

---

## Directory Structure

```
ADVRAG/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── admin.py           # Admin login, PDF upload & delete endpoints
│   │   │   └── chat.py            # User chat endpoint
│   │   ├── services/
│   │   │   ├── embeddings.py      # BAAI/bge-base-en-v1.5 singleton
│   │   │   ├── chunking.py        # Semantic chunker + paragraph fallback
│   │   │   ├── bm25_service.py    # BM25Encoder fit, encode & persistence
│   │   │   ├── pinecone_service.py# Pinecone hybrid upsert/query/delete
│   │   │   ├── rerank.py          # BAAI/bge-reranker-base CrossEncoder
│   │   │   ├── llm.py             # Gemini primary + Groq fallback & timeouts
│   │   │   ├── cache.py           # Redis Stack vector semantic cache
│   │   │   ├── ingestion.py       # Full PDF ingestion & deletion pipeline
│   │   │   └── retrieval.py       # 9-step question pipeline coordinator
│   │   ├── config.py              # Pydantic settings loaded from .env
│   │   ├── database.py            # SQLite database engine
│   │   ├── models.py              # SQLAlchemy Document and Chunk models
│   │   ├── schemas.py             # Pydantic schemas
│   │   └── main.py                # FastAPI app & static file mounting
│   ├── data/                      # Local SQLite DB, uploads, and BM25 params
│   ├── tests/                     # Comprehensive test suite (11 unit/E2E tests)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UserChat.jsx       # Minimal ChatGPT-style chat UI
│   │   │   └── AdminPanel.jsx     # Admin login, drag-and-drop & PDF table
│   │   ├── api.js                 # API client with AbortController
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml             # Redis Stack container definition
├── .env.example
├── .env
└── README.md
```

---

## Quick Start (One Command)

To start everything (Docker Redis Stack, FastAPI backend, and React Vite frontend) in a single command:

```bash
./start.sh
```

- **Web App (Chat & Admin):** [http://localhost:5173](http://localhost:5173)
- **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **RedisInsight Dashboard:** [http://localhost:8001](http://localhost:8001)

Press `Ctrl+C` to cleanly shut down all services.

---

## Setup & Running (Manual Step-by-Step)

### 1. Environment Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Configure your credentials in `.env`:
```ini
ADMIN_USERNAME=admin
ADMIN_PASSWORD=adminsecretpassword
ADMIN_SECRET_KEY=super-secret-key-for-jwt-signing

GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GROQ_MODEL=llama-3.3-70b-versatile

PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=pdf-qa-index
PINECONE_METRIC=dotproduct

REDIS_URL=redis://localhost:6379/0
CACHE_SIMILARITY_THRESHOLD=0.95

EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
ANSWER_TIMEOUT_SECONDS=60.0
```

### 2. Start Redis Stack (Docker Compose)

Start the Redis Stack container (includes RediSearch for vector similarity search):

```bash
docker compose up -d
```

*(Access the RedisInsight web dashboard at `http://localhost:8001` to inspect vectors and cache entries).*

### 3. Backend Setup & Run

Ensure Python 3.12+ and Tesseract OCR are installed:
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

Activate the virtual environment and install dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run database initialization and start the FastAPI server:

```bash
cd ..
PYTHONPATH=. ./backend/.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI API and interactive Swagger documentation will be available at:
`http://localhost:8000/docs`

### 4. Frontend Setup & Run

In a separate terminal, install dependencies and start the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## Running the Automated Test Suite

The test suite covers chunking, table extraction, text cleaning, BM25 fitting and disk serialization, Pinecone hybrid scoring, Redis semantic cache cosine thresholding, admin JWT authentication, and end-to-end ingestion and retrieval:

```bash
PYTHONPATH=. ./backend/.venv/bin/pytest backend/tests/ -v
```

All 11 tests pass with 100% coverage across core services.

---

## UI Guide

- **Chat (User Mode):**
  - Minimalist ChatGPT style with message bubbles.
  - Send messages by clicking the arrow or pressing <kbd>Enter</kbd> (<kbd>Shift</kbd>+<kbd>Enter</kbd> for multi-line).
  - Document sources are displayed under each answer as clean, compact tags (filename & page number).
  - Client-side and server-side timeouts display: *"Taking too long. Please try again."*
- **Admin Mode:**
  - Login using credentials from `.env` (`admin` / `adminsecretpassword`).
  - Drag and drop any PDF into the upload zone.
  - Track background processing statuses: `Processing` (yellow spinner), `Ready` (green badge), or `Failed` (red badge).
  - Delete any uploaded PDF with one click to remove its records, vectors, and clear outdated cache entries.
