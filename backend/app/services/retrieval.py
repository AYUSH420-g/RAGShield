import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from backend.app.config import settings
from backend.app.services.embeddings import embed_query
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.pinecone_service import get_pinecone_service
from backend.app.services.rerank import rerank_chunks
from backend.app.services.llm import (
    rewrite_question,
    generate_multi_queries,
    generate_answer_with_timeout
)
from backend.app.services.cache import get_semantic_cache

logger = logging.getLogger("pdf_qa.retrieval")

GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "who are you", "who are you?", "what can you do", "what can you do?",
    "help", "how does this work", "how does this work?"
}

def count_ready_documents() -> int:
    from backend.app.database import SessionLocal
    from backend.app.models import Document
    db = SessionLocal()
    try:
        return db.query(Document).filter(Document.status == "Ready").count()
    finally:
        db.close()

async def search_single_query(query: str) -> List[Dict[str, Any]]:
    """
    Compute dense + sparse vectors and execute Pinecone hybrid search.
    """
    bm25 = get_bm25_service()
    pinecone = get_pinecone_service()

    dense_vec = await asyncio.to_thread(embed_query, query)
    sparse_vec = await asyncio.to_thread(bm25.encode_query, query)
    
    matches = await asyncio.to_thread(
        pinecone.hybrid_query,
        dense_vec,
        sparse_vec,
        top_k=settings.RETRIEVAL_TOP_K,
        alpha=settings.HYBRID_ALPHA
    )
    return matches

async def process_question_pipeline(question: str) -> Dict[str, Any]:
    """
    Full Question Pipeline as specified:
    1. Cache check
    2. Rewrite question
    3. Multi-query generation (4 total)
    4. Parallel hybrid search
    5. Merge & deduplicate
    6. Rerank top 3
    7. Answer generation with hard 10s timeout
    8. Sources
    9. Cache in Redis
    """
    pipeline_start = time.time()
    clean_q = question.strip()
    logger.info(f"=== Starting question pipeline for: '{clean_q}' ===")

    # Friendly greeting / pleasantry handling
    norm_q = clean_q.lower().strip("!?. ")
    if norm_q in GREETINGS or (len(clean_q.split()) <= 2 and norm_q.startswith(("hi", "hello", "hey"))):
        return {
            "answer": "Hello! I am your PDF Q&A Assistant. Please upload documents in the Admin panel and ask me any questions about them.",
            "sources": [],
            "cached": False,
            "latency": round(time.time() - pipeline_start, 3),
            "llm_used": None,
            "status": "success"
        }

    # Guidance if no documents are uploaded yet
    ready_docs_count = await asyncio.to_thread(count_ready_documents)
    if ready_docs_count == 0:
        return {
            "answer": "No documents have been uploaded yet. Please switch to the Admin panel to upload a PDF document before asking questions.",
            "sources": [],
            "cached": False,
            "latency": round(time.time() - pipeline_start, 3),
            "llm_used": None,
            "status": "success"
        }

    # Step 1: Cache check
    cache = get_semantic_cache()
    query_embedding = await asyncio.to_thread(embed_query, clean_q)
    cached_result = await asyncio.to_thread(cache.get, query_embedding)

    if cached_result and cached_result.get("hit"):
        elapsed = time.time() - pipeline_start
        logger.info(
            f"[Cache Hit] Cosine similarity: {cached_result['similarity']:.4f} >= {settings.CACHE_SIMILARITY_THRESHOLD}. "
            f"Returning cached answer immediately in {elapsed:.3f}s."
        )
        return {
            "answer": cached_result["answer"],
            "sources": cached_result["sources"],
            "cached": True,
            "latency": round(elapsed, 3),
            "llm_used": "Semantic Cache (Redis)",
            "status": "success"
        }

    logger.info(f"[Cache Miss] No matching question found in cache (threshold={settings.CACHE_SIMILARITY_THRESHOLD}).")

    # Step 2: Rewrite question (timeout 5s guarded)
    rewritten_q = await rewrite_question(clean_q)

    # Step 3: Multi-query generation (timeout 5s guarded)
    queries = await generate_multi_queries(rewritten_q)

    # Step 4: Hybrid search for each of the 4 queries in parallel
    logger.info(f"Running hybrid search in parallel for {len(queries)} queries...")
    search_tasks = [search_single_query(q) for q in queries]
    search_results_nested = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Step 5: Merge and deduplicate by chunk ID
    all_chunks_dict: Dict[str, Dict[str, Any]] = {}
    for res in search_results_nested:
        if isinstance(res, Exception):
            logger.warning(f"One hybrid search query failed: {res}")
            continue
        for chunk in res:
            cid = chunk["id"]
            if cid not in all_chunks_dict or chunk["score"] > all_chunks_dict[cid]["score"]:
                all_chunks_dict[cid] = chunk

    unique_chunks = list(all_chunks_dict.values())
    logger.info(f"Merged search results: {len(unique_chunks)} unique chunks found.")

    if not unique_chunks:
        elapsed = time.time() - pipeline_start
        logger.info("No relevant chunks found in vector database.")
        return {
            "answer": "I couldn't find this in the uploaded documents.",
            "sources": [],
            "cached": False,
            "latency": round(elapsed, 3),
            "llm_used": None,
            "status": "success"
        }

    # Step 6: Rerank unique chunks against rewritten query
    top_chunks = await asyncio.to_thread(
        rerank_chunks,
        rewritten_q,
        unique_chunks,
        top_k=settings.RERANK_TOP_K
    )

    # Step 7 & 8: Answer generation with hard 10s timeout
    try:
        answer, llm_used, sources = await generate_answer_with_timeout(
            rewritten_q,
            top_chunks,
            timeout_seconds=settings.ANSWER_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - pipeline_start
        logger.error(f"Answer generation exceeded {settings.ANSWER_TIMEOUT_SECONDS}s hard limit! Request timed out.")
        return {
            "answer": "Taking too long. Please try again.",
            "sources": [],
            "cached": False,
            "latency": round(elapsed, 3),
            "llm_used": None,
            "status": "timeout"
        }
    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.error(f"Answer generation error: {e}")
        return {
            "answer": "An error occurred while generating the answer. Please try again.",
            "sources": [],
            "cached": False,
            "latency": round(elapsed, 3),
            "llm_used": None,
            "status": "error"
        }

    elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline finished in {elapsed:.3f}s. LLM used: {llm_used}")

    # Step 9: Save in Redis semantic cache (never cache errors or timeouts)
    if answer and "Taking too long" not in answer:
        try:
            await asyncio.to_thread(
                cache.set,
                clean_q,
                query_embedding,
                answer,
                sources
            )
        except Exception as cache_err:
            logger.warning(f"Failed saving to semantic cache: {cache_err}")

    return {
        "answer": answer,
        "sources": sources,
        "cached": False,
        "latency": round(elapsed, 3),
        "llm_used": llm_used,
        "status": "success"
    }
