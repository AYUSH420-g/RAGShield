import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas import ChatRequest, ChatResponse
from backend.app.services.retrieval import process_question_pipeline

logger = logging.getLogger("pdf_qa.chat_api")
router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    User asks a question:
    1. Semantic Cache check in Redis
    2. Rewrite question
    3. Multi-query generation
    4. Parallel hybrid search in Pinecone
    5. Deduplication & Merge
    6. Cross-encoder Rerank
    7. Answer generation with hard 10s timeout & fallback
    8. Sources
    9. Cache in Redis
    """
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )

    try:
        result = await process_question_pipeline(user_msg)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            cached=result["cached"],
            latency=result["latency"],
            llm_used=result["llm_used"],
            status=result.get("status", "success")
        )
    except Exception as e:
        logger.exception(f"Unexpected error handling chat request: {e}")
        return ChatResponse(
            answer="An unexpected error occurred. Please try again.",
            sources=[],
            cached=False,
            latency=0.0,
            llm_used=None,
            status="error"
        )
