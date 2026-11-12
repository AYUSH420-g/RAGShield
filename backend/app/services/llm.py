import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.llm")

def get_gemini_llm() -> Optional[ChatGoogleGenerativeAI]:
    if not settings.GOOGLE_API_KEY or "your_" in settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "mock-key-or-real-gemini-key":
        return None
    try:
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0,
            max_retries=1
        )
    except Exception as e:
        logger.warning(f"Failed to create ChatGoogleGenerativeAI: {e}")
        return None

def get_groq_llm() -> Optional[ChatGroq]:
    if not settings.GROQ_API_KEY or "your_" in settings.GROQ_API_KEY or settings.GROQ_API_KEY == "mock-key-or-real-groq-key":
        return None
    try:
        return ChatGroq(
            model_name=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.0
        )
    except Exception as e:
        logger.warning(f"Failed to create ChatGroq: {e}")
        return None

async def call_llm_with_fallback(messages: List[Any], step_name: str = "LLM Call") -> Tuple[str, str]:
    """
    Call Gemini as primary; if it fails (rate limit, error, quota, timeout),
    fallback to Groq. Returns (response_text, provider_name).
    """
    gemini = get_gemini_llm()
    groq = get_groq_llm()

    # 1. Try Gemini
    if gemini:
        try:
            start_t = time.time()
            # Fast failover if Gemini is unresponsive or errors
            response = await asyncio.wait_for(gemini.ainvoke(messages), timeout=3.5)
            elapsed = time.time() - start_t
            content = response.content if hasattr(response, "content") else str(response)
            logger.info(f"[{step_name}] Answered by Gemini ({settings.GEMINI_MODEL}) in {elapsed:.3f}s")
            return content.strip(), f"Gemini ({settings.GEMINI_MODEL})"
        except Exception as e:
            logger.warning(f"[{step_name}] Gemini failed or timed out ({e}). Attempting Groq fallback...")

    # 2. Try Groq fallback
    if groq:
        try:
            start_t = time.time()
            response = await groq.ainvoke(messages)
            elapsed = time.time() - start_t
            content = response.content if hasattr(response, "content") else str(response)
            logger.info(f"[{step_name}] Answered by Groq fallback ({settings.GROQ_MODEL}) in {elapsed:.3f}s")
            return content.strip(), f"Groq ({settings.GROQ_MODEL})"
        except Exception as e:
            logger.error(f"[{step_name}] Groq fallback also failed with error: {e}")
            raise e

    # Mock response if neither key is configured
    logger.warning(f"[{step_name}] Neither Gemini nor Groq is configured with a valid key. Returning mock response.")
    return "Mock response: Please configure GOOGLE_API_KEY or GROQ_API_KEY in .env.", "Mock-LLM"

async def rewrite_question(question: str) -> str:
    """
    Step 2: LLM fixes grammar and makes the question clear and complete without changing its meaning.
    Timeout guarded by caller (5s).
    """
    sys_prompt = (
        "You are an expert editor. Rewrite the user's question to fix grammar, remove ambiguities, "
        "and make it clear and complete. Do NOT change its original meaning. Return ONLY the rewritten question."
    )
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=question)
    ]
    try:
        rewritten, provider = await asyncio.wait_for(
            call_llm_with_fallback(messages, step_name="Question Rewrite"),
            timeout=settings.QUERY_REWRITE_TIMEOUT_SECONDS
        )
        # Clean quotes
        rewritten = rewritten.strip('"\' ')
        logger.info(f"Rewritten question: '{question}' -> '{rewritten}' (via {provider})")
        return rewritten if rewritten else question
    except Exception as e:
        logger.warning(f"Question rewrite failed or timed out ({e}). Continuing with original question.")
        return question

async def generate_multi_queries(rewritten_query: str) -> List[str]:
    """
    Step 3: LLM generates 3 alternative versions of the rewritten question, giving 4 queries in total.
    Timeout guarded (5s).
    """
    sys_prompt = (
        "Generate 3 diverse alternative formulations of the following question for search retrieval. "
        "Each should capture different keywords or aspects of the user's intent. "
        "Output exactly 3 numbered lines, for example:\n1. first alternative\n2. second alternative\n3. third alternative"
    )
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=rewritten_query)
    ]
    queries = [rewritten_query]
    try:
        output, provider = await asyncio.wait_for(
            call_llm_with_fallback(messages, step_name="Multi-Query Generation"),
            timeout=settings.QUERY_REWRITE_TIMEOUT_SECONDS
        )
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        for line in lines:
            # strip leading numbers like "1. ", "- ", etc.
            cleaned = line.lstrip("0123456789.-) ")
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
            if len(queries) == 4:
                break
        logger.info(f"Generated multi-queries ({len(queries)} total via {provider}): {queries}")
    except Exception as e:
        logger.warning(f"Multi-query generation failed or timed out ({e}). Continuing with 1 query.")

    return queries[:4]

async def generate_answer_with_timeout(
    question: str,
    context_chunks: List[Dict[str, Any]],
    timeout_seconds: float = settings.ANSWER_TIMEOUT_SECONDS
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Step 7: Send top chunks + question to LLM.
    Strict prompt: answer ONLY from context, simple language.
    If not in context, reply: 'I couldn't find this in the uploaded documents.'
    Hard timeout enforced via asyncio.wait_for.
    Returns: (answer_text, llm_used, sources)
    """
    sources: List[Dict[str, Any]] = []
    seen_sources = set()
    context_parts = []

    for idx, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page", 1)
        text = meta.get("text", "")
        
        src_key = (file_name, page)
        if src_key not in seen_sources:
            seen_sources.add(src_key)
            sources.append({"file_name": file_name, "page": page})

        context_parts.append(f"[Document: {file_name}, Page: {page}]\n{text}")

    context_str = "\n\n---\n\n".join(context_parts)

    sys_prompt = (
        "You are a helpful assistant for answering questions based ONLY on the provided document excerpts.\n"
        "Instructions:\n"
        "1. Answer strictly using ONLY the information present in the Context excerpts below.\n"
        "2. Use simple, clear, and direct language.\n"
        "3. Do not assume or extrapolate beyond the text.\n"
        "4. If the answer cannot be found in the provided excerpts, reply EXACTLY with:\n"
        "\"I couldn't find this in the uploaded documents.\""
    )

    user_content = (
        f"Context excerpts:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_content)
    ]

    # Hard timeout wrapping the async LLM call
    answer, llm_used = await asyncio.wait_for(
        call_llm_with_fallback(messages, step_name="Answer Generation"),
        timeout=timeout_seconds
    )

    return answer, llm_used, sources
