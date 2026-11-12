import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from backend.app.services.embeddings import get_embeddings_model
from backend.app.config import settings

logger = logging.getLogger("pdf_qa.chunking")

_paragraph_splitter: RecursiveCharacterTextSplitter = None
_semantic_chunker: SemanticChunker = None

def get_paragraph_splitter() -> RecursiveCharacterTextSplitter:
    global _paragraph_splitter
    if _paragraph_splitter is None:
        _paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.MAX_CHUNK_SIZE,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    return _paragraph_splitter

def get_semantic_chunker() -> SemanticChunker:
    global _semantic_chunker
    if _semantic_chunker is None:
        embeddings = get_embeddings_model()
        _semantic_chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile"
        )
    return _semantic_chunker

def chunk_text(text: str) -> List[str]:
    """
    Split text using SemanticChunker with paragraph-based chunking as fallback.
    Ensures no chunk exceeds settings.MAX_CHUNK_SIZE.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # If text is already shorter than max chunk size, return as single chunk
    if len(cleaned) <= settings.MAX_CHUNK_SIZE:
        return [cleaned]

    raw_chunks: List[str] = []
    used_method = "semantic"

    try:
        chunker = get_semantic_chunker()
        docs = chunker.create_documents([cleaned])
        raw_chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]
        if not raw_chunks:
            raise ValueError("Semantic chunker returned no chunks")
    except Exception as e:
        logger.warning(f"SemanticChunker failed ({str(e)}), falling back to paragraph-based chunking.")
        used_method = "paragraph_fallback"
        splitter = get_paragraph_splitter()
        raw_chunks = splitter.split_text(cleaned)

    # Post-process: enforce max_chunk_size on any chunk that exceeds it
    final_chunks: List[str] = []
    splitter = get_paragraph_splitter()
    for chunk in raw_chunks:
        if len(chunk) > settings.MAX_CHUNK_SIZE:
            sub_chunks = splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    logger.debug(f"Chunked text of length {len(text)} into {len(final_chunks)} chunks using {used_method}")
    return [c.strip() for c in final_chunks if c.strip()]
