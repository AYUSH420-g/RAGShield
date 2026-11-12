import os
import pymupdf
import pytest
from backend.app.services.ingestion import parse_pdf, ingest_pdf_pipeline, delete_pdf_pipeline
from backend.app.database import SessionLocal
from backend.app.models import Document, Chunk
from backend.app.services.retrieval import process_question_pipeline
from backend.app.services.cache import get_semantic_cache
from backend.app.services.pinecone_service import get_pinecone_service

def create_sample_pdf(file_path: str):
    """Generate a 2-page sample PDF with text and table."""
    doc = pymupdf.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "Company Annual Financial Report\n\n"
        "Executive Summary:\n"
        "In fiscal year 2026, the company achieved total global revenue of $142 million, representing an annual growth of 18%.\n"
        "Our cloud infrastructure and artificial intelligence initiatives were the primary growth drivers.\n"
    )
    
    # Page 2: with table-like structure
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "Company Annual Financial Report\n\n"
        "Quarterly Financial Breakdown:\n"
        "Quarter   Revenue     Net Income\n"
        "Q1        $32M        $5.1M\n"
        "Q2        $35M        $5.8M\n"
        "Q3        $37M        $6.2M\n"
        "Q4        $38M        $6.9M\n"
    )
    
    doc.save(file_path)
    doc.close()

def test_full_e2e_ingestion_and_retrieval(tmp_path):
    pdf_path = str(tmp_path / "financial_report_2026.pdf")
    create_sample_pdf(pdf_path)
    assert os.path.exists(pdf_path)

    # 1. Test parsing
    parsed_items = parse_pdf(pdf_path, "financial_report_2026.pdf")
    assert len(parsed_items) >= 2
    full_text = " ".join(item["text"] for item in parsed_items)
    assert "142 million" in full_text
    assert "Executive Summary" in full_text

    # 2. Insert into DB and run background ingestion pipeline
    doc_id = "test-e2e-doc-001"
    db = SessionLocal()
    # Clean any leftover
    db.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
    db.query(Document).filter(Document.id == doc_id).delete()
    db.commit()

    doc_record = Document(
        id=doc_id,
        name="financial_report_2026.pdf",
        file_path=pdf_path,
        file_size=os.path.getsize(pdf_path),
        status="Processing"
    )
    db.add(doc_record)
    db.commit()
    db.close()

    # Ensure Pinecone runs in mock mode for unit/CI test
    pinecone = get_pinecone_service()
    pinecone._init_local_mock()

    # Run ingestion pipeline
    ingest_pdf_pipeline(doc_id)

    db = SessionLocal()
    updated_doc = db.query(Document).filter(Document.id == doc_id).first()
    assert updated_doc is not None
    assert updated_doc.status == "Ready"
    assert updated_doc.chunks_count > 0

    chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).all()
    assert len(chunks) == updated_doc.chunks_count
    db.close()

    # 3. Test Question Pipeline with mock LLM / offline
    import asyncio
    res1 = asyncio.run(process_question_pipeline("What was the company's global revenue in fiscal year 2026?"))
    assert res1["status"] in ("success", "timeout", "error")
    assert "latency" in res1
    assert "cached" in res1
    assert res1["cached"] is False

    # 4. Test Semantic Cache HIT on repeating or identical query
    res2 = asyncio.run(process_question_pipeline("What was the company's global revenue in fiscal year 2026?"))
    # Second identical query should be a cache hit if response was cached
    if res1["status"] == "success" and res1["answer"]:
        assert res2["cached"] is True
        assert res2["llm_used"] == "Semantic Cache (Redis)"
        assert res2["answer"] == res1["answer"]

    # 5. Test Deletion Pipeline
    delete_pdf_pipeline(doc_id)
    db = SessionLocal()
    deleted_doc = db.query(Document).filter(Document.id == doc_id).first()
    assert deleted_doc is None
    remaining_chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).all()
    assert len(remaining_chunks) == 0
    db.close()

    # After deletion, cache should be cleared
    cache = get_semantic_cache()
    from backend.app.services.embeddings import embed_query
    q_vec = embed_query("What was the company's global revenue in fiscal year 2026?")
    assert cache.get(q_vec) is None
