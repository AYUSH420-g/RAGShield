import os
import io
import re
import logging
import time
from typing import List, Dict, Any, Tuple
from collections import Counter
import pymupdf  # fitz
import pdfplumber
import pytesseract
from PIL import Image

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import Document, Chunk
from backend.app.services.chunking import chunk_text
from backend.app.services.embeddings import embed_documents
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.pinecone_service import get_pinecone_service
from backend.app.services.cache import get_semantic_cache

logger = logging.getLogger("pdf_qa.ingestion")

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

def clean_text(raw_text: str) -> str:
    """
    Remove extra spaces and broken hyphenated words.
    """
    if not raw_text:
        return ""
    # Fix broken hyphenated words (e.g. "connec-\ntion" -> "connection")
    text = re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', raw_text)
    # Replace multiple horizontal spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def table_to_markdown(table_data: List[List[Any]]) -> str:
    """
    Convert 2D table grid to Markdown table format.
    """
    if not table_data or len(table_data) < 1:
        return ""
    cleaned_rows = []
    for row in table_data:
        cleaned_rows.append([str(c or "").replace("\n", " ").strip() for c in row])
    header = cleaned_rows[0]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |"
    ]
    for row in cleaned_rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)

def detect_repeated_headers_footers(page_texts: List[str]) -> Tuple[set, set]:
    """
    Identify lines that appear at the top or bottom of pages across >= 50% of the pages.
    """
    if len(page_texts) < 3:
        return set(), set()

    top_lines = []
    bottom_lines = []
    for p_text in page_texts:
        lines = [line.strip() for line in p_text.splitlines() if line.strip()]
        if lines:
            top_lines.append(lines[0])
        if len(lines) > 1:
            bottom_lines.append(lines[-1])

    threshold = len(page_texts) * 0.5
    top_counter = Counter(top_lines)
    bottom_counter = Counter(bottom_lines)

    common_headers = {line for line, cnt in top_counter.items() if cnt >= threshold}
    common_footers = {line for line, cnt in bottom_counter.items() if cnt >= threshold}
    return common_headers, common_footers

def is_obj_in_bbox(obj: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> bool:
    """Check if a pdfplumber object is inside a bounding box (x0, top, x1, bottom)."""
    ox0, otop, ox1, obottom = obj.get("x0", 0), obj.get("top", 0), obj.get("x1", 0), obj.get("bottom", 0)
    bx0, btop, bx1, bbottom = bbox
    # If overlapping significantly
    return not (ox1 < bx0 or ox0 > bx1 or obottom < btop or otop > bbottom)

def parse_pdf(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Step 1: Parse each page:
    - Text: PyMuPDF.
    - Tables: pdfplumber, convert to Markdown, keep as one chunk, remove table areas from text.
    - Images & scanned pages: OCR if little/no text, and OCR embedded images (> 100x100).
    """
    raw_pages: List[Dict[str, Any]] = []

    # Open with both pdfplumber and pymupdf
    plumber_doc = pdfplumber.open(file_path)
    pymupdf_doc = pymupdf.open(file_path)

    total_pages = len(pymupdf_doc)
    logger.info(f"Parsing PDF '{file_name}' ({total_pages} pages)...")

    # Pass 1: Extract tables and table bboxes per page via pdfplumber
    page_tables: Dict[int, List[Dict[str, Any]]] = {}
    for p_idx, plumber_page in enumerate(plumber_doc.pages):
        page_num = p_idx + 1
        page_tables[page_num] = []
        try:
            found_tables = plumber_page.find_tables()
            for t in found_tables:
                extracted = t.extract()
                md_table = table_to_markdown(extracted)
                if md_table.strip():
                    page_tables[page_num].append({
                        "bbox": t.bbox,
                        "markdown": md_table
                    })
        except Exception as e:
            logger.warning(f"Error extracting tables on page {page_num} of {file_name}: {e}")

    # Pass 2: Extract text outside tables and OCR if necessary
    page_texts: List[str] = []
    page_records = []

    for p_idx in range(total_pages):
        page_num = p_idx + 1
        pymupdf_page = pymupdf_doc[p_idx]
        plumber_page = plumber_doc.pages[p_idx] if p_idx < len(plumber_doc.pages) else None

        # Filter out text in table areas
        tables = page_tables.get(page_num, [])
        table_bboxes = [t["bbox"] for t in tables]

        normal_text = ""
        if plumber_page and table_bboxes:
            try:
                # Filter out objects falling inside table bounding boxes
                filtered = plumber_page.filter(lambda obj: not any(is_obj_in_bbox(obj, b) for b in table_bboxes))
                normal_text = filtered.extract_text() or ""
            except Exception as e:
                logger.warning(f"Table bbox filtering fallback on page {page_num}: {e}")
                normal_text = pymupdf_page.get_text() or ""
        else:
            normal_text = pymupdf_page.get_text() or ""

        # OCR check: if page has little or no text (< 50 chars), OCR whole page
        if len(normal_text.strip()) < 50:
            logger.info(f"Page {page_num} has sparse text ({len(normal_text.strip())} chars). OCRing whole page...")
            try:
                pix = pymupdf_page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_result = pytesseract.image_to_string(img)
                if ocr_result.strip():
                    normal_text = ocr_result
            except Exception as e:
                logger.warning(f"Whole-page OCR failed on page {page_num}: {e}")

        # OCR embedded images on this page
        embedded_ocr_texts = []
        try:
            image_list = pymupdf_page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                base_img = pymupdf_doc.extract_image(xref)
                w, h = base_img.get("width", 0), base_img.get("height", 0)
                # Skip small images like logos (< 100x100)
                if w >= 100 and h >= 100:
                    img_bytes = base_img.get("image")
                    if img_bytes:
                        try:
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            ocr_text = pytesseract.image_to_string(pil_img)
                            if len(ocr_text.strip()) > 15:
                                embedded_ocr_texts.append(ocr_text.strip())
                        except Exception as ocr_err:
                            logger.debug(f"Embedded image OCR skipped/failed: {ocr_err}")
        except Exception as img_err:
            logger.debug(f"Image extraction check failed on page {page_num}: {img_err}")

        page_texts.append(normal_text)
        page_records.append({
            "page_num": page_num,
            "text": normal_text,
            "tables": tables,
            "embedded_ocrs": embedded_ocr_texts
        })

    plumber_doc.close()
    pymupdf_doc.close()

    # Step 2: Clean text - remove repeated headers/footers
    common_headers, common_footers = detect_repeated_headers_footers(page_texts)

    cleaned_items: List[Dict[str, Any]] = []
    for rec in page_records:
        page_num = rec["page_num"]
        lines = rec["text"].splitlines()
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in common_headers or stripped in common_footers:
                continue
            filtered_lines.append(line)

        cleaned_page_text = clean_text("\n".join(filtered_lines))
        if cleaned_page_text:
            cleaned_items.append({
                "page": page_num,
                "type": "text",
                "text": cleaned_page_text
            })

        # Add tables (each table is kept intact as one chunk, never split)
        for t in rec["tables"]:
            cleaned_items.append({
                "page": page_num,
                "type": "table",
                "text": t["markdown"]
            })

        # Add embedded image OCR texts
        for ocr_t in rec["embedded_ocrs"]:
            cleaned_ocr = clean_text(ocr_t)
            if cleaned_ocr:
                cleaned_items.append({
                    "page": page_num,
                    "type": "image_ocr",
                    "text": cleaned_ocr
                })

    return cleaned_items

def ingest_pdf_pipeline(doc_id: str):
    """
    Full background ingestion pipeline for an uploaded PDF.
    """
    logger.info(f"Starting ingestion background task for doc_id: {doc_id}")
    db = SessionLocal()
    doc_record = db.query(Document).filter(Document.id == doc_id).first()
    if not doc_record:
        logger.error(f"Document {doc_id} not found in database.")
        db.close()
        return

    try:
        doc_record.status = "Processing"
        db.commit()

        # 1. Parse & Clean PDF
        file_path = doc_record.file_path
        file_name = doc_record.name
        parsed_items = parse_pdf(file_path, file_name)

        # 3. Chunk items
        # Table chunks are NEVER split. Text and OCR are semantically chunked.
        chunks_to_create: List[Dict[str, Any]] = []
        for item in parsed_items:
            item_type = item["type"]
            page = item["page"]
            content = item["text"]

            if item_type == "table":
                # Keep table as one single chunk (never split)
                chunks_to_create.append({
                    "page": page,
                    "type": "table",
                    "text": content
                })
            else:
                # Semantic chunking with paragraph fallback
                sub_chunks = chunk_text(content)
                for sc in sub_chunks:
                    chunks_to_create.append({
                        "page": page,
                        "type": item_type,
                        "text": sc
                    })

        logger.info(f"Document {file_name} generated {len(chunks_to_create)} chunks.")

        # Save chunks into SQLite database
        db_chunk_objects = []
        for idx, c in enumerate(chunks_to_create):
            chunk_id = f"{doc_id}#chunk-{idx}"
            db_chunk = Chunk(
                id=chunk_id,
                doc_id=doc_id,
                file_name=file_name,
                page=c["page"],
                chunk_type=c["type"],
                content=c["text"]
            )
            db_chunk_objects.append(db_chunk)

        db.add_all(db_chunk_objects)
        doc_record.chunks_count = len(db_chunk_objects)
        db.commit()

        # Refit BM25 encoder on ALL active chunks across all PDFs
        all_chunks = db.query(Chunk.content).all()
        all_texts = [row[0] for row in all_chunks]
        bm25_service = get_bm25_service()
        bm25_service.fit_and_save(all_texts)

        # 5. Compute vectors: Dense (BGE) + Sparse (BM25)
        texts_to_embed = [c["text"] for c in chunks_to_create]
        dense_vectors = embed_documents(texts_to_embed)
        sparse_vectors = bm25_service.encode_documents(texts_to_embed)

        # 6. Upsert to Pinecone
        pinecone_vectors = []
        for idx, c in enumerate(chunks_to_create):
            chunk_id = f"{doc_id}#chunk-{idx}"
            pinecone_vectors.append({
                "id": chunk_id,
                "values": dense_vectors[idx],
                "sparse_values": sparse_vectors[idx],
                "metadata": {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page": c["page"],
                    "type": c["type"],
                    "text": c["text"]
                }
            })

        pinecone_service = get_pinecone_service()
        pinecone_service.upsert_vectors(pinecone_vectors)

        # 7. Clear Redis semantic cache
        cache = get_semantic_cache()
        cache.clear()

        # Update status to Ready
        doc_record.status = "Ready"
        doc_record.error_message = None
        db.commit()
        logger.info(f"Ingestion succeeded for doc_id {doc_id} ({file_name}). Status is Ready.")

    except Exception as e:
        logger.exception(f"Ingestion failed for doc_id {doc_id}: {e}")
        doc_record.status = "Failed"
        doc_record.error_message = str(e)
        db.commit()
    finally:
        db.close()

def delete_pdf_pipeline(doc_id: str):
    """
    Handle deletion of a PDF:
    - Delete from Pinecone by prefix / chunk IDs
    - Delete from SQLite (cascades chunks)
    - Refit BM25 on remaining chunks
    - Clear Redis cache
    - Delete uploaded file from disk
    """
    logger.info(f"Starting deletion pipeline for doc_id: {doc_id}")
    db = SessionLocal()
    doc_record = db.query(Document).filter(Document.id == doc_id).first()
    if not doc_record:
        logger.warning(f"Doc {doc_id} not found in database for deletion.")
        db.close()
        return

    try:
        # Get all chunk IDs
        chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).all()
        chunk_ids = [c.id for c in chunks]

        # 1. Delete vectors from Pinecone
        pinecone_service = get_pinecone_service()
        pinecone_service.delete_by_doc_id(doc_id, chunk_ids=chunk_ids)

        # 2. Delete file from disk
        if os.path.exists(doc_record.file_path):
            try:
                os.remove(doc_record.file_path)
            except Exception as e:
                logger.warning(f"Could not remove file {doc_record.file_path}: {e}")

        # 3. Delete from DB
        db.delete(doc_record)
        db.commit()

        # 4. Refit BM25 on remaining chunks
        remaining_chunks = db.query(Chunk.content).all()
        remaining_texts = [r[0] for r in remaining_chunks]
        bm25_service = get_bm25_service()
        if remaining_texts:
            bm25_service.fit_and_save(remaining_texts)

        # 5. Clear Redis cache
        cache = get_semantic_cache()
        cache.clear()

        logger.info(f"Document {doc_id} successfully deleted.")
    except Exception as e:
        logger.exception(f"Error during deletion of doc {doc_id}: {e}")
        raise e
    finally:
        db.close()
