import os
import uuid
import hmac
import hashlib
import time
import base64
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import Document
from backend.app.schemas import (
    LoginRequest,
    Token,
    DocumentResponse,
    DocumentListResponse,
    StatusResponse
)
from backend.app.services.ingestion import ingest_pdf_pipeline, delete_pdf_pipeline

logger = logging.getLogger("pdf_qa.admin")
router = APIRouter(prefix="/api/admin", tags=["admin"])
security = HTTPBearer()

def create_admin_token(username: str) -> str:
    """Generate a signed HMAC token with expiration."""
    exp = int(time.time()) + (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    payload = f"{username}:{exp}"
    sig = hmac.new(
        settings.ADMIN_SECRET_KEY.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    raw = f"{payload}:{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError()
        username, exp_str, sig = parts
        exp = int(exp_str)
        if time.time() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please log in again."
            )
        expected_sig = hmac.new(
            settings.ADMIN_SECRET_KEY.encode(),
            f"{username}:{exp}".encode(),
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token."
            )
        return username
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials."
        )

@router.post("/login", response_model=Token)
async def login(req: LoginRequest):
    """Admin login endpoint."""
    if (
        req.username == settings.ADMIN_USERNAME and
        req.password == settings.ADMIN_PASSWORD
    ):
        token = create_admin_token(req.username)
        logger.info(f"Admin '{req.username}' logged in successfully.")
        return Token(access_token=token, token_type="bearer")

    logger.warning(f"Failed login attempt for username: {req.username}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password"
    )

@router.get("/pdfs", response_model=DocumentListResponse)
async def get_pdfs(
    db: Session = Depends(get_db),
    admin: str = Depends(verify_admin_token)
):
    """List all uploaded PDFs and their processing status."""
    docs = db.query(Document).order_by(Document.upload_date.desc()).all()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs)
    )

@router.post("/upload", response_model=DocumentResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: str = Depends(verify_admin_token)
):
    """Upload PDF and run background ingestion pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}_{os.path.basename(file.filename)}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    # Save uploaded file to disk
    content = await file.read()
    file_size = len(content)

    with open(file_path, "wb") as f:
        f.write(content)

    # Create document record with status 'Processing'
    doc = Document(
        id=doc_id,
        name=file.filename,
        file_path=file_path,
        file_size=file_size,
        status="Processing",
        chunks_count=0
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    logger.info(f"Uploaded file '{file.filename}' (ID: {doc_id}, {file_size} bytes). Scheduling background ingestion...")
    # Trigger background ingestion task
    background_tasks.add_task(ingest_pdf_pipeline, doc_id)

    return DocumentResponse.model_validate(doc)

@router.delete("/pdfs/{doc_id}", response_model=StatusResponse)
async def delete_pdf(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: str = Depends(verify_admin_token)
):
    """Delete a PDF and purge its vectors and cache."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    # Execute deletion
    try:
        delete_pdf_pipeline(doc_id)
        return StatusResponse(status="success", message=f"Document {doc_id} deleted successfully.")
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
