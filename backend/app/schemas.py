from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    username: str
    password: str

class DocumentResponse(BaseModel):
    id: str
    name: str
    file_size: int
    upload_date: datetime
    status: str
    error_message: Optional[str] = None
    chunks_count: int

    model_config = ConfigDict(from_attributes=True)

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

class Source(BaseModel):
    file_name: str
    page: int

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    cached: bool = False
    latency: float = 0.0
    llm_used: Optional[str] = None
    status: str = "success"  # "success" or "timeout" or "error"

class StatusResponse(BaseModel):
    status: str
    message: str
