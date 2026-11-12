import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="Processing")  # 'Processing', 'Ready', 'Failed'
    error_message = Column(Text, nullable=True)
    chunks_count = Column(Integer, default=0)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, index=True) # "{doc_id}#chunk-{n}"
    doc_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    file_name = Column(String, nullable=False)
    page = Column(Integer, nullable=False)
    chunk_type = Column(String, nullable=False)  # "text", "table", "image_ocr"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
