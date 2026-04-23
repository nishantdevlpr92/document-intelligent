from pydantic import BaseModel, Field
from typing import List, Optional, Any

class CreateDocumentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    text: str = Field(..., min_length=1)

class CreateDocumentResponse(BaseModel):
    document_id: int
    chunk_count: int

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=8, alias="topK")

    class Config:
        populate_by_name = True

class CitationResponse(BaseModel):
    chunkId: int
    documentId: int
    documentName: str
    chunkIndex: int
    excerpt: str
    score: float
    
    class Config:
        populate_by_name = True

class AskResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]

class DocumentResponse(BaseModel):
    id: int
    name: str
    created_at: str

class PaginationInfo(BaseModel):
    page: int
    limit: int
    total_count: int
    total_pages: int
    has_next: bool
    has_prev: bool

class DocumentsListResponse(BaseModel):
    documents: List[DocumentResponse]
    pagination: PaginationInfo

class DocumentTextResponse(BaseModel):
    text: str

class HealthResponse(BaseModel):
    ok: bool = True

class ErrorResponse(BaseModel):
    error: str
    details: Optional[Any] = None