import math
from fastapi import APIRouter, HTTPException, Query, Path
from app import schemas
from app import database as db
from app.chunking import chunk_text
from app.openai_service import embed_text
from app.cache import api_cache

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.get("", response_model=schemas.DocumentsListResponse)
async def list_documents_endpoint(
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1, le=100)
):
    """List all documents with pagination"""
    cache_key = f"documents:{page}:{limit}"
    cached = api_cache.get(cache_key)
    if cached:
        return cached
    
    all_documents = db.list_documents()
    total_count = len(all_documents)
    total_pages = math.ceil(total_count / limit) if total_count > 0 else 1
    
    start_index = (page - 1) * limit
    end_index = start_index + limit
    documents = all_documents[start_index:end_index]
    
    result = {
        "documents": [
            schemas.DocumentResponse(
                id=doc.id,
                name=doc.name,
                created_at=doc.created_at
            )
            for doc in documents
        ],
        "pagination": schemas.PaginationInfo(
            page=page,
            limit=limit,
            total_count=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    }
    
    # Cache for 2 minutes
    api_cache.set(cache_key, result, 1000 * 60 * 2)
    return result

@router.get("/{id}/text", response_model=schemas.DocumentTextResponse)
async def get_document_text(id: int = Path(..., gt=0)):
    """Get the full text of a document"""
    text = db.get_document_text(id)
    if text is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"text": text}

@router.delete("/{id}", response_model=dict)
async def delete_document_endpoint(id: int = Path(..., gt=0)):
    """Delete a document and all its chunks"""
    deleted = db.delete_document(id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Invalidate all caches
    api_cache.clear()
    return {"ok": True}

@router.post("", response_model=schemas.CreateDocumentResponse)
async def create_document(request: schemas.CreateDocumentRequest):
    """Create a new document with text chunks and embeddings"""
    text_chunks = chunk_text(request.text)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="Document text produced no chunks.")
    
    try:
        document_id = db.insert_document(request.name)
        
        embedded_chunks = []
        for idx, content in enumerate(text_chunks):
            embedding = await embed_text(content)
            embedded_chunks.append({
                "chunk_index": idx,
                "content": content,
                "embedding": embedding
            })
        
        db.insert_chunks(document_id, embedded_chunks)
        
        # Invalidate caches
        api_cache.clear()
        
        return schemas.CreateDocumentResponse(
            document_id=document_id,
            chunk_count=len(embedded_chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")