from fastapi import APIRouter, HTTPException
from app import schemas
from app import database as db
from app.config import config
from app.openai_service import embed_text, answer_question
from app.retrieval import top_k_chunks
from app.cache import api_cache, generate_question_cache_key

router = APIRouter(prefix="/api", tags=["questions"])

@router.post("/ask", response_model=schemas.AskResponse)
async def ask_question(request: schemas.AskRequest):
    """Ask a question and get an answer with citations"""
    chunks = db.get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="No document chunks found. Ingest at least one document first.")
    
    # Check cache
    cache_key = generate_question_cache_key(request.question, request.top_k)
    cached = api_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        query_embedding = await embed_text(request.question)
        document_names = db.get_document_name_map()
        
        citations = top_k_chunks(query_embedding, chunks, document_names, request.top_k)
        citations = [c for c in citations if c.score >= config.min_citation_score]
        
        if not citations:
            result = {
                "answer": "I do not have enough information in the document corpus to answer this question.",
                "citations": []
            }
            # Cache negative results for 2 minutes
            api_cache.set(cache_key, result, 1000 * 60 * 2)
            return result
        
        # Build context for answer generation
        context_parts = []
        for i, citation in enumerate(citations):
            context_parts.append(
                f"[{i + 1}] document=\"{citation.document_name}\" chunk={citation.chunk_index}\n{citation.excerpt}"
            )
        context = "\n\n".join(context_parts)
        
        answer = await answer_question(request.question, context)
        
        result = {
            "answer": answer,
            "citations": [
                schemas.CitationResponse(
                    chunkId=c.chunk_id,
                    documentId=c.document_id,
                    documentName=c.document_name,
                    chunkIndex=c.chunk_index,
                    excerpt=c.excerpt,
                    score=c.score
                )
                for c in citations
            ]
        }
        
        # Cache successful results for 15 minutes
        api_cache.set(cache_key, result, 1000 * 60 * 15)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")