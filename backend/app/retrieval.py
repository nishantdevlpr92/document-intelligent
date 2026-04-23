from typing import List, Dict
from app.models import ChunkRecord, Citation
from app.similarity import cosine_similarity

def top_k_chunks(
    query_embedding: List[float],
    all_chunks: List[ChunkRecord],
    document_names: Dict[int, str],
    top_k: int
) -> List[Citation]:
    """Find top-k most similar chunks to the query embedding"""
    scored = []
    
    for chunk in all_chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        scored.append((chunk, score))
    
    # Sort by score descending and take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_k]
    
    citations = []
    for chunk, score in top_scored:
        citations.append(Citation(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_name=document_names.get(chunk.document_id, f"Document {chunk.document_id}"),
            chunk_index=chunk.chunk_index,
            excerpt=chunk.content,
            score=score
        ))
    
    return citations