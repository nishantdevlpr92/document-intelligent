import pytest
from app.similarity import cosine_similarity
from app.retrieval import top_k_chunks
from app.models import ChunkRecord

def test_cosine_similarity_prefers_aligned_vectors():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    
    assert cosine_similarity(a, b) > cosine_similarity(a, c)

def test_cosine_similarity_handles_empty_vectors():
    assert cosine_similarity([], [1.0]) == -1.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == -1.0

def test_top_k_chunks_ranks_most_similar_first():
    chunks = [
        ChunkRecord(
            id=1, document_id=10, chunk_index=0, 
            content="roadmap", embedding=[1.0, 0.0, 0.0]
        ),
        ChunkRecord(
            id=2, document_id=10, chunk_index=1, 
            content="budget", embedding=[0.0, 1.0, 0.0]
        )
    ]
    
    results = top_k_chunks(
        [1.0, 0.0, 0.0], 
        chunks, 
        {10: "plan.txt"}, 
        2
    )
    
    assert results[0].chunk_id == 1
    assert len(results) == 2