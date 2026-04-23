#!/usr/bin/env python3
"""Evaluation script for retrieval quality"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking import chunk_text
from app.retrieval import top_k_chunks
from app.models import ChunkRecord, Citation
from typing import List, Tuple, Dict

class Fixture:
    def __init__(self, document_name: str, text: str):
        self.document_name = document_name
        self.text = text

class QueryCase:
    def __init__(self, question: str, expected_keyword: str):
        self.question = question
        self.expected_keyword = expected_keyword

fixtures = [
    Fixture(
        document_name="roadmap.txt",
        text="Q3 priorities are reliability improvements, latency reduction, and user onboarding quality."
    ),
    Fixture(
        document_name="support.txt",
        text="Customer support hours are Monday through Friday from 9am to 6pm local time."
    )
]

queries = [
    QueryCase(question="What are the Q3 priorities?", expected_keyword="reliability"),
    QueryCase(question="When is customer support available?", expected_keyword="Monday")
]

def fake_embed(text: str) -> List[float]:
    """Deterministic tiny embedding for offline evaluation"""
    t = text.lower()
    return [
        1.0 if "q3" in t or "priorities" in t else 0.0,
        1.0 if "support" in t or "monday" in t else 0.0,
        1.0 if "latency" in t else 0.0
    ]

def build_chunks() -> Tuple[List[ChunkRecord], Dict[int, str]]:
    """Build chunks from fixtures"""
    chunks: List[ChunkRecord] = []
    names: Dict[int, str] = {}
    chunk_id = 1
    
    for i, doc in enumerate(fixtures):
        document_id = i + 1
        names[document_id] = doc.document_name
        
        for chunk_index, content in enumerate(chunk_text(doc.text, 500, 50)):
            chunks.append(ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                embedding=fake_embed(content)
            ))
            chunk_id += 1
    
    return chunks, names

def run_eval():
    """Run evaluation and print results"""
    chunks, names = build_chunks()
    hits = 0
    
    for q in queries:
        retrieved = top_k_chunks(fake_embed(q.question), chunks, names, 1)
        top_excerpt = retrieved[0].excerpt if retrieved else ""
        is_hit = q.expected_keyword.lower() in top_excerpt.lower()
        if is_hit:
            hits += 1
        
        print(f"Q: {q.question}")
        print(f"Top-1 excerpt: {top_excerpt}")
        print(f"Hit expected keyword \"{q.expected_keyword}\": {'yes' if is_hit else 'no'}")
        print("---")
    
    p_at_1 = hits / len(queries)
    print(f"precision@1: {p_at_1:.2f} ({hits}/{len(queries)})")

if __name__ == "__main__":
    run_eval()