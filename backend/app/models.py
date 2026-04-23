from dataclasses import dataclass
from typing import List

@dataclass
class DocumentRecord:
    id: int
    name: str
    created_at: str

@dataclass
class ChunkRecord:
    id: int
    document_id: int
    chunk_index: int
    content: str
    embedding: List[float]

@dataclass
class Citation:
    chunk_id: int
    document_id: int
    document_name: str
    chunk_index: int
    excerpt: str
    score: float