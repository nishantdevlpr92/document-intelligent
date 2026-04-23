import os
import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import contextmanager
from app.config import config
from app.models import DocumentRecord, ChunkRecord

# Ensure data directory exists
db_dir = Path(config.db_path).parent
db_dir.mkdir(parents=True, exist_ok=True)

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initialize database tables and indexes"""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            
            -- Performance indexes
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_chunk ON chunks(document_id, chunk_index);
            CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
        """)

# Initialize database on module load
init_db()

def insert_document(name: str) -> int:
    """Insert a new document and return its ID"""
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO documents (name) VALUES (?)", (name,))
        return cursor.lastrowid

def insert_chunks(document_id: int, chunks: List[dict]) -> None:
    """Insert chunks for a document"""
    with get_db() as conn:
        for chunk in chunks:
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding_json) VALUES (?, ?, ?, ?)",
                (document_id, chunk["chunk_index"], chunk["content"], json.dumps(chunk["embedding"]))
            )

def list_documents() -> List[DocumentRecord]:
    """List all documents ordered by ID descending"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM documents ORDER BY id DESC"
        ).fetchall()
        return [DocumentRecord(id=row["id"], name=row["name"], created_at=row["created_at"]) for row in rows]

def delete_document(document_id: int) -> bool:
    """Delete a document and its chunks. Returns True if document existed."""
    with get_db() as conn:
        # Delete chunks first due to foreign key
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

def get_all_chunks() -> List[ChunkRecord]:
    """Get all chunks from all documents"""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, document_id, chunk_index, content, embedding_json 
            FROM chunks
        """).fetchall()
        
        return [
            ChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                embedding=json.loads(row["embedding_json"])
            )
            for row in rows
        ]

def get_document_name_map() -> Dict[int, str]:
    """Get a mapping of document IDs to names"""
    with get_db() as conn:
        rows = conn.execute("SELECT id, name FROM documents").fetchall()
        return {row["id"]: row["name"] for row in rows}

def get_document_text(document_id: int) -> Optional[str]:
    """Get the full text of a document by concatenating its chunks"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)
        ).fetchall()
        
        if not rows:
            return None
        
        return "\n\n".join(row["content"] for row in rows)