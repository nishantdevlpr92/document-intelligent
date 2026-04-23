# Document Q&A with Citations - Python Backend

A FastAPI backend service for document ingestion, embedding, and question answering with citations.

## Features

- Document ingestion with automatic chunking
- Text embedding using OpenAI API (with local fallback)
- Semantic search for relevant document chunks
- Question answering with citations
- In-memory caching for embeddings and API responses
- SQLite database for document and chunk storage
- Full compatibility with the Node.js version API

## Prerequisites

- Python 3.12+
- OpenAI API key (optional, falls back to local embeddings)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate