import hashlib
import secrets
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI
from app.config import config
from app.cache import embedding_cache, generate_cache_key

# Initialize OpenAI client if API key is provided
client: Optional[AsyncOpenAI] = None
if config.openai_api_key:
    client = AsyncOpenAI(api_key=config.openai_api_key)

def _local_embedding(input_text: str, dims: int = 256) -> List[float]:
    """Generate a deterministic local embedding when OpenAI is unavailable"""
    embedding = [0.0] * dims
    
    # Simple tokenization
    import re
    tokens = re.sub(r'[^a-z0-9\s]', ' ', input_text.lower()).split()
    tokens = [t for t in tokens if t]
    
    for token in tokens:
        # Use SHA-256 to get deterministic index
        digest = hashlib.sha256(token.encode()).digest()
        # Use first 4 bytes as unsigned int
        idx = int.from_bytes(digest[:4], 'little') % dims
        embedding[idx] += 1.0
    
    # Normalize
    norm = sum(v * v for v in embedding) ** 0.5
    if norm > 0:
        embedding = [v / norm for v in embedding]
    
    return embedding

async def embed_text(input_text: str) -> List[float]:
    """Generate embedding for text, using cache and fallback to local embedding"""
    cache_key = generate_cache_key(input_text)
    
    # Try cache first
    cached = embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    
    embedding: List[float] = []
    
    if client is None:
        embedding = _local_embedding(input_text)
    else:
        try:
            response = await client.embeddings.create(
                model=config.embedding_model,
                input=input_text
            )
            embedding = response.data[0].embedding if response.data else []
        except Exception:
            embedding = _local_embedding(input_text)
    
    # Cache for 24 hours
    embedding_cache.set(cache_key, embedding, 1000 * 60 * 60 * 24)
    return embedding

async def answer_question(question: str, context_blocks: str) -> str:
    """Generate an answer based on provided context"""
    if client is None:
        context = context_blocks.strip()
        if not context:
            return "The documents do not provide enough information."
        
        lines = [line for line in context.split("\n") if line.strip()]
        excerpt = "\n".join(lines[:12])
        return f"OpenAI is not configured, so I can't generate a model-based answer.\n\nRelevant context:\n{excerpt}"
    
    try:
        response = await client.chat.completions.create(
            model=config.chat_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "\n".join([
                        "You are a document Q&A assistant.",
                        "The provided contexts are ordered by relevance (most relevant first).",
                        "CRITICAL: Only use information from contexts that are directly relevant to the question.",
                        "If a context discusses unrelated topics, completely ignore it and do not reference it.",
                        "If none of the contexts contain relevant information, say that the documents do not provide enough information.",
                        "Do not invent details or mix information from different contexts.",
                        "If helpful, you may then provide a brief general-knowledge answer, clearly labeled as such."
                    ])
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context_blocks}\n\nReturn a concise answer."
                }
            ]
        )
        
        return response.choices[0].message.content.strip() if response.choices else "No answer generated."
    except Exception as e:
        return f"Error generating answer: {str(e)}"