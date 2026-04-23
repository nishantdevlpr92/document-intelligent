import hashlib
import time
from typing import Any, Optional, Dict

class MemoryCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, default_ttl: int = 1000 * 60 * 60):  # 1 hour default
        self._cache: Dict[str, dict] = {}
        self.default_ttl = default_ttl
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL in milliseconds"""
        self._cache[key] = {
            "value": value,
            "timestamp": time.time() * 1000,
            "ttl": ttl if ttl is not None else self.default_ttl
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, returns None if expired or not found"""
        entry = self._cache.get(key)
        if not entry:
            return None
        
        current_time = time.time() * 1000
        if current_time - entry["timestamp"] > entry["ttl"]:
            del self._cache[key]
            return None
        
        return entry["value"]
    
    def clear(self) -> None:
        """Clear all cached values"""
        self._cache.clear()
    
    def cleanup(self) -> None:
        """Remove expired entries"""
        current_time = time.time() * 1000
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry["timestamp"] > entry["ttl"]
        ]
        for key in expired_keys:
            del self._cache[key]

# Cache instances
embedding_cache = MemoryCache()
api_cache = MemoryCache()

def generate_cache_key(text: str) -> str:
    """Generate a SHA-256 hash for cache key"""
    return hashlib.sha256(text.encode()).hexdigest()

def generate_question_cache_key(question: str, top_k: int) -> str:
    """Generate cache key for question-answer pairs"""
    return f"question:{generate_cache_key(question)}:{top_k}"