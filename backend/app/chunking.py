def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks"""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    
    chunks = []
    start = 0
    text_length = len(normalized)
    
    while start < text_length:
        end = min(start + max_chars, text_length)
        slice_text = normalized[start:end]
        chunks.append(slice_text.strip())
        
        if end == text_length:
            break
        
        start = max(0, end - overlap)
    
    return [chunk for chunk in chunks if chunk]