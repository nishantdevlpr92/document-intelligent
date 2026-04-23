import math

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return -1.0
    
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    
    for i in range(len(a)):
        dot += a[i] * b[i]
        mag_a += a[i] * a[i]
        mag_b += b[i] * b[i]
    
    if mag_a == 0 or mag_b == 0:
        return -1.0
    
    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))