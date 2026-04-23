import pytest
from app.chunking import chunk_text

def test_chunk_text_returns_empty_for_blank_input():
    result = chunk_text("   \n  ")
    assert len(result) == 0

def test_chunk_text_creates_overlapping_chunks():
    source = "a" * 1000
    result = chunk_text(source, 300, 50)
    assert len(result) > 1
    assert len(result[0]) <= 300

def test_chunk_text_handles_small_text():
    result = chunk_text("hello world", 100, 20)
    assert len(result) == 1
    assert result[0] == "hello world"