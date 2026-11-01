"""Unit tests for CodeBERT integration on Apple Silicon / macOS."""
import pytest
import numpy as np
from app.ml.codebert_embedder import CodeBERTEmbedder
from app.ml.feature_extractor import MultimodalFeatureExtractor


def test_codebert_availability():
    """Verify CodeBERT model is loaded and available on macOS."""
    assert CodeBERTEmbedder.is_available() is True


def test_codebert_768d_embedding_extraction():
    """Verify CodeBERT extracts 768-D dense embeddings from Python code."""
    code = (
        "def binary_search(arr, target):\n"
        "    low, high = 0, len(arr) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: low = mid + 1\n"
        "        else: high = mid - 1\n"
        "    return -1\n"
    )
    emb = CodeBERTEmbedder.extract_embedding(code)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert not np.isnan(emb).any()
    # Check normalized
    norm = np.linalg.norm(emb)
    assert 0.99 <= norm <= 1.01


def test_codebert_projected_embedding():
    """Verify CodeBERT projected embeddings match requested target dimension."""
    code = "def solve(n): return n * 2"
    emb_16 = CodeBERTEmbedder.extract_projected(code, target_dim=16)
    assert emb_16.shape == (16,)
    assert not np.isnan(emb_16).any()


def test_multimodal_feature_extractor_with_codebert():
    """Verify MultimodalFeatureExtractor integrates CodeBERT with AST and Process features."""
    code = "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)"
    exp = "Calculates factorial recursively by multiplying n with the recursive call until base case n <= 1."
    proc = {
        "paste_ratio": 0.05,
        "burstiness_ratio": 0.35,
        "coding_duration_seconds": 120.0,
        "total_events": 45,
        "run_count": 2,
        "error_count": 0,
        "error_recovery_rate": 1.0,
        "typing_speed_cpm": 180.0
    }
    
    vec = MultimodalFeatureExtractor.extract(code, exp, proc, blend_codebert=True)
    assert vec.shape == (40,)
    assert not np.isnan(vec).any()

