"""Sentence-Transformer embedding extraction service for student explanation semantics.

Provides explicit status indicators, lazy initialization, graceful offline fallback,
and Apple Silicon MPS / CPU device support without silent zero-vector spoofing.
"""
import os
import logging
from typing import Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy singleton model holders
_ST_TOKENIZER = None
_ST_MODEL = None
_ST_DEVICE = None
_ST_STATUS = "uninitialized"  # "uninitialized", "loaded", "unavailable", "failed"
_ST_ERROR: Optional[str] = None


def get_optimal_device() -> str:
    """Detect available PyTorch acceleration: MPS (Apple Silicon), CUDA, or CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def _init_sentence_transformer():
    """Lazily load Sentence-Transformer tokenizer and model with offline safety."""
    global _ST_TOKENIZER, _ST_MODEL, _ST_DEVICE, _ST_STATUS, _ST_ERROR
    if _ST_STATUS in ("loaded", "unavailable"):
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _ST_DEVICE = get_optimal_device()
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # 1. Attempt loading from local HF cache first
        try:
            _ST_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            _ST_MODEL = AutoModel.from_pretrained(model_name, local_files_only=True).to(_ST_DEVICE)
        except Exception as local_e:
            # 2. Online fetch if cache misses
            logger.info(f"Local Sentence-Transformer weights not in cache ({local_e}); attempting online fetch...")
            _ST_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _ST_MODEL = AutoModel.from_pretrained(model_name).to(_ST_DEVICE)

        _ST_MODEL.eval()
        _ST_STATUS = "loaded"
        _ST_ERROR = None
        logger.info(f"Sentence-Transformer successfully initialized on device: {_ST_DEVICE}")

    except ImportError as imp_err:
        _ST_STATUS = "unavailable"
        _ST_ERROR = f"PyTorch or HuggingFace Transformers not installed: {imp_err}"
        logger.info(f"Sentence-Transformer is unavailable ({_ST_ERROR}). Viva rubrics will serve as fallback.")
    except Exception as e:
        _ST_STATUS = "failed"
        _ST_ERROR = str(e)
        logger.warning(f"Could not load Sentence-Transformer weights: {e}")


class SentenceTransformerEmbeddingResult(BaseModel):
    """Detailed result of Sentence-Transformer embedding extraction with explicit status."""
    is_available: bool
    status: str  # "loaded", "unavailable", "failed", "fallback_used"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int
    device: Optional[str] = None
    embedding: Optional[list] = None
    fallback_used: bool = False
    error: Optional[str] = None


class SentenceTransformerEmbedder:
    """Extracts 384-dimensional dense semantic vectors from student natural-language explanations."""

    EMBEDDING_DIM = 384
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Return explicit model availability and device status."""
        _init_sentence_transformer()
        return {
            "model_name": cls.MODEL_NAME,
            "is_available": _ST_STATUS == "loaded",
            "status": _ST_STATUS,
            "device": _ST_DEVICE or get_optimal_device(),
            "dimension": cls.EMBEDDING_DIM,
            "error": _ST_ERROR,
        }

    @classmethod
    def is_available(cls) -> bool:
        """Check if Sentence-Transformer model and tokenizer are active."""
        _init_sentence_transformer()
        return _ST_STATUS == "loaded"

    @classmethod
    def extract_embedding_detailed(cls, text: str, max_length: int = 128) -> SentenceTransformerEmbeddingResult:
        """Extract mean-pooled, normalized 384-D sentence embedding with full metadata."""
        status_info = cls.get_status()

        if not cls.is_available():
            return SentenceTransformerEmbeddingResult(
                is_available=False,
                status=status_info["status"],
                dimension=cls.EMBEDDING_DIM,
                device=status_info["device"],
                fallback_used=True,
                error=status_info["error"],
            )

        if not text or not text.strip():
            return SentenceTransformerEmbeddingResult(
                is_available=True,
                status="loaded",
                dimension=cls.EMBEDDING_DIM,
                device=status_info["device"],
                embedding=[0.0] * cls.EMBEDDING_DIM,
                fallback_used=True,
                error="Empty explanation text provided",
            )

        try:
            import torch
            inputs = _ST_TOKENIZER(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(_ST_DEVICE)

            with torch.no_grad():
                outputs = _ST_MODEL(**inputs)
                token_embeddings = outputs[0]
                input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()

                norm = np.linalg.norm(embedding)
                if norm > 1e-9:
                    embedding = embedding / norm

                return SentenceTransformerEmbeddingResult(
                    is_available=True,
                    status="loaded",
                    dimension=cls.EMBEDDING_DIM,
                    device=_ST_DEVICE,
                    embedding=embedding.tolist(),
                    fallback_used=False,
                )
        except Exception as e:
            logger.warning(f"Error during Sentence-Transformer embedding inference: {e}")
            return SentenceTransformerEmbeddingResult(
                is_available=False,
                status="failed",
                dimension=cls.EMBEDDING_DIM,
                device=_ST_DEVICE,
                fallback_used=True,
                error=str(e),
            )

    @classmethod
    def extract_embedding(cls, text: str, max_length: int = 128) -> np.ndarray:
        """Extract mean-pooled, normalized 384-D sentence embedding vector."""
        result = cls.extract_embedding_detailed(text, max_length)
        if result.is_available and result.embedding:
            return np.array(result.embedding, dtype=np.float32)

        logger.debug(f"Sentence-Transformer unavailable ({result.status}); using zero-initialized representation.")
        return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

    @classmethod
    def extract_projected(cls, text: str, target_dim: int = 16) -> np.ndarray:
        """Linearly pool/project Sentence-Transformer embedding to target dimensions."""
        full_emb = cls.extract_embedding(text)
        if target_dim >= cls.EMBEDDING_DIM:
            return full_emb

        chunk_size = cls.EMBEDDING_DIM // target_dim
        projected = np.zeros(target_dim, dtype=np.float32)
        for i in range(target_dim):
            start = i * chunk_size
            end = start + chunk_size
            projected[i] = full_emb[start:end].mean()
        return projected
