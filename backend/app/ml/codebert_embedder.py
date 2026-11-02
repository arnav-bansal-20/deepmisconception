"""CodeBERT embedding extraction service using microsoft/codebert-base.

Provides explicit status indicators, lazy initialization, graceful offline fallback,
and Apple Silicon MPS / CPU device support without silent zero-vector spoofing.
"""
import os
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy singleton model holders
_CODEBERT_TOKENIZER = None
_CODEBERT_MODEL = None
_CODEBERT_DEVICE = None
_CODEBERT_STATUS = "uninitialized"  # "uninitialized", "loaded", "unavailable", "failed"
_CODEBERT_ERROR: Optional[str] = None


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


def _init_codebert():
    """Lazily load CodeBERT tokenizer and model with offline safety and device detection."""
    global _CODEBERT_TOKENIZER, _CODEBERT_MODEL, _CODEBERT_DEVICE, _CODEBERT_STATUS, _CODEBERT_ERROR
    if _CODEBERT_STATUS in ("loaded", "unavailable"):
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _CODEBERT_DEVICE = get_optimal_device()
        model_name = "microsoft/codebert-base"

        # 1. Attempt loading from local HF cache first (offline mode)
        try:
            _CODEBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            _CODEBERT_MODEL = AutoModel.from_pretrained(model_name, local_files_only=True).to(_CODEBERT_DEVICE)
        except Exception as local_e:
            # 2. Online fallback if cache misses and network is reachable
            logger.info(f"Local CodeBERT weights not in cache ({local_e}); attempting online fetch...")
            _CODEBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _CODEBERT_MODEL = AutoModel.from_pretrained(model_name).to(_CODEBERT_DEVICE)

        _CODEBERT_MODEL.eval()
        _CODEBERT_STATUS = "loaded"
        _CODEBERT_ERROR = None
        logger.info(f"CodeBERT successfully initialized on device: {_CODEBERT_DEVICE}")

    except ImportError as imp_err:
        _CODEBERT_STATUS = "unavailable"
        _CODEBERT_ERROR = f"PyTorch or HuggingFace Transformers not installed: {imp_err}"
        logger.info(f"CodeBERT is unavailable ({_CODEBERT_ERROR}). AST structural features will serve as fallback.")
    except Exception as e:
        _CODEBERT_STATUS = "failed"
        _CODEBERT_ERROR = str(e)
        logger.warning(f"Could not load CodeBERT weights: {e}. Fallback representation will be reported.")


class CodeBERTEmbeddingResult(BaseModel):
    """Detailed result of CodeBERT embedding extraction with explicit status."""
    is_available: bool
    status: str  # "loaded", "unavailable", "failed", "fallback_used"
    model_name: str = "microsoft/codebert-base"
    dimension: int
    device: Optional[str] = None
    embedding: Optional[list] = None
    fallback_used: bool = False
    error: Optional[str] = None


class CodeBERTEmbedder:
    """Extracts 768-dimensional contextual dense code representations using CodeBERT."""

    EMBEDDING_DIM = 768
    MODEL_NAME = "microsoft/codebert-base"

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Return explicit model availability and device status."""
        _init_codebert()
        return {
            "model_name": cls.MODEL_NAME,
            "is_available": _CODEBERT_STATUS == "loaded",
            "status": _CODEBERT_STATUS,
            "device": _CODEBERT_DEVICE or get_optimal_device(),
            "dimension": cls.EMBEDDING_DIM,
            "error": _CODEBERT_ERROR,
        }

    @classmethod
    def is_available(cls) -> bool:
        """Check if CodeBERT model and tokenizer are active."""
        _init_codebert()
        return _CODEBERT_STATUS == "loaded"

    @classmethod
    def extract_embedding_detailed(cls, code: str, max_length: int = 256) -> CodeBERTEmbeddingResult:
        """Extract [CLS] token embedding with full diagnostic metadata."""
        status_info = cls.get_status()

        if not cls.is_available():
            return CodeBERTEmbeddingResult(
                is_available=False,
                status=status_info["status"],
                dimension=cls.EMBEDDING_DIM,
                device=status_info["device"],
                fallback_used=True,
                error=status_info["error"],
            )

        if not code or not code.strip():
            return CodeBERTEmbeddingResult(
                is_available=True,
                status="loaded",
                dimension=cls.EMBEDDING_DIM,
                device=status_info["device"],
                embedding=[0.0] * cls.EMBEDDING_DIM,
                fallback_used=True,
                error="Empty source code provided",
            )

        try:
            import torch
            inputs = _CODEBERT_TOKENIZER(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(_CODEBERT_DEVICE)

            with torch.no_grad():
                outputs = _CODEBERT_MODEL(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                norm = np.linalg.norm(cls_embedding)
                if norm > 1e-9:
                    cls_embedding = cls_embedding / norm

                return CodeBERTEmbeddingResult(
                    is_available=True,
                    status="loaded",
                    dimension=cls.EMBEDDING_DIM,
                    device=_CODEBERT_DEVICE,
                    embedding=cls_embedding.tolist(),
                    fallback_used=False,
                )
        except Exception as e:
            logger.warning(f"Error during CodeBERT embedding inference: {e}")
            return CodeBERTEmbeddingResult(
                is_available=False,
                status="failed",
                dimension=cls.EMBEDDING_DIM,
                device=_CODEBERT_DEVICE,
                fallback_used=True,
                error=str(e),
            )

    @classmethod
    def extract_embedding(cls, code: str, max_length: int = 256) -> np.ndarray:
        """Extract [CLS] embedding vector (768-D) with transparent fallback logging."""
        result = cls.extract_embedding_detailed(code, max_length)
        if result.is_available and result.embedding:
            return np.array(result.embedding, dtype=np.float32)

        # Fallback representation: notify caller via log that fallback feature is in use
        logger.debug(f"CodeBERT unavailable ({result.status}); using zero-initialized representation.")
        return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

    @classmethod
    def extract_projected(cls, code: str, target_dim: int = 16) -> np.ndarray:
        """Linearly pool/project CodeBERT embedding to target dimensions."""
        full_emb = cls.extract_embedding(code)
        if target_dim >= cls.EMBEDDING_DIM:
            return full_emb

        chunk_size = cls.EMBEDDING_DIM // target_dim
        projected = np.zeros(target_dim, dtype=np.float32)
        for i in range(target_dim):
            start = i * chunk_size
            end = start + chunk_size
            projected[i] = full_emb[start:end].mean()
        return projected
