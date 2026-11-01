"""CodeBERT embedding extraction service using microsoft/codebert-base on macOS."""
import os
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy singleton model holder
_CODEBERT_TOKENIZER = None
_CODEBERT_MODEL = None
_CODEBERT_DEVICE = None
_CODEBERT_FAILED = False


def _init_codebert():
    """Lazily load CodeBERT tokenizer and model onto Apple Silicon (MPS) or CPU."""
    global _CODEBERT_TOKENIZER, _CODEBERT_MODEL, _CODEBERT_DEVICE, _CODEBERT_FAILED
    if _CODEBERT_MODEL is not None or _CODEBERT_FAILED:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _CODEBERT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        model_name = "microsoft/codebert-base"

        # Try loading locally first without internet requirement
        try:
            _CODEBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            _CODEBERT_MODEL = AutoModel.from_pretrained(model_name, local_files_only=True).to(_CODEBERT_DEVICE)
        except Exception:
            _CODEBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _CODEBERT_MODEL = AutoModel.from_pretrained(model_name).to(_CODEBERT_DEVICE)

        _CODEBERT_MODEL.eval()
        logger.info(f"CodeBERT successfully initialized on device: {_CODEBERT_DEVICE}")
    except Exception as e:
        logger.warning(f"Could not initialize CodeBERT: {e}. Falling back to AST features.")
        _CODEBERT_FAILED = True


class CodeBERTEmbedder:
    """Extracts 768-dimensional contextual dense code representations using CodeBERT."""

    EMBEDDING_DIM = 768

    @classmethod
    def is_available(cls) -> bool:
        """Check if CodeBERT and PyTorch are loaded and ready."""
        _init_codebert()
        return _CODEBERT_MODEL is not None

    @classmethod
    def extract_embedding(cls, code: str, max_length: int = 256) -> np.ndarray:
        """Extract [CLS] token embedding vector (768-D) from source code."""
        if not code or not code.strip():
            return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

        _init_codebert()
        if _CODEBERT_MODEL is None or _CODEBERT_TOKENIZER is None:
            # Fallback zero vector if model is unavailable
            return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

        try:
            import torch
            inputs = _CODEBERT_TOKENIZER(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(_CODEBERT_DEVICE)

            with torch.no_grad():
                outputs = _CODEBERT_MODEL(**inputs)
                # Extract [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                # L2 normalize
                norm = np.linalg.norm(cls_embedding)
                if norm > 1e-9:
                    cls_embedding = cls_embedding / norm
                return cls_embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error during CodeBERT embedding inference: {e}")
            return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

    @classmethod
    def extract_projected(cls, code: str, target_dim: int = 16) -> np.ndarray:
        """Extract and linearly pool/project CodeBERT embedding to target dimensions."""
        full_emb = cls.extract_embedding(code)
        if target_dim >= cls.EMBEDDING_DIM:
            return full_emb

        # Average pool chunks to reduce dimension deterministically
        chunk_size = cls.EMBEDDING_DIM // target_dim
        projected = np.zeros(target_dim, dtype=np.float32)
        for i in range(target_dim):
            start = i * chunk_size
            end = start + chunk_size
            projected[i] = np.mean(full_emb[start:end])

        norm = np.linalg.norm(projected)
        if norm > 1e-9:
            projected = projected / norm
        return projected

