"""Sentence-Transformer embedding extraction service for student explanation semantics."""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy singleton model holder
_ST_TOKENIZER = None
_ST_MODEL = None
_ST_DEVICE = None
_ST_FAILED = False


def _init_sentence_transformer():
    """Lazily load Sentence-Transformer tokenizer and model onto Apple Silicon (MPS) or CPU."""
    global _ST_TOKENIZER, _ST_MODEL, _ST_DEVICE, _ST_FAILED
    if _ST_MODEL is not None or _ST_FAILED:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _ST_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        try:
            _ST_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            _ST_MODEL = AutoModel.from_pretrained(model_name, local_files_only=True).to(_ST_DEVICE)
        except Exception:
            _ST_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _ST_MODEL = AutoModel.from_pretrained(model_name).to(_ST_DEVICE)

        _ST_MODEL.eval()
        logger.info(f"Sentence-Transformer successfully initialized on device: {_ST_DEVICE}")
    except Exception as e:
        logger.warning(f"Could not initialize Sentence-Transformer: {e}")
        _ST_FAILED = True


class SentenceTransformerEmbedder:
    """Extracts 384-dimensional dense semantic vectors from student natural-language explanations."""

    EMBEDDING_DIM = 384

    @classmethod
    def is_available(cls) -> bool:
        """Check if Sentence-Transformer is ready."""
        _init_sentence_transformer()
        return _ST_MODEL is not None

    @classmethod
    def extract_embedding(cls, text: str, max_length: int = 128) -> np.ndarray:
        """Extract mean-pooled, normalized 384-D sentence embedding from text."""
        if not text or not text.strip():
            return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

        _init_sentence_transformer()
        if _ST_MODEL is None or _ST_TOKENIZER is None:
            return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

        try:
            import torch
            inputs = _ST_TOKENIZER(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
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
                return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error during Sentence-Transformer embedding inference: {e}")
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
            projected[i] = np.mean(full_emb[start:end])

        norm = np.linalg.norm(projected)
        if norm > 1e-9:
            projected = projected / norm
        return projected.astype(np.float32)

