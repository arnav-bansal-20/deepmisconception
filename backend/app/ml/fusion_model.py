"""Transformer-based Fusion Model (Proposed Model) implementing Cross-Attention between Code, Explanation, and Process representations."""
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np

LABELS = ["Understood", "Partial Understanding", "Possible Misconception"]
LABEL_MAP = {lbl: i for i, lbl in enumerate(LABELS)}
REV_LABEL_MAP = {i: lbl for i, lbl in enumerate(LABELS)}


class MultimodalTransformerFusionModel:
    """Proposed Multimodal Fusion Architecture from DeepMisconception Novelty Schematic.
    
    Architecture:
    1. Input Encoders:
       - Code Embedding (AST + CodeBERT): d_code
       - Explanation Embedding (Sentence-Transformer): d_exp
       - Process Embedding (Process Transformer Encoder): d_proc
    2. Cross-Attention Layer:
       - Q = Code_Emb * W_Q
       - K = Exp_Emb * W_K
       - V = Exp_Emb * W_V
       - A = Softmax(Q K^T / sqrt(d_k)) * V
    3. Fully Connected Dense Fusion Layer:
       - Fusion = [A | Process_Emb] * W_dense + b_dense
       - Activation: ReLU + LayerNorm
    4. Softmax Classification Layer:
       - Logits = Fusion * W_out + b_out
       - Output: Softmax distribution over 3 Comprehension classes
    """

    def __init__(
        self,
        d_code: int = 16,
        d_exp: int = 16,
        d_proc: int = 16,
        d_k: int = 16,
        hidden_dim: int = 32,
        num_classes: int = 3,
        seed: int = 42,
    ):
        self.d_code = d_code
        self.d_exp = d_exp
        self.d_proc = d_proc
        self.d_k = d_k
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        rng = np.random.RandomState(seed)
        # 1. Cross-Attention projection matrices
        scale_q = 1.0 / math.sqrt(d_code)
        scale_k = 1.0 / math.sqrt(d_exp)
        self.W_q = (rng.randn(d_code, d_k) * scale_q).astype(np.float32)
        self.W_k = (rng.randn(d_exp, d_k) * scale_k).astype(np.float32)
        self.W_v = (rng.randn(d_exp, d_k) * scale_k).astype(np.float32)

        # 2. Fully Connected Dense Layer (combines Cross-Attention output d_k and Process d_proc)
        fusion_in_dim = d_k + d_proc
        scale_dense = 1.0 / math.sqrt(fusion_in_dim)
        self.W_dense = (rng.randn(fusion_in_dim, hidden_dim) * scale_dense).astype(np.float32)
        self.b_dense = np.zeros(hidden_dim, dtype=np.float32)

        # 3. Final Classification Layer
        scale_out = 1.0 / math.sqrt(hidden_dim)
        self.W_out = (rng.randn(hidden_dim, num_classes) * scale_out).astype(np.float32)
        self.b_out = np.zeros(num_classes, dtype=np.float32)

    def forward(
        self,
        code_emb: np.ndarray,
        exp_emb: np.ndarray,
        proc_emb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through Cross-Attention Fusion and Classification Layer.
        
        Args:
            code_emb: (B, d_code) or (d_code,)
            exp_emb: (B, d_exp) or (d_exp,)
            proc_emb: (B, d_proc) or (d_proc,)
            
        Returns:
            probs: (B, 3) classification probabilities
            attn_weights: Cross-attention alignment matrix
        """
        is_single = (code_emb.ndim == 1)
        if is_single:
            code_emb = code_emb[np.newaxis, :]
            exp_emb = exp_emb[np.newaxis, :]
            proc_emb = proc_emb[np.newaxis, :]

        # 1. Compute Cross-Attention: Code Query attends to Explanation Key/Value
        Q = np.matmul(code_emb, self.W_q)  # (B, d_k)
        K = np.matmul(exp_emb, self.W_k)   # (B, d_k)
        V = np.matmul(exp_emb, self.W_v)   # (B, d_k)

        # Pairwise scalar products
        scale = math.sqrt(self.d_k)
        raw_attn = np.sum(Q * K, axis=-1, keepdims=True) / scale  # (B, 1)
        attn_weights = 1.0 / (1.0 + np.exp(-raw_attn))            # Sigmoid alignment score
        cross_out = attn_weights * V                               # (B, d_k)

        # 2. Fuse with Process Representation
        fused = np.concatenate([cross_out, proc_emb], axis=-1)     # (B, d_k + d_proc)
        dense = np.maximum(0, np.matmul(fused, self.W_dense) + self.b_dense)  # ReLU

        # LayerNorm
        mean = np.mean(dense, axis=-1, keepdims=True)
        std = np.std(dense, axis=-1, keepdims=True) + 1e-6
        dense_norm = (dense - mean) / std

        # 3. Softmax Classification
        logits = np.matmul(dense_norm, self.W_out) + self.b_out    # (B, 3)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)

        if is_single:
            return probs[0], attn_weights[0]
        return probs, attn_weights

    def predict(
        self,
        code_emb: np.ndarray,
        exp_emb: np.ndarray,
        proc_emb: np.ndarray,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict comprehension diagnosis label, confidence score, and distribution."""
        probs, _ = self.forward(code_emb, exp_emb, proc_emb)
        pred_idx = int(np.argmax(probs))
        label = REV_LABEL_MAP[pred_idx]
        conf = float(probs[pred_idx])
        distribution = {REV_LABEL_MAP[i]: float(probs[i]) for i in range(self.num_classes)}
        return label, conf, distribution

