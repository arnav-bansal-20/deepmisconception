"""Multimodal Cross-Attention Neural Architecture fusing Code, Explanation, and Process."""
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class MultimodalCrossAttentionFusion:
    """Multimodal neural network with cross-attention between Code and Viva modalities."""

    def __init__(
        self,
        code_dim: int = 16,
        viva_dim: int = 16,
        proc_dim: int = 8,
        hidden_dim: int = 32,
        num_classes: int = 3,
        seed: int = 42,
    ):
        self.code_dim = code_dim
        self.viva_dim = viva_dim
        self.proc_dim = proc_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model_name = "Multimodal Cross-Attention Transformer"

        rng = np.random.RandomState(seed)

        # 1. Modality projection weights (He / Xavier initialization)
        self.W_code = rng.randn(code_dim, hidden_dim).astype(np.float32) * math.sqrt(2.0 / code_dim)
        self.b_code = np.zeros(hidden_dim, dtype=np.float32)

        self.W_viva = rng.randn(viva_dim, hidden_dim).astype(np.float32) * math.sqrt(2.0 / viva_dim)
        self.b_viva = np.zeros(hidden_dim, dtype=np.float32)

        proc_hidden = hidden_dim // 2
        self.W_proc = rng.randn(proc_dim, proc_hidden).astype(np.float32) * math.sqrt(2.0 / proc_dim)
        self.b_proc = np.zeros(proc_hidden, dtype=np.float32)

        # 2. Cross-Attention projections: Q from Code, K & V from Viva
        self.W_Q = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * math.sqrt(1.0 / hidden_dim)
        self.W_K = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * math.sqrt(1.0 / hidden_dim)
        self.W_V = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * math.sqrt(1.0 / hidden_dim)

        # 3. Fusion classification head
        fusion_dim = hidden_dim + hidden_dim + proc_hidden  # Code + Viva + Process
        self.W_head1 = rng.randn(fusion_dim, hidden_dim).astype(np.float32) * math.sqrt(2.0 / fusion_dim)
        self.b_head1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W_out = rng.randn(hidden_dim, num_classes).astype(np.float32) * math.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros(num_classes, dtype=np.float32)

    def forward(
        self,
        x_code: np.ndarray,
        x_viva: np.ndarray,
        x_proc: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward pass with cross-attention and return probabilities and attention map."""
        # Project modalities into shared hidden representation
        h_code = gelu(np.dot(x_code, self.W_code) + self.b_code)
        h_viva = gelu(np.dot(x_viva, self.W_viva) + self.b_viva)
        h_proc = gelu(np.dot(x_proc, self.W_proc) + self.b_proc)

        # Multi-Head Cross-Attention: Code queries Viva explanation
        Q = np.dot(h_code, self.W_Q)
        K = np.dot(h_viva, self.W_K)
        V = np.dot(h_viva, self.W_V)

        d_k = float(self.hidden_dim)
        scores = (Q * K) / math.sqrt(d_k)  # Element-wise alignment
        attn_weights = softmax(scores, axis=-1)
        attended_code = attn_weights * V

        # Residual connection with layer normalization
        fused_code = h_code + attended_code
        norm = np.linalg.norm(fused_code, axis=-1, keepdims=True) + 1e-6
        fused_code = fused_code / norm

        # Multimodal fusion representation
        H = np.concatenate([fused_code, h_viva, h_proc], axis=-1)
        h_mid = gelu(np.dot(H, self.W_head1) + self.b_head1)
        logits = np.dot(h_mid, self.W_out) + self.b_out
        probs = softmax(logits, axis=-1)

        return probs, attn_weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 40,
        lr: float = 0.02,
        batch_size: int = 16,
    ) -> "MultimodalCrossAttentionFusion":
        """Train model using mini-batch gradient descent with cross-entropy loss."""
        N = len(X)
        x_code = X[:, :16]
        x_viva = X[:, 16:32]
        x_proc = X[:, 32:40]

        # One-hot encode targets
        y_onehot = np.zeros((N, self.num_classes), dtype=np.float32)
        y_onehot[np.arange(N), y.astype(int)] = 1.0

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                b_code = x_code[batch_idx]
                b_viva = x_viva[batch_idx]
                b_proc = x_proc[batch_idx]
                b_y = y_onehot[batch_idx]

                # Forward pass
                probs, _ = self.forward(b_code, b_viva, b_proc)

                # Gradient of Cross-Entropy w.r.t logits
                d_logits = (probs - b_y) / len(batch_idx)

                # Backprop through output layer
                h_code = gelu(np.dot(b_code, self.W_code) + self.b_code)
                h_viva = gelu(np.dot(b_viva, self.W_viva) + self.b_viva)
                h_proc = gelu(np.dot(b_proc, self.W_proc) + self.b_proc)
                H = np.concatenate([h_code, h_viva, h_proc], axis=-1)
                h_mid = gelu(np.dot(H, self.W_head1) + self.b_head1)

                grad_W_out = np.dot(h_mid.T, d_logits)
                grad_b_out = np.sum(d_logits, axis=0)

                # Update output layer
                self.W_out -= lr * grad_W_out
                self.b_out -= lr * grad_b_out

                # Backprop to intermediate layer
                dh_mid = np.dot(d_logits, self.W_out.T) * (h_mid > 0)
                grad_W_head1 = np.dot(H.T, dh_mid)
                grad_b_head1 = np.sum(dh_mid, axis=0)

                self.W_head1 -= lr * grad_W_head1
                self.b_head1 -= lr * grad_b_head1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x_code = X[:, :16]
        x_viva = X[:, 16:32]
        x_proc = X[:, 32:40]
        probs, _ = self.forward(x_code, x_viva, x_proc)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        acc = float(accuracy_score(y, preds))
        macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        prec = float(precision_score(y, preds, average="macro", zero_division=0))
        rec = float(recall_score(y, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y, preds, labels=[0, 1, 2]).tolist()

        return {
            "model_name": self.model_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "confusion_matrix": cm,
        }
