"""Process Feature Extractor (Transformer Encoder) for student coding process events."""
import math
from typing import List, Dict, Any, Optional
import numpy as np

EVENT_TYPE_MAP = {
    "idle": 0,
    "edit": 1,
    "delete": 2,
    "paste": 3,
    "run": 4,
    "compile_error": 5,
    "runtime_error": 6,
    "navigation": 7,
}


class ProcessTransformerEncoder:
    """Encodes sequences of coding-process events (keystrokes, edits, pastes, runs, pauses) using a Transformer Encoder."""

    EVENT_DIM = 8
    HIDDEN_DIM = 16
    OUTPUT_DIM = 16

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Weight matrices for Transformer Encoder
        # Linear projection: EVENT_DIM -> HIDDEN_DIM
        self.W_in = (self.rng.randn(self.EVENT_DIM, self.HIDDEN_DIM) * 0.1).astype(np.float32)
        self.b_in = np.zeros(self.HIDDEN_DIM, dtype=np.float32)

        # Self-Attention weights
        self.W_q = (self.rng.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.1).astype(np.float32)
        self.W_k = (self.rng.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.1).astype(np.float32)
        self.W_v = (self.rng.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.1).astype(np.float32)

        # Feed-forward sublayer
        self.W_ff = (self.rng.randn(self.HIDDEN_DIM, self.OUTPUT_DIM) * 0.1).astype(np.float32)
        self.b_ff = np.zeros(self.OUTPUT_DIM, dtype=np.float32)

    def _event_to_vector(self, event: Dict[str, Any], prev_time: Optional[float] = None) -> np.ndarray:
        """Convert a single telemetry event into an 8-D numerical feature slice."""
        vec = np.zeros(self.EVENT_DIM, dtype=np.float32)
        etype = str(event.get("event_type", "edit")).lower()
        vec[0] = EVENT_TYPE_MAP.get(etype, 1) / 7.0

        # Character delta (positive for inserts, negative for deletions)
        delta = float(event.get("delta_chars", 0))
        vec[1] = max(-1.0, min(1.0, delta / 50.0))

        # Paste length
        paste_len = float(event.get("paste_length", 0))
        vec[2] = min(1.0, paste_len / 100.0)

        # Cursor line & column normalized
        line = float(event.get("cursor_line", 1))
        col = float(event.get("cursor_column", 1))
        vec[3] = min(1.0, line / 50.0)
        vec[4] = min(1.0, col / 80.0)

        # Inter-event pause / time gap
        curr_time = float(event.get("timestamp_s", 0) or 0)
        gap = (curr_time - prev_time) if prev_time is not None else 0.5
        vec[5] = min(1.0, max(0.0, gap / 10.0))

        # Is compilation / run attempt
        vec[6] = 1.0 if etype == "run" else 0.0

        # Is error occurrence
        vec[7] = 1.0 if "error" in etype else 0.0

        return vec

    def encode_sequence(self, events: List[Dict[str, Any]], max_len: int = 32) -> np.ndarray:
        """Pass event sequence through Transformer Encoder and produce fixed Process Representation."""
        if not events:
            return np.zeros(self.OUTPUT_DIM, dtype=np.float32)

        # Truncate or sample sequence
        seq_events = events[-max_len:] if len(events) > max_len else events
        T = len(seq_events)

        # 1. Convert events to input feature matrix (T x EVENT_DIM)
        X = np.zeros((T, self.EVENT_DIM), dtype=np.float32)
        prev_time = None
        for t, ev in enumerate(seq_events):
            X[t] = self._event_to_vector(ev, prev_time)
            prev_time = float(ev.get("timestamp_s", 0) or 0)

        # 2. Linear input projection: (T x HIDDEN_DIM)
        H = np.matmul(X, self.W_in) + self.b_in

        # Add sinusoidal positional encoding
        pos = np.arange(T)[:, np.newaxis]
        i = np.arange(self.HIDDEN_DIM)[np.newaxis, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(self.HIDDEN_DIM))
        pos_encoding = np.zeros_like(H)
        pos_encoding[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])
        H = H + pos_encoding

        # 3. Self-Attention layer: Softmax(Q K^T / sqrt(d)) V
        Q = np.matmul(H, self.W_q)
        K = np.matmul(H, self.W_k)
        V = np.matmul(H, self.W_v)

        scale = math.sqrt(self.HIDDEN_DIM)
        attn_scores = np.matmul(Q, K.T) / scale
        exp_scores = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

        attn_out = np.matmul(attn_weights, V)

        # Residual connection + Layer normalization
        H_attn = H + attn_out
        mean = np.mean(H_attn, axis=-1, keepdims=True)
        std = np.std(H_attn, axis=-1, keepdims=True) + 1e-6
        H_norm = (H_attn - mean) / std

        # 4. Feed-forward sublayer + Mean pooling across time sequence
        FF = np.maximum(0, np.matmul(H_norm, self.W_ff) + self.b_ff)  # ReLU
        process_rep = np.mean(FF, axis=0)  # (OUTPUT_DIM,)

        # L2 Normalize
        norm = np.linalg.norm(process_rep)
        if norm > 1e-9:
            process_rep = process_rep / norm

        return process_rep.astype(np.float32)

