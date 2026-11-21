"""Process Feature Extractor (Transformer Encoder) for student coding process events.

Implements a genuine PyTorch Transformer Encoder architecture that processes ordered
sequences of student coding telemetry events (keystrokes, edits, pastes, test executions,
errors, time gaps) together with aggregated numerical telemetry features into a learned
dense process representation.
"""
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object

from app.ml.config import PROC_DIM, D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS

EVENT_TYPE_MAP = {
    "pad": 0,
    "idle": 1,
    "edit": 2,
    "insert": 2,           # Unified edit/text-insertion
    "delete": 3,
    "paste": 4,
    "run": 5,
    "success_run": 5,       # Execution success
    "compile_error": 6,
    "syntax_error": 6,      # Syntax / compilation error
    "runtime_error": 7,
    "error": 7,             # General runtime error
    "test_run": 8,          # Sandbox test execution
    "debug_action": 9,      # Debugger / inspect action
    "navigation": 10,
    "unknown": 11,
}
NUM_EVENT_TYPES = 16


def parse_timestamp_to_utc(ts: Any) -> Optional[datetime]:
    """Safely convert diverse timestamp types (datetime, ISO string, epoch float/int) to UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    if isinstance(ts, (int, float)):
        val = float(ts)
        if val > 1e11:  # Milliseconds epoch
            val /= 1000.0
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    return None


def normalize_process_event_timestamps(events: List[Any]) -> List[Dict[str, Any]]:
    """Accepts heterogeneous events (DB models, dicts) with datetime, ISO strings, or numeric timestamps,
    sorts them chronologically, and converts timestamps into numerical elapsed seconds from reference event T0.
    
    Guarantees:
    1. Preserves chronological event ordering (stable sort on parsed UTC timestamp).
    2. Converts timestamps into numerical elapsed seconds from T0 (Event 1 -> 0.0s, Event 2 -> delta, ...).
    3. Handles timezone-aware, naive, ISO strings, and epoch values consistently.
    4. Safely handles missing/invalid timestamps without crashing.
    5. Calculates time gaps between consecutive events.
    """
    if not events:
        return []

    # 1. Standardize to raw dicts with parsed UTC timestamp or numerical elapsed seconds
    extracted: List[Dict[str, Any]] = []
    has_precomputed_elapsed = False

    for idx, ev in enumerate(events):
        if isinstance(ev, dict):
            etype = ev.get("event_type", "edit")
            raw_ts = ev.get("timestamp") or ev.get("timestamp_s") or ev.get("created_at")
            raw_elapsed = ev.get("elapsed_seconds") if "elapsed_seconds" in ev else ev.get("t_sec")
            delta_chars = ev.get("delta_chars", ev.get("chars", 0))
            paste_len = ev.get("paste_length", 0)
            line = ev.get("cursor_line", ev.get("line", 1))
            col = ev.get("cursor_column", 1)
            res = ev.get("result", "")
            err_type = ev.get("error_type", None)
            is_err = bool(ev.get("is_error") or res == "error" or err_type or "error" in str(etype).lower())
        else:
            etype = getattr(ev, "event_type", "edit")
            raw_ts = getattr(ev, "timestamp", None)
            raw_elapsed = getattr(ev, "elapsed_seconds", getattr(ev, "t_sec", None))
            delta_chars = getattr(ev, "delta_chars", getattr(ev, "chars", 0))
            paste_len = getattr(ev, "paste_length", 0)
            line = getattr(ev, "cursor_line", getattr(ev, "line", 1))
            col = getattr(ev, "cursor_column", 1)
            is_err = bool(getattr(ev, "is_error", False) or "error" in str(etype).lower())

        # Validate and sanitize numerical attributes against non-finite values
        def sanitize_num(val, default, min_val=None):
            try:
                fval = float(val)
                if not math.isfinite(fval):
                    return default
                if min_val is not None:
                    fval = max(min_val, fval)
                return fval
            except (ValueError, TypeError):
                return default

        delta_chars = sanitize_num(delta_chars, 0.0)
        paste_len = sanitize_num(paste_len, 0.0, min_val=0.0)
        line = sanitize_num(line, 1.0, min_val=1.0)
        col = sanitize_num(col, 1.0, min_val=1.0)

        # Parse elapsed seconds if available
        parsed_elapsed = None
        if raw_elapsed is not None:
            try:
                f_el = float(raw_elapsed)
                if math.isfinite(f_el) and f_el >= 0:
                    parsed_elapsed = f_el
                    has_precomputed_elapsed = True
            except (ValueError, TypeError):
                parsed_elapsed = None

        parsed_ts = parse_timestamp_to_utc(raw_ts)
        extracted.append({
            "original_index": idx,
            "event_type": str(etype).lower(),
            "delta_chars": int(delta_chars),
            "paste_length": int(paste_len),
            "cursor_line": int(line),
            "cursor_column": int(col),
            "parsed_dt": parsed_ts,
            "parsed_elapsed": parsed_elapsed,
            "is_error": is_err,
        })

    # 2. Sort chronologically:
    # If precomputed elapsed seconds are present, sort on parsed_elapsed;
    # otherwise sort on parsed_dt UTC datetimes.
    fallback_dt = datetime.max.replace(tzinfo=timezone.utc)
    fallback_elapsed = float("inf")

    if has_precomputed_elapsed:
        sorted_events = sorted(
            extracted,
            key=lambda item: (item["parsed_elapsed"] if item["parsed_elapsed"] is not None else fallback_elapsed, item["original_index"])
        )
    else:
        sorted_events = sorted(
            extracted,
            key=lambda item: (item["parsed_dt"] if item["parsed_dt"] is not None else fallback_dt, item["original_index"])
        )

    # 3. Determine reference timestamp T0
    t0_dt: Optional[datetime] = None
    t0_el: Optional[float] = None
    if has_precomputed_elapsed:
        for ev in sorted_events:
            if ev["parsed_elapsed"] is not None:
                t0_el = ev["parsed_elapsed"]
                break
    else:
        for ev in sorted_events:
            if ev["parsed_dt"] is not None:
                t0_dt = ev["parsed_dt"]
                break

    # 4. Compute elapsed seconds relative to T0 and consecutive time gaps
    normalized: List[Dict[str, Any]] = []
    prev_elapsed = 0.0

    for idx, ev in enumerate(sorted_events):
        if has_precomputed_elapsed and ev["parsed_elapsed"] is not None and t0_el is not None:
            elapsed = max(0.0, ev["parsed_elapsed"] - t0_el)
        elif t0_dt is not None and ev["parsed_dt"] is not None:
            elapsed = max(0.0, (ev["parsed_dt"] - t0_dt).total_seconds())
        else:
            elapsed = prev_elapsed + (0.5 if idx > 0 else 0.0)

        # Ensure elapsed is strictly non-negative and finite
        if not math.isfinite(elapsed) or elapsed < 0:
            elapsed = prev_elapsed + 0.5

        gap = max(0.0, elapsed - prev_elapsed) if idx > 0 else 0.0
        prev_elapsed = elapsed

        normalized.append({
            "event_type": ev["event_type"],
            "delta_chars": ev["delta_chars"],
            "paste_length": ev["paste_length"],
            "cursor_line": ev["cursor_line"],
            "cursor_column": ev["cursor_column"],
            "elapsed_seconds": round(float(elapsed), 4),
            "time_gap_seconds": round(float(gap), 4),
            "timestamp_s": round(float(elapsed), 4),  # backward compatibility alias
            "is_error": ev["is_error"],
        })

    return normalized


class PositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Sinusoidal positional encoding for sequential process events."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to input tensor (B, seq_len, d_model)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


if TORCH_AVAILABLE:
    class PyTorchProcessTransformer(nn.Module):
        """Genuine PyTorch Transformer Encoder for coding process events."""

        def __init__(
            self,
            event_emb_dim: int = 16,
            num_event_features: int = 7,
            global_feature_dim: int = 8,
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            output_dim: int = PROC_DIM,
            max_seq_len: int = 256,
        ):
            super().__init__()
            self.d_model = d_model
            self.output_dim = output_dim
            self.max_seq_len = max_seq_len

            # 1. Event Type Embedding
            self.event_embedding = nn.Embedding(NUM_EVENT_TYPES, event_emb_dim, padding_idx=0)

            # 2. Per-event numerical attribute projection
            self.event_num_proj = nn.Linear(num_event_features, event_emb_dim)

            # 3. Combined event token projection to d_model
            self.token_proj = nn.Linear(event_emb_dim * 2, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
            self.token_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            # 4. Transformer Encoder Layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

            # 5. Global numerical telemetry projection (fuses macro metrics with micro sequence)
            self.global_proj = nn.Sequential(
                nn.Linear(global_feature_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.global_norm = nn.LayerNorm(d_model)

            # 6. Final representation projection
            self.out_proj = nn.Sequential(
                nn.Linear(d_model * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
            self.out_norm = nn.LayerNorm(output_dim)

        def forward(
            self,
            event_types: torch.Tensor,              # (B, T) LongTensor
            event_num_features: torch.Tensor,       # (B, T, 7) FloatTensor
            padding_mask: Optional[torch.Tensor] = None,  # (B, T) BoolTensor (True where padded)
            global_features: Optional[torch.Tensor] = None, # (B, 8) FloatTensor
        ) -> torch.Tensor:
            """Forward pass through Process Transformer.
            
            Returns:
                process_representation: (B, output_dim)
            """
            batch_size, seq_len = event_types.shape
            device = event_types.device

            # Check if all events are padded or empty
            if padding_mask is not None and padding_mask.all(dim=1).all():
                # All tokens masked: produce default baseline
                if global_features is not None:
                    g_rep = self.global_norm(self.global_proj(global_features))
                    return self.out_norm(self.out_proj(torch.cat([g_rep, g_rep], dim=-1)))
                return torch.zeros(batch_size, self.output_dim, device=device)

            # 1. Embed categorical event types: (B, T, event_emb_dim)
            emb_types = self.event_embedding(event_types)

            # 2. Project numerical per-event attributes: (B, T, event_emb_dim)
            emb_nums = torch.relu(self.event_num_proj(event_num_features))

            # 3. Concatenate and project to d_model
            tokens = torch.cat([emb_types, emb_nums], dim=-1)  # (B, T, event_emb_dim * 2)
            h = self.token_norm(self.token_proj(tokens))
            h = self.dropout(self.pos_encoder(h))

            # 4. Pass through Transformer Encoder
            # src_key_padding_mask: (B, T) with True for padded positions
            encoded = self.transformer_encoder(h, src_key_padding_mask=padding_mask)  # (B, T, d_model)

            # 5. Masked Pooling over sequence
            if padding_mask is not None:
                # Invert mask for multiplying: 1.0 for valid, 0.0 for padded
                valid_mask = (~padding_mask).unsqueeze(-1).float()  # (B, T, 1)
                seq_sum = torch.sum(encoded * valid_mask, dim=1)
                valid_counts = torch.clamp(valid_mask.sum(dim=1), min=1.0)
                seq_pooled = seq_sum / valid_counts  # (B, d_model)
            else:
                seq_pooled = encoded.mean(dim=1)

            # 6. Global telemetry representation fusion
            if global_features is not None:
                g_rep = self.global_norm(self.global_proj(global_features))
            else:
                g_rep = torch.zeros(batch_size, self.d_model, device=device)

            # Combine pooled sequence representation and macro telemetry
            fused = torch.cat([seq_pooled, g_rep], dim=-1)  # (B, d_model * 2)
            process_rep = self.out_norm(self.out_proj(fused))

            return process_rep


class ProcessTransformerEncoder:
    """High-level service class wrapping the PyTorch Process Transformer with tensor batching and NumPy fallback."""

    def __init__(
        self,
        output_dim: int = PROC_DIM,
        d_model: int = D_MODEL,
        nhead: int = NUM_HEADS,
        num_encoder_layers: int = 2,
        seed: int = 42,
    ):
        self.output_dim = output_dim
        self.d_model = d_model
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            self.model = PyTorchProcessTransformer(
                event_emb_dim=16,
                num_event_features=7,
                global_feature_dim=8,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                output_dim=output_dim,
            )
            self.model.eval()
        else:
            self.model = None

    @staticmethod
    def _event_to_feature_slice(event: Dict[str, Any], prev_time: Optional[float] = None) -> Tuple[int, np.ndarray]:
        """Extracts event type ID and 7-dimensional normalized numerical attribute slice."""
        etype_str = str(event.get("event_type", "edit")).lower()
        type_id = EVENT_TYPE_MAP.get(etype_str, EVENT_TYPE_MAP["unknown"])

        delta = float(event.get("delta_chars", 0) or 0)
        paste_len = float(event.get("paste_length", 0) or 0)
        line = float(event.get("cursor_line", 1) or 1)
        col = float(event.get("cursor_column", 1) or 1)
        
        # Numerical elapsed seconds & time gap normalized safely
        curr_time = float(event.get("elapsed_seconds", event.get("timestamp_s", 0)) or 0)
        if "time_gap_seconds" in event:
            gap = float(event["time_gap_seconds"])
        else:
            gap = (curr_time - prev_time) if prev_time is not None else 0.5

        num_features = np.zeros(7, dtype=np.float32)
        num_features[0] = max(-1.0, min(1.0, delta / 50.0))          # Delta chars normalized
        num_features[1] = min(1.0, paste_len / 100.0)                 # Paste length
        num_features[2] = min(1.0, line / 50.0)                       # Cursor line
        num_features[3] = min(1.0, col / 80.0)                        # Cursor col
        num_features[4] = min(1.0, max(0.0, gap / 10.0))             # Pause / gap
        num_features[5] = 1.0 if etype_str in ("run", "execute") else 0.0  # Run attempt
        num_features[6] = 1.0 if ("error" in etype_str or event.get("is_error")) else 0.0  # Error flag

        return type_id, num_features

    def encode_sequence(
        self,
        events: List[Any],
        max_len: int = 32,
        global_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode a single event sequence and return an (output_dim,) numpy array.
        
        Automatically normalizes diverse event types and timestamps into chronological order
        and numerical elapsed seconds from reference event T0.
        """
        if not events:
            return np.zeros(self.output_dim, dtype=np.float32)

        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(self.output_dim, dtype=np.float32)

        # 1. Normalize heterogeneous timestamps and order chronologically
        norm_events = normalize_process_event_timestamps(events)
        seq_events = norm_events[-max_len:] if len(norm_events) > max_len else norm_events
        T = len(seq_events)

        type_ids = np.zeros((1, T), dtype=np.int64)
        num_feats = np.zeros((1, T, 7), dtype=np.float32)

        prev_time = None
        for t, ev in enumerate(seq_events):
            tid, nslice = self._event_to_feature_slice(ev, prev_time)
            type_ids[0, t] = tid
            num_feats[0, t] = nslice
            prev_time = float(ev.get("elapsed_seconds", 0) or 0)

        t_types = torch.from_numpy(type_ids)
        t_nums = torch.from_numpy(num_feats)

        t_glob = None
        if global_features is not None:
            t_glob = torch.from_numpy(np.array(global_features, dtype=np.float32)).unsqueeze(0)

        with torch.no_grad():
            out = self.model(t_types, t_nums, padding_mask=None, global_features=t_glob)
            vec = out.squeeze(0).cpu().numpy()

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_batch(
        self,
        batch_events: List[List[Dict[str, Any]]],
        max_len: int = 64,
        global_features_batch: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Encode a batch of variable-length event sequences into a (B, output_dim) torch.Tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for encode_batch")

        B = len(batch_events)
        if B == 0:
            return torch.zeros(0, self.output_dim)

        actual_max_len = min(max_len, max(len(evs) for evs in batch_events))
        actual_max_len = max(1, actual_max_len)

        type_ids = np.zeros((B, actual_max_len), dtype=np.int64)
        num_feats = np.zeros((B, actual_max_len, 7), dtype=np.float32)
        pad_mask = np.ones((B, actual_max_len), dtype=bool)  # True = padded

        for b, evs in enumerate(batch_events):
            norm_evs = normalize_process_event_timestamps(evs)
            seq = norm_evs[-actual_max_len:] if len(norm_evs) > actual_max_len else norm_evs
            prev_time = None
            for t, ev in enumerate(seq):
                tid, nslice = self._event_to_feature_slice(ev, prev_time)
                type_ids[b, t] = tid
                num_feats[b, t] = nslice
                pad_mask[b, t] = False  # Valid token
                prev_time = float(ev.get("elapsed_seconds", 0) or 0)

        t_types = torch.from_numpy(type_ids)
        t_nums = torch.from_numpy(num_feats)
        t_pad = torch.from_numpy(pad_mask)
        t_glob = torch.from_numpy(global_features_batch).float() if global_features_batch is not None else None

        return self.model(t_types, t_nums, padding_mask=t_pad, global_features=t_glob)
