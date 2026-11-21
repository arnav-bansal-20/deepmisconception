#!/usr/bin/env python3
"""Process Transformer architecture, PyTorch Dataset, and utilities for DeepMisconception."""
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

EXACT_24_FEATURES = [
    "n_events", "duration_s", "n_insert", "n_edit", "n_delete", "n_paste",
    "typed_chars", "paste_chars", "paste_char_ratio", "largest_insert_share",
    "n_runs", "n_syntax_errors", "n_runtime_errors", "n_errors", "n_debug_actions",
    "n_tests", "n_test_pass", "n_revisions", "n_errors_resolved", "idle_total_s",
    "max_idle_s", "mean_gap_s", "time_to_first_run_s", "final_state_pass"
]

CLASS_TO_IDX = {
    "Understood": 0,
    "Partial Understanding": 1,
    "Possible Misconception": 2
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class ProcessDataset(Dataset):
    def __init__(self, partition="train", dataset_dir="DeepMisconception_Corrected_Dataset",
                 max_seq_len=32, norm_stats=None, vocab=None, filter_sufficient_evidence=True):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_len = max_seq_len
        self.partition = partition

        # Load sessions
        sessions = pd.read_csv(self.dataset_dir / "data/integrated/multimodal_sessions.csv")
        part_sessions = sessions[sessions["partition"] == partition].copy()

        # Exclude insufficient evidence from supervised training/eval
        if filter_sufficient_evidence:
            self.sessions = part_sessions[part_sessions["evidence_status"] == "Sufficient Evidence"].copy()
            self.excluded_count = len(part_sessions) - len(self.sessions)
        else:
            self.sessions = part_sessions.copy()
            self.excluded_count = 0

        self.session_ids = self.sessions["session_id"].tolist()
        self.labels = [CLASS_TO_IDX[lbl] for lbl in self.sessions["ground_truth_label"]]

        # Load 24 numerical features
        features_df = pd.read_csv(self.dataset_dir / "data/process/process_features_24.csv")
        features_df = features_df.set_index("session_id").reindex(self.session_ids)
        raw_feats = features_df[EXACT_24_FEATURES].astype(float).values

        if norm_stats is not None:
            means = np.array([norm_stats["means"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
            stds = np.array([norm_stats["stds"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
            self.num_features = (raw_feats - means) / stds
        else:
            self.num_features = raw_feats.astype(np.float32)

        # Load events
        events_df = pd.read_csv(self.dataset_dir / "data/process/process_events.csv")
        events_df = events_df[events_df["session_id"].isin(set(self.session_ids))].copy()
        self.events_grouped = {sid: grp for sid, grp in events_df.groupby("session_id")}

        self.vocab = vocab or {}

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        sid = self.session_ids[idx]
        label = self.labels[idx]
        num_f = torch.tensor(self.num_features[idx], dtype=torch.float32)

        # Parse event sequence
        evts = self.events_grouped.get(sid, pd.DataFrame())
        evts_sorted = evts.sort_values(by="timestamp").reset_index(drop=True)
        n = min(len(evts_sorted), self.max_seq_len)

        event_ids = np.zeros(self.max_seq_len, dtype=np.int64)
        time_feats = np.zeros((self.max_seq_len, 4), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=bool)

        if n > 0:
            from datetime import datetime
            def parse_s(t_str):
                try:
                    return datetime.fromisoformat(t_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    return 0.0

            base_t = parse_s(evts_sorted.loc[0, "timestamp"])
            prev_t = base_t

            for i in range(n):
                row = evts_sorted.iloc[i]
                cur_t = parse_s(row["timestamp"])
                e_id = self.vocab.get(row["event_type"].lower().strip(), 0)
                gap = max(0.0, cur_t - prev_t)
                elapsed = max(0.0, cur_t - base_t)

                event_ids[i] = e_id
                time_feats[i, 0] = math.log1p(min(gap, 600.0))
                time_feats[i, 1] = math.log1p(min(elapsed, 3600.0))
                time_feats[i, 2] = math.log1p(float(row.get("characters_inserted", 0)))
                time_feats[i, 3] = math.log1p(float(row.get("characters_deleted", 0)))
                mask[i] = True

                prev_t = cur_t

        return {
            "session_id": sid,
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
            "time_features": torch.tensor(time_feats, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "num_features": num_f,
            "label": torch.tensor(label, dtype=torch.long)
        }


# -----------------------------------------------------------------------------
# Model Architectures
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ProcessTemporalEncoder(nn.Module):
    def __init__(self, vocab_size=13, event_embed_dim=48, time_dim=16,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.15):
        super().__init__()
        self.event_embed = nn.Embedding(vocab_size, event_embed_dim, padding_idx=0)
        self.time_proj = nn.Linear(4, time_dim)
        self.input_proj = nn.Linear(event_embed_dim + time_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, event_ids, time_features, mask):
        # Embeddings
        e_emb = self.event_embed(event_ids)
        t_emb = self.time_proj(time_features)
        x = torch.cat([e_emb, t_emb], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Key padding mask: True for positions to be IGNORED
        padding_mask = ~mask
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        out = self.layer_norm(out)

        # Masked average pooling
        mask_expanded = mask.unsqueeze(-1).float()
        sum_out = (out * mask_expanded).sum(dim=1)
        count_out = mask_expanded.sum(dim=1).clamp(min=1.0)
        pooled = sum_out / count_out
        return pooled


class NumericalFeatureBranch(nn.Module):
    def __init__(self, in_features=24, hidden_dim=64, out_dim=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


# Experiment A: Event-only Process Transformer
class EventOnlyProcessTransformer(nn.Module):
    def __init__(self, vocab_size=13, d_model=64, num_classes=3):
        super().__init__()
        self.temporal_encoder = ProcessTemporalEncoder(vocab_size=vocab_size, d_model=d_model)
        self.output_dim = d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, event_ids, time_features, mask, num_features=None):
        temporal_repr = self.temporal_encoder(event_ids, time_features, mask)
        logits = self.classifier(temporal_repr)
        return {
            "logits": logits,
            "process_embedding": temporal_repr
        }


# Experiment B: Numerical-only Baseline MLP
class NumericalProcessBaseline(nn.Module):
    def __init__(self, in_features=24, hidden_dim=64, out_dim=64, num_classes=3, dropout=0.15):
        super().__init__()
        self.num_branch = NumericalFeatureBranch(in_features, hidden_dim, out_dim, dropout)
        self.output_dim = out_dim
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, num_features, event_ids=None, time_features=None, mask=None):
        num_repr = self.num_branch(num_features)
        logits = self.classifier(num_repr)
        return {
            "logits": logits,
            "process_embedding": num_repr
        }


# Experiment C: Combined Process Transformer (Official Architecture)
class CombinedProcessTransformer(nn.Module):
    def __init__(self, vocab_size=13, in_features=24, d_model=64,
                 num_classes=3, dropout=0.15):
        super().__init__()
        self.temporal_encoder = ProcessTemporalEncoder(vocab_size=vocab_size, d_model=d_model, dropout=dropout)
        self.num_branch = NumericalFeatureBranch(in_features=in_features, hidden_dim=d_model, out_dim=d_model, dropout=dropout)

        # Learned Gated Fusion
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.output_dim = d_model  # exactly 64
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, event_ids, time_features, mask, num_features):
        temporal_repr = self.temporal_encoder(event_ids, time_features, mask)
        num_repr = self.num_branch(num_features)

        # Gated fusion
        concat = torch.cat([temporal_repr, num_repr], dim=-1)
        gate = self.gate_fc(concat)
        fused = gate * temporal_repr + (1.0 - gate) * num_repr
        process_embedding = self.fusion_norm(fused)

        logits = self.classifier(process_embedding)
        return {
            "logits": logits,
            "process_embedding": process_embedding,
            "gate": gate
        }
