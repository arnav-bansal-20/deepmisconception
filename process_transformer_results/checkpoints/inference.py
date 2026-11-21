"""Reusable inference API for DeepMisconception Process Transformer."""
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EXACT_24_FEATURES = [
    "n_events", "duration_s", "n_insert", "n_edit", "n_delete", "n_paste",
    "typed_chars", "paste_chars", "paste_char_ratio", "largest_insert_share",
    "n_runs", "n_syntax_errors", "n_runtime_errors", "n_errors", "n_debug_actions",
    "n_tests", "n_test_pass", "n_revisions", "n_errors_resolved", "idle_total_s",
    "max_idle_s", "mean_gap_s", "time_to_first_run_s", "final_state_pass"
]

class ProcessTransformerPredictor:
    def __init__(self, model_dir=None, device="cpu"):
        self.device = torch.device(device)
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent

        # 1. Load vocabulary and normalization stats
        with open(self.model_dir / "event_vocab.json") as f:
            self.vocab = json.load(f)

        with open(self.model_dir / "normalization_stats.json") as f:
            self.norm_stats = json.load(f)

        with open(self.model_dir / "model_config.json") as f:
            self.config = json.load(f)

        # 2. Build model architecture
        from process_model_and_dataset import CombinedProcessTransformer
        self.model = CombinedProcessTransformer(
            vocab_size=len(self.vocab),
            in_features=24,
            d_model=self.config["transformer_hidden_dim"],
            num_classes=self.config["num_classes"],
            dropout=0.0
        )

        ckpt_path = self.model_dir / "combined_process_transformer_best.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_events(self, events_list, max_seq_len=32):
        """Converts raw list of event dicts or DataFrame to model tensors."""
        if isinstance(events_list, pd.DataFrame):
            events = events_list.to_dict("records")
        else:
            events = list(events_list)

        n = min(len(events), max_seq_len)
        event_ids = np.zeros(max_seq_len, dtype=np.int64)
        time_feats = np.zeros((max_seq_len, 4), dtype=np.float32)
        mask = np.zeros(max_seq_len, dtype=bool)

        if n > 0:
            def parse_s(t_str):
                try:
                    return datetime.fromisoformat(str(t_str).replace("Z", "+00:00")).timestamp()
                except Exception:
                    return 0.0

            base_t = parse_s(events[0].get("timestamp", ""))
            prev_t = base_t

            for i in range(n):
                ev = events[i]
                cur_t = parse_s(ev.get("timestamp", ""))
                e_type = str(ev.get("event_type", "")).lower().strip()
                e_id = self.vocab.get(e_type, 0)
                gap = max(0.0, cur_t - prev_t)
                elapsed = max(0.0, cur_t - base_t)

                event_ids[i] = e_id
                time_feats[i, 0] = math.log1p(min(gap, 600.0))
                time_feats[i, 1] = math.log1p(min(elapsed, 3600.0))
                time_feats[i, 2] = math.log1p(float(ev.get("characters_inserted", 0)))
                time_feats[i, 3] = math.log1p(float(ev.get("characters_deleted", 0)))
                mask[i] = True

                prev_t = cur_t

        return {
            "event_ids": torch.tensor(event_ids, dtype=torch.long).unsqueeze(0).to(self.device),
            "time_features": torch.tensor(time_feats, dtype=torch.float32).unsqueeze(0).to(self.device),
            "mask": torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        }

    def preprocess_features(self, features_dict):
        """Validates 24 features and normalizes using training statistics."""
        raw_vals = []
        for feat in EXACT_24_FEATURES:
            val = float(features_dict.get(feat, 0.0))
            raw_vals.append(val)
        raw_arr = np.array(raw_vals, dtype=np.float32)

        means = np.array([self.norm_stats["means"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
        stds = np.array([self.norm_stats["stds"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
        norm_arr = (raw_arr - means) / stds

        return torch.tensor(norm_arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def extract_process_embedding(self, process_events, process_features_24):
        """Primary extraction function returning [batch_size, 64] process embedding tensor."""
        evt_tensors = self.preprocess_events(process_events)
        feat_tensor = self.preprocess_features(process_features_24)

        with torch.no_grad():
            out = self.model(
                evt_tensors["event_ids"],
                evt_tensors["time_features"],
                evt_tensors["mask"],
                feat_tensor
            )
            embedding = out["process_embedding"]
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)

        return {
            "process_embedding": embedding.cpu(),  # [1, 64]
            "logits": logits.cpu(),               # [1, 3]
            "class_probabilities": probs.cpu()    # [1, 3]
        }

_default_predictor = None

def extract_process_embedding(process_events, process_features_24, model_dir=None):
    global _default_predictor
    if _default_predictor is None or model_dir is not None:
        _default_predictor = ProcessTransformerPredictor(model_dir=model_dir)
    res = _default_predictor.extract_process_embedding(process_events, process_features_24)
    return res["process_embedding"]
