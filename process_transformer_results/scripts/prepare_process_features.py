#!/usr/bin/env python3
"""Prepare, validate, and compute training-only normalization for the exact 24 numerical process features."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

DATASET_DIR = Path("DeepMisconception_Corrected_Dataset")
OUT_DIR = Path("DeepMisconception/models/process_transformer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXACT_24_FEATURES = [
    "n_events", "duration_s", "n_insert", "n_edit", "n_delete", "n_paste",
    "typed_chars", "paste_chars", "paste_char_ratio", "largest_insert_share",
    "n_runs", "n_syntax_errors", "n_runtime_errors", "n_errors", "n_debug_actions",
    "n_tests", "n_test_pass", "n_revisions", "n_errors_resolved", "idle_total_s",
    "max_idle_s", "mean_gap_s", "time_to_first_run_s", "final_state_pass"
]

def compute_training_normalization():
    sessions = pd.read_csv(DATASET_DIR / "data/integrated/multimodal_sessions.csv")
    features = pd.read_csv(DATASET_DIR / "data/process/process_features_24.csv")

    # Align by session_id
    df = sessions[["session_id", "partition", "evidence_status"]].merge(features, on="session_id")
    
    # Strictly training partition ONLY
    train_df = df[df["partition"] == "train"]
    print(f"Total training sessions for feature normalization: {len(train_df)}")

    means = {}
    stds = {}

    for col in EXACT_24_FEATURES:
        vals = train_df[col].astype(float).values
        m = float(np.mean(vals))
        s = float(np.std(vals))
        if s < 1e-6:
            s = 1.0  # Avoid zero division
        means[col] = m
        stds[col] = s

    stats = {
        "feature_names": EXACT_24_FEATURES,
        "means": means,
        "stds": stds,
        "num_features": len(EXACT_24_FEATURES),
        "source_partition": "train",
        "train_sample_count": len(train_df)
    }

    out_file = OUT_DIR / "normalization_stats.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved training-only normalization stats to {out_file}")
    return stats

def normalize_features(features_array: np.ndarray, stats: dict) -> np.ndarray:
    means = np.array([stats["means"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
    stds = np.array([stats["stds"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
    return (features_array - means) / stds

if __name__ == "__main__":
    compute_training_normalization()
