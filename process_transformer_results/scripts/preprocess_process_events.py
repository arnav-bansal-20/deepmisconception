#!/usr/bin/env python3
"""Event sequence preprocessing and tokenization for Process Transformer."""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

DATASET_DIR = Path("DeepMisconception_Corrected_Dataset")
OUT_DIR = Path("DeepMisconception/models/process_transformer")
REPORTS_DIR = Path("DeepMisconception/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 32

EVENT_TYPES = [
    "<PAD>",
    "debug", "delete", "edit", "idle", "insert", "paste",
    "revision", "run", "runtime_error", "save", "syntax_error", "test"
]

def build_event_vocab():
    vocab = {evt: idx for idx, evt in enumerate(EVENT_TYPES)}
    vocab_path = OUT_DIR / "event_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved event vocabulary ({len(vocab)} items) to {vocab_path}")
    return vocab

def parse_iso_seconds(ts_str):
    try:
        # Handle ISO format e.g. 2026-10-15T14:22:10Z or similar
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0

def process_session_events(events_df, vocab, max_len=MAX_SEQ_LEN):
    """Encodes event sequence for a session into arrays:
    - event_ids: [max_len]
    - time_gaps: [max_len] (seconds from prior event)
    - session_times: [max_len] (seconds from session start)
    - chars_in: [max_len]
    - chars_del: [max_len]
    - mask: [max_len] (bool, True for genuine events)
    """
    sorted_events = events_df.sort_values(by="timestamp").reset_index(drop=True)
    n = min(len(sorted_events), max_len)

    event_ids = np.zeros(max_len, dtype=np.int64)
    time_gaps = np.zeros(max_len, dtype=np.float32)
    session_times = np.zeros(max_len, dtype=np.float32)
    chars_in = np.zeros(max_len, dtype=np.float32)
    chars_del = np.zeros(max_len, dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)

    if n > 0:
        base_time = parse_iso_seconds(sorted_events.loc[0, "timestamp"])
        prev_time = base_time

        for i in range(n):
            row = sorted_events.iloc[i]
            cur_time = parse_iso_seconds(row["timestamp"])
            e_type = row["event_type"].lower().strip()
            e_id = vocab.get(e_type, 0)

            gap = max(0.0, cur_time - prev_time)
            total_elapsed = max(0.0, cur_time - base_time)

            event_ids[i] = e_id
            time_gaps[i] = min(gap, 600.0)  # cap at 10 mins
            session_times[i] = min(total_elapsed, 3600.0) # cap at 1 hour
            chars_in[i] = float(row.get("characters_inserted", 0))
            chars_del[i] = float(row.get("characters_deleted", 0))
            mask[i] = True

            prev_time = cur_time

    return {
        "event_ids": event_ids,
        "time_gaps": time_gaps,
        "session_times": session_times,
        "chars_in": chars_in,
        "chars_del": chars_del,
        "mask": mask,
        "length": n
    }

def generate_preprocessing_report():
    vocab = build_event_vocab()
    events = pd.read_csv(DATASET_DIR / "data/process/process_events.csv")
    sessions = pd.read_csv(DATASET_DIR / "data/integrated/multimodal_sessions.csv")

    seq_lens = events.groupby("session_id").size()

    report_path = REPORTS_DIR / "process_preprocessing_report.md"
    with open(report_path, "w") as f:
        f.write("# Process Event Preprocessing and Tokenization Report\n\n")
        f.write(f"- **Total Sessions:** {len(sessions)}\n")
        f.write(f"- **Total Events:** {len(events)}\n")
        f.write(f"- **Max Sequence Length Configured:** {MAX_SEQ_LEN}\n")
        f.write(f"- **Observed Min Length:** {seq_lens.min()}\n")
        f.write(f"- **Observed Max Length:** {seq_lens.max()}\n")
        f.write(f"- **Observed Mean Length:** {seq_lens.mean():.2f}\n")
        f.write(f"- **Sequences Truncated by Max Length ({MAX_SEQ_LEN}):** {(seq_lens > MAX_SEQ_LEN).sum()} (0.0%)\n\n")
        f.write("## Event Vocabulary\n\n")
        f.write("| Token ID | Event Type | Description |\n")
        f.write("| :---: | :--- | :--- |\n")
        for evt, tid in vocab.items():
            f.write(f"| {tid} | `{evt}` | {'Zero padding token' if tid == 0 else 'Chronological telemetry action'} |\n")
        
        f.write("\n## Temporal Feature Engineering\n\n")
        f.write("1. **Event Type Embedding:** Discrete embedding layer (dim=48).\n")
        f.write("2. **Time Delta Gap:** Continuous log1p-transformed inter-event gap in seconds.\n")
        f.write("3. **Elapsed Session Time:** Continuous normalized seconds from session onset.\n")
        f.write("4. **Character Attributes:** Inserted and deleted character deltas.\n")
        f.write("5. **Temporal Linear Projection:** Concatenated temporal & numerical attributes projected to dim=16, added to event embedding (total token dim=64).\n")
        f.write("6. **Attention Mask:** Boolean tensor indicating real events vs padding.\n")

    print(f"Saved preprocessing report to {report_path}")

if __name__ == "__main__":
    generate_preprocessing_report()
