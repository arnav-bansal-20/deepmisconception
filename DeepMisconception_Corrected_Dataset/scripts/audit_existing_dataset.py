"""
DeepMisconception Baseline Audit Script
Audits the existing dataset in DeepMisconception_Multimodal_Dataset/
and reproduces all baseline metrics, sequence diversity counts, and leakage indicators.
"""

import os
import pandas as pd
import numpy as np

def run_baseline_audit(base_dir: str):
    code_path = os.path.join(base_dir, "data", "code", "code_submissions.csv")
    viva_path = os.path.join(base_dir, "data", "viva", "viva_questions_answers.csv")
    events_path = os.path.join(base_dir, "data", "process", "process_events.csv")
    feat_path = os.path.join(base_dir, "data", "process", "process_features_24.csv")
    labels_path = os.path.join(base_dir, "data", "labels", "session_labels.csv")
    
    code_df = pd.read_csv(code_path)
    viva_df = pd.read_csv(viva_path)
    events_df = pd.read_csv(events_path)
    feat_df = pd.read_csv(feat_path)
    labels_df = pd.read_csv(labels_path)
    
    # 1. Total sessions and classes
    n_sessions = len(labels_df)
    class_dist = labels_df["ground_truth_label"].value_counts().to_dict()
    evidence_dist = labels_df["evidence_status"].value_counts().to_dict()
    
    # 2. Process sequences
    seqs = events_df.groupby("session_id")["event_type"].apply(tuple)
    n_unique_seqs = seqs.nunique()
    top_seqs = seqs.value_counts().head(5).to_dict()
    
    # 3. Viva diversity
    n_unique_q = viva_df["question"].nunique()
    n_unique_a = viva_df["student_answer"].nunique()
    
    # 4. Code correctness vs Label
    merged_code = pd.merge(labels_df, code_df, on="session_id")
    ct_code = pd.crosstab(merged_code["ground_truth_label"], merged_code["code_correctness_status"]).to_dict()
    
    # 5. final_state_pass vs Label
    merged_feat = pd.merge(labels_df, feat_df, on="session_id")
    ct_pass = pd.crosstab(merged_feat["ground_truth_label"], merged_feat["final_state_pass"]).to_dict()
    
    return {
        "n_sessions": n_sessions,
        "class_dist": class_dist,
        "evidence_dist": evidence_dist,
        "n_events": len(events_df),
        "n_unique_seqs": n_unique_seqs,
        "top_seqs": top_seqs,
        "n_viva_turns": len(viva_df),
        "n_unique_q": n_unique_q,
        "n_unique_a": n_unique_a,
        "code_vs_label_crosstab": ct_code,
        "pass_vs_label_crosstab": ct_pass
    }

if __name__ == "__main__":
    orig_dir = "/Users/arnavbansal/Desktop/datasets/DeepMisconception_Multimodal_Dataset"
    res = run_baseline_audit(orig_dir)
    print("Baseline Audit Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
