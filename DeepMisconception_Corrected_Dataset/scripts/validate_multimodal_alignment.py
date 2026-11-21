"""
DeepMisconception Multimodal Alignment Validation Engine (Corrected)
Verifies 100% referential integrity, cross-modal coherence, and temporal consistency
across Code, Viva, Process Events, Numerical Features, and Labels.
"""

import pandas as pd
from datetime import datetime

def validate_multimodal_alignment(code_df: pd.DataFrame,
                                  viva_df: pd.DataFrame,
                                  events_df: pd.DataFrame,
                                  feat_df: pd.DataFrame,
                                  labels_df: pd.DataFrame,
                                  integrated_df: pd.DataFrame):
    results = {}
    
    # 1. Session IDs consistency
    s_labels = set(labels_df["session_id"])
    s_code = set(code_df["session_id"])
    s_viva = set(viva_df["session_id"])
    s_events = set(events_df["session_id"])
    s_feat = set(feat_df["session_id"])
    s_integ = set(integrated_df["session_id"])
    
    results["all_session_ids_match"] = (
        len(s_labels) == 5000 and
        s_labels == s_code and
        s_labels == s_viva and
        s_labels == s_events and
        s_labels == s_feat and
        s_labels == s_integ
    )
    
    # 2. Viva question count per session (3-5)
    viva_counts = viva_df.groupby("session_id").size()
    results["viva_counts_valid"] = bool((viva_counts >= 3).all() and (viva_counts <= 5).all())
    
    # 3. Process events timestamp monotonicity
    events_sorted = events_df.sort_values(by=["session_id", "timestamp"])
    mono_valid = True
    for sid, group in events_df.groupby("session_id"):
        ts_list = [datetime.fromisoformat(t.replace("Z", "")) for t in group["timestamp"]]
        for i in range(1, len(ts_list)):
            if ts_list[i] < ts_list[i-1]:
                mono_valid = False
                break
        if not mono_valid:
            break
    results["timestamps_monotonic"] = mono_valid
    
    # 4. Feature constraint verification
    c1 = (feat_df["n_errors"] == feat_df["n_syntax_errors"] + feat_df["n_runtime_errors"]).all()
    c2 = (feat_df["n_test_pass"] <= feat_df["n_tests"]).all()
    c3 = (feat_df["n_errors_resolved"] <= feat_df["n_errors"]).all()
    c4 = ((feat_df["paste_char_ratio"] >= 0.0) & (feat_df["paste_char_ratio"] <= 1.0)).all()
    c5 = ((feat_df["largest_insert_share"] >= 0.0) & (feat_df["largest_insert_share"] <= 1.0)).all()
    c6 = feat_df["final_state_pass"].isin([0, 1]).all()
    results["all_feature_constraints_met"] = bool(c1 and c2 and c3 and c4 and c5 and c6)
    
    # 5. Provenance field presence
    results["label_provenance_valid"] = (labels_df["label_provenance"] == "synthetic_pedagogical_rule").all()
    
    results["all_alignment_checks_passed"] = all(results.values())
    return results
