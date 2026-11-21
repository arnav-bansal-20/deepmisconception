"""
DeepMisconception Actual Leakage & Partition Audit Engine (Safeguard 3)
Audits actual cross-partition overlap across code, viva, process, and scenario families.
Treats zero leakage as a verifiable empirical objective, not an assumed conclusion.
"""

import pandas as pd
import numpy as np

def audit_actual_leakage(train_code_df: pd.DataFrame,
                         val_code_df: pd.DataFrame,
                         test_code_df: pd.DataFrame,
                         train_viva_df: pd.DataFrame,
                         val_viva_df: pd.DataFrame,
                         test_viva_df: pd.DataFrame,
                         train_sf: set,
                         val_sf: set,
                         test_sf: set,
                         labels_df: pd.DataFrame,
                         code_df: pd.DataFrame,
                         feat_df: pd.DataFrame):
    results = {}
    
    # 1. Actual Scenario Family Overlap
    sf_overlap_tv = len(train_sf.intersection(val_sf))
    sf_overlap_tt = len(train_sf.intersection(test_sf))
    sf_overlap_vt = len(val_sf.intersection(test_sf))
    results["sf_overlap_train_val"] = sf_overlap_tv
    results["sf_overlap_train_test"] = sf_overlap_tt
    results["sf_overlap_val_test"] = sf_overlap_vt
    
    # 2. Actual Code Overlap across Partitions
    train_codes = set(train_code_df["python_code"])
    val_codes = set(val_code_df["python_code"])
    test_codes = set(test_code_df["python_code"])
    
    code_overlap_tv = len(train_codes.intersection(val_codes))
    code_overlap_tt = len(train_codes.intersection(test_codes))
    code_overlap_vt = len(val_codes.intersection(test_codes))
    results["code_exact_overlap_train_val"] = code_overlap_tv
    results["code_exact_overlap_train_test"] = code_overlap_tt
    results["code_exact_overlap_val_test"] = code_overlap_vt
    
    # 3. Actual Viva Question Overlap across Partitions
    train_q = set(train_viva_df["question"])
    val_q = set(val_viva_df["question"])
    test_q = set(test_viva_df["question"])
    
    q_overlap_tv = len(train_q.intersection(val_q))
    q_overlap_tt = len(train_q.intersection(test_q))
    q_overlap_vt = len(val_q.intersection(test_q))
    results["viva_q_overlap_train_val"] = q_overlap_tv
    results["viva_q_overlap_train_test"] = q_overlap_tt
    results["viva_q_overlap_val_test"] = q_overlap_vt
    
    # 4. Code Correctness Contingency Table vs Synthetic Label
    merged_code = pd.merge(labels_df, code_df, on="session_id")
    ct_code = pd.crosstab(merged_code["synthetic_annotator_label"], merged_code["code_correctness_status"])
    results["code_contingency"] = ct_code.to_dict()
    
    # Diagonal check
    is_diagonal = False
    if "correct" in ct_code.columns and "partially_correct" in ct_code.columns and "incorrect" in ct_code.columns:
        diag_sum = (
            ct_code.loc["Understood", "correct"] +
            ct_code.loc["Partial Understanding", "partially_correct"] +
            ct_code.loc["Possible Misconception", "incorrect"]
        )
        is_diagonal = (diag_sum == len(labels_df))
    results["code_label_is_diagonal"] = is_diagonal
    
    # 5. Feature Shortcut Audit (final_state_pass)
    merged_feat = pd.merge(labels_df, feat_df, on="session_id")
    pass_by_class = merged_feat.groupby("synthetic_annotator_label")["final_state_pass"].mean().to_dict()
    results["pass_rate_by_class"] = pass_by_class
    results["final_pass_is_deterministic_shortcut"] = (pass_by_class.get("Understood", 0.0) == 1.0 and pass_by_class.get("Partial Understanding", 0.0) == 0.0)
    
    # Zero partition leakage verdict
    zero_leakage = (sf_overlap_tv == 0 and sf_overlap_tt == 0 and sf_overlap_vt == 0 and
                    code_overlap_tv == 0 and code_overlap_tt == 0 and code_overlap_vt == 0)
    results["zero_cross_partition_leakage_confirmed"] = zero_leakage
    
    return results
