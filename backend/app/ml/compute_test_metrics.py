#!/usr/bin/env python3
"""Compute all 9 comprehensive evaluation metrics on untouched test split (N=627)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Understood", "Partial Understanding", "Possible Misconception"]

def main():
    pred_path = Path("DeepMisconception/models/complete_multimodal_model/test_multimodal_predictions.csv")
    df = pd.read_csv(pred_path)
    
    y_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_true = np.array([y_map[c] for c in df["true_class"]])
    y_pred = np.array([y_map[c] for c in df["pred_class"]])
    y_probs = df[["prob_Understood", "prob_Partial", "prob_Misconception"]].values

    acc = float(accuracy_score(y_true, y_pred))
    cls_err = float((1.0 - acc) * 100.0)

    # Precision
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    # Recall
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # F1 Score
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total_samples = len(y_true)

    # Sensitivity and Specificity (One-vs-Rest)
    sens_per = []
    spec_per = []

    for k in range(3):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total_samples - tp - fn - fp

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        sens_per.append(sens)
        spec_per.append(spec)

    sens_macro = float(np.mean(sens_per))
    spec_macro = float(np.mean(spec_per))

    bin_y = label_binarize(y_true, classes=[0, 1, 2])
    roc_auc = float(roc_auc_score(bin_y, y_probs, average="macro", multi_class="ovr"))
    pr_auc = float(average_precision_score(bin_y, y_probs, average="macro"))

    test_metrics = {
        "status": "success",
        "dataset_partition": "test",
        "total_test_sessions": 693,
        "evaluated_supervised_sessions": 627,
        "excluded_insufficient_evidence_sessions": 66,
        "metrics": {
            "accuracy": acc,
            "classification_error_pct": cls_err,
            "precision_macro": p_macro,
            "precision_weighted": p_weighted,
            "recall_macro": r_macro,
            "recall_weighted": r_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "sensitivity_macro": sens_macro,
            "specificity_macro": spec_macro,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        },
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "sensitivity": float(sens_per[i]),
                "specificity": float(spec_per[i]),
                "support": int(cm[i, :].sum())
            } for i in range(3)
        },
        "confusion_matrix": {
            "raw_counts": cm.tolist(),
            "normalized_percentages": (cm / cm.sum(axis=1, keepdims=True) * 100.0).round(2).tolist(),
            "classes": CLASS_NAMES
        }
    }

    out_paths = [
        Path("backend/app/ml/checkpoints/test_evaluation_metrics.json"),
        Path("DeepMisconception/reports/test_evaluation_metrics.json"),
        Path("figures/evaluation/test_evaluation_metrics.json")
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved test metrics to {p}")

    print("\nOVERALL TEST PERFORMANCE (N=627):")
    for k, v in test_metrics["metrics"].items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")

if __name__ == "__main__":
    main()
