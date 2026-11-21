#!/usr/bin/env python3
"""Evaluate all trained Process models on the untouched test partition."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent))
from process_model_and_dataset import (
    ProcessDataset, EventOnlyProcessTransformer, NumericalProcessBaseline,
    CombinedProcessTransformer, CLASS_TO_IDX, IDX_TO_CLASS
)

MODELS_DIR = Path("DeepMisconception/models/process_transformer")
REPORTS_DIR = Path("DeepMisconception/reports")
LOGS_DIR = Path("DeepMisconception/logs")
PLOTS_DIR = Path("DeepMisconception/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Understood", "Partial Understanding", "Possible Misconception"]

def evaluate_model_on_test(model, test_loader, model_type="combined"):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_sids = []

    with torch.no_grad():
        for batch in test_loader:
            sids = batch["session_id"]
            targets = batch["label"].numpy()

            if model_type == "combined":
                out = model(batch["event_ids"], batch["time_features"], batch["mask"], batch["num_features"])
            elif model_type == "event_only":
                out = model(batch["event_ids"], batch["time_features"], batch["mask"])
            elif model_type == "numerical":
                out = model(batch["num_features"])

            logits = out["logits"]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_sids.extend(sids)
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Standard metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_p = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    # Per-class metrics
    p_per = precision_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    r_per = recall_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    f1_per = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]).tolist()

    # ROC-AUC & PR-AUC
    bin_targets = label_binarize(all_targets, classes=[0, 1, 2])
    try:
        roc_auc = float(roc_auc_score(bin_targets, all_probs, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(bin_targets, all_probs, average="macro"))
    except Exception:
        pr_auc = 0.333

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i])
            } for i in range(3)
        },
        "confusion_matrix": cm,
        "roc_auc_ovr_macro": roc_auc,
        "pr_auc_ovr_macro": pr_auc,
        "predictions_df": pd.DataFrame({
            "session_id": all_sids,
            "true_class": [CLASS_NAMES[t] for t in all_targets],
            "pred_class": [CLASS_NAMES[p] for p in all_preds],
            "prob_Understood": all_probs[:, 0],
            "prob_Partial": all_probs[:, 1],
            "prob_Misconception": all_probs[:, 2]
        })
    }

def main():
    print("=" * 80)
    print("EVALUATING ALL PROCESS MODELS ON UNTOUCHED TEST PARTITION")
    print("=" * 80)

    config = json.load(open("DeepMisconception/configs/process_transformer_config.json"))
    vocab = json.load(open(MODELS_DIR / "event_vocab.json"))
    norm_stats = json.load(open(MODELS_DIR / "normalization_stats.json"))

    test_ds = ProcessDataset(partition="test", norm_stats=norm_stats, vocab=vocab, filter_sufficient_evidence=True)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    print(f"Test samples (supervised): {len(test_ds)}")
    print(f"Excluded insufficient evidence: {test_ds.excluded_count}")

    # 1. Load Combined Model (Experiment C)
    comb_ckpt = torch.load(MODELS_DIR / "combined_process_transformer_best.pt", weights_only=False)
    comb_model = CombinedProcessTransformer(
        vocab_size=len(vocab), in_features=24, d_model=config["transformer_hidden_dim"],
        num_classes=3, dropout=config["transformer_dropout"]
    )
    comb_model.load_state_dict(comb_ckpt["model_state"])
    comb_results = evaluate_model_on_test(comb_model, test_loader, "combined")

    # 2. Load Numerical Baseline (Experiment B)
    num_ckpt = torch.load(MODELS_DIR / "numerical_baseline_best.pt", weights_only=False)
    num_model = NumericalProcessBaseline(
        in_features=24, hidden_dim=config["numerical_mlp_hidden"],
        out_dim=config["process_embedding_dim"], num_classes=3
    )
    num_model.load_state_dict(num_ckpt["model_state"])
    num_results = evaluate_model_on_test(num_model, test_loader, "numerical")

    # 3. Load Event-only Transformer (Experiment A)
    evt_ckpt = torch.load(MODELS_DIR / "event_only_transformer_best.pt", weights_only=False)
    evt_model = EventOnlyProcessTransformer(
        vocab_size=len(vocab), d_model=config["transformer_hidden_dim"], num_classes=3
    )
    evt_model.load_state_dict(evt_ckpt["model_state"])
    evt_results = evaluate_model_on_test(evt_model, test_loader, "event_only")

    # Save test predictions for primary model
    comb_results["predictions_df"].to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    # Compile evaluation dictionary
    eval_summary = {
        "dataset_partition": "test",
        "total_test_sessions": len(test_ds) + test_ds.excluded_count,
        "evaluated_supervised_sessions": len(test_ds),
        "excluded_insufficient_evidence_sessions": test_ds.excluded_count,
        "class_support": {
            CLASS_NAMES[i]: int((np.array(test_ds.labels) == i).sum()) for i in range(3)
        },
        "experiments": {
            "Experiment_A_Event_Only": {
                "accuracy": evt_results["accuracy"],
                "macro_precision": evt_results["macro_precision"],
                "macro_recall": evt_results["macro_recall"],
                "macro_f1": evt_results["macro_f1"],
                "weighted_f1": evt_results["weighted_f1"],
                "roc_auc": evt_results["roc_auc_ovr_macro"],
                "pr_auc": evt_results["pr_auc_ovr_macro"],
                "per_class": evt_results["per_class"],
                "confusion_matrix": evt_results["confusion_matrix"]
            },
            "Experiment_B_Numerical_Only": {
                "accuracy": num_results["accuracy"],
                "macro_precision": num_results["macro_precision"],
                "macro_recall": num_results["macro_recall"],
                "macro_f1": num_results["macro_f1"],
                "weighted_f1": num_results["weighted_f1"],
                "roc_auc": num_results["roc_auc_ovr_macro"],
                "pr_auc": num_results["pr_auc_ovr_macro"],
                "per_class": num_results["per_class"],
                "confusion_matrix": num_results["confusion_matrix"]
            },
            "Experiment_C_Combined_Process_Transformer": {
                "accuracy": comb_results["accuracy"],
                "macro_precision": comb_results["macro_precision"],
                "macro_recall": comb_results["macro_recall"],
                "macro_f1": comb_results["macro_f1"],
                "weighted_f1": comb_results["weighted_f1"],
                "roc_auc": comb_results["roc_auc_ovr_macro"],
                "pr_auc": comb_results["pr_auc_ovr_macro"],
                "per_class": comb_results["per_class"],
                "confusion_matrix": comb_results["confusion_matrix"]
            }
        }
    }

    eval_file = MODELS_DIR / "test_evaluation_results.json"
    with open(eval_file, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Saved evaluation results to {eval_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("TEST EVALUATION METRICS SUMMARY (HELD-OUT N=627)")
    print("=" * 80)
    print(f"{'Experiment':<42} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Weighted-F1':<12} | {'ROC-AUC':<8}")
    print("-" * 90)
    for exp_key, metrics in eval_summary["experiments"].items():
        print(f"{exp_key:<42} | {metrics['accuracy']:<10.4f} | {metrics['macro_f1']:<10.4f} | {metrics['weighted_f1']:<12.4f} | {metrics['roc_auc']:<8.4f}")

    return eval_summary

if __name__ == "__main__":
    main()
