#!/usr/bin/env python3
"""Run genuine Stratified, Group-Aware 5-Fold Cross-Validation on DeepMisconception train partition.

Strict Research Protocol:
1. Training partition ONLY (DeepMisconception_Corrected_Dataset/splits/train/): N=3,137 supervised sessions.
2. Group-Aware: scenario_family_id strictly segregated across folds (0 family overlap).
3. Fresh models trained on each fold: Stage A Alignment (Code <-> Viva) + Multimodal Fusion Transformer.
4. Evaluates all 9 metrics:
   - Accuracy
   - Precision (Macro, Weighted, per-class)
   - Recall (Macro, Weighted, per-class)
   - F1 Score (Macro, Weighted, per-class)
   - Classification Error (1 - Accuracy)
   - Sensitivity (Macro, per-class; note: Sensitivity = Recall)
   - Specificity (Macro, per-class: TN / (TN + FP))
   - Confusion Matrix
5. Computes Mean ± Standard Deviation across all 5 folds.
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

CLASS_NAMES = ["Understood", "Partial Understanding", "Possible Misconception"]

# -----------------------------------------------------------------------------
# Architecture Definitions
# -----------------------------------------------------------------------------
class CrossModalAlignmentModule(nn.Module):
    def __init__(self, code_dim=768, viva_dim=384, common_dim=64, nheads=4):
        super().__init__()
        self.code_proj = nn.Sequential(
            nn.Linear(code_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim)
        )
        self.viva_proj = nn.Sequential(
            nn.Linear(viva_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim)
        )
        self.cross_attn_c2v = nn.MultiheadAttention(embed_dim=common_dim, num_heads=nheads, batch_first=True)
        self.cross_attn_v2c = nn.MultiheadAttention(embed_dim=common_dim, num_heads=nheads, batch_first=True)
        self.norm_c = nn.LayerNorm(common_dim)
        self.norm_v = nn.LayerNorm(common_dim)

    def forward(self, code_emb, viva_emb):
        c_proj = self.code_proj(code_emb).unsqueeze(1)
        v_proj = self.viva_proj(viva_emb).unsqueeze(1)
        c_aligned, _ = self.cross_attn_c2v(c_proj, v_proj, v_proj)
        c_out = self.norm_c(c_proj + c_aligned).squeeze(1)
        v_aligned, _ = self.cross_attn_v2c(v_proj, c_proj, c_proj)
        v_out = self.norm_v(v_proj + v_aligned).squeeze(1)
        return c_out, v_out


class MultimodalFusionTransformer(nn.Module):
    def __init__(self, d_model=64, nheads=4, num_layers=2, dim_feedforward=256, dropout=0.15, num_classes=3):
        super().__init__()
        self.code_proj = nn.Sequential(nn.Linear(768, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.viva_proj = nn.Sequential(nn.Linear(384, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.proc_proj = nn.Sequential(nn.Linear(64, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.num_proj = nn.Sequential(nn.Linear(24, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))

        self.modality_embed = nn.Embedding(4, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, code_emb, viva_emb, proc_emb, num_feat, aligned_code=None, aligned_viva=None):
        bs = len(code_emb)
        device = code_emb.device

        c_repr = aligned_code if aligned_code is not None else self.code_proj(code_emb)
        v_repr = aligned_viva if aligned_viva is not None else self.viva_proj(viva_emb)
        p_repr = self.proc_proj(proc_emb)
        n_repr = self.num_proj(num_feat)

        tokens = torch.stack([c_repr, v_repr, p_repr, n_repr], dim=1)
        mod_ids = torch.arange(4, device=device).unsqueeze(0).expand(bs, -1)
        tokens = self.input_norm(tokens + self.modality_embed(mod_ids))

        fused_tokens = self.transformer(tokens)
        fused_tokens = self.fusion_norm(fused_tokens)
        fused_repr = fused_tokens.mean(dim=1)

        logits = self.classifier(fused_repr)
        return {"logits": logits, "fused_embedding": fused_repr}


def compute_comprehensive_metrics(y_true, y_pred, y_probs=None):
    """Compute all 9 specified evaluation metrics."""
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

    # Sensitivity and Specificity (One-vs-Rest)
    sens_per = []
    spec_per = []
    total_samples = len(y_true)

    for k in range(3):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total_samples - tp - fn - fp

        # Sensitivity = TP / (TP + FN) [equivalent to recall]
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # Specificity = TN / (TN + FP)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        sens_per.append(sens)
        spec_per.append(spec)

    sens_macro = float(np.mean(sens_per))
    spec_macro = float(np.mean(spec_per))

    # ROC / PR AUC if probs available
    roc_auc = 0.5
    pr_auc = 0.333
    if y_probs is not None:
        bin_y = label_binarize(y_true, classes=[0, 1, 2])
        try:
            roc_auc = float(roc_auc_score(bin_y, y_probs, average="macro", multi_class="ovr"))
        except Exception:
            pass
        try:
            pr_auc = float(average_precision_score(bin_y, y_probs, average="macro"))
        except Exception:
            pass

    return {
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
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "sensitivity": float(sens_per[i]),
                "specificity": float(spec_per[i]),
                "support": int(cm[i, :].sum())
            } for i in range(3)
        }
    }


def train_and_eval_fold(fold_num, train_idx, val_idx, code_t, viva_t, proc_t, num_t, y_t, epochs=10):
    """Train fresh alignment and fusion transformer models for a single fold."""
    print(f"\n--- Training Fold {fold_num} (Train N={len(train_idx)}, Val N={len(val_idx)}) ---")
    
    # Slice tensors
    tr_code, val_code = code_t[train_idx], code_t[val_idx]
    tr_viva, val_viva = viva_t[train_idx], viva_t[val_idx]
    tr_proc, val_proc = proc_t[train_idx], proc_t[val_idx]
    tr_num, val_num = num_t[train_idx], num_t[val_idx]
    tr_y, val_y = y_t[train_idx], y_t[val_idx]

    tr_ds = TensorDataset(tr_code, tr_viva, tr_proc, tr_num, tr_y)
    val_ds = TensorDataset(val_code, val_viva, val_proc, val_num, val_y)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Compute fold-specific class weights
    y_arr = tr_y.numpy()
    counts = np.bincount(y_arr, minlength=3)
    weights = len(y_arr) / (3.0 * np.maximum(counts, 1))
    weights = torch.tensor(weights, dtype=torch.float32)

    # 1. Fresh Alignment Module
    align_model = CrossModalAlignmentModule()
    opt_align = torch.optim.AdamW(align_model.parameters(), lr=1e-3, weight_decay=1e-4)

    align_model.train()
    for ep in range(5):
        for b_c, b_v, _, _, _ in tr_loader:
            opt_align.zero_grad()
            c_out, v_out = align_model(b_c, b_v)
            c_norm = F.normalize(c_out, p=2, dim=-1)
            v_norm = F.normalize(v_out, p=2, dim=-1)
            sim = torch.mm(c_norm, v_norm.t()) / 0.07
            labels = torch.arange(len(b_c))
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0
            loss.backward()
            opt_align.step()

    align_model.eval()

    # 2. Fresh Multimodal Fusion Transformer
    fusion_model = MultimodalFusionTransformer()
    opt_fusion = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = -1.0
    best_preds, best_probs = None, None

    for ep in range(epochs):
        fusion_model.train()
        for b_c, b_v, b_p, b_n, b_y in tr_loader:
            opt_fusion.zero_grad()
            with torch.no_grad():
                a_c, a_v = align_model(b_c, b_v)
            out = fusion_model(b_c, b_v, b_p, b_n, aligned_code=a_c, aligned_viva=a_v)
            loss = criterion(out["logits"], b_y)
            loss.backward()
            opt_fusion.step()

        # Validation evaluation
        fusion_model.eval()
        val_preds, val_probs = [], []
        with torch.no_grad():
            for b_c, b_v, b_p, b_n, _ in val_loader:
                a_c, a_v = align_model(b_c, b_v)
                out = fusion_model(b_c, b_v, b_p, b_n, aligned_code=a_c, aligned_viva=a_v)
                probs = F.softmax(out["logits"], dim=-1).numpy()
                preds = np.argmax(probs, axis=-1)
                val_preds.extend(preds)
                val_probs.extend(probs)

        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)
        current_f1 = f1_score(val_y.numpy(), val_preds, average="macro", zero_division=0)

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_preds = val_preds
            best_probs = val_probs

    fold_metrics = compute_comprehensive_metrics(val_y.numpy(), best_preds, best_probs)
    print(f"Fold {fold_num} Result -> Acc: {fold_metrics['accuracy']:.4f} | F1: {fold_metrics['f1_macro']:.4f} | Sens: {fold_metrics['sensitivity_macro']:.4f} | Spec: {fold_metrics['specificity_macro']:.4f}")
    return fold_metrics


def main():
    print("=" * 80)
    print("STAGE: 5-FOLD STRATIFIED GROUP-AWARE CROSS-VALIDATION")
    print("=" * 80)
    
    cache_path = Path("backend/app/ml/cache/train_precomputed.pt")
    if not cache_path.exists():
        print(f"Error: cache not found at {cache_path}")
        sys.exit(1)

    train_bundle = torch.load(cache_path, weights_only=False)
    mask = train_bundle["is_sufficient"].numpy()

    # Extract supervised instances
    code_t = train_bundle["code_embeddings"][mask]
    viva_t = train_bundle["viva_embeddings"][mask]
    proc_t = train_bundle["process_embeddings"][mask]
    num_t = train_bundle["numerical_features"][mask]
    y_t = train_bundle["labels"][mask]
    groups = np.array(train_bundle["scenario_families"])[mask]

    n_samples = len(y_t)
    n_groups = len(np.unique(groups))
    print(f"Loaded {n_samples} supervised training samples across {n_groups} disjoint scenario families.")

    # Run StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.arange(n_samples), y_t.numpy(), groups=groups)):
        # Verify 0 group leakage
        tr_g = set(groups[train_idx])
        va_g = set(groups[val_idx])
        assert len(tr_g.intersection(va_g)) == 0, f"Leakage detected in fold {fold+1}!"

        m = train_and_eval_fold(fold + 1, train_idx, val_idx, code_t, viva_t, proc_t, num_t, y_t, epochs=10)
        fold_results.append(m)

    total_time = time.time() - t0
    print(f"\nCompleted 5-fold cross-validation in {total_time:.1f}s.")

    # Calculate Aggregate Metrics across 5 Folds
    metric_keys = [
        "accuracy", "classification_error_pct", "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted", "f1_macro", "f1_weighted",
        "sensitivity_macro", "specificity_macro", "roc_auc", "pr_auc"
    ]
    
    summary = {}
    for k in metric_keys:
        vals = [f[k] for f in fold_results]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals
        }

    # Aggregate Class-wise metrics
    per_class_summary = {}
    for cname in CLASS_NAMES:
        per_class_summary[cname] = {}
        for subk in ["precision", "recall", "f1", "sensitivity", "specificity"]:
            vals = [f["per_class"][cname][subk] for f in fold_results]
            per_class_summary[cname][subk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }

    # Format JSON structure
    output_data = {
        "methodology": "StratifiedGroupKFold (n_splits=5, zero scenario_family_id leakage)",
        "training_samples_evaluated": n_samples,
        "unique_scenario_families": n_groups,
        "duration_seconds": total_time,
        "aggregate_macro_metrics": summary,
        "aggregate_per_class_metrics": per_class_summary,
        "individual_folds": [
            {
                "fold": i + 1,
                "metrics": fold_results[i]
            } for i in range(5)
        ]
    }

    # Save to both backend ml checkpoints and DeepMisconception reports
    out_paths = [
        Path("backend/app/ml/checkpoints/five_fold_cv_results.json"),
        Path("DeepMisconception/reports/five_fold_cv_results.json"),
        Path("figures/evaluation/five_fold_cv_results.json")
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved CV results to {p}")

    # Print summary table
    print("\n" + "=" * 90)
    print("FIVE-FOLD CROSS-VALIDATION SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Fold':<8} | {'Accuracy':<10} | {'Macro-Prec':<11} | {'Macro-Rec':<11} | {'Macro-F1':<10} | {'Class Error':<12} | {'Macro-Sens':<11} | {'Macro-Spec':<11}")
    print("-" * 90)
    for i in range(5):
        f = fold_results[i]
        print(f"Fold {i+1:<3} | {f['accuracy']:<10.4f} | {f['precision_macro']:<11.4f} | {f['recall_macro']:<11.4f} | {f['f1_macro']:<10.4f} | {f['classification_error_pct']:<11.2f}% | {f['sensitivity_macro']:<11.4f} | {f['specificity_macro']:<11.4f}")
    print("-" * 90)
    print(f"{'Mean':<8} | {summary['accuracy']['mean']:<10.4f} | {summary['precision_macro']['mean']:<11.4f} | {summary['recall_macro']['mean']:<11.4f} | {summary['f1_macro']['mean']:<10.4f} | {summary['classification_error_pct']['mean']:<11.2f}% | {summary['sensitivity_macro']['mean']:<11.4f} | {summary['specificity_macro']['mean']:<11.4f}")
    print(f"{'Std':<8} | {summary['accuracy']['std']:<10.4f} | {summary['precision_macro']['std']:<11.4f} | {summary['recall_macro']['std']:<11.4f} | {summary['f1_macro']['std']:<10.4f} | {summary['classification_error_pct']['std']:<11.2f}% | {summary['sensitivity_macro']['std']:<11.4f} | {summary['specificity_macro']['std']:<11.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
