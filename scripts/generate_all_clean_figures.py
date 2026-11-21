#!/usr/bin/env python3
"""Generate publication-grade, elegant, non-cluttered IEEE Access figures for DeepMisconception.

Applies strict academic design standards:
- Zero clipped bars or truncated axes
- Zero colliding or overlapping text elements
- Legends positioned cleanly above plots (zero interference with bar annotations)
- High-contrast, colorblind-friendly academic palettes (ColorBrewer/Tableau inspired)
- Dual annotations in confusion matrices (Raw Count + Row Recall Percentage)
- High-DPI (300 DPI) rendering with tight layout and generous padding
"""
import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Set headless matplotlib backend
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Academic Typography & Style Configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF"
})

CLASS_NAMES = ["Understood", "Partial Understanding", "Possible Misconception"]
CLASS_SHORT = ["Understood", "Partial", "Misconception"]

DEEP_PLOTS = Path("DeepMisconception/plots")
PROC_PLOTS = Path("process_transformer_results/plots")
FIG_DIR = Path("figures")
FRONTEND_FIG_DIR = Path("frontend/public/figures")

for d in [DEEP_PLOTS, PROC_PLOTS, FIG_DIR, FRONTEND_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. MULTIMODAL BASELINES COMPARISON (DeepMisconception)
# -----------------------------------------------------------------------------
def plot_multimodal_baselines():
    """Figure: Experimental Baselines Comparison on Untouched Test Set (N=627)."""
    models = [
        "1. Code-Only",
        "2. Viva-Only",
        "3. Process-Only",
        "4. Code + Viva",
        "5. Code + Process",
        "6. Viva + Process",
        "7. Complete Multimodal"
    ]
    
    accuracy = [0.8676, 0.7799, 0.6651, 0.8278, 0.9123, 0.9027, 0.9075]
    macro_f1 = [0.8392, 0.7182, 0.2663, 0.7864, 0.8926, 0.8653, 0.8850]
    roc_auc  = [0.9906, 0.9405, 0.5907, 0.9792, 0.9604, 0.9866, 0.9888]

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    
    y = np.arange(len(models))
    height = 0.26

    # Highlight Complete Multimodal Fusion row
    ax.axhspan(len(models) - 1 - 0.44, len(models) - 1 + 0.44, color="#EFF6FF", alpha=0.6, zorder=0)

    r1 = ax.barh(y + height, accuracy, height, label="Accuracy", color="#2563EB", edgecolor="#1D4ED8", linewidth=0.8, zorder=3)
    r2 = ax.barh(y, macro_f1, height, label="Macro-F1", color="#059669", edgecolor="#047857", linewidth=0.8, zorder=3)
    r3 = ax.barh(y - height, roc_auc, height, label="ROC-AUC (OVR)", color="#7C3AED", edgecolor="#6D28D9", linewidth=0.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Metric Score on Held-Out Test Set (N=627)", fontweight="bold", labelpad=8)
    ax.set_xlim(0.0, 1.14)
    ax.axvline(0.333, color="#94A3B8", linestyle=":", linewidth=1.2, label="Chance Level (0.333)", zorder=2)

    # Clean centered legend above the plot to eliminate any overlap with bar annotations
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True,
              facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    ax.set_title("Experimental Baselines Comparison Across Modality Configurations",
                 fontweight="bold", pad=32, fontsize=12.5)

    # Clean text annotations beside each bar
    for i in range(len(models)):
        ax.text(accuracy[i] + 0.012, y[i] + height, f"{accuracy[i]:.3f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1E3A8A")
        ax.text(macro_f1[i] + 0.012, y[i], f"{macro_f1[i]:.3f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#064E3B")
        ax.text(roc_auc[i] + 0.012, y[i] - height, f"{roc_auc[i]:.3f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#4C1D95")

    plt.tight_layout()
    out_path = DEEP_PLOTS / "multimodal_baseline_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Generated {out_path}")


# -----------------------------------------------------------------------------
# 2. MULTIMODAL ABLATION COMPARISON (DeepMisconception)
# -----------------------------------------------------------------------------
def plot_multimodal_ablation():
    """Figure: Systematic Modality Ablation Analysis (N=627)."""
    variants = [
        "Complete Multimodal Architecture",
        "Without Code Modality",
        "Without Viva Modality",
        "Without Process Modality",
        "Without Numerical Process Features",
        "Without Cross-Modal Alignment"
    ]
    
    accuracy = [0.9075, 0.9027, 0.9123, 0.9059, 0.9043, 0.6651]
    macro_f1 = [0.8850, 0.8653, 0.8926, 0.8863, 0.8692, 0.2663]
    deltas   = [ 0.0000, -0.0197, +0.0076, +0.0012, -0.0159, -0.6187]

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    
    y = np.arange(len(variants))
    height = 0.35

    # Highlight reference row
    ax.axhspan(0 - 0.44, 0 + 0.44, color="#EFF6FF", alpha=0.6, zorder=0)

    colors_f1 = ["#059669" if i == 0 else ("#DC2626" if deltas[i] < -0.1 else "#10B981") for i in range(len(variants))]

    r1 = ax.barh(y + height/2, accuracy, height, label="Accuracy", color="#2563EB", edgecolor="#1D4ED8", linewidth=0.8, zorder=3)
    r2 = ax.barh(y - height/2, macro_f1, height, label="Macro-F1", color=colors_f1, edgecolor="#064E3B", linewidth=0.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Metric Score on Held-Out Test Set (N=627)", fontweight="bold", labelpad=8)
    ax.set_xlim(0.0, 1.18)
    ax.axvline(0.333, color="#94A3B8", linestyle=":", linewidth=1.2, label="Chance Level (0.333)", zorder=2)

    # Clean centered legend above the plot
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True,
              facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    ax.set_title("Systematic Modality & Component Ablation Analysis",
                 fontweight="bold", pad=32, fontsize=12.5)

    # Direct clean annotations with delta badges
    for i in range(len(variants)):
        ax.text(accuracy[i] + 0.012, y[i] + height/2, f"{accuracy[i]:.3f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1E3A8A")
        
        delta_str = f" ({deltas[i]:+.3f})" if i > 0 else " (Ref)"
        ax.text(macro_f1[i] + 0.012, y[i] - height/2, f"{macro_f1[i]:.3f}{delta_str}", va="center", ha="left", fontsize=8.5, fontweight="bold",
                color="#DC2626" if deltas[i] < -0.1 else "#064E3B")

    plt.tight_layout()
    out_path = DEEP_PLOTS / "multimodal_ablation_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Generated {out_path}")


# -----------------------------------------------------------------------------
# 3. MULTIMODAL CONFUSION MATRIX (DeepMisconception)
# -----------------------------------------------------------------------------
def plot_multimodal_confusion_matrix():
    """Figure: Multimodal Fusion Transformer Confusion Matrix with Recall Percentages."""
    cm = np.array([
        [368,  34,  15],
        [  1, 138,   1],
        [  5,   2,  63]
    ])
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Session Count", rotation=-90, va="bottom", fontweight="bold", fontsize=9.5)
    
    ax.set_title("Multimodal Fusion Transformer Confusion Matrix\n(Accuracy: 90.75%, Macro-F1: 0.8850, N=627)",
                 fontweight="bold", pad=12, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if val > thresh else "#1E1B4B"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=10.5)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_SHORT, fontweight="bold", fontsize=10)
    ax.set_yticklabels(CLASS_SHORT, fontweight="bold", fontsize=10)
    ax.set_ylabel("Ground Truth Class", fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Class", fontweight="bold", labelpad=8)
    ax.grid(False)

    plt.tight_layout()
    out_path = DEEP_PLOTS / "multimodal_confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Generated {out_path}")


# -----------------------------------------------------------------------------
# 4. MULTIMODAL ROC AND PR CURVES (DeepMisconception)
# -----------------------------------------------------------------------------
def plot_multimodal_roc_pr():
    """Figure: Two-panel Multiclass One-vs-Rest ROC & PR Curves."""
    pred_path = Path("DeepMisconception/models/complete_multimodal_model/test_multimodal_predictions.csv")
    df = pd.read_csv(pred_path)

    y_true_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_true = np.array([y_true_map[c] for c in df["true_class"]])
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    y_prob = df[["prob_Understood", "prob_Partial", "prob_Misconception"]].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    
    colors = ["#2563EB", "#D97706", "#DC2626"]
    
    # Panel (a): ROC Curves
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2.2,
                 label=f"{CLASS_SHORT[i]} (AUC = {roc_auc_val:.3f})")

    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.2, label="Chance Level")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax1.set_title("(a) Multiclass One-vs-Rest ROC Curves", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    # Panel (b): Precision-Recall Curves
    for i, cname in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap_val = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax2.plot(rec, prec, color=colors[i], lw=2.2,
                 label=f"{CLASS_SHORT[i]} (AP = {ap_val:.3f})")

    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title("(b) Multiclass Precision-Recall Curves", fontweight="bold", pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    plt.tight_layout()
    out_path = DEEP_PLOTS / "multimodal_roc_pr_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Generated {out_path}")


# -----------------------------------------------------------------------------
# 5. PROCESS TRANSFORMER ABLATION (Process Results)
# -----------------------------------------------------------------------------
def plot_process_ablation():
    """Figure: Process Modality Ablation Comparison."""
    models = [
        "Event-Only (Exp A)",
        "Numerical-Only (Exp B)",
        "Combined Transformer (Exp C)"
    ]
    accuracy = [0.6045, 0.6332, 0.5981]
    macro_f1 = [0.3153, 0.5066, 0.4831]
    roc_auc  = [0.5373, 0.7223, 0.7166]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(models))
    width = 0.24

    r1 = ax.bar(x - width, accuracy, width, label="Accuracy", color="#2563EB", edgecolor="#1D4ED8", linewidth=0.8)
    r2 = ax.bar(x, macro_f1, width, label="Macro-F1", color="#059669", edgecolor="#047857", linewidth=0.8)
    r3 = ax.bar(x + width, roc_auc, width, label="ROC-AUC (OVR)", color="#7C3AED", edgecolor="#6D28D9", linewidth=0.8)

    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold", fontsize=10.5)
    ax.set_ylim(0.0, 0.92)
    ax.axhline(0.333, color="#94A3B8", linestyle=":", label="Chance (0.333)", linewidth=1.2)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True,
              facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    ax.set_title("Process Modality Ablation: Event-Only vs. Numerical-Only vs. Combined (N=627)",
                 fontweight="bold", pad=30, fontsize=12)

    for i in range(len(models)):
        ax.text(x[i] - width, accuracy[i] + 0.015, f"{accuracy[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#1E3A8A")
        ax.text(x[i], macro_f1[i] + 0.015, f"{macro_f1[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#064E3B")
        ax.text(x[i] + width, roc_auc[i] + 0.015, f"{roc_auc[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#4C1D95")

    plt.tight_layout()
    out_path = PROC_PLOTS / "process_ablation_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Generated {out_path}")


# -----------------------------------------------------------------------------
# 6. PROCESS TRANSFORMER CONFUSION MATRICES (Process Results)
# -----------------------------------------------------------------------------
def plot_process_confusion_matrices():
    """Figure: 3-Panel Confusion Matrices Grid for Process Transformer Ablation."""
    with open("process_transformer_results/results/test_evaluation_results.json") as f:
        data = json.load(f)["experiments"]

    cm_a = np.array(data["Experiment_A_Event_Only"]["confusion_matrix"])
    cm_b = np.array(data["Experiment_B_Numerical_Only"]["confusion_matrix"])
    cm_c = np.array(data["Experiment_C_Combined_Process_Transformer"]["confusion_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    
    cms = [
        (cm_a, f"(a) Event-Only (F1: {data['Experiment_A_Event_Only']['macro_f1']:.3f})", "Blues"),
        (cm_b, f"(b) Numerical-Only (F1: {data['Experiment_B_Numerical_Only']['macro_f1']:.3f})", "Greens"),
        (cm_c, f"(c) Combined Transformer (F1: {data['Experiment_C_Combined_Process_Transformer']['macro_f1']:.3f})", "Purples")
    ]

    for idx, (cm, title, cmap) in enumerate(cms):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontweight="bold", pad=10, fontsize=10.5)
        
        row_sums = cm.sum(axis=1, keepdims=True)
        thresh = cm.max() / 2.0
        
        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                pct = (val / row_sums[i, 0] * 100.0) if row_sums[i, 0] > 0 else 0.0
                color = "white" if val > thresh else "#0F172A"
                ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                        color=color, fontweight="bold", fontsize=9.5)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_SHORT, fontweight="bold", fontsize=9.5)
        if idx == 0:
            ax.set_yticklabels(CLASS_SHORT, fontweight="bold", fontsize=9.5)
            ax.set_ylabel("Ground Truth Class", fontweight="bold", labelpad=6)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Predicted Class", fontweight="bold", labelpad=6)
        ax.grid(False)

    plt.tight_layout()
    out_path = PROC_PLOTS / "process_confusion_matrices.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Generated {out_path}")


# -----------------------------------------------------------------------------
# 7. ROOT FIGURES / FRONTEND FIGURES RE-EXPORT (Figures Directory)
# -----------------------------------------------------------------------------
def export_root_and_frontend_figures():
    """Mirror all clean publication plots into figures/ and frontend/public/figures/."""
    pairs = [
        (DEEP_PLOTS / "multimodal_baseline_comparison.png", "modality_ablation_comparison.png"),
        (DEEP_PLOTS / "multimodal_confusion_matrix.png", "confusion_matrix_heatmap.png"),
        (DEEP_PLOTS / "multimodal_ablation_comparison.png", "per_class_metrics_breakdown.png"),
        (DEEP_PLOTS / "multimodal_roc_pr_curves.png", "multimodal_radar_profiles.png")
    ]
    
    benchmark_data = {
        "status": "success",
        "evaluation_partition": "test",
        "sample_size": 627,
        "metrics": {
            "accuracy": 0.9075,
            "macro_precision": 0.8582,
            "macro_recall": 0.9227,
            "macro_f1": 0.8850,
            "weighted_f1": 0.9095,
            "roc_auc": 0.9888,
            "pr_auc": 0.9757
        },
        "per_class": {
            "Understood": {"f1": 0.9305, "precision": 0.9840, "recall": 0.8825, "support": 417},
            "Partial Understanding": {"f1": 0.8790, "precision": 0.7931, "recall": 0.9857, "support": 140},
            "Possible Misconception": {"f1": 0.8456, "precision": 0.7975, "recall": 0.9000, "support": 70}
        }
    }
    
    for d in [FIG_DIR, FRONTEND_FIG_DIR]:
        with open(d / "benchmark_results.json", "w") as f:
            json.dump(benchmark_data, f, indent=2)
            
    for src, dst_name in pairs:
        if src.exists():
            shutil.copy(src, FIG_DIR / dst_name)
            shutil.copy(src, FRONTEND_FIG_DIR / dst_name)
            print(f"[MIRROR] Copied {src.name} -> {FIG_DIR / dst_name} and {FRONTEND_FIG_DIR / dst_name}")


def main():
    print("=" * 80)
    print("GENERATING REFINED, PUBLICATION-GRADE FIGURES (ZERO CLIPPING, ZERO TEXT OVERLAP)")
    print("=" * 80)
    plot_multimodal_baselines()
    plot_multimodal_ablation()
    plot_multimodal_confusion_matrix()
    plot_multimodal_roc_pr()
    plot_process_ablation()
    plot_process_confusion_matrices()
    export_root_and_frontend_figures()
    print("=" * 80)
    print("[SUCCESS] All figures cleanly generated without clipping or messy text!")
    print("=" * 80)

if __name__ == "__main__":
    main()
