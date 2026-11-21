#!/usr/bin/env python3
"""Generate all 5 required academic evaluation graphs for DeepMisconception.

Graphs:
1. overall_performance_metrics.png (Graph 1: 0-100% bar chart)
2. classwise_performance_comparison.png (Graph 2: Grouped bar chart comparing 3 classes)
3. confusion_matrix_heatmap.png (Graph 3: Professional 3x3 heatmap with raw counts and %)
4. five_fold_cv_performance.png (Graph 4: Fold-by-fold comparison with Mean lines)
5. five_fold_cv_stability.png (Graph 5: Mean +/- Std stability error bars)

Applies strict IEEE Access publication design rules:
- Zero clipping, zero colliding annotations
- Professional muted academic palettes
- High-DPI (300 DPI) rendering
"""
import os
import json
from pathlib import Path
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF"
})

OUTPUT_DIRS = [
    Path("figures/evaluation"),
    Path("DeepMisconception/plots"),
    Path("frontend/public/figures/evaluation")
]
for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def save_to_all(fig, filename):
    for d in OUTPUT_DIRS:
        fig.savefig(d / filename)
    plt.close(fig)
    print(f"[EVAL FIG] Saved {filename}")


# -----------------------------------------------------------------------------
# GRAPH 1: Overall Performance Metrics (0-100% Scale)
# -----------------------------------------------------------------------------
def plot_graph1_overall_performance(test_metrics):
    m = test_metrics["metrics"]
    labels = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Macro Sensitivity",
        "Macro Specificity"
    ]
    values = [
        m["accuracy"] * 100.0,
        m["precision_macro"] * 100.0,
        m["recall_macro"] * 100.0,
        m["f1_macro"] * 100.0,
        m["sensitivity_macro"] * 100.0,
        m["specificity_macro"] * 100.0
    ]
    colors = ["#2563EB", "#0284C7", "#059669", "#10B981", "#7C3AED", "#4F46E5"]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="#1E293B", width=0.55, linewidth=0.8, zorder=3)

    ax.set_ylabel("Metric Score (%)", fontweight="bold", labelpad=8)
    ax.set_title("Graph 1: Overall Model Performance Metrics on Untouched Test Set (N=627)",
                 fontweight="bold", pad=14, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.axhline(33.33, color="#94A3B8", linestyle=":", linewidth=1.2, label="Chance Level (33.33%)", zorder=2)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.8, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0F172A")

    plt.tight_layout()
    save_to_all(fig, "overall_performance_metrics.png")


# -----------------------------------------------------------------------------
# GRAPH 2: Class-wise Performance Comparison
# -----------------------------------------------------------------------------
def plot_graph2_classwise_comparison(test_metrics):
    pc = test_metrics["per_class"]
    classes = ["Understood", "Partial Understanding", "Possible Misconception"]
    metrics = ["Precision", "Recall", "F1 Score", "Sensitivity", "Specificity"]

    data = {
        m: [pc[c][m.lower().replace(" score", "")] * 100.0 for c in classes]
        for m in metrics
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(classes))
    width = 0.15
    colors = ["#2563EB", "#059669", "#7C3AED", "#D97706", "#475569"]

    for i, m in enumerate(metrics):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, data[m], width, label=m, color=colors[i],
                      edgecolor="#1E293B", linewidth=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E293B")

    ax.set_ylabel("Metric Score (%)", fontweight="bold", labelpad=8)
    ax.set_title("Graph 2: Class-wise Performance Profiles Across Diagnostic Dimensions",
                 fontweight="bold", pad=28, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(N={pc[c]['support']})" for c in classes], fontweight="bold", fontsize=10)
    ax.set_ylim(0, 118)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=True,
              facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9)

    plt.tight_layout()
    save_to_all(fig, "classwise_performance_comparison.png")


# -----------------------------------------------------------------------------
# GRAPH 3: Confusion Matrix Heatmap
# -----------------------------------------------------------------------------
def plot_graph3_confusion_matrix(test_metrics):
    cm = np.array(test_metrics["confusion_matrix"]["raw_counts"])
    pct = np.array(test_metrics["confusion_matrix"]["normalized_percentages"])
    short_classes = ["Understood", "Partial", "Misconception"]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Session Count", rotation=-90, va="bottom", fontweight="bold", fontsize=9.5)

    ax.set_title("Graph 3: Multimodal Fusion Transformer Confusion Matrix Heatmap\n"
                 f"(Accuracy: {test_metrics['metrics']['accuracy']*100:.2f}%, Macro-F1: {test_metrics['metrics']['f1_macro']:.4f}, N=627)",
                 fontweight="bold", pad=12, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            p = pct[i, j]
            color = "white" if val > thresh else "#1E1B4B"
            ax.text(j, i, f"{val}\n({p:.1f}%)", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=10.5)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(short_classes, fontweight="bold", fontsize=10)
    ax.set_yticklabels(short_classes, fontweight="bold", fontsize=10)
    ax.set_ylabel("Ground Truth Class", fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Class", fontweight="bold", labelpad=8)
    ax.grid(False)

    plt.tight_layout()
    save_to_all(fig, "confusion_matrix_heatmap.png")


# -----------------------------------------------------------------------------
# GRAPH 4: Five-Fold Cross-Validation Performance (Fold-by-Fold Comparison)
# -----------------------------------------------------------------------------
def plot_graph4_five_fold_comparison(cv_data):
    folds = [f"Fold {i+1}" for i in range(5)]
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    colors = ["#2563EB", "#0284C7", "#059669", "#7C3AED"]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(folds))
    width = 0.18

    for i, (m, k) in enumerate(zip(metrics, metric_keys)):
        vals = [f["metrics"][k] * 100.0 for f in cv_data["individual_folds"]]
        mean_val = cv_data["aggregate_macro_metrics"][k]["mean"] * 100.0
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"{m} (Mean: {mean_val:.1f}%)",
                      color=colors[i], edgecolor="#1E293B", linewidth=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E293B")

    ax.set_ylabel("Validation Score (%)", fontweight="bold", labelpad=8)
    ax.set_title("Graph 4: Stratified Group-Aware 5-Fold Cross-Validation Performance Across Folds",
                 fontweight="bold", pad=28, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontweight="bold", fontsize=10)
    ax.set_ylim(60, 110)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=True,
              facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9)

    plt.tight_layout()
    save_to_all(fig, "five_fold_cv_performance.png")


# -----------------------------------------------------------------------------
# GRAPH 5: Cross-Validation Stability (Mean +/- Std Error Bars)
# -----------------------------------------------------------------------------
def plot_graph5_cv_stability(cv_data):
    agg = cv_data["aggregate_macro_metrics"]
    labels = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Macro Sensitivity",
        "Macro Specificity"
    ]
    keys = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "sensitivity_macro", "specificity_macro"
    ]

    means = [agg[k]["mean"] * 100.0 for k in keys]
    stds = [agg[k]["std"] * 100.0 for k in keys]
    colors = ["#2563EB", "#0284C7", "#059669", "#10B981", "#7C3AED", "#4F46E5"]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="#1E293B",
                  width=0.52, linewidth=0.8, alpha=0.92, zorder=3,
                  error_kw=dict(ecolor="#0F172A", lw=1.5, capthick=1.5))

    ax.set_ylabel("Metric Score (Mean ± Std %)", fontweight="bold", labelpad=8)
    ax.set_title("Graph 5: Cross-Validation Generalization Stability Across Disjoint Scenario Families",
                 fontweight="bold", pad=14, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=9.5)
    ax.set_ylim(60, 108)

    for bar, m_val, s_val in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m_val + s_val + 1.8,
                f"{m_val:.1f}±{s_val:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")

    plt.tight_layout()
    save_to_all(fig, "five_fold_cv_stability.png")


def main():
    print("=" * 80)
    print("GENERATING ALL 5 REQUIRED EVALUATION GRAPHS")
    print("=" * 80)
    with open("backend/app/ml/checkpoints/test_evaluation_metrics.json") as f:
        test_metrics = json.load(f)
    with open("backend/app/ml/checkpoints/five_fold_cv_results.json") as f:
        cv_data = json.load(f)

    plot_graph1_overall_performance(test_metrics)
    plot_graph2_classwise_comparison(test_metrics)
    plot_graph3_confusion_matrix(test_metrics)
    plot_graph4_five_fold_comparison(cv_data)
    plot_graph5_cv_stability(cv_data)
    print("=" * 80)
    print("[SUCCESS] All 5 graphs successfully generated and mirrored!")
    print("=" * 80)


if __name__ == "__main__":
    main()
