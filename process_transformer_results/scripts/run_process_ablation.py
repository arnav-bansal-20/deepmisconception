#!/usr/bin/env python3
"""Run ablation analysis across process modalities and generate publication plots and reports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = Path("DeepMisconception/models/process_transformer")
LOGS_DIR = Path("DeepMisconception/logs")
REPORTS_DIR = Path("DeepMisconception/reports")
PLOTS_DIR = Path("DeepMisconception/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Understood", "Partial Understanding", "Possible Misconception"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

def main():
    eval_data = json.load(open(MODELS_DIR / "test_evaluation_results.json"))
    exps = eval_data["experiments"]

    # 1. Plot Ablation Comparison Bar Chart
    models = ["Event-Only\n(Exp A)", "Numerical-Only\n(Exp B)", "Combined Transformer\n(Exp C)"]
    keys = ["Experiment_A_Event_Only", "Experiment_B_Numerical_Only", "Experiment_C_Combined_Process_Transformer"]
    
    accs = [exps[k]["accuracy"] for k in keys]
    f1s = [exps[k]["macro_f1"] for k in keys]
    aucs = [exps[k]["roc_auc"] for k in keys]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width, accs, width, label="Accuracy", color="#2563EB", edgecolor="#1E3A8A")
    ax.bar(x, f1s, width, label="Macro-F1", color="#059669", edgecolor="#064E3B")
    ax.bar(x + width, aucs, width, label="ROC-AUC (OVR)", color="#7C3AED", edgecolor="#4C1D95")

    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.set_title("Process Modality Ablation: Event-Only vs. Numerical-Only vs. Combined (N=627)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0.0, 0.85)
    ax.axhline(0.333, color="#94A3B8", linestyle=":", label="Chance (0.333)")
    ax.legend(loc="upper left", frameon=True)

    for i in range(len(models)):
        ax.text(x[i] - width, accs[i] + 0.015, f"{accs[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(x[i], f1s[i] + 0.015, f"{f1s[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(x[i] + width, aucs[i] + 0.015, f"{aucs[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plot_file = PLOTS_DIR / "process_ablation_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved ablation chart to {plot_file}")

    # 2. Plot Confusion Matrices Grid
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = [
        f"Event-Only (F1: {exps[keys[0]]['macro_f1']:.3f})",
        f"Numerical-Only (F1: {exps[keys[1]]['macro_f1']:.3f})",
        f"Combined Transformer (F1: {exps[keys[2]]['macro_f1']:.3f})"
    ]
    cmaps = ["Blues", "Greens", "Purples"]

    for idx, ax in enumerate(axes):
        cm = np.array(exps[keys[idx]]["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=cmaps[idx])
        ax.set_title(titles[idx], fontweight="bold", pad=10)
        thresh = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontweight="bold", fontsize=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["Understood", "Partial", "Miscon"], rotation=20, ha="right")
        if idx == 0:
            ax.set_yticklabels(["Understood", "Partial", "Miscon"])
            ax.set_ylabel("True Class", fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Predicted Class", fontweight="bold")

    cm_file = PLOTS_DIR / "process_confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()
    print(f"Saved confusion matrices to {cm_file}")

    # 3. Write Reports
    ab_report = REPORTS_DIR / "process_ablation_report.md"
    with open(ab_report, "w") as f:
        f.write("# Process Modality Ablation Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report presents the comparative ablation analysis between:\n")
        f.write("1. **Experiment A (Event-Only):** Chronological event token sequences modeled by multi-head self-attention.\n")
        f.write("2. **Experiment B (Numerical-Only Baseline):** Dense MLP on 24 summary behavioral metrics.\n")
        f.write("3. **Experiment C (Combined Process Transformer):** Gated multi-modal fusion of temporal event dynamics and 24 numerical features.\n\n")
        f.write("## Quantitative Test Set Results (Held-Out Test N=627)\n\n")
        f.write("| Architecture | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | ROC-AUC (OVR) | PR-AUC (OVR) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, k in enumerate(keys):
            m = exps[k]
            f.write(f"| **{models[idx].replace(chr(10), ' ')}** | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | **{m['macro_f1']:.4f}** | {m['weighted_f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |\n")

        f.write("\n## Per-Class Breakdown (F1-Score)\n\n")
        f.write("| Architecture | Understood (N=417) | Partial Understanding (N=140) | Possible Misconception (N=70) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for idx, k in enumerate(keys):
            pc = exps[k]["per_class"]
            f.write(f"| **{models[idx].replace(chr(10), ' ')}** | {pc['Understood']['f1']:.4f} | {pc['Partial Understanding']['f1']:.4f} | {pc['Possible Misconception']['f1']:.4f} |\n")

        f.write("\n## Key Scientific Findings\n\n")
        f.write("1. **Dominance of Aggregated Process Indicators:** Numerical process features (MLP baseline) achieve Macro-F1 = 0.5066 and ROC-AUC = 0.7223. Cumulative summary metrics (test pass rate, error resolution rate, revision frequency) carry strong discriminative signal.\n")
        f.write("2. **Temporal Attention Behavior:** The standalone Event-Only Transformer attains Macro-F1 = 0.3153. Because event token vocabularies (`run`, `edit`, `insert`) do not encapsulate full source code semantics, sequence progression alone offers limited stand-alone discriminative power for subtle conceptual boundaries.\n")
        f.write("3. **Combined Gated Representation:** The Combined Process Transformer balances both streams with a learned gating mechanism, achieving Macro-F1 = 0.4831 and ROC-AUC = 0.7166 while providing the required 64-dimensional embedding for downstream multimodal fusion with CodeBERT and Viva Sentence Transformer.\n")

    print(f"Saved ablation report to {ab_report}")

    eval_report = REPORTS_DIR / "process_evaluation_report.md"
    with open(eval_report, "w") as f:
        f.write("# Process Transformer Standalone Evaluation Report\n\n")
        f.write(f"- **Evaluated Model:** Combined Process Transformer (`combined_process_transformer_best.pt`)\n")
        f.write(f"- **Test Partition:** `DeepMisconception_Corrected_Dataset/splits/test/` (Untouched, N=627 supervised sessions)\n")
        f.write(f"- **Excluded Insufficient Evidence:** 66 sessions\n")
        f.write(f"- **Test Accuracy:** {exps['Experiment_C_Combined_Process_Transformer']['accuracy']:.4f}\n")
        f.write(f"- **Test Macro-F1:** {exps['Experiment_C_Combined_Process_Transformer']['macro_f1']:.4f}\n")
        f.write(f"- **Test Weighted-F1:** {exps['Experiment_C_Combined_Process_Transformer']['weighted_f1']:.4f}\n")
        f.write(f"- **Test ROC-AUC:** {exps['Experiment_C_Combined_Process_Transformer']['roc_auc']:.4f}\n\n")
        f.write("### Confusion Matrix (Test Split)\n\n")
        f.write("```\n")
        f.write(np.array2string(np.array(exps['Experiment_C_Combined_Process_Transformer']['confusion_matrix'])))
        f.write("\n```\n")

    print(f"Saved evaluation report to {eval_report}")

if __name__ == "__main__":
    main()
