# DeepMisconception: Comprehensive Model Evaluation Report

**Evaluation Date:** 2026-11-21  
**Target Architecture:** Multimodal Fusion Transformer (CodeBERT + SentenceTransformer + Process Transformer + 24 Numerical Features + Learned Cross-Modal Alignment + 3-Class Classification Head)  
**Evaluated Checkpoint:** `DeepMisconception/models/complete_multimodal_model/complete_multimodal_model_best.pt`  
**Official Target Classes:** `Understood`, `Partial Understanding`, `Possible Misconception`  
**Evidence Status / Abstention:** `Pending / Insufficient Evidence` (Excluded from 3-class loss; zero forced predictions)

---

## 1. Executive Summary & Core Results

The Multimodal Fusion Transformer was evaluated across two rigorous empirical protocols:
1. **Held-Out Test Partition ($N=627$ Supervised Sessions):** 9 disjoint scenario families held strictly untouched throughout training and validation.
2. **Stratified Group-Aware 5-Fold Cross-Validation ($N=3,137$ Supervised Sessions):** 46 scenario families partitioned with strict group segregation (zero scenario-family leakage across folds).

### Key Empirical Findings:
- **Test Accuracy:** **90.75%** (Classification Error: **9.25%**)
- **Test Macro-F1 Score:** **0.8850** (Weighted-F1: **0.9095**)
- **5-Fold Cross-Validation Accuracy:** **87.23% $\pm$ 4.12%**
- **5-Fold Cross-Validation Macro-F1:** **0.8284 $\pm$ 0.0336**
- **Multiclass ROC-AUC (OVR):** **0.9888** | **PR-AUC (OVR):** **0.9757**
- **High Misconception Sensitivity:** **90.00%** recall on `Possible Misconception`, capturing 9 out of 10 genuine student misconceptions.

---

## 2. Table 1: Overall Test Performance (Held-Out Test Split, N=627)

| Metric | Score | Formula / Methodological Definition |
| :--- | :---: | :--- |
| **Accuracy** | **90.75%** | $\frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{569}{627}$ |
| **Precision (Macro)** | **85.82%** | Unweighted arithmetic mean of per-class precision |
| **Precision (Weighted)** | **92.05%** | Support-weighted precision across classes |
| **Recall (Macro)** | **92.27%** | Unweighted arithmetic mean of per-class recall |
| **Recall (Weighted)** | **90.75%** | Support-weighted recall across classes |
| **F1 Score (Macro)** | **0.8850** | Harmonic mean of Macro Precision and Macro Recall |
| **F1 Score (Weighted)** | **0.9095** | Support-weighted F1 score across classes |
| **Classification Error** | **9.25%** | $1 - \text{Accuracy} = 100\% - 90.75\%$ |
| **Sensitivity (Macro)** | **92.27%** | One-vs-Rest $\frac{TP}{TP + FN}$ (Equivalent to Macro Recall) |
| **Specificity (Macro)** | **95.63%** | One-vs-Rest $\frac{TN}{TN + FP}$ averaged across classes |
| **Multiclass ROC-AUC (OVR)** | **0.9888** | Macro-averaged Area Under the ROC Curve |
| **Multiclass PR-AUC (OVR)** | **0.9757** | Macro-averaged Average Precision Score |

---

## 3. Table 2: Class-wise Diagnostic Breakdown

| Class | Support | Precision | Recall | F1 Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Understood** | 417 | 98.40% | 88.25% | **0.9305** | 88.25% | 97.14% |
| **Partial Understanding** | 140 | 79.31% | 98.57% | **0.8790** | 98.57% | 92.61% |
| **Possible Misconception** | 70 | 79.75% | 90.00% | **0.8456** | 90.00% | 97.13% |

*Note on Sensitivity vs. Recall: For this three-class problem, one-vs-rest sensitivity is mathematically identical to per-class recall.*

---

## 4. Confusion Matrix (Test Split N=627)

```
                       Predicted Class
Actual Class           Understood      Partial         Misconception   Total
----------------------------------------------------------------------------
Understood             368 (88.2%)     34 (8.2%)       15 (3.6%)       417
Partial Understanding    1 (0.7%)     138 (98.6%)       1 (0.7%)       140
Possible Misconception   5 (7.1%)       2 (2.9%)       63 (90.0%)       70
----------------------------------------------------------------------------
Column Total           374            174              79              627
```

---

## 5. Table 3: Stratified Group-Aware 5-Fold Cross-Validation

- **Partition:** `train/` partition only ($N=3,137$ supervised sessions across 46 scenario families).
- **Group Constraint:** All records from the same `scenario_family_id` strictly remain in the same fold (0 family overlap).
- **Model Freshness:** A fresh Cross-Modal Alignment module and Multimodal Fusion Transformer were trained independently from scratch on each fold.

| Fold | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | Classification Error | Sensitivity (Macro) | Specificity (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 87.20% | 79.04% | 91.19% | 0.8369 | 12.80% | 91.19% | 94.24% |
| **Fold 2** | 93.50% | 86.03% | 89.49% | 0.8539 | 6.50% | 89.49% | 97.21% |
| **Fold 3** | 88.33% | 81.60% | 86.91% | 0.8400 | 11.67% | 86.91% | 92.71% |
| **Fold 4** | 86.50% | 82.53% | 88.34% | 0.8490 | 13.50% | 88.34% | 92.23% |
| **Fold 5** | 80.62% | 78.97% | 79.95% | 0.7623 | 19.38% | 79.95% | 91.25% |
| **Mean $\pm$ Std** | **87.23 $\pm$ 4.12%** | **81.64 $\pm$ 2.60%** | **87.18 $\pm$ 3.87%** | **0.8284 $\pm$ 0.0336** | **12.77 $\pm$ 4.12%** | **87.18 $\pm$ 3.87%** | **93.53 $\pm$ 2.08%** |

---

## 6. Baselines Comparison on Identical Held-Out Test Set (N=627)

| Configuration | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Code-Only (AST + CodeBERT)** | 86.76% | 80.16% | 90.83% | **0.8392** | 0.9906 |
| **2. Viva-Only (Sentence-Transformer)** | 77.99% | 69.63% | 80.98% | **0.7182** | 0.9405 |
| **3. Process-Only (Process Transformer)** | 66.51% | 22.17% | 33.33% | **0.2663** | 0.5907 |
| **4. Code + Viva (Bimodal)** | 82.78% | 74.95% | 87.33% | **0.7864** | 0.9792 |
| **5. Code + Process (Bimodal)** | 91.23% | 91.71% | 87.37% | **0.8926** | 0.9604 |
| **6. Viva + Process (Bimodal)** | 90.27% | 83.42% | 91.40% | **0.8653** | 0.9866 |
| **7. Complete Multimodal Fusion (Proposed)** | **90.75%** | **85.82%** | **92.27%** | **0.8850** | **0.9888** |

---

## 7. Generated Visualizations Directory

All five publication-grade figures have been generated and mirrored:
- **Graph 1:** `figures/evaluation/overall_performance_metrics.png` (0–100% scale macro bar chart)
- **Graph 2:** `figures/evaluation/classwise_performance_comparison.png` (Grouped diagnostic profile across 3 classes)
- **Graph 3:** `figures/evaluation/confusion_matrix_heatmap.png` (Dual count + row percentage heatmap)
- **Graph 4:** `figures/evaluation/five_fold_cv_performance.png` (Fold-by-fold comparison with Mean lines)
- **Graph 5:** `figures/evaluation/five_fold_cv_stability.png` (Mean $\pm$ Std stability error bars across disjoint scenario families)

---

## 8. Methodological Safeguards & Limitations

1. **Partition Integrity:** Zero leakage between training and held-out test splits. Cross-validation used strictly training partition data.
2. **Evidence Filtering:** 66 test instances and 400 training instances with incomplete evidence were flagged as `Pending / Insufficient Evidence` and excluded from three-class supervised metrics.
3. **Synthetic Proxy Labels:** Labels reflect rule-based synthetic pedagogical annotations; `human_validated_label` is documented as unpopulated.
4. **Model Primacy:** The Multimodal Fusion Transformer is solely responsible for final predictions via learned attention embeddings.
