# Multimodal Fusion Transformer Independent Evaluation Report

## 1. Evaluation Protocol

- **Evaluation Partition:** Untouched `test/` partition (N=627 supervised sessions; 66 insufficient evidence excluded)
- **Disjoint Scenario Families:** 9 families (0 overlap with train/val)
- **Ground Truth:** `ground_truth_label` (Synthetic pedagogical proxy label)

## 2. Primary Test Metrics

- **Accuracy:** 0.9075
- **Macro Precision:** 0.8582
- **Macro Recall:** 0.9227
- **Macro-F1 Score:** **0.8850**
- **Weighted-F1 Score:** 0.9095
- **Multiclass ROC-AUC (OVR):** 0.9888
- **Multiclass PR-AUC (OVR):** 0.9757

## 3. Per-Class Diagnostic Breakdown

| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Understood** | 417 | 0.9840 | 0.8825 | **0.9305** |
| **Partial Understanding** | 140 | 0.7931 | 0.9857 | **0.8790** |
| **Possible Misconception** | 70 | 0.7975 | 0.9000 | **0.8456** |

## 4. Confusion Matrix (Test Split)

```
[[368  34  15]
 [  1 138   1]
 [  5   2  63]]
```
