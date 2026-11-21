# Systematic Modality Ablation Report

## 1. Modality Removal Analysis

| Ablation Variant | Accuracy | Macro-F1 | Macro-F1 Delta | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Complete Multimodal Architecture** | 0.9075 | **0.8850** | +0.0000 | 0.9888 |
| **Without Code Modality** | 0.9027 | **0.8653** | -0.0197 | 0.9866 |
| **Without Viva Modality** | 0.9123 | **0.8926** | +0.0076 | 0.9604 |
| **Without Process Modality** | 0.9059 | **0.8863** | +0.0012 | 0.9895 |
| **Without Numerical Process Features** | 0.9043 | **0.8692** | -0.0159 | 0.9893 |
| **Without Cross-Modal Alignment** | 0.6651 | **0.2663** | -0.6187 | 0.6638 |
