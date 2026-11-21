# Experimental Baselines Comparison Report

## 1. Experimental Conditions

All 7 configurations were evaluated on the identical held-out test split ($N=627$).

| Configuration | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Code-Only** | 0.8676 | 0.8016 | 0.9083 | **0.8392** | 0.8725 | 0.9906 |
| **2. Viva-Only** | 0.7799 | 0.6963 | 0.8098 | **0.7182** | 0.7995 | 0.9405 |
| **3. Process-Only** | 0.6651 | 0.2217 | 0.3333 | **0.2663** | 0.5313 | 0.5907 |
| **4. Code + Viva** | 0.8278 | 0.7495 | 0.8733 | **0.7864** | 0.8380 | 0.9792 |
| **5. Code + Process** | 0.9123 | 0.9171 | 0.8737 | **0.8926** | 0.9123 | 0.9604 |
| **6. Viva + Process** | 0.9027 | 0.8342 | 0.9140 | **0.8653** | 0.9070 | 0.9866 |
| **7. Complete Multimodal Fusion** | 0.9075 | 0.8582 | 0.9227 | **0.8850** | 0.9095 | 0.9888 |
