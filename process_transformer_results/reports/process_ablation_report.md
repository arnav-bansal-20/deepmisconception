# Process Modality Ablation Analysis Report

## Overview

This report presents the comparative ablation analysis between:
1. **Experiment A (Event-Only):** Chronological event token sequences modeled by multi-head self-attention.
2. **Experiment B (Numerical-Only Baseline):** Dense MLP on 24 summary behavioral metrics.
3. **Experiment C (Combined Process Transformer):** Gated multi-modal fusion of temporal event dynamics and 24 numerical features.

## Quantitative Test Set Results (Held-Out Test N=627)

| Architecture | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | ROC-AUC (OVR) | PR-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Event-Only (Exp A)** | 0.6045 | 0.3033 | 0.3393 | **0.3153** | 0.5430 | 0.5373 | 0.3575 |
| **Numerical-Only (Exp B)** | 0.6332 | 0.5030 | 0.5248 | **0.5066** | 0.6502 | 0.7223 | 0.4902 |
| **Combined Transformer (Exp C)** | 0.5981 | 0.4906 | 0.5056 | **0.4831** | 0.6274 | 0.7166 | 0.4843 |

## Per-Class Breakdown (F1-Score)

| Architecture | Understood (N=417) | Partial Understanding (N=140) | Possible Misconception (N=70) |
| :--- | :---: | :---: | :---: |
| **Event-Only (Exp A)** | 0.7511 | 0.1949 | 0.0000 |
| **Numerical-Only (Exp B)** | 0.7797 | 0.4388 | 0.3011 |
| **Combined Transformer (Exp C)** | 0.7523 | 0.4412 | 0.2557 |

## Key Scientific Findings

1. **Dominance of Aggregated Process Indicators:** Numerical process features (MLP baseline) achieve Macro-F1 = 0.5066 and ROC-AUC = 0.7223. Cumulative summary metrics (test pass rate, error resolution rate, revision frequency) carry strong discriminative signal.
2. **Temporal Attention Behavior:** The standalone Event-Only Transformer attains Macro-F1 = 0.3153. Because event token vocabularies (`run`, `edit`, `insert`) do not encapsulate full source code semantics, sequence progression alone offers limited stand-alone discriminative power for subtle conceptual boundaries.
3. **Combined Gated Representation:** The Combined Process Transformer balances both streams with a learned gating mechanism, achieving Macro-F1 = 0.4831 and ROC-AUC = 0.7166 while providing the required 64-dimensional embedding for downstream multimodal fusion with CodeBERT and Viva Sentence Transformer.
