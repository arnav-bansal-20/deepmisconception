# Multimodal Fusion Transformer Training Report

## 1. Overview and Setup

- **Architecture:** Multimodal Fusion Transformer (2 layers, 4 attention heads, d_model=64, feedforward=256, dropout=0.15)
- **Encoders:** CodeBERT (768-D, frozen), SentenceTransformer (384-D, frozen), Process Transformer (64-D, frozen), Numerical Features (24-D, normalized)
- **Cross-Modal Alignment:** Pre-aligned Code <-> Viva representations (Stage A)
- **Optimizer:** AdamW ($lr=10^{-3}$, weight_decay=$10^{-4}$) with CosineAnnealingLR
- **Best Checkpoint:** Epoch 9 with Validation Loss 0.0016
- **Validation Accuracy:** 1.0000 | **Validation Macro-F1:** 1.0000

## 2. Loss and Validation Trajectory

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.4021 | 0.0706 | 0.9957 | 0.9891 |
| 2 | 0.1367 | 0.0391 | 0.9971 | 0.9927 |
| 3 | 0.0859 | 0.0173 | 0.9957 | 0.9916 |
| 4 | 0.0693 | 0.1040 | 0.9740 | 0.9459 |
| 5 | 0.0457 | 0.0177 | 0.9986 | 0.9963 |
| 6 | 0.0339 | 0.0160 | 0.9957 | 0.9891 |
| 7 | 0.0302 | 0.0075 | 0.9971 | 0.9927 |
| 8 | 0.0204 | 0.0027 | 1.0000 | 1.0000 |
| 9 | 0.0140 | 0.0016 | 1.0000 | 1.0000 |
| 10 | 0.0119 | 0.0040 | 0.9986 | 0.9985 |
| 11 | 0.0077 | 0.0072 | 0.9971 | 0.9927 |
| 12 | 0.0100 | 0.0101 | 0.9986 | 0.9963 |
| 13 | 0.0092 | 0.0274 | 0.9971 | 0.9927 |
| 14 | 0.0068 | 0.0080 | 0.9971 | 0.9948 |
| 15 | 0.0063 | 0.0229 | 0.9971 | 0.9927 |

## 3. Phase 3: Controlled End-to-End Fine-Tuning Comparison

We evaluated whether selectively unfreezing upper CodeBERT layers improves generalization on held-out test scenario families.

| Architecture Variant | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Empirical Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frozen Encoders + Alignment (Selected)** | 1.0000 | 1.0000 | **0.9075** | **0.8850** | **Best Generalization on Disjoint Families** |
| Selectively Fine-Tuned Upper Layers | 0.9986 | 0.9985 | 0.8421 | 0.8175 | Overfits to training scenario-family naming quirks |

**Conclusion:** Freezing pretrained encoders and training dedicated cross-modal alignment and fusion layers yields the highest robustness and generalizability on unseen problem families.
