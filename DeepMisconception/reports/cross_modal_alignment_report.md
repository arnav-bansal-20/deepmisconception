# Cross-Modal Semantic Alignment Report (Code <-> Viva)

## 1. Overview and Architecture

- **Input Code Modality:** `microsoft/codebert-base` (768-D representation)
- **Input Viva Modality:** `all-MiniLM-L6-v2` (384-D representation)
- **Aligned Dimension:** 64-D for each modality
- **Alignment Mechanism:** Feed-forward projections + Bidirectional Multi-Head Cross-Attention (4 heads)
- **Objective:** Symmetric InfoNCE Contrastive Alignment Loss (temperature=0.07)

## 2. Quantitative Alignment Performance (Validation Partition)

| Evaluation Metric | Unaligned Baseline | Learned Alignment Module |
| :--- | :---: | :---: |
| **Positive Pair Cosine Similarity** | 0.0053 | **0.9524** |
| **Negative Pair Cosine Similarity** | 0.0097 | **0.0485** |
| **Cross-Modal Retrieval Recall@1** | 0.0013 | **0.9987** |
| **Cross-Modal Retrieval Recall@5** | 0.0052 | **1.0000** |
| **Mean Reciprocal Rank (MRR)** | 0.0094 | **0.9994** |
| **Best Validation Loss** | N/A | **0.1716** (Epoch 16) |

## 3. Conclusions

Learned cross-modal semantic alignment significantly pulls genuine corresponding student code submissions and viva explanations closer in the shared 64-D latent space while cleanly separating non-corresponding pairs.
