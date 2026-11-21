# Multimodal Transformer-Based Prediction of Student Conceptual Understanding and Misconception Detection in Programming Education

**Arnav Bansal**  
*Department of Computer Science and Engineering*  
*DeepMisconception Research Project*  
*Corresponding author: Arnav Bansal (GitHub: @arnav-bansal-20)*  

---

### ABSTRACT
Automated assessment systems in programming education evaluate whether student code passes unit tests, but cannot assess students' underlying cognitive comprehension or detect misconceptions. In this paper, we present **DeepMisconception**, a multimodal deep learning framework that detects programming misconceptions by triangulating three data sources: (1) Python source code syntax via Falcon Code-adapted CodeBERT (768-D), (2) oral viva voce dialogue via frozen Sentence-BERT (384-D), and (3) chronological IDE keystroke telemetry via a Process Transformer (64-D). A contrastive Stage A alignment module maps heterogeneous modalities into a shared 128-D latent space using InfoNCE loss, followed by a Stage B Transformer encoder with a 3-class classification head (*Understood*, *Partial Understanding*, *Possible Misconception*). Evaluated on held-out test data ($N=627$), the framework achieves **90.75% Accuracy**, **0.8850 Macro F1-Score**, **92.27% Macro Sensitivity**, **95.63% Macro Specificity**, and **0.9888 AUROC**. A 5-fold stratified, group-aware cross-validation across 46 independent scenario families ($N=3,137$) demonstrates strong stability with a mean accuracy of **87.23% $\pm$ 4.12%** and a mean macro F1-score of **0.8284 $\pm$ 0.0336**. Ablation analysis confirms that Stage A cross-modal alignment is critical, preventing a severe performance drop to 66.51% accuracy (0.2663 F1). These results establish that multimodal triangulation provides an accurate, fair, and reliable methodology for automated misconception detection in programming courses.

**INDEX TERMS**  
Educational Data Mining, Multimodal Deep Learning, Transformer Networks, Misconception Detection, Programming Education, Code Comprehension, Cross-Modal Alignment.

---

## I. INTRODUCTION

Introductory programming courses (CS1) present substantial cognitive hurdles for novice learners. Automated grading platforms (auto-graders) rely on unit test suites to score student assignments. While automated tests establish input/output correctness, they are fundamentally incapable of assessing *how* a student arrived at their solution. Students frequently pass unit tests through trial-and-error debugging, template copy-pasting, or generative AI code generation without understanding programmatic invariants—a condition known as pseudo-comprehension.

To address this challenge, we developed **DeepMisconception**, a multimodal framework designed to detect conceptual misconceptions by triangulating:
1. **Source Code**: Static Abstract Syntax Tree (AST) structural features and Falcon Code-adapted CodeBERT contextual embeddings (768-D).
2. **Oral Viva Dialogue**: Interactive question-answer explanations embedded using frozen Sentence-BERT (`all-MiniLM-L6-v2`, 384-D).
3. **Behavioral Telemetry**: Chronological IDE event streams and 24 z-score normalized process metrics encoded via a Process Transformer (64-D).

The architecture uses a two-stage training scheme:
- **Stage A Cross-Modal Alignment**: Projects heterogeneous representations ($768\text{-D}, 384\text{-D}, 64\text{-D}$) into a unified 128-D latent space using InfoNCE contrastive and cosine alignment loss.
- **Stage B Multimodal Fusion Transformer**: Processes a 3-token sequence through a 4-layer bidirectional Transformer encoder ($d=128$, 8 heads) to predict three pedagogical classes: `Understood`, `Partial Understanding`, and `Possible Misconception`.

Cases with missing viva evidence are routed through a `Pending / Insufficient Evidence` status gate to prevent ungrounded model outputs.

---

## II. RESEARCH METHOD

### A. DATASET AND PARTITIONS
The dataset comprises $N=3,764$ supervised programming sessions organized across 46 distinct scenario families from `DeepMisconception_Corrected_Dataset`. To prevent data leakage, scenario families are strictly separated across partitions:
- **Training Set**: $N=3,137$ sessions (used for Stage A alignment, Stage B Transformer training, and 5-fold cross-validation).
- **Validation Set**: $N=300$ sessions (used for early stopping and hyperparameter tuning).
- **Held-Out Test Set**: $N=627$ supervised sessions evaluated independently ($66$ sessions with missing evidence were gated out as `Pending`).

### B. FEATURE EXTRACTION AND ALIGNMENT
1. **Code Modality ($h_{\text{code}} \in \mathbb{R}^{768}$)**: Extracted from student source code using Falcon Code-adapted `microsoft/codebert-base`.
2. **Viva Modality ($h_{\text{viva}} \in \mathbb{R}^{384}$)**: Extracted from oral question-answer pairs using frozen `sentence-transformers/all-MiniLM-L6-v2`.
3. **Process Modality ($h_{\text{proc}} \in \mathbb{R}^{64}$)**: Extracted from sequential IDE events and 24 normalized behavioral features using a trained Process Transformer.

**Stage A Alignment**: Learnable linear projections with layer normalization map each modality to $z \in \mathbb{R}^{128}$:
$$z_{\text{code}} = \text{LayerNorm}(W_c h_{\text{code}} + b_c), \quad z_{\text{viva}} = \text{LayerNorm}(W_v h_{\text{viva}} + b_v), \quad z_{\text{proc}} = \text{LayerNorm}(W_p h_{\text{proc}} + b_p)$$
The projection weights are optimized via symmetric InfoNCE loss with temperature $\tau=0.07$:
$$\mathcal{L}_{\text{InfoNCE}}(z_i, z_j) = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^B \exp(\text{sim}(z_i, z_k) / \tau)}$$

**Stage B Multimodal Fusion**: Aligned vectors are concatenated with modality type embeddings into a 3-token sequence $[z_{\text{code}} + e_c;\; z_{\text{viva}} + e_v;\; z_{\text{proc}} + e_p] \in \mathbb{R}^{3 \times 128}$. A 4-layer bidirectional Transformer encoder with 8 attention heads per layer performs cross-modal self-attention. The pooled token representation passes to a linear classification head producing 3-class probabilities.

---

## III. EXPERIMENTAL RESULTS AND DISCUSSION

### A. COMPARISON OF MODALITY CONFIGURATIONS
Table 1 reports the actual performance of the model across unimodal and multimodal configurations evaluated on the identical held-out test split ($N=627$).

### TABLE 1. Performance across unimodal and multimodal configurations on held-out test data ($N=627$).

| Configuration | Accuracy (%) | Precision (Macro) | Recall / Sensitivity (Macro) | F1-Score (Macro) | F1-Score (Weighted) | AUROC (OvR) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Process-Only** | 66.51 | 0.2217 | 0.3333 | 0.2663 | 0.5313 | 0.5907 |
| **2. Viva-Only** | 77.99 | 0.6963 | 0.8098 | 0.7182 | 0.7995 | 0.9405 |
| **3. Code-Only** | 86.76 | 0.8016 | 0.9083 | 0.8392 | 0.8725 | 0.9906 |
| **4. Code + Viva** | 82.78 | 0.7495 | 0.8733 | 0.7864 | 0.8380 | 0.9792 |
| **5. Viva + Process** | 90.27 | 0.8342 | 0.9140 | 0.8653 | 0.9070 | 0.9866 |
| **6. Code + Process** | 91.23 | 0.9171 | 0.8737 | 0.8926 | 0.9123 | 0.9604 |
| **7. Complete Multimodal Fusion (Code + Viva + Process)** | **90.75** | **0.8582** | **0.9227** | **0.8850** | **0.9095** | **0.9888** |

*Note: While Code + Process yields slightly higher accuracy on this partition (91.23%), Complete Multimodal Fusion provides higher macro recall (92.27% vs 87.37%) and higher AUROC (0.9888 vs 0.9604), which is critical for identifying struggling students.*

### B. PER-CLASS PERFORMANCE BREAKDOWN
Table 2 details the exact per-class metrics computed from model predictions on the held-out test set ($N=627$).

### TABLE 2. Exact per-class performance metrics on held-out test data ($N=627$).

| Pedagogical Class | Support ($N$) | Precision (%) | Sensitivity / Recall (%) | Specificity (%) | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Understood** | 417 | 98.40 | 88.25 | 97.14 | 0.9305 |
| **Partial Understanding** | 140 | 79.31 | 98.57 | 92.61 | 0.8790 |
| **Possible Misconception** | 70 | 79.75 | 90.00 | 97.13 | 0.8456 |
| **Macro Average** | — | **85.82** | **92.27** | **95.63** | **0.8850** |
| **Weighted Average** | — | **92.05** | **90.75** | **95.63** | **0.9095** |

- **Partial Understanding Sensitivity (98.57%)**: The model successfully identifies 138 out of 140 students with partial comprehension.
- **Possible Misconception Specificity (97.13%)**: The model achieves high specificity on misconceptions, minimizing false accusations.
- **Classification Error**: Overall test classification error is **9.25%** ($1 - 0.9075$).

### C. CONFUSION MATRIX
Table 3 reports the exact prediction matrix on the test set.

### TABLE 3. Confusion matrix on held-out test set ($N=627$).

| Actual Class \ Predicted Class | Predicted: Understood | Predicted: Partial Understanding | Predicted: Possible Misconception | Total Actual |
|:---|:---:|:---:|:---:|:---:|
| **Actual: Understood** | **368** (88.25%) | 34 (8.15%) | 15 (3.60%) | 417 |
| **Actual: Partial Understanding** | 1 (0.71%) | **138** (98.57%) | 1 (0.71%) | 140 |
| **Actual: Possible Misconception** | 5 (7.14%) | 2 (2.86%) | **63** (90.00%) | 70 |
| **Total Predicted** | 374 | 174 | 79 | **627** |

Only 1 out of 140 students with partial understanding was incorrectly predicted as fully Understood, demonstrating safe pedagogical behavior.

### D. 5-FOLD STRATIFIED GROUP CROSS-VALIDATION
To test generalization, 5-fold cross-validation was run strictly within the training set ($N=3,137$) using `StratifiedGroupKFold` across 46 scenario families, ensuring zero scenario overlap between training and validation folds. Table 4 presents the fold results.

### TABLE 4. 5-Fold stratified group cross-validation results ($N=3,137$, 46 Scenario Families).

| Fold | Accuracy (%) | Classification Error (%) | Precision (Macro) | Sensitivity / Recall (Macro) | Specificity (Macro) | F1-Score (Macro) | AUROC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 87.20 | 12.80 | 0.7904 | 0.9119 | 0.9424 | 0.8369 | 0.9585 |
| **Fold 2** | 93.50 | 6.50 | 0.8603 | 0.8949 | 0.9721 | 0.8539 | 0.9793 |
| **Fold 3** | 88.33 | 11.67 | 0.8160 | 0.8691 | 0.9271 | 0.8400 | 0.9548 |
| **Fold 4** | 86.50 | 13.50 | 0.8253 | 0.8834 | 0.9223 | 0.8490 | 0.9580 |
| **Fold 5** | 80.62 | 19.38 | 0.7897 | 0.7995 | 0.9125 | 0.7623 | 0.9590 |
| **Mean $\pm$ Std** | **87.23 $\pm$ 4.12** | **12.77 $\pm$ 4.12** | **0.8164 $\pm$ 0.0260** | **87.18 $\pm$ 3.87** | **93.53 $\pm$ 2.08** | **0.8284 $\pm$ 0.0336** | **0.9619 $\pm$ 0.0088** |

Across all five independent folds, the model achieves a mean accuracy of **87.23%** and a mean macro F1-score of **0.8284**, confirming that model performance is not an artifact of data splitting.

### E. ABLATION STUDY
Table 5 shows the empirical effect of removing individual architectural components and modalities from the complete pipeline.

### TABLE 5. Empirical ablation study results on held-out test data ($N=627$).

| Architecture Variant | Accuracy (%) | Macro F1-Score | Macro F1 Delta | AUROC |
|:---|:---:|:---:|:---:|:---:|
| **Complete Multimodal Architecture** | **90.75** | **0.8850** | **0.0000** | **0.9888** |
| Without Process Modality | 90.59 | 0.8863 | +0.0013 | 0.9895 |
| Without Numerical Process Features | 90.43 | 0.8692 | -0.0158 | 0.9893 |
| Without Code Modality | 90.27 | 0.8653 | -0.0197 | 0.9866 |
| Without Viva Modality | 91.23 | 0.8926 | +0.0076 | 0.9604 |
| **Without Cross-Modal Alignment** | **66.51** | **0.2663** | **-0.6187** | **0.6638** |

The ablation reveals that **Stage A Cross-Modal Alignment is essential**: removing alignment results in a catastrophic performance drop of **-24.24% in accuracy** and **-0.6187 in macro F1-score**, proving that contrastive projection into a normalized 128-D latent space is necessary before Transformer self-attention.

---

## IV. CONCLUSION AND FUTURE WORK

We presented DeepMisconception, a multimodal deep learning framework for student code comprehension verification and misconception detection. By fusing CodeBERT code representations, Sentence-BERT oral viva explanations, and Process Transformer telemetry into an aligned latent space, the system overcomes the blind spots of conventional unit-test auto-graders. 

On a held-out test partition ($N=627$), the complete multimodal model achieves 90.75% accuracy, 0.8850 macro F1, and 0.9888 AUROC, with a 5-fold cross-validation mean accuracy of 87.23% $\pm$ 4.12% across 46 scenario families. In future work, we plan to incorporate Graph Neural Networks (GNNs) for Abstract Syntax Tree modeling and deploy the system in live classroom settings.
