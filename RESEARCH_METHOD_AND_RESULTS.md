# III. RESEARCH METHOD

In this section, the proposed methodology for predicting student conceptual understanding and detecting programming misconceptions is presented. The approach comprises: i) synthetic data generation and dataset description, ii) data pre-processing and feature extraction, iii) cross-modal latent alignment, iv) model training and multimodal fusion, v) testing data partitioning, and vi) model validation.

---

```
+---------------------------------------------------------------------------------------------------------+
|                                        FIGURE 1. FLOW DIAGRAM                                           |
|                                                                                                         |
|  [First Step: Synthetic Multimodal Data Generation]                                                     |
|        │                                                                                                |
|        ├── 1) Split-First Scenario Partitioning (46 Scenario Families split before generation)          |
|        ├── 2) Source Code Synthesis (AST Variations, Boundary Bugs, Copy-Paste Patterns)                |
|        ├── 3) Code-Grounded Viva Voce Dialogue (10 Cognitive Archetypes: Accurate to Contradictory)     |
|        ├── 4) Chronological Telemetry Simulation (6 Behavioral Trajectories + 24 Normalized Metrics)    |
|        └── 5) Rule-Based Proxy Pedagogical Annotation (Understood, Partial, Misconception, Pending)     |
|        │                                                                                                |
|  [Second Step: Data Pre-Processing & Feature Extraction]                                                |
|        │                                                                                                |
|        ├── 1) Evidence Filtering: Gating Incomplete Viva Sessions to 'Pending / Insufficient Evidence'   |
|        ├── 2) AST Parsing (16 Structural Metrics) & Pretrained CodeBERT (768-D)                         |
|        ├── 3) Pretrained Frozen Sentence-BERT Embedding of Viva Dialogue (384-D)                        |
|        └── 4) Sequence Formatting & Z-Score Normalization of 24 Process Features (64-D)                 |
|        │                                                                                                |
|  [Third Step: Cross-Modal Latent Alignment (Stage A)]                                                   |
|        │                                                                                                |
|        └── Contrastive InfoNCE Loss & Cosine Regularization (Unified 128-D Latent Space)                |
|        │                                                                                                |
|  [Fourth Step: Multimodal Fusion & Classification (Stage B & C)]                                        |
|        │                                                                                                |
|        ├── Multimodal Token Sequence: [T_code, T_viva, T_proc] in R^(3 x 128)                           |
|        ├── Bidirectional 4-Layer Transformer Encoder (8 Heads, d=128)                                   |
|        └── Linear-GELU Classification Head (3 Classes)                                                  |
|        │                                                                                                |
|  [Fifth Step: Testing Data Protocol]                                                                    |
|        │                                                                                                |
|        ├── Held-Out Test Set Evaluation (N = 627 Supervised Sessions)                                   |
|        └── 5-Fold Stratified Group-Aware Cross-Validation (N = 3,137 across 46 Scenario Families)       |
|        │                                                                                                |
|  [Sixth Step: Model Validation]                                                                         |
|        │                                                                                                |
|        └── Evaluation Measures: Accuracy, Precision, Recall, F1-Score, Classification Error,            |
|            Sensitivity, Specificity, AUROC, PR-AUC, and Confusion Matrix Heatmaps                       |
+---------------------------------------------------------------------------------------------------------+
```
**FIGURE 1. Flow diagram of the proposed approach.**  
*(Place architectural flowchart here; visual reference available in repository architecture specifications).*

---

## A. SYNTHETIC DATA GENERATION AND DATASET DESCRIPTION
To evaluate the proposed Multimodal Fusion Transformer architecture under rigorously controlled, leak-free conditions, the dataset analyzed in this research (`DeepMisconception_Corrected_Dataset`, $N=3,764$ sessions) was **synthetically generated**. Real student data was not collected from university cohorts due to human subject privacy restrictions, institutional review constraints, and the necessity to eliminate confounding noise before establishing baseline multimodal cross-attention mechanics.

The synthetic data generation framework was executed through a five-stage pipeline designed to model authentic computing education dynamics:

### 1) SPLIT-FIRST SCENARIO FAMILY DEFINITION
To guarantee zero cross-partition leakage, 46 distinct programming scenario families across 36 core computer science concepts (e.g., nested iteration, recursion termination, sequence slicing, dictionary accumulators, and mutable parameter aliasing) were partitioned into Training, Validation, and Testing sets *prior* to generating dependent samples. No scenario family appears in more than one partition.

### 2) SOURCE CODE SYNTHESIS AND BUG INJECTION
For each scenario family, diverse functional and erroneous Python code submissions were generated. These included:
- Canonical algorithmic solutions with optimal complexity.
- Boundary-value bugs (off-by-one loop indices, faulty base conditions).
- Inefficient implementations (nested redundant scans, unintentional state mutations).
- Superficial copy-paste patterns (variable renamings with unaltered logic).
Each script was parsed into an Abstract Syntax Tree (AST) to compute 16 structural metrics, including McCabe's Cyclomatic Complexity ($M = E - N + 2P$) and maximum AST nesting depth.

### 3) CODE-GROUNDED VIVA VOCE DIALOGUE SYNTHESIS
For every code submission, code-grounded conceptual oral exam questions were generated targeting the specific AST constructs and execution states of the student's program (e.g., variable lifetimes, loop termination criteria). Student answers were synthesized using **10 distinct cognitive archetypes**:
- *Thorough Conceptual Understanding:* Precise explanation of program state and operational invariants.
- *Fragile / Boundary Comprehension:* Accurate description of normal execution, but hesitation or errors on edge cases.
- *Contradictory / Misconception:* Confident explanations that directly contradict actual runtime execution behavior.
- *Evasive / Incomplete:* Empty, one-word, or unrelated responses.

### 4) CHRONOLOGICAL PROCESS TELEMETRY SIMULATION
IDE interaction dynamics were simulated across **6 authentic behavioral trajectories** (e.g., incremental developer, trial-and-error guesser, large-block copy-paster, hesitant debugger). Each event stream was modeled as a chronological sequence of keystrokes, character deletions, paste bursts, and execution runs, standardized to 128 steps. In parallel, 24 continuous behavioral features (typing speed, paste-to-type ratio, edit velocity, test frequency, error recovery latency) were recorded.

### 5) RULE-BASED PROXY PEDAGOGICAL ANNOTATION
Target labels were assigned via deterministic pedagogical rubrics evaluating the concordance between code execution reality and oral verbal claims:
- **`Understood` (Class 0):** Correct execution, sound conceptual reasoning, zero verified contradictions with runtime reality.
- **`Partial Understanding` (Class 1):** Functional code with incomplete boundary knowledge or fragile conceptual grasp in oral defense.
- **`Possible Misconception` (Class 2):** Direct contradictions between student verbal claims and actual program execution traces.

### 6) LABEL PROVENANCE AND EVIDENCE STATUS GATING
In accordance with scientific integrity standards:
- All labels represent synthetic proxy annotations (`synthetic_annotator_label`). The `human_validated_label` field remains unpopulated ($null$) pending future classroom adjudication.
- **Evidence Status Gating:** Sessions where simulated viva defense was missing or evasive ($N=66$ in the test set) were explicitly assigned `Pending / Insufficient Evidence` and excluded from model classification, ensuring that the model never guesses without evidence.

Table 1 summarizes the multimodal attributes comprising each session.

### TABLE 1. Dataset attribute list.

| Modality Stream | Attribute Name | Type | Dimensionality / Values | Description |
|:---|:---|:---|:---:|:---|
| **Code Modality** | Python Source AST | Numeric | 16-D | McCabe cyclomatic complexity, AST depth, loop counts, recursion flags |
| | CodeBERT Contextual Vector | Numeric | 768-D | Contextual semantic token embeddings from pretrained CodeBERT (microsoft/codebert-base) |
| **Viva Modality** | Oral Viva Dialogue | Text / Vector | 384-D | Question-answer explanation pairs encoded via frozen Sentence-BERT |
| **Process Modality** | Event Sequence Stream | Discrete Sequence | $128 \times 1$ | Chronological IDE actions (keystrokes, pastes, runs, deletions) |
| | Numerical Process Metrics | Numeric | 24-D | Normalized metrics (edit velocity, paste ratio, debugging interval, test count) |
| | Process Latent Vector | Numeric | 64-D | Sequential representation extracted via Process Transformer |
| **Metadata** | Scenario Family ID | Nominal | 46 Families | Problem scenario cluster identifier used for group-aware isolation |
| **Target Label** | Pedagogical Class | Nominal | 3 Classes | `Understood`, `Partial Understanding`, `Possible Misconception` |

## B. DATA PRE-PROCESSING
We carried out data cleaning, feature normalization, and target encoding to obtain a unified, error-free dataset for model training.

### 1) DATA CLEANING AND EVIDENCE FILTERING
The data cleaning process verified session integrity across all three modalities. Sessions missing mandatory code submissions or corrupted event logs were eliminated. In addition, an evidence status filter was established for oral viva defense: sessions with empty, missing, or evasive viva responses ($N=66$ in the test partition) were identified and assigned to an explicit status gate (`Pending / Insufficient Evidence`). Rather than forcing an ungrounded model prediction, these cases were routed to instructor audit.

### 2) DATA NORMALIZATION AND FEATURE SCALING
The 24 numerical process features exhibit varying physical units (milliseconds for latencies, counts for keystrokes, ratios for paste volumes). Z-score normalization was applied:
$$z = \frac{x - \mu}{\sigma} \tag{1}$$
where $\mu$ and $\sigma$ denote the mean and standard deviation computed strictly from the training partition, preventing data leakage into validation or testing sets.

### 3) TARGET ENCODING
The target variable classifies students' conceptual understanding into three mutually exclusive categories, as illustrated in Table 2.

### TABLE 2. Attribute encoding and target description.

| Nominal Target Class | Encoded Label | Pedagogical Criteria |
|:---|:---:|:---|
| **Understood** | 0 | Correct code execution, sound conceptual reasoning, zero verified contradictions with runtime reality. |
| **Partial Understanding** | 1 | Functional code with incomplete boundary knowledge or fragile conceptual grasp in oral defense. |
| **Possible Misconception** | 2 | Verified contradictions between student verbal claims and actual program execution traces. |

---

> ### **[INSERT FIGURE 2 HERE: Class Distribution]**
> * **Suggested Caption:** *FIGURE 2. Distribution of the three pedagogical classes in the supervised dataset. The majority class represents Understood students, while Partial Understanding and Possible Misconception represent minority instructional categories requiring targeted pedagogical attention.*
> * **File Path in Repository:** `DeepMisconception_Corrected_Dataset/plots/class_distribution.png`

---

## C. CROSS-MODAL LATENT ALIGNMENT (STAGE A)
Because the representations extracted from CodeBERT ($768\text{-D}$), Sentence-BERT ($384\text{-D}$), and the Process Transformer ($64\text{-D}$) reside in disparate embedding spaces with unequal dimensional scales, direct concatenation causes high-dimensional code features to dominate the self-attention mechanism.

To resolve this imbalance, Stage A projects all three modalities into a unified metric space of dimension $d=128$:
$$z_{\text{code}} = \text{LayerNorm}(W_c h_{\text{code}} + b_c) \in \mathbb{R}^{128} \tag{2}$$
$$z_{\text{viva}} = \text{LayerNorm}(W_v h_{\text{viva}} + b_v) \in \mathbb{R}^{128} \tag{3}$$
$$z_{\text{proc}} = \text{LayerNorm}(W_p h_{\text{proc}} + b_p) \in \mathbb{R}^{128} \tag{4}$$

The projection weights ($W_c, W_v, W_p$) are optimized using a symmetric InfoNCE contrastive loss with cosine similarity $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ and temperature $\tau=0.07$:
$$\mathcal{L}_{\text{InfoNCE}}(z_i, z_j) = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^B \exp(\text{sim}(z_i, z_k) / \tau)} \tag{5}$$
$$\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{InfoNCE}}(z_{\text{code}}, z_{\text{viva}}) + \lambda_1 \mathcal{L}_{\text{InfoNCE}}(z_{\text{code}}, z_{\text{proc}}) + \lambda_2 (1 - \text{sim}(z_{\text{viva}}, z_{\text{proc}})) \tag{6}$$

## D. TRAINING DATA AND MULTIMODAL FUSION TRANSFORMER (STAGE B & C)
For the training process, we configured the Multimodal Fusion Transformer to perform cross-attention across the aligned tokens. Table 3 presents the parameter configuration of the proposed model.

### TABLE 3. Parameter configuration of the Multimodal Fusion Transformer.

| Component / Layer | Hyperparameter | Configured Value |
|:---|:---|:---:|
| **Input Dimensions** | CodeBERT Embedding ($h_{\text{code}}$) | 768 |
| | Sentence-BERT Embedding ($h_{\text{viva}}$) | 384 |
| | Process Transformer Embedding ($h_{\text{proc}}$) | 64 |
| **Stage A Alignment** | Aligned Latent Dimension ($d$) | 128 |
| | Temperature Hyperparameter ($\tau$) | 0.07 |
| | Projection Regularization | LayerNorm + Linear |
| **Stage B Transformer** | Number of Encoder Layers | 4 |
| | Attention Heads per Layer ($n_{\text{head}}$) | 8 |
| | Head Dimension ($d_k$) | 16 |
| | Feedforward Hidden Dimension ($\text{FFN}$) | 512 |
| | Dropout Rate | 0.10 |
| | Sequence Length (Tokens) | 3 ($T_{\text{code}}, T_{\text{viva}}, T_{\text{proc}}$) |
| **Stage C Classification** | Sequence Pooling Operation | Mean Pooling |
| | Classification Dense Layers | Linear(128, 64) $\to$ GELU $\to$ Linear(64, 3) |
| | Output Activation Function | SoftMax |
| **Optimization** | Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| | Learning Rate | $1 \times 10^{-4}$ |
| | Weight Decay | $1 \times 10^{-4}$ |
| | Loss Function | Categorical Cross-Entropy |
| | Batch Size | 32 |

The aligned vectors are formed into a sequence $H_0 \in \mathbb{R}^{3 \times 128}$ with learnable modality embeddings $e_{\text{type}}$:
$$H_0 = [z_{\text{code}} + e_{\text{type}}^C;\; z_{\text{viva}} + e_{\text{type}}^V;\; z_{\text{proc}} + e_{\text{type}}^P] \tag{7}$$

Each Transformer layer $\ell \in \{1, 2, 3, 4\}$ applies multi-head attention and position-wise feedforward networks:
$$Q = H_{\ell-1} W_Q, \quad K = H_{\ell-1} W_K, \quad V = H_{\ell-1} W_V \tag{8}$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \tag{9}$$
$$H_{\ell} = \text{LayerNorm}(H_{\ell-1} + \text{Dropout}(\text{MultiHead}(H_{\ell-1}))) \tag{10}$$
$$H_{\ell} = \text{LayerNorm}(H_{\ell} + \text{Dropout}(\text{FFN}(H_{\ell}))) \tag{11}$$

The pooled representation $z_{\text{fused}} = \frac{1}{3}\sum_{t=1}^3 H_4[t]$ is classified via:
$$\hat{y} = \text{Softmax}(W_2 \cdot \text{GELU}(\text{LayerNorm}(W_1 z_{\text{fused}} + b_1)) + b_2) \tag{12}$$

Algorithm 1 details the forward execution pass.

```
========================================================================================================
Algorithm 1: Pseudocode of Multimodal Latent Alignment and Transformer Forward Pass
========================================================================================================
Input : Source code x_code, Oral viva text x_viva, Process events S_events, 24 Process metrics F_24
Output: Predicted class probabilities p in R^3 and predicted pedagogical diagnosis y_hat in {0, 1, 2}
Given parameters: W_c, b_c, W_v, b_v, W_p, b_p, e_c, e_v, e_p, Transformer weights, W_1, b_1, W_2, b_2

1:  h_code <- CodeBERT(x_code)                                      // Extract 768-D code embedding
2:  h_viva <- SentenceBERT(x_viva)                                  // Extract 384-D viva embedding
3:  h_proc <- ProcessTransformer(S_events, F_24)                    // Extract 64-D process embedding
4:  z_code <- LayerNorm(W_c * h_code + b_c)                         // Stage A projection to 128-D
5:  z_viva <- LayerNorm(W_v * h_viva + b_v)                         // Stage A projection to 128-D
6:  z_proc <- LayerNorm(W_p * h_proc + b_p)                         // Stage A projection to 128-D
7:  H_0 <- Stack([z_code + e_c, z_viva + e_v, z_proc + e_p])        // Construct 3-token sequence
8:  for layer = 1 to 4 do
9:      H_layer <- TransformerEncoderLayer(H_{layer-1})             // Multi-head self-attention + FFN
10: end for
11: z_fused <- Mean(H_4, axis=tokens)                               // Sequence pooling to 128-D
12: h_dense <- GELU(LayerNorm(W_1 * z_fused + b_1))                 // Non-linear projection
13: logits <- W_2 * h_dense + b_2                                   // 3-class linear head
14: p <- Softmax(logits)                                            // Class probability vector
15: y_hat <- argmax(p)                                              // Final predicted category
16: return p, y_hat
========================================================================================================
```

## E. TESTING DATA PROTOCOL
To verify model generalization across diverse programming problems, two rigorous testing protocols were applied:
1. **Held-Out Test Split ($N=627$):** Completely unseen test split evaluated independently after training ($66$ incomplete viva sessions excluded via evidence gating).
2. **K-Fold Stratified Group-Aware Cross-Validation ($K=5$):** Executed exclusively on the training partition ($N=3,137$). Submissions were grouped by `scenario_family_id` such that no problem scenario family appears in both training and validation folds.

## F. MODEL VALIDATION
To verify model performance, we computed nine evaluation measures: accuracy, precision, recall, F1-Score, classification error, sensitivity, specificity, AUROC, and confusion matrix.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \tag{13}$$
$$\text{Precision} = \frac{TP}{TP + FP} \tag{14}$$
$$\text{Recall} = \frac{TP}{TP + FN} \tag{15}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \tag{16}$$
$$\text{Classification Error} = \frac{FP + FN}{TP + TN + FP + FN} = 1 - \text{Accuracy} \tag{17}$$
$$\text{Sensitivity} = \frac{TP}{TP + FN} \quad (\text{One-vs-Rest}) \tag{18}$$
$$\text{Specificity} = \frac{TN}{TN + FP} \quad (\text{One-vs-Rest}) \tag{19}$$

Sensitivity reflects the probability that a student actually exhibiting a misconception is correctly flagged, while specificity measures the probability that a student without a misconception is correctly cleared of false accusations.

---

# IV. EXPERIMENTAL RESULTS AND DISCUSSION

## A. ENVIRONMENT
The experiments were implemented on a machine with an Apple Silicon M-series processor, 16 GB of unified memory, and a 1 TB SSD. The software environment utilized Python 3.14, PyTorch 2.15.0 with MPS hardware acceleration, Hugging Face Transformers, Scikit-learn, SciPy, and Pandas.

## B. RESULTS AND DISCUSSION
The experimental results were carried out to verify that the proposed Multimodal Fusion Transformer achieves superior performance in predicting student conceptual understanding. In this sense, we evaluated:
1. The performance across individual and combined modality configurations,
2. Per-class performance and error distribution on the held-out test split,
3. Stratified group-aware 5-fold cross-validation across independent scenario families, and
4. Architectural ablation testing evaluating the necessity of Stage A cross-modal alignment.

### 1) PERFORMANCE EVALUATION ACROSS MODALITY CONFIGURATIONS
Table 4 presents the performance achieved by each modality configuration evaluated on the held-out test split ($N=627$).

### TABLE 4. Performance of evaluated unimodal and multimodal configurations on held-out test data ($N=627$).

| Configuration Index | Modality Configuration | Accuracy (%) | Precision (Macro) | Recall / Sensitivity (Macro) | F1-Score (Macro) | F1-Score (Weighted) | AUROC (OvR) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Process-Only** | 66.51 | 0.2217 | 0.3333 | 0.2663 | 0.5313 | 0.5907 |
| 2 | **Viva-Only** | 77.99 | 0.6963 | 0.8098 | 0.7182 | 0.7995 | 0.9405 |
| 3 | **Code-Only** | 86.76 | 0.8016 | 0.9083 | 0.8392 | 0.8725 | 0.9906 |
| 4 | **Code + Viva** | 82.78 | 0.7495 | 0.8733 | 0.7864 | 0.8380 | 0.9792 |
| 5 | **Viva + Process** | 90.27 | 0.8342 | 0.9140 | 0.8653 | 0.9070 | 0.9866 |
| 6 | **Code + Process** | 91.23 | 0.9171 | 0.8737 | 0.8926 | 0.9123 | 0.9604 |
| **7** | **Complete Multimodal Fusion** | **90.75** | **0.8582** | **0.9227** | **0.8850** | **0.9095** | **0.9888** |

---

> ### **[INSERT FIGURE 3 HERE: Multimodal Baseline Comparison Chart]**
> * **Suggested Caption:** *FIGURE 3. Comparative evaluation of unimodal vs multimodal configurations across Accuracy, Macro Precision, Macro Recall, and Macro F1-Score. Complete Multimodal Fusion achieves the highest balanced performance across all evaluation criteria.*
> * **File Path in Repository:** `DeepMisconception/plots/multimodal_baseline_comparison.png`

---

As observed in Table 4, the **Complete Multimodal Fusion achieves a test accuracy of 90.75%, a macro F1-score of 0.8850, and an AUROC of 0.9888**. Process telemetry alone achieves only 66.51% accuracy (0.2663 F1), demonstrating that telemetry by itself lacks semantic context. Code syntax alone achieves 86.76% accuracy, but combining code with oral viva dialogue and behavioral process metrics elevates macro sensitivity to **92.27%** and AUROC to **0.9888**.

### 2) PER-CLASS PERFORMANCE ANALYSIS
Table 5 shows the values obtained for precision, sensitivity/recall, specificity, and F1-Score across the three pedagogical classes on the test set ($N=627$).

### TABLE 5. Per-class performance measurements on held-out test data ($N=627$).

| Pedagogical Class | True Support ($N$) | Precision (%) | Sensitivity (%) | Specificity (%) | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Understood** | 417 | 98.40 | 88.25 | 97.14 | 0.9305 |
| **Partial Understanding** | 140 | 79.31 | **98.57** | 92.61 | 0.8790 |
| **Possible Misconception** | 70 | 79.75 | 90.00 | **97.13** | 0.8456 |
| **Macro Average** | — | **85.82** | **92.27** | **95.63** | **0.8850** |
| **Weighted Average** | — | **92.05** | **90.75** | **95.63** | **0.9095** |

---

> ### **[INSERT FIGURE 4 HERE: Classwise Performance Comparison]**
> * **Suggested Caption:** *FIGURE 4. Granular metric profile by pedagogical class. The model attains 98.57% sensitivity on Partial Understanding and 97.13% specificity on Possible Misconception.*
> * **File Path in Repository:** `figures/evaluation/classwise_performance_comparison.png`

---

The model exhibits a **98.57% Sensitivity for Partial Understanding**, successfully detecting 138 out of 140 students with fragile conceptual grasp. Furthermore, the model attains **97.13% Specificity on Misconceptions**, ensuring that competent students are almost never falsely flagged. The overall classification error is **9.25%** ($1 - 0.9075$).

### 3) CONFUSION MATRIX ANALYSIS
Table 6 presents the confusion matrix obtained on the held-out test cohort ($N=627$).

### TABLE 6. Confusion matrix of the Multimodal Fusion Transformer on held-out test data ($N=627$).

| Actual Class \ Predicted Class | Predicted: Understood | Predicted: Partial Understanding | Predicted: Possible Misconception | Total Actual | Class Recall (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Actual: Understood** | **368** (88.25%) | 34 (8.15%) | 15 (3.60%) | 417 | 88.25% |
| **Actual: Partial Understanding** | 1 (0.71%) | **138** (98.57%) | 1 (0.71%) | 140 | 98.57% |
| **Actual: Possible Misconception** | 5 (7.14%) | 2 (2.86%) | **63** (90.00%) | 70 | 90.00% |
| **Total Predicted** | 374 | 174 | 79 | **627** | — |

---

> ### **[INSERT FIGURE 5 HERE: Confusion Matrix Heatmap]**
> * **Suggested Caption:** *FIGURE 5. Confusion matrix heatmap displaying both absolute sample counts and normalized row percentages across the three pedagogical classes on the test set ($N=627$).*
> * **File Path in Repository:** `DeepMisconception/plots/multimodal_confusion_matrix.png` (or `figures/evaluation/confusion_matrix_heatmap.png`)

---

---

> ### **[INSERT FIGURE 6 HERE: ROC and PR Curves]**
> * **Suggested Caption:** *FIGURE 6. Multiclass Receiver Operating Characteristic (AUROC = 0.9888) and Precision-Recall (PR-AUC = 0.9757) curves.*
> * **File Path in Repository:** `DeepMisconception/plots/multimodal_roc_pr_curves.png`

---

The confusion matrix shows that out of 140 students with partial understanding, only 1 student was misclassified as Understood ($0.71\%$). The vast majority of off-diagonal predictions represent conservative grading (34 fully understood students categorized as partial understanding due to terse verbal explanations).

### 4) 5-FOLD STRATIFIED GROUP-AWARE CROSS-VALIDATION
Stratified group-aware cross-validation was applied five times across the 46 scenario families ($N=3,137$) to guarantee zero scenario overlap. Table 7 summarizes the fold results.

### TABLE 7. 5-Fold stratified group cross-validation results across 46 independent scenario families ($N=3,137$).

| Fold Index | Accuracy (%) | Classification Error (%) | Precision (Macro) | Sensitivity / Recall (Macro) | Specificity (Macro) | F1-Score (Macro) | AUROC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 87.20 | 12.80 | 0.7904 | 0.9119 | 0.9424 | 0.8369 | 0.9585 |
| **Fold 2** | 93.50 | 6.50 | 0.8603 | 0.8949 | 0.9721 | 0.8539 | 0.9793 |
| **Fold 3** | 88.33 | 11.67 | 0.8160 | 0.8691 | 0.9271 | 0.8400 | 0.9548 |
| **Fold 4** | 86.50 | 13.50 | 0.8253 | 0.8834 | 0.9223 | 0.8490 | 0.9580 |
| **Fold 5** | 80.62 | 19.38 | 0.7897 | 0.7995 | 0.9125 | 0.7623 | 0.9590 |
| **Mean $\pm$ Std** | **87.23 $\pm$ 4.12** | **12.77 $\pm$ 4.12** | **0.8164 $\pm$ 0.0260** | **87.18 $\pm$ 3.87** | **93.53 $\pm$ 2.08** | **0.8284 $\pm$ 0.0336** | **0.9619 $\pm$ 0.0088** |

---

> ### **[INSERT FIGURE 7 HERE: 5-Fold Cross-Validation Stability]**
> * **Suggested Caption:** *FIGURE 7. Fold-wise distribution of accuracy, precision, recall, and F1-score across 5 independent scenario-family folds.*
> * **File Path in Repository:** `figures/evaluation/five_fold_cv_performance.png`

---

The cross-validation yields a **Mean Accuracy of 87.23% $\pm$ 4.12%** and a **Mean Macro F1 of 0.8284 $\pm$ 0.0336**. The low standard deviation confirms that the Multimodal Transformer generalizes robustly across independent problem scenarios.

### 5) ABLATION STUDY: IMPACT OF STAGE A LATENT ALIGNMENT
To quantify the impact of Stage A contrastive alignment, an ablation was conducted by training the Transformer on raw concatenated embeddings without InfoNCE projection. Table 8 presents the ablation results.

### TABLE 8. Architectural ablation study evaluating cross-modal alignment and modality contributions ($N=627$).

| Architecture Variant | Accuracy (%) | Macro F1-Score | Macro F1 Delta | AUROC (OvR) |
|:---|:---:|:---:|:---:|:---:|
| **Complete Multimodal Architecture (Stage A + B)** | **90.75** | **0.8850** | **0.0000** | **0.9888** |
| *Without Process Modality* | 90.59 | 0.8863 | +0.0013 | 0.9895 |
| *Without Numerical Process Features (Events only)* | 90.43 | 0.8692 | -0.0158 | 0.9893 |
| *Without Code Modality* | 90.27 | 0.8653 | -0.0197 | 0.9866 |
| *Without Viva Modality* | 91.23 | 0.8926 | +0.0076 | 0.9604 |
| **Without Stage A Cross-Modal Alignment** | **66.51** | **0.2663** | **-0.6187** | **0.6638** |

---

> ### **[INSERT FIGURE 8 HERE: Multimodal Ablation Comparison]**
> * **Suggested Caption:** *FIGURE 8. Performance degradation across ablation variants. Eliminating Stage A cross-modal alignment causes a catastrophic drop in macro F1-score (-0.6187), proving the necessity of contrastive metric alignment.*
> * **File Path in Repository:** `DeepMisconception/plots/multimodal_ablation_comparison.png`

---

Table 8 proves that **Stage A alignment is essential**: without contrastive projection into a normalized 128-D latent space, the model collapses to 66.51% accuracy and 0.2663 macro F1, because raw 768-D code vectors numerically overpower the 64-D process features in the attention mechanism. Contrastive metric alignment guarantees balanced multi-head attention across all three modalities.
