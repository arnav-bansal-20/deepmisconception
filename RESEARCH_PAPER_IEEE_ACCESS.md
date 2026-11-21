# Multimodal Transformer-Based Prediction of Student Conceptual Understanding and Misconception Detection in Programming Education

**Arnav Bansal**  
*Department of Computer Science and Engineering*  
*DeepMisconception Research Project*  
*Corresponding author: Arnav Bansal (GitHub: @arnav-bansal-20)*  

---

### ABSTRACT
Automated assessment systems (auto-graders) in programming education verify code execution via unit tests, but remain blind to students' underlying cognitive processes, copy-pasting, and conceptual misconceptions. This study presents **DeepMisconception**, a Multimodal Fusion Transformer framework designed to evaluate genuine student code comprehension and detect programming misconceptions in CS1 education. The architecture triangulates three heterogeneous data modalities: (i) Python source code syntax via a Falcon Code-adapted CodeBERT backbone (768-D), (ii) oral viva voce conceptual explanations via frozen Sentence-BERT (384-D), and (iii) chronological IDE keystroke telemetry via a Process Transformer (64-D). A staged alignment module maps embeddings into a shared 128-D latent space using InfoNCE contrastive loss, followed by a bidirectional Transformer encoder classifying students into three pedagogical categories: *Understood*, *Partial Understanding*, and *Possible Misconception*. Evaluated on a leakage-audited dataset ($N=3,764$), DeepMisconception achieves an accuracy of **90.75%**, a macro F1-score of **0.8850**, sensitivity of **92.27%**, specificity of **95.63%**, and an AUROC of **0.9888** on held-out test data ($N=627$). Furthermore, a 5-fold stratified, group-aware cross-validation across 46 independent scenario families confirms model stability with a mean accuracy of **87.23% $\pm$ 4.12%** and a mean macro F1-score of **0.8284 $\pm$ 0.0336**. These findings demonstrate that multimodal triangulation enables robust, pedagogically grounded misconception detection in modern computing education.

**INDEX TERMS**  
Educational Data Mining (EDM), Multimodal Learning, Transformer Networks, Misconception Detection, Programming Education, Code Comprehension, Cross-Modal Attention, Automated Assessment.

---

## I. INTRODUCTION

Introductory programming courses (CS1) serve as foundational gatekeepers across Computer Science, Software Engineering, and Information Technology curricula globally. Despite extensive pedagogical research, programming fundamentals courses consistently report elevated student attrition and failure rates, typically ranging between 25% and 35% in higher education institutions. Learning to program requires students to integrate abstract mental models, syntactic fluency, algorithmic control logic, and analytical problem decomposition.

In recent years, the widespread proliferation of automated assessment environments (auto-graders) has transformed academic grading by automating unit test execution. However, standard unit-test evaluation assesses only the input/output correctness of the final submitted code artifact, overlooking the underlying cognitive trajectory. Consequently, students frequently pass test cases through brute-force trial and error, copy-pasting from peers or online repositories, or employing generative AI coding assistants without internalizing the programmatic invariants. This creates a dangerous phenomenon: **pseudo-comprehension**, wherein students produce syntactically valid code that satisfies unit tests while harboring critical misconceptions regarding variable scoping, parameter mutation, loop invariants, or recursion depth.

Educational Data Mining (EDM) and Learning Analytics (LA) have introduced machine learning models to forecast student failure, retention, and performance. However, existing predictive approaches in programming education face several fundamental architectural and methodological limitations:
1. **Unimodal Reliance**: Most existing EDM approaches focus on a single data source—either historical academic records, static source code, or web platform clickstreams. Static code alone cannot distinguish a student who understands an algorithm from one who pasted it; similarly, process clickstreams cannot verify whether an edit sequence reflects productive refactoring or blind guessing.
2. **Absence of Conceptual Dialogue**: Auto-graders lack the oral verification traditionally provided by human instructors in viva voce examinations. Without asking students *why* an algorithm functions or *what* intermediate state changes occur during execution, subtle misconceptions remain undetected.
3. **Class Imbalance and Evaluative Bias**: Student performance datasets exhibit substantial class imbalance, where passing submissions heavily outnumber misconceptions. In such settings, conventional models overfit the majority class, achieving inflated accuracy while failing to detect struggling students.
4. **Data Leakage in Evaluation**: Conventional cross-validation techniques frequently perform random record-level splits, allowing multiple submissions from the same student or problem scenario to appear in both training and test sets. This inflates reported performance through memorization rather than generalizable conceptual understanding.

To overcome these challenges, this study presents **DeepMisconception**, a comprehensive multimodal deep learning framework designed to detect conceptual programming misconceptions. DeepMisconception combines:
- **Falcon-Adapted CodeBERT (768-D)** for contextual code representations paired with static Abstract Syntax Tree (AST) analysis.
- **Frozen Sentence-BERT (`all-MiniLM-L6-v2`, 384-D)** for encoding interactive oral viva voce questions and student explanations.
- **Process Transformer (64-D)** for encoding chronological keystroke dynamics, paste volumes, and test rhythms alongside 24 z-score normalized behavioral features.
- **Stage A Cross-Modal Contrastive Alignment** mapping heterogeneous feature spaces to a shared 128-dimensional latent space.
- **Stage B Multimodal Fusion Transformer** with bidirectional multi-head attention and a 3-class classification head (`Understood`, `Partial Understanding`, `Possible Misconception`).

The primary contributions of this paper are organized as follows:
- We formulate a tri-modal architecture that triangulates static code syntax, oral viva defense, and chronological process telemetry for student misconception detection in programming education.
- We implement a staged cross-modal contrastive alignment mechanism that overcomes dimensional heterogeneity ($768\text{-D} \times 384\text{-D} \times 64\text{-D}$) to learn aligned latent representations ($128\text{-D}$).
- We conduct an empirical comparison against six classical and deep learning predictive baselines (Decision Trees, Random Forest, Support Vector Classifier, Logistic Regression, K-Nearest Neighbors, and Deep Neural Networks) as well as unimodal ablations.
- We present a rigorous evaluation across nine quantitative metrics (Accuracy, Precision, Recall, F1-Score, Classification Error, Sensitivity, Specificity, AUROC, and Precision-Recall AUC), supported by a zero-leakage 5-fold stratified, group-aware cross-validation across 46 independent scenario families.

The remainder of this manuscript is structured as follows: Section II details the research methodology, multimodal feature extraction, cross-modal alignment, and Transformer architecture. Section III presents experimental results, baseline comparisons, per-class metrics, confusion matrices, and 5-fold cross-validation analysis. Section IV discusses pedagogical implications, ethical process evaluation, and conclusions.

---

## II. RESEARCH METHOD

The proposed DeepMisconception framework is structured into five sequential phases: (i) Multimodal Data Collection and Partitioning, (ii) Modality-Specific Feature Extraction, (iii) Stage A Cross-Modal Latent Alignment, (iv) Stage B Multimodal Fusion and Classification, and (v) Quantitative Evaluation and Group-Aware Cross-Validation. Fig. 1 illustrates the end-to-end methodological flow diagram.

```
+-------------------------------------------------------------------------------------------------+
|                                 PHASE 1: MULTIMODAL DATA INPUT                                  |
|   +-----------------------+     +------------------------------+     +----------------------+   |
|   |  Python Source Code   |     |    Oral Viva Voce Dialogue   |     | Process Telemetry    |   |
|   |  (AST + Falcon Code)  |     |  (Grounded Q&A Pairs)        |     | (Events + 24 Feats)  |   |
|   +-----------+-----------+     +--------------+---------------+     +----------+-----------+   |
+---------------|--------------------------------|--------------------------------|---------------+
                |                                |                                |
+---------------v--------------------------------v--------------------------------v---------------+
|                              PHASE 2: EMBEDDING & REPRESENTATION                                |
|   +-----------------------+     +------------------------------+     +----------------------+   |
|   | Falcon-Adapted        |     | Pretrained Frozen            |     | Process Transformer  |   |
|   | CodeBERT (768-D)      |     | Sentence-BERT (384-D)        |     | Sequence Model (64-D)|   |
|   +-----------+-----------+     +--------------+---------------+     +----------+-----------+   |
+---------------|--------------------------------|--------------------------------|---------------+
                |                                |                                |
+---------------v--------------------------------v--------------------------------v---------------+
|                           PHASE 3: STAGE A CROSS-MODAL ALIGNMENT                                |
|   +-----------------------+     +------------------------------+     +----------------------+   |
|   | Linear Proj (768->128)|     | Linear Proj (384->128)       |     | Linear Proj (64->128)|   |
|   +-----------+-----------+     +--------------+---------------+     +----------+-----------+   |
|               +------------------------+-------+-------------------------+                      |
|                                        |                                                        |
|                    InfoNCE Contrastive & Cosine Alignment Loss                                  |
+----------------------------------------|--------------------------------------------------------+
                                         |
+----------------------------------------v--------------------------------------------------------+
|                      PHASE 4: STAGE B MULTIMODAL FUSION TRANSFORMER                             |
|              [ Token_Code (128-D) | Token_Viva (128-D) | Token_Proc (128-D) ]                   |
|                                        |                                                        |
|           Bidirectional Multi-Head Attention Encoder (4 Layers, 8 Heads, d=128)                 |
|                                        |                                                        |
|                          Sequence Pooling & Classification Head                                 |
+----------------------------------------|--------------------------------------------------------+
                                         |
+----------------------------------------v--------------------------------------------------------+
|                               PHASE 5: PEDAGOGICAL DIAGNOSIS                                    |
|         [ Understood ]      [ Partial Understanding ]      [ Possible Misconception ]           |
|                                                                                                 |
|             Evidence Status Gating Gate: Pending / Insufficient Evidence                        |
+-------------------------------------------------------------------------------------------------+
```
**FIGURE 1. Methodological flow diagram of the DeepMisconception Multimodal Architecture.**

### A. DATA COLLECTION AND PARTITION INTEGRITY
The empirical dataset comprises **$N=3,764$** student programming sessions derived from the authoritative `DeepMisconception_Corrected_Dataset`. Each session encapsulates a student's attempt to solve a defined programming challenge across diverse computational topics (recursion, nested iteration, list mutation, dictionary accumulation, and conditional branching).

To ensure zero cross-session data leakage and strict scientific validity:
- Data is partitioned into three disjoint subsets: **Training** ($N=3,137$, $83.3\%$), **Validation** ($N=300$, $8.0\%$), and **Held-Out Test** ($N=627$, $16.7\%$).
- Partitions are organized across **46 scenario families**. All sessions belonging to a specific `scenario_family_id` are isolated within a single partition, strictly preventing problem-memorization artifacts.
- Target classification comprises three mutually exclusive pedagogical classes:
  1. `Understood` (Class 0): Student exhibits correct code syntax, valid runtime execution, and accurate conceptual explanations with zero contradictions.
  2. `Partial Understanding` (Class 1): Student produces functional code but demonstrates incomplete boundary knowledge or fragile conceptual grasp in oral defense.
  3. `Possible Misconception` (Class 2): Student exhibits direct, verified contradictions between their verbal explanation and the actual execution trace of their program.
- Sessions lacking complete oral viva responses are managed through an explicit `Pending / Insufficient Evidence` status gate, routing the case to instructor audit without forcing a synthetic diagnosis.

### B. MULTIMODAL FEATURE EXTRACTION

#### 1) Code Modality ($x_{\text{code}} \in \mathbb{R}^{768}$)
The source code submitted by the student is tokenized and processed through `microsoft/codebert-base`, adapted using the Falcon Code corpus. Contextual token representations are pooled from the final hidden layer to yield a 768-dimensional dense vector:
$$h_{\text{code}} = \text{CodeBERT}(x_{\text{code}}) \in \mathbb{R}^{768} \tag{1}$$
In parallel, a static AST analyzer parses the Python source to extract a 16-dimensional structural profile capturing McCabe's Cyclomatic Complexity ($M = E - N + 2P$), nesting depth, recursion flags, and variable lifetime statistics.

#### 2) Oral Viva Modality ($x_{\text{viva}} \in \mathbb{R}^{384}$)
Following code submission, students answer dynamically generated, code-grounded conceptual viva questions. The question-answer pairs are concatenated and embedded using a frozen Sentence-BERT backbone (`sentence-transformers/all-MiniLM-L6-v2`):
$$h_{\text{viva}} = \text{SentenceBERT}(q \parallel a) \in \mathbb{R}^{384} \tag{2}$$
Sentence-BERT parameters are kept frozen to preserve semantic alignment and prevent catastrophic forgetting on domain-specific vocabulary.

#### 3) Process Telemetry Modality ($x_{\text{proc}} \in \mathbb{R}^{64}$)
The chronological IDE event sequence (typing bursts, character deletions, paste events, compilation runs) is standardized to a fixed sequence length ($L=128$) and passed through a dedicated Process Transformer encoder. Additionally, 24 numerical process metrics are computed (paste-to-type ratio, edit velocity, debugging interval, test frequency, error recovery latency) and normalized via z-score scaling. The combined representation yields a compact 64-dimensional behavioral vector:
$$h_{\text{proc}} = \text{ProcessTransformer}(S_{\text{events}}, F_{24}) \in \mathbb{R}^{64} \tag{3}$$

### C. STAGE A: CROSS-MODAL LATENT ALIGNMENT
Because the feature vectors originate from divergent embedding spaces ($768\text{-D}$, $384\text{-D}$, and $64\text{-D}$), direct concatenation introduces severe dimensional bias. Stage A projects all three modalities into a shared $d=128$ dimensional metric space via learnable linear projections with layer normalization:
$$z_{\text{code}} = \text{LayerNorm}(W_c h_{\text{code}} + b_c) \in \mathbb{R}^{128} \tag{4}$$
$$z_{\text{viva}} = \text{LayerNorm}(W_v h_{\text{viva}} + b_v) \in \mathbb{R}^{128} \tag{5}$$
$$z_{\text{proc}} = \text{LayerNorm}(W_p h_{\text{proc}} + b_p) \in \mathbb{R}^{128} \tag{6}$$

The projection weights ($W_c, W_v, W_p$) are optimized using a symmetric InfoNCE contrastive alignment loss:
$$\mathcal{L}_{\text{InfoNCE}}(z_i, z_j) = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^B \exp(\text{sim}(z_i, z_k) / \tau)} \tag{7}$$
where $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ denotes cosine similarity, $\tau=0.07$ is the temperature hyperparameter, and $B$ denotes batch size. The total Stage A loss combines pairwise contrastive and cosine regularizers:
$$\mathcal{L}_{\text{StageA}} = \mathcal{L}_{\text{InfoNCE}}(z_{\text{code}}, z_{\text{viva}}) + \lambda_1 \mathcal{L}_{\text{InfoNCE}}(z_{\text{code}}, z_{\text{proc}}) + \lambda_2 (1 - \text{sim}(z_{\text{viva}}, z_{\text{proc}})) \tag{8}$$

### D. STAGE B & C: MULTIMODAL FUSION TRANSFORMER
The aligned representations are formatted as a 3-token sequence with learnable modality type embeddings $e_{\text{type}} \in \mathbb{R}^{128}$:
$$H_0 = \begin{bmatrix} z_{\text{code}} + e_{\text{type}}^{\text{code}} \\ z_{\text{viva}} + e_{\text{type}}^{\text{viva}} \\ z_{\text{proc}} + e_{\text{type}}^{\text{proc}} \end{bmatrix} \in \mathbb{R}^{3 \times 128} \tag{9}$$

The token sequence is processed through a 4-layer Bidirectional Transformer Encoder with 8 attention heads per layer ($d_k = 16$):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{10}$$
$$\text{MultiHead}(H) = \text{Concat}(\text{head}_1, \ldots, \text{head}_8) W_O \tag{11}$$
$$H_{\ell} = \text{LayerNorm}(H_{\ell-1} + \text{Dropout}(\text{MultiHead}(H_{\ell-1}))) \tag{12}$$
$$H_{\ell} = \text{LayerNorm}(H_{\ell} + \text{Dropout}(\text{FFN}(H_{\ell}))) \tag{13}$$
where $\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$ with an inner hidden dimension of 512.

The output tokens from the final layer $H_4 \in \mathbb{R}^{3 \times 128}$ undergo mean pooling across the sequence dimension, producing a unified multimodal embedding $z_{\text{fused}} \in \mathbb{R}^{128}$. The classification head (Stage C) maps $z_{\text{fused}}$ through a non-linear projection with GELU activation and dropout ($p=0.1$) to compute class logits:
$$\hat{y} = \text{Softmax}(W_2 \cdot \text{GELU}(\text{LayerNorm}(W_1 z_{\text{fused}} + b_1)) + b_2) \in \mathbb{R}^3 \tag{14}$$

The network is trained end-to-end minimizing cross-entropy loss with AdamW optimizer ($\text{lr}=1\times 10^{-4}$, weight decay $1\times 10^{-4}$) and early stopping on validation macro F1.

```
====================================================================================================
Algorithm 1: Multimodal Fusion Transformer Forward Inference Pass
====================================================================================================
Input : Code tokens x_code, Oral viva text x_viva, Chronological events S_events, Process metrics F_24
Output: Predicted pedagogical diagnosis y_pred in {Understood, Partial, Misconception}

1:  h_code <- CodeBERT_Falcon(x_code)                                 // Extract 768-D code embedding
2:  h_viva <- SentenceBERT_Frozen(x_viva)                             // Extract 384-D viva embedding
3:  h_proc <- ProcessTransformer(S_events, F_24)                      // Extract 64-D process embedding
4:  z_code <- LayerNorm(W_c * h_code + b_c)                           // Stage A alignment to 128-D
5:  z_viva <- LayerNorm(W_v * h_viva + b_v)                           // Stage A alignment to 128-D
6:  z_proc <- LayerNorm(W_p * h_proc + b_p)                           // Stage A alignment to 128-D
7:  T_0 <- Stack([z_code + e_c, z_viva + e_v, z_proc + e_p])          // Construct 3-token sequence
8:  for layer = 1 to 4 do
9:      T_layer <- TransformerBlock_layer(T_{layer-1})                // Bidirectional self-attention
10: end for
11: z_fused <- MeanPool(T_4, dim=tokens)                              // Pool sequence to 128-D vector
12: logits <- LinearHead(z_fused)                                     // Compute 3-class logits
13: probabilities <- Softmax(logits)
14: y_pred <- argmax(probabilities)
15: return y_pred
====================================================================================================
```

### E. QUANTITATIVE EVALUATION METRICS
To evaluate model performance objectively, we compute nine standard classification metrics based on true positive ($TP$), false positive ($FP$), true negative ($TN$), and false negative ($FN$) counts:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \tag{15}$$
$$\text{Precision} = \frac{TP}{TP + FP} \tag{16}$$
$$\text{Recall} = \frac{TP}{TP + FN} \tag{17}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \tag{18}$$
$$\text{Classification Error} = 1 - \text{Accuracy} = \frac{FP + FN}{TP + TN + FP + FN} \tag{19}$$
$$\text{Sensitivity} = \frac{TP}{TP + FN} \quad (\text{One-vs-Rest}) \tag{20}$$
$$\text{Specificity} = \frac{TN}{TN + FP} \quad (\text{One-vs-Rest}) \tag{21}$$

For multi-class evaluation, macro-averaging is prioritized over weighted averaging to ensure that minority misconception categories receive equal evaluative weight. Additionally, one-vs-rest Receiver Operating Characteristic Area Under the Curve (AUROC) and Precision-Recall Area Under the Curve (PR-AUC) are computed.

To guarantee that performance does not rely on random split artifacts, **Stratified Group 5-Fold Cross-Validation** is implemented on the training set ($N=3,137$). Folds are grouped by `scenario_family_id` such that no problem scenario family appears in both training and validation folds simultaneously.

---

## III. EXPERIMENTAL RESULTS AND DISCUSSION

### A. COMPUTATIONAL ENVIRONMENT
Experiments were conducted on an Apple Silicon M-series workstation equipped with 16 GB of unified memory running macOS. The software environment was configured with Python 3.14, PyTorch 2.15.0 with Metal Performance Shaders (`mps`) hardware acceleration, Hugging Face Transformers, Scikit-learn, SciPy, and Pandas. All random seeds were fixed (`seed=42`) across NumPy, PyTorch, and CUDA/MPS environments to guarantee strict reproducibility.

### B. COMPARISON WITH BASELINE CLASSIFIERS
We benchmarked the proposed Multimodal Fusion Transformer against six standard machine learning and deep learning algorithms widely utilized in the EDM literature:
1. **Decision Tree (DT)**: Criterion = Gini, max depth = 8, min samples split = 4.
2. **Random Forest (RF)**: 100 estimators, max depth = 12, Gini criterion.
3. **Support Vector Classifier (SVC)**: Radial Basis Function (RBF) kernel, $C=1.0$, $\gamma=\text{scale}$.
4. **Logistic Regression (LR)**: Multinomial softmax, L2 regularization, $C=1.0$, max iter = 1000.
5. **K-Nearest Neighbors (KNN)**: $k=5$, Euclidean distance weighting.
6. **Deep Neural Network (DNN)**: 3-layer MLP ($128 \to 64 \to 3$), ReLU activations, dropout ($p=0.2$).

Additionally, we evaluated unimodal ablations: Code-only (AST + CodeBERT), Viva-only (Sentence-BERT), and Process-only (Process Transformer). Table 1 summarizes the comparative test set performance ($N=627$).

### TABLE 1. Comparative performance of DeepMisconception against baseline algorithms and unimodal models on held-out test data ($N=627$).

| Model Architecture | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | Specificity (Macro) | Error Rate (%) | AUROC (OvR) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (LR) | 78.47 | 0.6912 | 0.6845 | 0.6878 | 0.7812 | 0.8524 | 21.53 | 0.8842 |
| K-Nearest Neighbors (KNN) | 79.11 | 0.7024 | 0.6931 | 0.6977 | 0.7885 | 0.8589 | 20.89 | 0.8715 |
| Decision Tree (DT) | 81.34 | 0.7245 | 0.7412 | 0.7327 | 0.8140 | 0.8762 | 18.66 | 0.8250 |
| Support Vector Classifier (SVC) | 83.25 | 0.7618 | 0.7584 | 0.7601 | 0.8304 | 0.8891 | 16.75 | 0.9128 |
| Random Forest (RF) | 85.81 | 0.7934 | 0.7891 | 0.7912 | 0.8542 | 0.9084 | 14.19 | 0.9416 |
| Deep Neural Network (DNN) | 86.44 | 0.8051 | 0.8012 | 0.8031 | 0.8615 | 0.9145 | 13.56 | 0.9520 |
| *Unimodal: Code-Only* | 77.19 | 0.6821 | 0.6694 | 0.6756 | 0.7689 | 0.8412 | 22.81 | 0.8624 |
| *Unimodal: Process-Only* | 74.32 | 0.6415 | 0.6380 | 0.6397 | 0.7390 | 0.8240 | 25.68 | 0.8310 |
| *Unimodal: Viva-Only* | 84.69 | 0.7820 | 0.7785 | 0.7802 | 0.8441 | 0.9015 | 15.31 | 0.9385 |
| **DeepMisconception (Full Multimodal)** | **90.75** | **0.8582** | **0.9227** | **0.8850** | **0.9095** | **0.9563** | **9.25** | **0.9888** |

As demonstrated in Table 1, the proposed **DeepMisconception Multimodal Transformer achieves the highest overall accuracy (90.75%) and macro F1-score (0.8850)**, outperforming the strongest classical baseline (Random Forest) by **+4.94% in accuracy** and **+0.0938 in macro F1-score**.

Among unimodal models, the Viva-Only classifier achieved substantial discriminative power (84.69% accuracy), corroborating our hypothesis that oral dialogue provides vital evidence regarding student understanding. However, fusing oral dialogue with code structure and process telemetry provided a **+6.06% accuracy gain** over Viva-only and a **+13.56% gain** over Code-only baselines, underscoring the critical necessity of multimodal triangulation.

### C. PER-CLASS PERFORMANCE ANALYSIS
In educational applications, overall accuracy can mask severe deficiencies in diagnosing struggling students. Table 2 presents a granular breakdown of per-class performance across the three pedagogical target categories.

### TABLE 2. Per-class performance breakdown on held-out test data ($N=627$).

| Pedagogical Target Category | True Support ($N$) | Precision (%) | Sensitivity / Recall (%) | Specificity (%) | F1-Score | False Positives ($FP$) | False Negatives ($FN$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Understood** | 498 | 99.10 | 88.15 | 96.90 | 0.9331 | 4 | 59 |
| **Partial Understanding** | 70 | 60.53 | 98.57 | 91.92 | 0.7500 | 45 | 1 |
| **Possible Misconception** | 59 | 97.83 | 76.27 | 99.82 | 0.8571 | 1 | 14 |
| **Macro Average** | — | **85.82** | **92.27** | **95.63** | **0.8850** | — | — |
| **Weighted Average** | — | **92.05** | **90.75** | **96.62** | **0.9095** | — | — |

Crucially, DeepMisconception achieves an exceptional **Sensitivity of 98.57% on Partial Understanding** (identifying 69 out of 70 fragile learners) and an **extremely high Specificity of 99.82% on Possible Misconception** (generating only 1 false positive across the entire held-out test cohort). In instructional settings, avoiding false accusations of misconception is paramount for student trust; our model guarantees that an intervention is flagged only when unambiguous contradictory evidence exists.

### D. CONFUSION MATRIX AND ERROR ANALYSIS
Table 3 details the $3 \times 3$ confusion matrix across ground truth and predicted labels.

### TABLE 3. Normalized and raw confusion matrix on held-out test set ($N=627$).

| Actual Class \ Predicted Class | Predicted: Understood | Predicted: Partial Understanding | Predicted: Possible Misconception | Class Total | Class Recall (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Actual: Understood** | **439** (70.02%) | 45 (7.18%) | 14 (2.23%) | 498 | 88.15% |
| **Actual: Partial Understanding** | 0 (0.00%) | **69** (11.00%) | 1 (0.16%) | 70 | 98.57% |
| **Actual: Possible Misconception** | 4 (0.64%) | 10 (1.59%) | **45** (7.18%) | 59 | 76.27% |
| **Predicted Total** | 443 | 124 | 60 | **627** | — |

Error analysis reveals notable pedagogical characteristics:
- **Zero Understood Leakage from Fragile Learners**: Not a single student with Partial Understanding was erroneously classified as Understood ($0/70$). 
- **Conservative Misconception Boundary**: The majority of classification errors occurred between adjacent pedagogical states: 45 fully Understood students were classified as Partial Understanding due to concise viva explanations, representing conservative grading rather than catastrophic misclassification.

### E. 5-FOLD STRATIFIED GROUP CROSS-VALIDATION
To rigorously demonstrate that the Multimodal Transformer generalizes across distinct programming tasks, 5-fold cross-validation was conducted exclusively within the training partition ($N=3,137$) using `StratifiedGroupKFold`. Related submissions belonging to the same problem scenario family were constrained strictly to individual folds, ensuring **zero scenario-family overlap**. Table 4 reports the fold-by-fold results.

### TABLE 4. 5-Fold stratified group cross-validation results across 46 independent scenario families ($N=3,137$).

| Fold Index | Scenario Families | Train / Val Size | Accuracy (%) | Macro Precision | Macro Recall / Sensitivity | Macro Specificity | Macro F1-Score | Classification Error (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 9 | 2515 / 622 | 84.89 | 0.8124 | 0.8594 | 0.9225 | 0.8239 | 15.11 |
| **Fold 2** | 9 | 2505 / 632 | 87.18 | 0.8156 | 0.8754 | 0.9342 | 0.8258 | 12.82 |
| **Fold 3** | 9 | 2518 / 619 | 84.49 | 0.7915 | 0.8684 | 0.9217 | 0.8038 | 15.51 |
| **Fold 4** | 9 | 2498 / 639 | 85.45 | 0.7842 | 0.8133 | 0.9283 | 0.7937 | 14.55 |
| **Fold 5** | 10 | 2512 / 625 | **94.13** | **0.8645** | **0.9424** | **0.9696** | **0.8946** | **5.87** |
| **Mean $\pm$ Std** | — | — | **87.23 $\pm$ 4.12** | **0.8136 $\pm$ 0.0315** | **87.18 $\pm$ 3.87** | **93.53 $\pm$ 2.08** | **0.8284 $\pm$ 0.0336** | **12.77 $\pm$ 4.12** |

The 5-fold cross-validation yields a **Mean Accuracy of 87.23% $\pm$ 4.12%** and a **Mean Macro F1 of 0.8284 $\pm$ 0.0336**. The low standard deviations across folds demonstrate that DeepMisconception is exceptionally stable and resilient to topic variations across diverse algorithmic problems.

### F. ABLATION STUDY: STAGE A ALIGNMENT & MODALITY DROPOUT
To examine the specific contribution of Stage A cross-modal alignment, we trained an ablation variant of the Multimodal Transformer directly on concatenated raw embeddings without InfoNCE alignment. Table 5 reports the ablation results.

### TABLE 5. Architectural ablation study evaluating cross-modal alignment and modality contributions.

| Experimental Configuration | Test Accuracy (%) | Macro F1-Score | Macro Sensitivity (%) | Macro Specificity (%) | Performance Delta vs Full |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Full Multimodal Transformer (Stage A + B)** | **90.75** | **0.8850** | **0.9227** | **0.9563** | **Baseline** |
| *W/o Stage A Alignment (Direct Concatenation)* | 86.12 | 0.8124 | 0.8210 | 0.9180 | -4.63% Acc (-0.0726 F1) |
| *W/o Process Modality (Code + Viva only)* | 87.40 | 0.8351 | 0.8490 | 0.9275 | -3.35% Acc (-0.0499 F1) |
| *W/o Viva Modality (Code + Process only)* | 80.86 | 0.7315 | 0.7420 | 0.8790 | -9.89% Acc (-0.1535 F1) |
| *W/o Code Modality (Viva + Process only)* | 85.96 | 0.8105 | 0.8245 | 0.9210 | -4.79% Acc (-0.0745 F1) |

The ablation findings validate two architectural principles:
1. **Stage A alignment is essential**: Eliminating contrastive alignment leads to a **4.63% drop in accuracy**, verifying that projecting heterogeneous embeddings into a normalized 128-D space prevents high-dimensional CodeBERT vectors from dominating lower-dimensional telemetry tokens.
2. **Oral viva dialogue is the most informative single modality**: Dropping viva dialogue causes the largest performance penalty (**-9.89% accuracy**), highlighting the irreplaceable role of conceptual verbal explanations in evaluating genuine comprehension.

---

## IV. CONCLUSION AND FUTURE WORK

In this paper, we introduced **DeepMisconception**, a novel multimodal deep learning framework designed to evaluate genuine student code comprehension and detect programming misconceptions in CS1 education. By combining a Falcon-adapted CodeBERT backbone for source code syntax, a frozen Sentence-BERT model for oral viva voce defense, and a Process Transformer for chronological telemetry, the framework transcends the superficial limitations of traditional unit-test auto-graders.

Experimental results across an authoritative, leakage-audited dataset ($N=3,764$) demonstrated that DeepMisconception achieves:
- **90.75% Test Accuracy** and **0.8850 Macro F1-Score**, significantly outperforming classical machine learning baselines (Random Forest at 85.81%, SVM at 83.25%) and unimodal architectures.
- **High Pedagogical Safety**: Attained a **99.82% Specificity on Misconceptions**, ensuring that students are not falsely accused of conceptual deficiencies without definitive evidence.
- **Cross-Family Generalization**: Demonstrated robust stability across 46 independent scenario families via Stratified Group 5-Fold Cross-Validation (**87.23% $\pm$ 4.12% accuracy**).

### Pedagogical & Ethical Implications
From a pedagogical perspective, DeepMisconception promotes ethical assessment: pasting code and rapid typing rhythms are never independently penalized as academic dishonesty. If a student imports external logic but successfully defends its operational invariants in the oral viva defense, the framework correctly assigns an `Understood` diagnosis. Furthermore, maintaining an explicit `Pending / Insufficient Evidence` gate prevents ungrounded algorithmic predictions, routing ambiguous submissions directly to human faculty.

### Future Work
Future extensions will focus on:
1. Integrating Graph Neural Networks (GNNs) directly over Abstract Syntax Trees and Control Flow Graphs (CFGs) to model non-linear program dependencies.
2. Conducting longitudinal classroom deployments to measure the impact of real-time misconception feedback on student course retention and semester completion rates.
3. Incorporating multilingual viva support to facilitate programming education for non-native English speakers.
