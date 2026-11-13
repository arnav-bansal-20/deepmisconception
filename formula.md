# DeepMisconception: Mathematical & Algorithmic Formulas

This document provides a comprehensive mathematical reference of all formulas, weighting schemes, neural equations, and statistical metrics implemented in the **DeepMisconception** platform.

---

## 1. Multimodal Composite Comprehension Index ($S_{\text{overall}}$)

The core diagnostic engine fuses four orthogonal modalities of student evidence into a single continuous score bounded in $[0.0, 1.0]$:

$$S_{\text{overall}} = w_{\text{ast}} \cdot S_{\text{ast}} + w_{\text{exec}} \cdot S_{\text{exec}} + w_{\text{viva}} \cdot S_{\text{viva}} + w_{\text{proc}} \cdot S_{\text{proc}}$$

### Weights Configuration
$$\sum w_i = 0.15 + 0.20 + 0.45 + 0.20 = 1.00$$

| Symbol | Component | Weight | Implementation File |
|---|---|---|---|
| $w_{\text{ast}}$ | Abstract Syntax Tree Structural Score | **0.15** | [`backend/app/config.py`](backend/app/config.py) |
| $w_{\text{exec}}$ | Sandbox Execution Fidelity Score | **0.20** | [`backend/app/config.py`](backend/app/config.py) |
| $w_{\text{viva}}$ | Code-Grounded Oral Viva Reasoning Score | **0.45** | [`backend/app/config.py`](backend/app/config.py) |
| $w_{\text{proc}}$ | Process Telemetry Behavioral Score | **0.20** | [`backend/app/config.py`](backend/app/config.py) |

*Code Reference*: [`backend/app/services/comprehension_engine.py`](backend/app/services/comprehension_engine.py)

---

## 2. Process Telemetry Scoring System ($S_{\text{process}}$)

Process telemetry evaluates fine-grained editor behavioral dynamics. The total process score is a weighted linear combination of four sub-dimensions:

$$S_{\text{process}} = 0.30 \cdot S_{\text{INC}} + 0.30 \cdot S_{\text{DEBUG}} + 0.25 \cdot S_{\text{REV}} + 0.15 \cdot S_{\text{TEST}}$$

*Code Reference*: [`backend/app/services/process_telemetry.py`](backend/app/services/process_telemetry.py)

```
                            S_process (100%)
                                   │
      ┌─────────────────┬──────────┴──────────┬─────────────────┐
      ▼                 ▼                     ▼                 ▼
   S_INC (30%)     S_DEBUG (30%)         S_REV (25%)       S_TEST (15%)
Incremental Dev    Error Recovery       Revision Cycles   Sandbox Testing
```

---

### 2.1 Incremental Development Score ($S_{\text{INC}}$)

Accounts for continuous sequential edits while discounting non-incremental block pasting:

#### Step 1: Base Incremental Score ($S_{\text{INC-BASE}}$)
Derived from sequential edit volume:
$$S_{\text{INC-BASE}} = \begin{cases}
1.00 & \text{if } \text{edit count} \ge 15 \\
0.92 & \text{if } \text{edit count} \ge 8 \\
0.85 & \text{if } \text{edit count} \ge 3 \\
0.80 & \text{otherwise (baseline for clean complete drafts)}
\end{cases}$$

#### Step 2: Paste Adjustment Factor ($\text{PASTE-FACTOR}$)
$$\text{PASTE-FACTOR} = \min\left(1.0, \max\left(0.0, \frac{1.0 - \text{paste ratio}}{0.70}\right)\right)$$

#### Step 3: Final Incremental Score
$$S_{\text{INC}} = \begin{cases}
S_{\text{INC-BASE}} & \text{if } \text{paste ratio} \le 0.30 \\
S_{\text{INC-BASE}} \times \text{PASTE-FACTOR} & \text{if } 0.30 < \text{paste ratio} < 1.00 \\
0.0 & \text{if } \text{paste ratio} \ge 1.00
\end{cases}$$

- $\text{paste ratio} \in [0.00, 0.30] \implies \text{factor} = 1.00$ (no reduction)
- $\text{paste ratio} = 0.50 \implies \text{factor} \approx 0.714$ (partial proportional reduction)
- $\text{paste ratio} = 0.70 \implies \text{factor} \approx 0.429$ (proportional reduction)
- $\text{paste ratio} = 1.00 \implies \text{factor} = 0.00$ (score forced to exactly $0.0$)

---

### 2.2 Debugging & Error Recovery Score ($S_{\text{DEBUG}}$)

Assesses how the student handles runtime errors, syntax errors, and test failures:

$$S_{\text{DEBUG}} = \begin{cases}
0.90 & \text{if } \text{error count} = 0 \text{ (clean execution without errors)} \\
\min\left(1.0, \max\left(0.40, 0.50 + 0.50 \cdot \text{error recovery rate}\right)\right) & \text{if } \text{error count} > 0
\end{cases}$$

Where the error recovery rate measures whether error events were resolved by subsequent corrective edits or a clean execution:
$$\text{error recovery rate} = \frac{\sum_{i \in \text{errors}} \mathbb{I}(\text{followed by edit or clean run})}{\text{total errors}}$$

---

### 2.3 Revision Iteration Score ($S_{\text{REV}}$)

Measures iterative refinement loops (cycles of editing code after running or checking output):

$$S_{\text{REV}} = \begin{cases}
1.00 & \text{if } \text{revision cycles} \ge 2 \\
0.88 & \text{if } \text{revision cycles} = 1 \\
0.82 & \text{if } \text{edit count} \ge 5 \\
0.75 & \text{baseline for direct first-pass submissions}
\end{cases}$$

---

### 2.4 Sandbox Testing Score ($S_{\text{TEST}}$)

Measures active verification habits inside the isolated execution environment:

$$S_{\text{TEST}} = \begin{cases}
1.00 & \text{if } \text{run count} \ge 3 \\
0.90 & \text{if } \text{run count} = 2 \\
0.82 & \text{if } \text{run count} = 1 \\
0.70 & \text{if } \text{run count} = 0 \text{ (baseline when final submission executes)}
\end{cases}$$

---

### 2.5 Raw Telemetry Metrics & Extraction Ratios

$$\text{Paste Ratio} = \frac{\text{total chars pasted}}{\text{total chars pasted} + \text{total chars edited}} \in [0.0, 1.0]$$

$$\text{Burstiness Ratio} = \frac{\sum \mathbb{I}(\Delta t_i < 0.5\text{ s})}{\text{total edits}} \in [0.0, 1.0]$$

$$\text{Typing Speed (CPM)} = \frac{\text{total chars edited}}{\max(0.1, \text{duration minutes})}$$

---

## 3. AST Structural Score ($S_{\text{ast}}$)

Evaluates the algorithmic sophistication, syntactic validity, and programmatic structure of the student's solution:

$$S_{\text{ast}} = \begin{cases}
\min\left(1.00, 0.75 + 0.10 \cdot \mathbb{I}(\text{cyclomatic complexity} \ge 2) + 0.10 \cdot \mathbb{I}(\text{loops or functions present})\right) & \text{if AST is syntactically valid} \\
0.15 & \text{if syntax error / parse failure}
\end{cases}$$

Where Cyclomatic Complexity $M$ is derived from the AST control flow graph:
$$M = E - N + 2P$$
- $E$ = edges in the control-flow graph
- $N$ = nodes in the control-flow graph
- $P$ = connected components (typically $1$)

*Code Reference*: [`backend/app/services/ast_analyzer.py`](backend/app/services/ast_analyzer.py)

---

## 4. Sandbox Execution Fidelity Score ($S_{\text{exec}}$)

Evaluates runtime correctness in an isolated subprocess execution environment:

$$S_{\text{exec}} = \begin{cases}
1.00 & \text{exit code } 0 \text{ with clean matching output} \\
0.50 & \text{output produced, but non-zero exit code} \\
0.25 & \text{runtime exception / error stacktrace generated} \\
0.10 & \text{timeout / memory breach / killed process}
\end{cases}$$

*Code Reference*: [`backend/app/services/safe_executor.py`](backend/app/services/safe_executor.py)

---

## 5. Oral Viva Reasoning Score ($S_{\text{viva}}$)

Evaluates student verbal explanations to dynamically generated oral viva questions:

$$S_{\text{viva}} = \frac{1}{N} \sum_{i=1}^{N} \text{score}(a_i)$$

Where each student answer $a_i$ is evaluated by the LLM Evaluator against the code AST and runtime facts:

$$\text{score}(a_i) = \begin{cases}
1.00 & \text{"Correct"} \quad (\text{accurate conceptual and mechanical explanation}) \\
0.65 & \text{"Partially Correct"} \quad (\text{fragile boundary understanding / incomplete reason}) \\
0.40 & \text{"Insufficient Evidence"} \quad (\text{evasive / non-committal answer}) \\
0.15 & \text{"Incorrect"} \quad (\text{hallucinated or mechanically flawed explanation})
\end{cases}$$

*When viva answers are not yet submitted, $S_{\text{viva}} = 0.50$ (neutral unverified prior).*

*Code Reference*: [`backend/app/services/answer_evaluator.py`](backend/app/services/answer_evaluator.py)

---

## 6. Diagnostic Verdict Classification & Confidence Calibration

### 6.1 Verdict Rules

The categorical verdict is determined by combining $S_{\text{overall}}$ with cross-modal consistency rules:

$$\text{Verdict} = \begin{cases}
\text{"Insufficient Evidence"} & \text{if } N_{\text{answers}} = 0 \text{ or all answers empty whitespace} \\
\text{"Possible Misconception"} & \text{if } (S_{\text{overall}} < 0.50 \text{ with errors}) \lor (N_{\text{incorrect}} \ge 2) \lor (N_{\text{contradictions}} \ge 2) \\
\text{"Understood"} & \text{if } S_{\text{overall}} \ge 0.70 \land N_{\text{incorrect}} = 0 \land N_{\text{contradictions}} = 0 \land \forall a \in \text{valid}: a = \text{"Correct"} \\
\text{"Partial Understanding"} & \text{otherwise (default balanced state: } 0.50 \le S_{\text{overall}} < 0.70\text{)}
\end{cases}$$

### 6.2 Diagnostic Confidence ($\text{Conf}$)

$$\text{Conf} = \begin{cases}
\max(0.75, \min(0.98, S_{\text{overall}})) & \text{for "Understood"} \\
\max(0.70, \min(0.95, 0.60 + 0.10 \cdot \max(N_{\text{incorrect}}, N_{\text{misconceptions}}))) & \text{for "Possible Misconception"} \\
\max(0.60, \min(0.85, S_{\text{overall}} + 0.05)) & \text{for "Partial Understanding"} \\
0.35 & \text{for "Insufficient Evidence"}
\end{cases}$$

---

## 7. Deep Learning: Multimodal Cross-Attention Transformer

Implemented in PyTorch in [`backend/app/ml/multimodal_transformer.py`](backend/app/ml/multimodal_transformer.py).

### 7.1 Input Modality Projections
Transforms heterogeneous raw feature spaces to a shared latent dimension $d = 32$:

$$h_{\text{code}} = \text{Dropout}\left(\text{LayerNorm}\left(\text{ReLU}\left(X_{\text{code}} W_{\text{code}} + b_{\text{code}}\right)\right)\right), \quad X_{\text{code}} \in \mathbb{R}^{16}$$

$$h_{\text{viva}} = \text{Dropout}\left(\text{LayerNorm}\left(\text{ReLU}\left(X_{\text{viva}} W_{\text{viva}} + b_{\text{viva}}\right)\right)\right), \quad X_{\text{viva}} \in \mathbb{R}^{16}$$

$$h_{\text{proc}} = \text{Dropout}\left(\text{LayerNorm}\left(\text{ReLU}\left(X_{\text{proc}} W_{\text{proc}} + b_{\text{proc}}\right)\right)\right), \quad X_{\text{proc}} \in \mathbb{R}^{8}$$

### 7.2 Inter-Modality Sequence Encoding
Forms token sequence $H_{\text{seq}} = [h_{\text{code}}, h_{\text{viva}}, h_{\text{proc}}] \in \mathbb{R}^{B \times 3 \times d}$ and passes through a 2-layer Transformer Encoder:

$$H_{\text{enc}} = \text{TransformerEncoder}(H_{\text{seq}})$$

### 7.3 Cross-Attention Mechanism ($Q = \text{Code}, K, V = \text{Explanation}$)
Code query attends to verbal explanation key-value tokens:

$$Q = h_{\text{code}} W_Q, \quad K = h_{\text{viva}} W_K, \quad V = h_{\text{viva}} W_V$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

$$h_{\text{attended}} = \text{LayerNorm}\left(h_{\text{code}} + \text{Attention}(Q, K, V)\right)$$

### 7.4 Dense Fusion & Classification Head
$$H_{\text{fused}} = [h_{\text{attended}} \,\|\, \bar{H}_{\text{enc}} \,\|\, h_{\text{proc}}] \in \mathbb{R}^{B \times 3d}$$

$$\text{Logits} = W_2 \cdot \text{Dropout}\left(\text{GELU}\left(W_1 H_{\text{fused}} + b_1\right)\right) + b_2, \quad \text{Logits} \in \mathbb{R}^{B \times 3}$$

$$P(y = c \mid X) = \frac{\exp(\text{Logits}_c)}{\sum_{j=1}^{3} \exp(\text{Logits}_j)}, \quad c \in \{\text{Understood}, \text{Partial}, \text{Misconception}\}$$

