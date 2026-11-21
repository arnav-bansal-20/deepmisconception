# Process Transformer Standalone Training Report

## 1. Overview and Scope

This report documents the standalone training of the **Process Transformer** for DeepMisconception. 
Per the master project requirements, neither CodeBERT, the Sentence Transformer, nor the Multimodal Fusion Transformer were modified or trained during this phase.

## 2. Dataset Partitions and Class Balancing

- **Dataset Source:** `DeepMisconception_Corrected_Dataset/`
- **Training Partition:** `train/` (N=3,537 total sessions; 3,137 sufficient evidence, 400 insufficient evidence excluded from supervised loss)
- **Validation Partition:** `validate/` (N=770 total sessions; 693 sufficient evidence, 77 insufficient evidence excluded)
- **Test Partition:** `test/` (N=693 total sessions; 627 sufficient evidence, 66 insufficient evidence excluded; held completely untouched)
- **Scenario Families:** 46 train, 10 validation, 9 test (100% disjoint, zero leakage)
- **Supervised Training Distribution:**
  - `Understood`: 2,251 (71.76%)
  - `Partial Understanding`: 587 (18.71%)
  - `Possible Misconception`: 299 (9.53%)
- **Inverse Class Weights:** `[0.4645, 1.7814, 3.4972]` applied via `nn.CrossEntropyLoss(weight=...)`

## 3. Model Architecture & Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Event Vocabulary Size** | 13 tokens (12 types + `<PAD>` at index 0) |
| **Max Sequence Length** | 32 (0% sequence truncation across all 5,000 sessions) |
| **Event Embedding Dimension** | 48 |
| **Temporal Feature Dimension** | 16 (log1p time gap, session elapsed time, chars inserted, chars deleted) |
| **Token Representation Dimension** | 64 |
| **Transformer Layers** | 2 layers |
| **Attention Heads** | 4 heads |
| **Feedforward Dimension** | 256 |
| **Numerical Feature MLP** | `Linear(24, 64) -> LayerNorm -> GELU -> Dropout(0.15) -> Linear(64, 64) -> LayerNorm` |
| **Fusion Mechanism** | Learned Gated Fusion: `gate * temporal + (1 - gate) * numerical` |
| **Output Process Embedding** | Exactly 64-D `[batch_size, 64]` |
| **Classification Head** | `Linear(64, 3)` |
| **Total Trainable Parameters** | 123,715 parameters |
| **Optimizer** | AdamW ($lr=10^{-3}$, weight_decay=$10^{-4}$) with CosineAnnealingLR |
| **Early Stopping** | Patience = 6 epochs monitoring validation loss |

## 4. Empirical Training Results Across Experiments

| Experiment | Model Architecture | Trainable Params | Best Epoch | Best Val Loss | Val Macro-F1 | Val Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Experiment A** | Event-Only Process Transformer | 105,155 | 5 | 1.1030 | 0.3219 | 0.6364 |
| **Experiment B** | Numerical-Only Baseline MLP | 6,211 | 1 | 1.0469 | 0.4366 | 0.5960 |
| **Experiment C** | **Combined Process Transformer** | **123,715** | **2** | **1.0435** | **0.4280** | **0.5743** |

## 5. Checkpoints and Artifacts

- Best Combined Model: `DeepMisconception/models/process_transformer/combined_process_transformer_best.pt`
- Numerical Baseline Checkpoint: `DeepMisconception/models/process_transformer/numerical_baseline_best.pt`
- Event-Only Checkpoint: `DeepMisconception/models/process_transformer/event_only_transformer_best.pt`
- Normalization Statistics: `DeepMisconception/models/process_transformer/normalization_stats.json`
- Event Vocabulary: `DeepMisconception/models/process_transformer/event_vocab.json`
