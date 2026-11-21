# Process Transformer: Training Deliverables & Experimental Results

This directory contains the complete, self-contained standalone training pipeline, trained model checkpoints, evaluation metrics, diagnostic audits, and precomputed embeddings for the **Process Transformer** (Modality 3: Chronological Coding-Process Telemetry and 24 Numerical Behavioral Features).

---

## Directory Overview

```
process_transformer_results/
├── checkpoints/             # Trained PyTorch weights, vocabularies, normalizations & inference API
│   ├── combined_process_transformer_best.pt   # Primary Combined Model (Best Val Loss: 1.0435)
│   ├── event_only_transformer_best.pt         # Event-Only Architecture (Exp A)
│   ├── numerical_baseline_best.pt             # 24 Numerical Features MLP Baseline (Exp B)
│   ├── precomputed_process_embeddings.pt      # 64-D embeddings for all 5,000 sessions [N, 64]
│   ├── event_vocab.json                       # 13 event tokens (with <PAD> at 0)
│   ├── normalization_stats.json               # Z-score params fitted solely on training split
│   ├── label_mapping.json                     # Official 3-class target mappings
│   └── inference.py                           # Standalone inference predictor module
├── results/                 # Programmatic evaluation outputs on held-out test split (N=627)
│   ├── test_evaluation_results.json           # Comprehensive quantitative metrics across all 3 models
│   └── test_predictions.csv                   # Per-session predicted classes & probabilities
├── reports/                 # Markdown audit and experimental reports
│   ├── process_dataset_audit.md               # Session counts, disjoint splits, evidence filtering
│   ├── process_preprocessing_report.md        # Event vocab, max sequence len, temporal projections
│   ├── process_training_report.md             # Model architectures, hyperparams, epoch losses
│   ├── process_evaluation_report.md           # Standalone test metrics and confusion matrices
│   ├── process_ablation_report.md             # Quantitative comparison: Event vs Num vs Combined
│   ├── process_leakage_audit.md               # Permutation importance & shortcut analysis
│   └── process_integration_report.md          # Multimodal compatibility checklist
├── plots/                   # Publication-grade figures
│   ├── process_ablation_comparison.png        # Bar chart comparing Accuracy, Macro-F1, ROC-AUC
│   └── process_confusion_matrices.png         # 3-panel confusion matrix grid across experiments
├── logs/                    # Epoch-by-epoch loss & metric curves
│   ├── combined_training_history.json
│   ├── event_only_training_history.json
│   └── numerical_training_history.json
├── scripts/                 # Standalone reproducible execution scripts
│   ├── inspect_process_dataset.py
│   ├── preprocess_process_events.py
│   ├── prepare_process_features.py
│   ├── process_model_and_dataset.py
│   ├── train_process_transformer.py
│   ├── train_process_baseline.py
│   ├── evaluate_process_transformer.py
│   ├── run_process_ablation.py
│   ├── audit_process_shortcuts.py
│   ├── export_process_embeddings.py
│   └── verify_process_integration.py
└── configs/                 # Hyperparameters & model configurations
    └── process_transformer_config.json
```

---

## Key Experimental Results (Test Partition, $N=627$ Supervised Sessions)

| Experiment | Model Architecture | Trainable Params | Accuracy | Macro-Precision | Macro-Recall | **Macro-F1** | Weighted-F1 | ROC-AUC (OVR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A** | Event-Only Process Transformer | 105,155 | 0.6045 | 0.4079 | 0.3802 | **0.3153** | 0.5430 | 0.5373 |
| **Experiment B** | Numerical-Only Baseline MLP | 6,211 | 0.6332 | 0.5186 | 0.5057 | **0.5066** | 0.6502 | 0.7223 |
| **Experiment C** | **Combined Process Transformer** | **123,715** | **0.5981** | **0.4907** | **0.4851** | **0.4831** | **0.6274** | **0.7166** |

---

## Quick Usage / Inference

To extract a 64-dimensional process embedding using Python:

```python
import sys
from pathlib import Path
sys.path.insert(0, "process_transformer_results/checkpoints")
from inference import extract_process_embedding

# process_events: list of dicts with 'timestamp', 'event_type', 'characters_inserted', 'characters_deleted'
# process_features_24: dict containing the exact 24 numerical process features
embedding = extract_process_embedding(process_events, process_features_24, model_dir="process_transformer_results/checkpoints")
print(embedding.shape)  # torch.Size([1, 64])
```
