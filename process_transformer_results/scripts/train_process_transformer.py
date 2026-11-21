#!/usr/bin/env python3
"""Train the Combined Process Transformer (Experiment C) for DeepMisconception."""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure scripts dir is in path
sys.path.insert(0, str(Path(__file__).parent))
from process_model_and_dataset import (
    ProcessDataset, CombinedProcessTransformer, CLASS_TO_IDX, EXACT_24_FEATURES
)

MODELS_DIR = Path("DeepMisconception/models/process_transformer")
LOGS_DIR = Path("DeepMisconception/logs")
REPORTS_DIR = Path("DeepMisconception/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def train_combined_model():
    print("=" * 80)
    print("EXPERIMENT C: TRAINING COMBINED PROCESS TRANSFORMER")
    print("=" * 80)

    # 1. Load configuration and stats
    config_path = Path("DeepMisconception/configs/process_transformer_config.json")
    with open(config_path) as f:
        config = json.load(f)

    with open(MODELS_DIR / "event_vocab.json") as f:
        vocab = json.load(f)

    with open(MODELS_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)

    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    # 2. Datasets & Loaders
    train_ds = ProcessDataset(partition="train", norm_stats=norm_stats, vocab=vocab, filter_sufficient_evidence=True)
    val_ds = ProcessDataset(partition="validation", norm_stats=norm_stats, vocab=vocab, filter_sufficient_evidence=True)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)} ({train_ds.excluded_count} insufficient evidence excluded)")
    print(f"Val samples:   {len(val_ds)} ({val_ds.excluded_count} insufficient evidence excluded)")

    # 3. Class-weighted Cross-Entropy Loss
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels, minlength=3)
    total_samples = len(train_labels)
    class_weights = total_samples / (3.0 * class_counts.astype(np.float32))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Training Class Counts:  Understood={class_counts[0]}, Partial={class_counts[1]}, Misconception={class_counts[2]}")
    print(f"Inverse Class Weights:  {weight_tensor.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # 4. Model, Optimizer, Scheduler
    model = CombinedProcessTransformer(
        vocab_size=len(vocab),
        in_features=24,
        d_model=config["transformer_hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["transformer_dropout"]
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameter Count: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_epochs"], eta_min=1e-5
    )

    # 5. Training Loop with Validation-based Early Stopping
    max_epochs = config["training"]["max_epochs"]
    patience = config["training"]["early_stopping_patience"]

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = {}
    epochs_no_improve = 0
    training_history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["event_ids"],
                batch["time_features"],
                batch["mask"],
                batch["num_features"]
            )
            loss = criterion(out["logits"], batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            optimizer.step()

            train_loss_total += loss.item() * len(batch["label"])
            preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(batch["label"].numpy())

        lr_scheduler.step()
        train_loss = train_loss_total / len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro")

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    batch["event_ids"],
                    batch["time_features"],
                    batch["mask"],
                    batch["num_features"]
                )
                loss = criterion(out["logits"], batch["label"])
                val_loss_total += loss.item() * len(batch["label"])
                preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["label"].numpy())

        val_loss = val_loss_total / len(val_ds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        epoch_stat = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        }
        training_history.append(epoch_stat)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")

        # Checkpoint selection on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_val_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1
            }
            # Save best checkpoint
            ckpt_path = MODELS_DIR / "combined_process_transformer_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "param_count": param_count,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "norm_stats": norm_stats,
                "vocab": vocab,
                "class_weights": class_weights.tolist()
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no val loss improvement for {patience} epochs).")
                break

    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Save training history
    history_file = LOGS_DIR / "combined_training_history.json"
    with open(history_file, "w") as f:
        json.dump({
            "model": "CombinedProcessTransformer",
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "duration_s": duration,
            "history": training_history
        }, f, indent=2)

    return best_val_metrics

if __name__ == "__main__":
    train_combined_model()
