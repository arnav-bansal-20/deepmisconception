#!/usr/bin/env python3
"""Train standalone baseline models:
- Experiment A: Event-only Process Transformer
- Experiment B: Numerical-only Baseline MLP
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from process_model_and_dataset import (
    ProcessDataset, EventOnlyProcessTransformer, NumericalProcessBaseline
)

MODELS_DIR = Path("DeepMisconception/models/process_transformer")
LOGS_DIR = Path("DeepMisconception/logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def train_single_model(model_type="numerical"):
    assert model_type in ["numerical", "event_only"]
    tag = "EXPERIMENT B: NUMERICAL-ONLY BASELINE" if model_type == "numerical" else "EXPERIMENT A: EVENT-ONLY PROCESS TRANSFORMER"
    print("\n" + "=" * 80)
    print(tag)
    print("=" * 80)

    config_path = Path("DeepMisconception/configs/process_transformer_config.json")
    with open(config_path) as f:
        config = json.load(f)

    with open(MODELS_DIR / "event_vocab.json") as f:
        vocab = json.load(f)

    with open(MODELS_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)

    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    train_ds = ProcessDataset(partition="train", norm_stats=norm_stats, vocab=vocab, filter_sufficient_evidence=True)
    val_ds = ProcessDataset(partition="validation", norm_stats=norm_stats, vocab=vocab, filter_sufficient_evidence=True)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = len(train_labels) / (3.0 * class_counts.astype(np.float32))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    if model_type == "numerical":
        model = NumericalProcessBaseline(
            in_features=24,
            hidden_dim=config["numerical_mlp_hidden"],
            out_dim=config["process_embedding_dim"],
            num_classes=config["num_classes"],
            dropout=config["transformer_dropout"]
        )
        ckpt_name = "numerical_baseline_best.pt"
        log_name = "numerical_training_history.json"
    else:
        model = EventOnlyProcessTransformer(
            vocab_size=len(vocab),
            d_model=config["transformer_hidden_dim"],
            num_classes=config["num_classes"]
        )
        ckpt_name = "event_only_transformer_best.pt"
        log_name = "event_only_training_history.json"

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Type: {model_type} | Parameter Count: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_epochs"], eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = {}
    epochs_no_improve = 0
    training_history = []
    patience = config["training"]["early_stopping_patience"]

    start_time = time.time()

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        train_loss_total = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == "numerical":
                out = model(batch["num_features"])
            else:
                out = model(batch["event_ids"], batch["time_features"], batch["mask"])

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
        train_f1 = f1_score(train_targets, train_preds, average="macro")

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                if model_type == "numerical":
                    out = model(batch["num_features"])
                else:
                    out = model(batch["event_ids"], batch["time_features"], batch["mask"])
                loss = criterion(out["logits"], batch["label"])
                val_loss_total += loss.item() * len(batch["label"])
                preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["label"].numpy())

        val_loss = val_loss_total / len(val_ds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")

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
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "param_count": param_count,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "model_type": model_type
            }, MODELS_DIR / ckpt_name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    duration = time.time() - start_time
    print(f"Training completed in {duration:.1f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    with open(LOGS_DIR / log_name, "w") as f:
        json.dump({
            "model_type": model_type,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "duration_s": duration,
            "history": training_history
        }, f, indent=2)

    return best_val_metrics

def main():
    train_single_model("event_only")
    train_single_model("numerical")

if __name__ == "__main__":
    main()
