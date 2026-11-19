"""Future training infrastructure and PyTorch dataset pipeline for DeepMisconception.

Architecturally prepares:
- MultimodalDataset & DataLoader with batch collation and padding masks
- Train/Validation/Test split interfaces
- CrossEntropyLoss with class weights
- AdamW optimizer with configurable weight decay and learning rate scheduling
- Complete Training & Validation loops with Early Stopping
- Checkpoint persistence and loading
- Reproducibility seed configuration

NOTE: This module provides the training pipeline architecture. As specified in the project
status, dataset generation and actual model training will be performed in a future stage.
"""
import os
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.ml.config import (
    CODE_DIM,
    EXP_DIM,
    PROC_DIM,
    ALIGN_DIM,
    NUM_FEATURES_DIM,
    D_MODEL,
    NUM_CLASSES,
    LABELS,
    LABEL_MAP,
    REV_LABEL_MAP,
)
from app.ml.multimodal_transformer import (
    MultimodalFusionTransformerModel,
    MultimodalCrossAttentionFusion,
)
from app.ml.cross_modal_alignment import CrossModalSemanticAlignment
from app.ml.process_transformer import PyTorchProcessTransformer

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class MultimodalSample:
    """Schema for a single student session multimodal training instance."""
    session_id: str
    code_emb: np.ndarray             # (768,) CodeBERT embedding
    exp_emb: np.ndarray              # (384,) Sentence-Transformer embedding
    proc_emb: np.ndarray             # (64,) Process Transformer representation
    align_rep: Optional[np.ndarray]  # (64,) Cross-modal semantic alignment representation
    num_features: np.ndarray         # (24,) Normalized formula and telemetry features
    label: int                       # 0: Understood, 1: Partial Understanding, 2: Possible Misconception
    modality_mask: Optional[List[bool]] = None # Length 5 mask (True if padded/missing)


class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal student sessions."""

    def __init__(self, samples: List[MultimodalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        mask = s.modality_mask
        if mask is None:
            mask = [False, False, False, False, False]

        align = s.align_rep if s.align_rep is not None else np.zeros(64, dtype=np.float32)

        return {
            "session_id": s.session_id,
            "proc_emb": torch.from_numpy(s.proc_emb).float(),
            "code_emb": torch.from_numpy(s.code_emb).float(),
            "exp_emb": torch.from_numpy(s.exp_emb).float(),
            "align_rep": torch.from_numpy(align).float(),
            "num_features": torch.from_numpy(s.num_features).float(),
            "label": torch.tensor(s.label, dtype=torch.long),
            "modality_mask": torch.tensor(mask, dtype=torch.bool),
        }


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate individual samples into batched PyTorch tensors."""
    return {
        "session_ids": [item["session_id"] for item in batch],
        "proc_emb": torch.stack([item["proc_emb"] for item in batch]),
        "code_emb": torch.stack([item["code_emb"] for item in batch]),
        "exp_emb": torch.stack([item["exp_emb"] for item in batch]),
        "align_rep": torch.stack([item["align_rep"] for item in batch]),
        "num_features": torch.stack([item["num_features"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "modality_masks": torch.stack([item["modality_mask"] for item in batch]),
    }


class EarlyStopping:
    """Monitors validation loss and stops training when improvement plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class MultimodalTrainer:
    """Configurable training and validation pipeline for MultimodalFusionTransformerModel."""

    def __init__(
        self,
        model: MultimodalFusionTransformerModel,
        alignment_module: Optional[CrossModalSemanticAlignment] = None,
        process_transformer: Optional[PyTorchProcessTransformer] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[List[float]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.alignment_module = alignment_module.to(self.device) if alignment_module else None
        self.process_transformer = process_transformer.to(self.device) if process_transformer else None

        # Trainable parameters across all trainable components
        params = list(self.model.parameters())
        if self.alignment_module:
            params += list(self.alignment_module.parameters())
        if self.process_transformer:
            params += list(self.process_transformer.parameters())

        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch."""
        self.model.train()
        if self.alignment_module:
            self.alignment_module.train()
        if self.process_transformer:
            self.process_transformer.train()

        total_loss = 0.0
        for batch in dataloader:
            proc_emb = batch["proc_emb"].to(self.device)
            code_emb = batch["code_emb"].to(self.device)
            exp_emb = batch["exp_emb"].to(self.device)
            num_feat = batch["num_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            masks = batch["modality_masks"].to(self.device)

            # Compute or use alignment
            if self.alignment_module:
                ast_feat = num_feat[:, 15:31] if num_feat.size(1) >= 31 else None
                align_rep, _, _ = self.alignment_module(code_emb, exp_emb, ast_feat)
            else:
                align_rep = batch["align_rep"].to(self.device)

            self.optimizer.zero_grad()
            logits, _, _ = self.model(
                proc_emb=proc_emb,
                code_emb=code_emb,
                exp_emb=exp_emb,
                align_rep=align_rep,
                num_features=num_feat,
                modality_key_padding_mask=masks,
            )

            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation on a dataset split."""
        self.model.eval()
        if self.alignment_module:
            self.alignment_module.eval()
        if self.process_transformer:
            self.process_transformer.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                proc_emb = batch["proc_emb"].to(self.device)
                code_emb = batch["code_emb"].to(self.device)
                exp_emb = batch["exp_emb"].to(self.device)
                num_feat = batch["num_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                masks = batch["modality_masks"].to(self.device)

                if self.alignment_module:
                    align_rep, _, _ = self.alignment_module(code_emb, exp_emb)
                else:
                    align_rep = batch["align_rep"].to(self.device)

                logits, probs, _ = self.model(
                    proc_emb=proc_emb,
                    code_emb=code_emb,
                    exp_emb=exp_emb,
                    align_rep=align_rep,
                    num_features=num_feat,
                    modality_key_padding_mask=masks,
                )

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(probs, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
        return {
            "val_loss": total_loss / max(1, len(dataloader)),
            "accuracy": round(acc, 4),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        checkpoint_dir: str = "backend/app/ml/checkpoints",
        checkpoint_name: str = "fusion_transformer_best.pt",
        patience: int = 5,
    ) -> Dict[str, Any]:
        """Execute full training pipeline with early stopping and best checkpoint persistence."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
        early_stopping = EarlyStopping(patience=patience)

        history = []
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["accuracy"]

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": val_acc,
            })

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "alignment_state": self.alignment_module.state_dict() if self.alignment_module else None,
                    "process_state": self.process_transformer.state_dict() if self.process_transformer else None,
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "is_trained": True,
                    "model_version": "v1.0-multimodal-transformer",
                }, ckpt_path)

            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return {
            "epochs_completed": len(history),
            "best_val_loss": round(best_val_loss, 4),
            "checkpoint_path": ckpt_path,
            "history": history,
        }

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model, alignment module, and process transformer states from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if self.alignment_module and checkpoint.get("alignment_state"):
            self.alignment_module.load_state_dict(checkpoint["alignment_state"])
        if self.process_transformer and checkpoint.get("process_state"):
            self.process_transformer.load_state_dict(checkpoint["process_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint
