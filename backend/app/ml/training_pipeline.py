"""Future training infrastructure and PyTorch dataset pipeline for DeepMisconception.

Architecturally prepares:
- Raw chronological process event sequencing through PyTorchProcessTransformer
- MultimodalDataset & DataLoader with batch collation, variable-length sequence padding, and modality masks
- Train/Validation/Test split interfaces
- CrossEntropyLoss with class weights
- AdamW optimizer with joint parameter groups (Fusion Transformer + Alignment Module + Process Transformer)
- Gradient flow preservation across all components
- Complete Training & Validation loops with Early Stopping
- Checkpoint persistence and loading with full component validation
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
    NUM_HEADS,
    NUM_ENCODER_LAYERS,
    NUM_CLASSES,
    LABELS,
    LABEL_MAP,
    REV_LABEL_MAP,
    FeatureDimensionContract,
)
from app.ml.multimodal_transformer import (
    MultimodalFusionTransformerModel,
    MultimodalCrossAttentionFusion,
)
from app.ml.cross_modal_alignment import CrossModalSemanticAlignment
from app.ml.process_transformer import (
    PyTorchProcessTransformer,
    ProcessTransformerEncoder,
    normalize_process_event_timestamps,
    EVENT_TYPE_MAP,
)

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = "2.0-multimodal-fusion"


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_events_to_sequence_tensors(
    events: List[Any],
    max_len: int = 32,
    global_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Convert raw chronological process events to model-ready sequence arrays.

    Returns:
        type_ids: (T,) int64 array of event type tokens
        num_features: (T, 7) float32 array of normalized event attributes
        global_features: (8,) float32 array of session macro telemetry
        has_events: boolean indicating whether valid events were present
    """
    if not events:
        empty_types = np.zeros(0, dtype=np.int64)
        empty_features = np.zeros((0, 7), dtype=np.float32)
        glob = global_features if global_features is not None else np.zeros(8, dtype=np.float32)
        return empty_types, empty_features, glob, False

    norm_events = normalize_process_event_timestamps(events)
    seq_events = norm_events[-max_len:] if len(norm_events) > max_len else norm_events
    T = len(seq_events)

    type_ids = np.zeros(T, dtype=np.int64)
    num_features = np.zeros((T, 7), dtype=np.float32)

    prev_time = None
    for t, ev in enumerate(seq_events):
        tid, nslice = ProcessTransformerEncoder._event_to_feature_slice(ev, prev_time)
        type_ids[t] = tid
        num_features[t] = nslice
        prev_time = float(ev.get("elapsed_seconds", 0) or 0)

    glob = global_features if global_features is not None else np.zeros(8, dtype=np.float32)
    return type_ids, num_features, glob, True


@dataclass
class MultimodalSample:
    """Schema for a single student session multimodal training instance."""
    session_id: str
    code_emb: np.ndarray             # (768,) CodeBERT embedding
    exp_emb: np.ndarray              # (384,) Sentence-Transformer embedding
    num_features: np.ndarray         # (24,) Normalized formula and telemetry features
    label: int                       # 0: Understood, 1: Partial Understanding, 2: Possible Misconception
    align_rep: Optional[np.ndarray] = None  # (64,) Cross-modal semantic alignment representation
    proc_emb: Optional[np.ndarray] = None   # (64,) Precomputed process representation (optional caching)
    proc_events: Optional[List[Dict[str, Any]]] = None # Raw chronological events for end-to-end training
    proc_event_types: Optional[np.ndarray] = None      # (T,) Long array of event type IDs
    proc_event_features: Optional[np.ndarray] = None   # (T, 7) Float32 array of per-event numerical attributes
    proc_global_features: Optional[np.ndarray] = None  # (8,) Float32 array of macro process telemetry
    modality_mask: Optional[List[bool]] = None         # Length 5 mask (True if padded/missing)


class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal student sessions supporting raw process sequences."""

    def __init__(self, samples: List[MultimodalSample], max_seq_len: int = 32):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        mask = s.modality_mask
        if mask is None:
            mask = [False, False, False, False, False]

        align = s.align_rep if s.align_rep is not None else np.zeros(ALIGN_DIM, dtype=np.float32)

        # Process sequence handling: check explicit arrays, then raw events, then precomputed proc_emb
        has_process_seq = False
        if s.proc_event_types is not None and s.proc_event_features is not None:
            types = s.proc_event_types
            nums = s.proc_event_features
            glob = s.proc_global_features if s.proc_global_features is not None else np.zeros(8, dtype=np.float32)
            has_process_seq = len(types) > 0
        elif s.proc_events is not None and len(s.proc_events) > 0:
            types, nums, glob, has_process_seq = convert_events_to_sequence_tensors(
                s.proc_events,
                max_len=self.max_seq_len,
                global_features=s.proc_global_features,
            )
        else:
            types = np.zeros(0, dtype=np.int64)
            nums = np.zeros((0, 7), dtype=np.float32)
            glob = np.zeros(8, dtype=np.float32)

        proc_emb = s.proc_emb if s.proc_emb is not None else np.zeros(PROC_DIM, dtype=np.float32)

        return {
            "session_id": s.session_id,
            "has_process_seq": has_process_seq,
            "proc_event_types": torch.from_numpy(types).long(),
            "proc_event_features": torch.from_numpy(nums).float(),
            "proc_global_features": torch.from_numpy(glob).float(),
            "proc_emb": torch.from_numpy(proc_emb).float(),
            "code_emb": torch.from_numpy(s.code_emb).float(),
            "exp_emb": torch.from_numpy(s.exp_emb).float(),
            "align_rep": torch.from_numpy(align).float(),
            "num_features": torch.from_numpy(s.num_features).float(),
            "label": torch.tensor(s.label, dtype=torch.long),
            "modality_mask": torch.tensor(mask, dtype=torch.bool),
        }


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate individual samples into batched PyTorch tensors with sequence padding."""
    session_ids = [item["session_id"] for item in batch]
    code_emb = torch.stack([item["code_emb"] for item in batch])
    exp_emb = torch.stack([item["exp_emb"] for item in batch])
    align_rep = torch.stack([item["align_rep"] for item in batch])
    num_features = torch.stack([item["num_features"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    modality_masks = torch.stack([item["modality_mask"] for item in batch])
    proc_emb = torch.stack([item["proc_emb"] for item in batch])

    # Check if any sample has process sequence events for end-to-end training
    any_seq = any(item["has_process_seq"] for item in batch)
    collate_dict = {
        "session_ids": session_ids,
        "code_emb": code_emb,
        "exp_emb": exp_emb,
        "align_rep": align_rep,
        "num_features": num_features,
        "labels": labels,
        "modality_masks": modality_masks,
        "proc_emb": proc_emb,
        "has_process_seq": any_seq,
    }

    if any_seq:
        lengths = [len(item["proc_event_types"]) for item in batch]
        max_len = max(lengths)
        if max_len == 0:
            max_len = 1

        batch_size = len(batch)
        padded_types = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_features = torch.zeros((batch_size, max_len, 7), dtype=torch.float32)
        proc_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
        global_features = torch.stack([item["proc_global_features"] for item in batch])

        for b, item in enumerate(batch):
            T = len(item["proc_event_types"])
            if T > 0:
                padded_types[b, :T] = item["proc_event_types"]
                padded_features[b, :T] = item["proc_event_features"]
                proc_padding_mask[b, :T] = False  # False means valid token

        collate_dict["proc_event_types"] = padded_types
        collate_dict["proc_event_features"] = padded_features
        collate_dict["proc_padding_mask"] = proc_padding_mask
        collate_dict["proc_global_features"] = global_features

    return collate_dict


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
    """Configurable training and validation pipeline for MultimodalFusionTransformerModel.

    Guarantees joint gradient flow across:
    1. Multimodal Fusion Transformer
    2. Cross-Modal Semantic Alignment Module
    3. PyTorch Process Transformer (raw chronological events)
    """

    def __init__(
        self,
        model: MultimodalFusionTransformerModel,
        alignment_module: Optional[CrossModalSemanticAlignment] = None,
        process_transformer: Optional[PyTorchProcessTransformer] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[List[float]] = None,
        device: Optional[str] = None,
        mask_formula_leakage: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.alignment_module = alignment_module.to(self.device) if alignment_module else None
        self.process_transformer = process_transformer.to(self.device) if process_transformer else None
        self.mask_formula_leakage = mask_formula_leakage

        # Trainable parameters across all active trainable components
        self.trainable_params = list(self.model.parameters())
        if self.alignment_module:
            self.trainable_params += list(self.alignment_module.parameters())
        if self.process_transformer:
            self.trainable_params += list(self.process_transformer.parameters())

        self.optimizer = optim.AdamW(self.trainable_params, lr=lr, weight_decay=weight_decay)

        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch with connected autograd across all modules."""
        self.model.train()
        if self.alignment_module:
            self.alignment_module.train()
        if self.process_transformer:
            self.process_transformer.train()

        total_loss = 0.0
        for batch in dataloader:
            code_emb = batch["code_emb"].to(self.device)
            exp_emb = batch["exp_emb"].to(self.device)
            num_feat = batch["num_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            masks = batch["modality_masks"].to(self.device)

            # Target leakage elimination: mask s_overall (index 0) and s_viva (index 3) if formula scores are present
            if self.mask_formula_leakage:
                num_feat = num_feat.clone()
                num_feat[:, 0] = 0.0
                num_feat[:, 3] = 0.0

            # End-to-end Process Transformer forward pass
            if self.process_transformer is not None and batch.get("has_process_seq") and "proc_event_types" in batch:
                proc_types = batch["proc_event_types"].to(self.device)
                proc_nums = batch["proc_event_features"].to(self.device)
                proc_pad_mask = batch["proc_padding_mask"].to(self.device)
                proc_glob = batch["proc_global_features"].to(self.device)

                # Process Transformer forward pass remains connected in computation graph!
                proc_emb = self.process_transformer(
                    event_types=proc_types,
                    event_num_features=proc_nums,
                    padding_mask=proc_pad_mask,
                    global_features=proc_glob,
                )
            else:
                proc_emb = batch["proc_emb"].to(self.device)

            # Compute cross-modal alignment representation if active
            # Strictly masked if either Code (index 1) or Explanation (index 2) is missing
            if self.alignment_module:
                ast_feat = num_feat[:, 15:20] if num_feat.size(1) >= 20 else None
                align_rep, _, _ = self.alignment_module(
                    code_emb, exp_emb, ast_feat,
                    code_mask=masks[:, 1], exp_mask=masks[:, 2]
                )
                either_missing = masks[:, 1] | masks[:, 2]
                if either_missing.any():
                    masks = masks.clone()
                    masks[:, 3] = masks[:, 3] | either_missing
                    align_rep = align_rep.clone()
                    align_rep[either_missing] = 0.0
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

            # Clip gradient norm across all registered parameters
            nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
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
                code_emb = batch["code_emb"].to(self.device)
                exp_emb = batch["exp_emb"].to(self.device)
                num_feat = batch["num_features"].to(self.device).clone()
                # Target leakage elimination in validation: mask s_overall and s_viva if formula scores are present
                if self.mask_formula_leakage:
                    num_feat = num_feat.clone()
                    num_feat[:, 0] = 0.0
                    num_feat[:, 3] = 0.0
                labels = batch["labels"].to(self.device)
                masks = batch["modality_masks"].to(self.device)

                if self.process_transformer is not None and batch.get("has_process_seq") and "proc_event_types" in batch:
                    proc_types = batch["proc_event_types"].to(self.device)
                    proc_nums = batch["proc_event_features"].to(self.device)
                    proc_pad_mask = batch["proc_padding_mask"].to(self.device)
                    proc_glob = batch["proc_global_features"].to(self.device)

                    proc_emb = self.process_transformer(
                        event_types=proc_types,
                        event_num_features=proc_nums,
                        padding_mask=proc_pad_mask,
                        global_features=proc_glob,
                    )
                else:
                    proc_emb = batch["proc_emb"].to(self.device)

                if self.alignment_module:
                    ast_feat = num_feat[:, 15:20] if num_feat.size(1) >= 20 else None
                    align_rep, _, _ = self.alignment_module(
                        code_emb, exp_emb, ast_feat,
                        code_mask=masks[:, 1], exp_mask=masks[:, 2]
                    )
                    either_missing = masks[:, 1] | masks[:, 2]
                    if either_missing.any():
                        masks = masks.clone()
                        masks[:, 3] = masks[:, 3] | either_missing
                        align_rep = align_rep.clone()
                        align_rep[either_missing] = 0.0
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
        """Execute full training pipeline with early stopping and canonical checkpoint persistence."""
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

            # Save best checkpoint conforming strictly to canonical checkpoint schema
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "align_state": self.alignment_module.state_dict() if self.alignment_module else None,
                    "process_state": self.process_transformer.state_dict() if self.process_transformer else None,
                    "optimizer_state": self.optimizer.state_dict(),
                    "feature_config": {
                        "code_dim": self.model.code_dim,
                        "exp_dim": self.model.exp_dim,
                        "proc_dim": self.model.proc_dim,
                        "align_dim": self.model.align_dim,
                        "num_features_dim": self.model.num_features_dim,
                        "d_model": self.model.d_model,
                        "num_classes": self.model.num_classes,
                        "num_heads": self.model.nhead,
                        "num_encoder_layers": self.model.num_encoder_layers,
                    },
                    "class_mapping": dict(LABEL_MAP),
                    "classes": list(LABELS),
                    "preprocessing_config": {
                        "mask_s_overall": True,
                        "mask_s_viva": True,
                        "num_features_dim": self.model.num_features_dim,
                    },
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "is_trained": True,
                    "model_version": "v2.0-multimodal-transformer",
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
        """Load model, alignment module, and process transformer states with strict validation."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(f"Incompatible or corrupted checkpoint at '{checkpoint_path}': missing 'model_state'")

        if "feature_config" in checkpoint:
            cfg = checkpoint["feature_config"]
            if cfg.get("code_dim") != self.model.code_dim:
                raise ValueError(f"Dimension mismatch code_dim: checkpoint has {cfg.get('code_dim')}, expected {self.model.code_dim}")
            if cfg.get("exp_dim") != self.model.exp_dim:
                raise ValueError(f"Dimension mismatch exp_dim: checkpoint has {cfg.get('exp_dim')}, expected {self.model.exp_dim}")
            if cfg.get("proc_dim") != self.model.proc_dim:
                raise ValueError(f"Dimension mismatch proc_dim: checkpoint has {cfg.get('proc_dim')}, expected {self.model.proc_dim}")
            if cfg.get("num_classes") != self.model.num_classes:
                raise ValueError(f"Dimension mismatch num_classes: checkpoint has {cfg.get('num_classes')}, expected {self.model.num_classes}")

        self.model.load_state_dict(checkpoint["model_state"])

        # Load alignment state (support canonical align_state and legacy alignment_state)
        align_st = checkpoint.get("align_state") or checkpoint.get("alignment_state")
        if self.alignment_module:
            if align_st is None:
                raise ValueError(f"Checkpoint at '{checkpoint_path}' missing align_state for active alignment_module")
            self.alignment_module.load_state_dict(align_st)

        # Load process transformer state
        proc_st = checkpoint.get("process_state")
        if self.process_transformer:
            if proc_st is None:
                raise ValueError(f"Checkpoint at '{checkpoint_path}' missing process_state for active process_transformer")
            self.process_transformer.load_state_dict(proc_st)

        if "optimizer_state" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint
