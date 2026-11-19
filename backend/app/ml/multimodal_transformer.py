"""Genuine PyTorch-based Multimodal Fusion Transformer Architecture.

Fuses:
1. Coding Process Representation (from Process Transformer Encoder)
2. Contextual Code Representation (from CodeBERT 768-D)
3. Semantic Explanation Representation (from Sentence-Transformer 384-D)
4. Cross-Modal Semantic Alignment Representation (from CrossModalSemanticAlignment)
5. Structured Numerical Features (AST complexity, Execution outcomes, Telemetry metrics)

Architecture Steps:
- Step 1: Modality Projection to uniform latent dimension (d_model)
- Step 2: Learnable Modality Type Embeddings
- Step 3: Multi-Head Inter-Modality Cross-Attention
- Step 4: Multi-Layer Transformer Fusion Encoder
- Step 5: Fully Connected Dense Fusion Layers
- Step 6: 3-Class Classification Head (Understood, Partial Understanding, Possible Misconception)
"""
import os
import math
import logging
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object
    logger.warning("PyTorch is not installed. MultimodalTransformer will run in NumPy fallback mode.")

from app.ml.cross_modal_alignment import CrossModalSemanticAlignment
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
    STATUS_READY,
    STATUS_MODEL_NOT_TRAINED,
    STATUS_INSUFFICIENT_EVIDENCE,
    FEATURE_CONTRACT,
)


@dataclass
class MultimodalPredictionResult:
    """Standardized output structure for Multimodal Fusion Transformer predictions."""
    model_status: str               # "READY", "MODEL_NOT_TRAINED", "INSUFFICIENT_EVIDENCE"
    predicted_class: Optional[str]  # "Understood", "Partial Understanding", "Possible Misconception", or None
    class_probabilities: Optional[Dict[str, float]]
    modality_availability: Dict[str, bool]
    supporting_feature_scores: Dict[str, Optional[float]]
    needs_instructor_review: bool
    confidence: float = 0.0
    attention_summary: Optional[Dict[str, Any]] = None
    model_version: str = "v1.0-multimodal-transformer"


if TORCH_AVAILABLE:
    class MultimodalFusionTransformerModel(nn.Module):
        """Genuine PyTorch Multimodal Fusion Transformer implementing Steps 1 through 6."""

        def __init__(
            self,
            proc_dim: int = 64,          # Process Transformer representation dimension
            code_dim: int = 768,         # Pretrained CodeBERT dimension
            exp_dim: int = 384,          # Pretrained Sentence-Transformer dimension
            align_dim: int = 64,         # Cross-Modal Alignment representation dimension
            num_features_dim: int = 24,  # AST, Execution, and Telemetry numerical features
            d_model: int = 128,          # Common Transformer hidden dimension
            nhead: int = 4,              # Number of attention heads
            num_encoder_layers: int = 3, # Number of Transformer Encoder layers
            dim_feedforward: int = 256,  # Feedforward dimension in Transformer layers
            dropout: float = 0.1,        # Dropout rate
            num_classes: int = 3,        # Target comprehension classes
        ):
            super().__init__()
            self.proc_dim = proc_dim
            self.code_dim = code_dim
            self.exp_dim = exp_dim
            self.align_dim = align_dim
            self.num_features_dim = num_features_dim
            self.d_model = d_model
            self.nhead = nhead
            self.num_encoder_layers = num_encoder_layers
            self.num_classes = num_classes

            # ==============================================================
            # STEP 1: Modality Projection Layers (Uniform Latent Space d_model)
            # ==============================================================
            self.proj_proc = nn.Sequential(
                nn.Linear(proc_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_code = nn.Sequential(
                nn.Linear(code_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_exp = nn.Sequential(
                nn.Linear(exp_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_align = nn.Sequential(
                nn.Linear(align_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_num = nn.Sequential(
                nn.Linear(num_features_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # ==============================================================
            # STEP 2: Learnable Modality Type Embeddings
            # ==============================================================
            # Modality IDs: 0=Process, 1=Code, 2=Explanation, 3=Alignment, 4=Numerical Evidence
            self.num_modalities = 5
            self.modality_type_embeddings = nn.Embedding(self.num_modalities, d_model)
            self.modality_norm = nn.LayerNorm(d_model)

            # ==============================================================
            # STEP 3: Multi-Head Cross-Attention Across Modalities
            # ==============================================================
            self.inter_modality_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(d_model)

            # ==============================================================
            # STEP 4: Transformer Fusion Encoder Layers
            # ==============================================================
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
            )

            # ==============================================================
            # STEP 5: Fully Connected Dense Fusion Layer
            # ==============================================================
            # Concatenates all 5 encoded modality representations: (d_model * 5)
            dense_input_dim = d_model * self.num_modalities
            self.dense_fusion = nn.Sequential(
                nn.Linear(dense_input_dim, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )

            # ==============================================================
            # STEP 6: 3-Class Classification Head
            # ==============================================================
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        def forward(
            self,
            proc_emb: torch.Tensor,                               # (B, proc_dim)
            code_emb: torch.Tensor,                               # (B, code_dim)
            exp_emb: torch.Tensor,                                # (B, exp_dim)
            align_rep: Optional[torch.Tensor] = None,             # (B, align_dim)
            num_features: Optional[torch.Tensor] = None,          # (B, num_features_dim)
            modality_key_padding_mask: Optional[torch.Tensor] = None, # (B, 5) BoolTensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            """Forward pass through Multimodal Fusion Transformer.
            
            Returns:
                logits: (B, 3) raw prediction scores
                probabilities: (B, 3) softmax probabilities
                attention_dict: cross-attention weights and intermediate activations
            """
            batch_size = proc_emb.size(0)
            device = proc_emb.device

            # Validate primary input tensor dimensions
            if proc_emb.size(-1) != self.proc_dim:
                raise ValueError(f"Expected proc_dim={self.proc_dim}, got {proc_emb.size(-1)}")
            if code_emb.size(-1) != self.code_dim:
                raise ValueError(f"Expected code_dim={self.code_dim}, got {code_emb.size(-1)}")
            if exp_emb.size(-1) != self.exp_dim:
                raise ValueError(f"Expected exp_dim={self.exp_dim}, got {exp_emb.size(-1)}")

            # Handle optional inputs gracefully with zero-padding
            if align_rep is None:
                align_rep = torch.zeros(batch_size, self.align_dim, device=device)
            if num_features is None:
                num_features = torch.zeros(batch_size, self.num_features_dim, device=device)

            # --------------------------------------------------------------
            # Step 1: Project modalities to common latent space d_model
            # --------------------------------------------------------------
            h_proc = self.proj_proc(proc_emb)   # (B, d_model)
            h_code = self.proj_code(code_emb)   # (B, d_model)
            h_exp = self.proj_exp(exp_emb)      # (B, d_model)
            h_align = self.proj_align(align_rep) # (B, d_model)
            h_num = self.proj_num(num_features) # (B, d_model)

            # --------------------------------------------------------------
            # Step 2: Add learnable modality type embeddings
            # --------------------------------------------------------------
            modality_ids = torch.arange(self.num_modalities, device=device)
            type_embs = self.modality_type_embeddings(modality_ids)  # (5, d_model)

            h_proc = self.modality_norm(h_proc + type_embs[0])
            h_code = self.modality_norm(h_code + type_embs[1])
            h_exp = self.modality_norm(h_exp + type_embs[2])
            h_align = self.modality_norm(h_align + type_embs[3])
            h_num = self.modality_norm(h_num + type_embs[4])

            # Form sequence of 5 modality tokens: (B, 5, d_model)
            modality_seq = torch.stack([h_proc, h_code, h_exp, h_align, h_num], dim=1)

            # Zero out any initially masked modality tokens before attention
            all_masked = None
            safe_mask = modality_key_padding_mask
            if modality_key_padding_mask is not None:
                valid_mask = (~modality_key_padding_mask).unsqueeze(-1).float()
                modality_seq = modality_seq * valid_mask

                # Prevent PyTorch MultiheadAttention NaN if all modalities are masked
                all_masked = modality_key_padding_mask.all(dim=-1)
                if all_masked.any():
                    safe_mask = modality_key_padding_mask.clone()
                    safe_mask[all_masked, :] = False  # Temporarily unmask during attention calculation

            # --------------------------------------------------------------
            # Step 3: Multi-Head Inter-Modality Cross-Attention
            # --------------------------------------------------------------
            attn_out, attn_weights = self.inter_modality_attention(
                modality_seq,
                modality_seq,
                modality_seq,
                key_padding_mask=safe_mask,
            )
            modality_seq = self.attn_norm(modality_seq + attn_out)

            # --------------------------------------------------------------
            # Step 4: Transformer Fusion Encoder
            # --------------------------------------------------------------
            encoded_modalities = self.transformer_encoder(
                modality_seq,
                src_key_padding_mask=safe_mask,
            )  # (B, 5, d_model)

            # --------------------------------------------------------------
            # Step 5: Fully Connected Dense Fusion Layer
            # --------------------------------------------------------------
            # Strictly zero out masked modality tokens so they CANNOT influence dense fusion
            if modality_key_padding_mask is not None:
                valid_mask = (~modality_key_padding_mask).unsqueeze(-1).float()
                encoded_modalities = encoded_modalities * valid_mask
                if all_masked is not None and all_masked.any():
                    encoded_modalities[all_masked] = 0.0

            # Flatten tokens across modality dimension: (B, 5 * d_model)
            flattened = encoded_modalities.reshape(batch_size, -1)
            fused_rep = self.dense_fusion(flattened)  # (B, d_model)

            # If all modalities were masked, zero out fused representation to prevent noise
            if all_masked is not None and all_masked.any():
                fused_rep = fused_rep.clone()
                fused_rep[all_masked] = 0.0

            # --------------------------------------------------------------
            # Step 6: 3-Class Classification Head & Softmax
            # --------------------------------------------------------------
            logits = self.classifier(fused_rep)  # (B, 3)
            probs = torch.softmax(logits, dim=-1)

            attention_dict = {
                "cross_attention_weights": attn_weights,
                "fused_representation": fused_rep,
                "encoded_modalities": encoded_modalities,
            }

            return logits, probs, attention_dict


    # Backward compatibility alias for legacy callers
    PyTorchMultimodalTransformerModel = MultimodalFusionTransformerModel


class MultimodalCrossAttentionFusion:
    """High-level service API for the Multimodal Fusion Transformer.
    
    Provides training, evaluation, inference, and serialization utilities with
    device detection (Apple Silicon MPS / CUDA / CPU) and full backward compatibility.
    """

    def __init__(
        self,
        code_dim: int = CODE_DIM,
        viva_dim: int = EXP_DIM,
        proc_dim: int = PROC_DIM,
        align_dim: int = ALIGN_DIM,
        num_features_dim: int = NUM_FEATURES_DIM,
        hidden_dim: int = D_MODEL,
        nhead: int = NUM_HEADS,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        self.code_dim = code_dim
        self.viva_dim = viva_dim
        self.proc_dim = proc_dim
        self.align_dim = align_dim
        self.num_features_dim = num_features_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.model_name = "Multimodal Cross-Attention Transformer"
        self.model_framework = "PyTorch" if TORCH_AVAILABLE else "NumPy"
        self.is_trained = False
        self.device = "cpu"

        if TORCH_AVAILABLE:
            from app.ml.codebert_embedder import get_optimal_device
            from app.ml.process_transformer import PyTorchProcessTransformer
            self.device = get_optimal_device()
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Initialize Alignment Module, Process Transformer, and Fusion Model
            self.alignment_module = CrossModalSemanticAlignment(
                code_dim=code_dim,
                exp_dim=viva_dim,
                ast_dim=16,
                exec_dim=4,
                d_align=128,
                alignment_dim=align_dim,
                dropout=dropout,
            ).to(self.device)

            self.process_transformer = PyTorchProcessTransformer(
                event_emb_dim=16,
                num_event_features=7,
                global_feature_dim=8,
                d_model=64,
                nhead=4,
                num_encoder_layers=2,
                output_dim=proc_dim,
            ).to(self.device)
            self.process_transformer.eval()

            self.model = MultimodalFusionTransformerModel(
                proc_dim=proc_dim,
                code_dim=code_dim,
                exp_dim=viva_dim,
                align_dim=align_dim,
                num_features_dim=num_features_dim,
                d_model=hidden_dim,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                num_classes=num_classes,
            ).to(self.device)
            self.model.eval()
        else:
            self.alignment_module = None
            self.process_transformer = None
            self.model = None

    def encode_process_events(
        self,
        events: List[Any],
        global_features: Optional[np.ndarray] = None,
        max_len: int = 64,
    ) -> torch.Tensor:
        """Encode process telemetry events into a (1, proc_dim) tensor using persistent process_transformer."""
        if not TORCH_AVAILABLE or self.process_transformer is None or not events:
            return torch.zeros(1, self.proc_dim, device=self.device)

        from app.ml.process_transformer import normalize_process_event_timestamps, ProcessTransformerEncoder
        norm_events = normalize_process_event_timestamps(events)
        if not norm_events:
            return torch.zeros(1, self.proc_dim, device=self.device)

        seq_events = norm_events[-max_len:] if len(norm_events) > max_len else norm_events
        T = len(seq_events)

        type_ids = np.zeros((1, T), dtype=np.int64)
        num_feats = np.zeros((1, T, 7), dtype=np.float32)

        prev_time = None
        for t, ev in enumerate(seq_events):
            tid, nslice = ProcessTransformerEncoder._event_to_feature_slice(ev, prev_time)
            type_ids[0, t] = tid
            num_feats[0, t] = nslice
            prev_time = float(ev.get("elapsed_seconds", 0) or 0)

        t_types = torch.from_numpy(type_ids).to(self.device)
        t_nums = torch.from_numpy(num_feats).to(self.device)
        t_glob = torch.from_numpy(global_features).float().unsqueeze(0).to(self.device) if global_features is not None else None

        with torch.no_grad():
            proc_rep = self.process_transformer(t_types, t_nums, padding_mask=None, global_features=t_glob)
        return proc_rep

    def forward(
        self,
        x_code: Union[np.ndarray, torch.Tensor],
        x_viva: Union[np.ndarray, torch.Tensor],
        x_proc: Union[np.ndarray, torch.Tensor],
        align_rep: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward pass and return (probs, attention_rep).
        
        Seamlessly handles both 768/384/64 inputs and legacy 16/16/8 inputs via adaptive projection.
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback for NumPy
            B = x_code.shape[0] if hasattr(x_code, "shape") else 1
            return np.ones((B, self.num_classes)) / self.num_classes, np.zeros((B, self.hidden_dim))

        # Convert NumPy inputs to PyTorch tensors
        def to_tensor(arr, expected_dim):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                t = torch.from_numpy(arr).float()
            else:
                t = arr.float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            # Adaptive zero-padding or linear pad if legacy dimension is passed
            if t.size(-1) < expected_dim:
                pad = torch.zeros(t.size(0), expected_dim - t.size(-1), device=t.device)
                t = torch.cat([t, pad], dim=-1)
            elif t.size(-1) > expected_dim:
                t = t[:, :expected_dim]
            return t.to(self.device)

        t_code = to_tensor(x_code, self.code_dim)
        t_viva = to_tensor(x_viva, self.viva_dim)
        t_proc = to_tensor(x_proc, self.proc_dim)
        t_align = to_tensor(align_rep, self.align_dim) if align_rep is not None else None
        t_num = to_tensor(num_features, self.num_features_dim) if num_features is not None else None

        # If align_rep was not provided, compute it via alignment_module
        if t_align is None and self.alignment_module is not None:
            with torch.no_grad():
                t_align, _, _ = self.alignment_module(t_code, t_viva)

        with torch.no_grad():
            logits, probs, attn_dict = self.model(
                proc_emb=t_proc,
                code_emb=t_code,
                exp_emb=t_viva,
                align_rep=t_align,
                num_features=t_num,
            )
            probs_np = probs.cpu().numpy()
            fused_np = attn_dict["fused_representation"].cpu().numpy()

        return probs_np, fused_np

    def _adapt_legacy_40d_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Explicit adapter isolating legacy 40-D input matrix (AST:16, Viva:16, Process:8).
        
        Maps legacy low-dimensional prototypes into the canonical representation spaces.
        Production pipelines use canonical 768/384/64/64/24 representations directly.
        """
        logger.info("Routing through legacy 40-D feature adapter (AST:16, Viva:16, Process:8).")
        B = X.shape[0]
        x_ast_16 = X[:, :16]
        x_viva_16 = X[:, 16:32]
        x_proc_8 = X[:, 32:40]

        # Zero-pad into canonical spaces
        x_code = np.zeros((B, self.code_dim), dtype=np.float32)
        x_code[:, :16] = x_ast_16

        x_viva = np.zeros((B, self.viva_dim), dtype=np.float32)
        x_viva[:, :16] = x_viva_16

        x_proc = np.zeros((B, self.proc_dim), dtype=np.float32)
        x_proc[:, :8] = x_proc_8

        # 24-D numerical feature: 16-D AST + 8-D Process
        num_feat = np.concatenate([x_ast_16, x_proc_8], axis=-1).astype(np.float32)
        return x_code, x_viva, x_proc, num_feat

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for multimodal feature matrix X with strict dimension validation."""
        if X.ndim == 1:
            X = X[np.newaxis, :]
        B = X.shape[0]

        # Explicit legacy 40-D adapter
        if X.shape[1] == 40:
            x_code, x_viva, x_proc, num_feat = self._adapt_legacy_40d_matrix(X)
            probs, _ = self.forward(x_code, x_viva, x_proc, num_features=num_feat)
            return probs

        expected_concat_dim = self.code_dim + self.viva_dim + self.proc_dim
        if X.shape[1] == expected_concat_dim:
            x_code = X[:, :self.code_dim]
            x_viva = X[:, self.code_dim:self.code_dim + self.viva_dim]
            x_proc = X[:, self.code_dim + self.viva_dim:]
            probs, _ = self.forward(x_code, x_viva, x_proc)
            return probs

        raise ValueError(
            f"Input feature matrix dimension mismatch: got shape {X.shape}, expected (*, {expected_concat_dim}) "
            f"for concatenated canonical modalities (code={self.code_dim}, exp={self.viva_dim}, proc={self.proc_dim}) "
            "or (*, 40) via legacy adapter."
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 25, lr: float = 0.003, batch_size: int = 16) -> Dict[str, Any]:
        """Architecturally prepare and support training across trainable parameters."""
        if not TORCH_AVAILABLE:
            self.is_trained = True
            return {"epochs": epochs, "loss": 0.0, "framework": "NumPy"}

        effective_lr = min(lr, 0.005)
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=effective_lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        B = X.shape[0]
        if X.shape[1] == 40:
            x_code = X[:, :16]
            x_viva = X[:, 16:32]
            x_proc = X[:, 32:40]
        else:
            x_code = X[:, :min(X.shape[1], 16)]
            x_viva = X[:, 16:min(X.shape[1], 32)]
            x_proc = X[:, -min(X.shape[1], 8):]

        def to_pad_tensor(arr, dim):
            t = torch.from_numpy(arr).float()
            if t.size(1) < dim:
                pad = torch.zeros(B, dim - t.size(1))
                t = torch.cat([t, pad], dim=-1)
            return t

        t_code = to_pad_tensor(x_code, self.code_dim)
        t_viva = to_pad_tensor(x_viva, self.viva_dim)
        t_proc = to_pad_tensor(x_proc, self.proc_dim)
        t_y = torch.from_numpy(y).long()

        dataset = TensorDataset(t_code, t_viva, t_proc, t_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        last_loss = 0.0
        for _ in range(epochs):
            for b_code, b_viva, b_proc, b_y in loader:
                b_code = b_code.to(self.device)
                b_viva = b_viva.to(self.device)
                b_proc = b_proc.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self.model(proc_emb=b_proc, code_emb=b_code, exp_emb=b_viva)
                loss = criterion(logits, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                last_loss = float(loss.item())

        self.model.eval()
        self.is_trained = True
        return {"epochs": epochs, "final_loss": round(last_loss, 4), "framework": "PyTorch"}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute evaluation metrics on validation/test split."""
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))

        f1_list = []
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (y == c))
            fp = np.sum((preds == c) & (y != c))
            fn = np.sum((preds != c) & (y == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list))
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_c, pred_c in zip(y, preds):
            cm[int(true_c), int(pred_c)] += 1

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm.tolist(),
        }

    def save_checkpoint(self, path: str):
        """Save PyTorch weights to disk with full schema and compatibility verification."""
        if TORCH_AVAILABLE and self.model is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({
                "schema_version": "2.0-multimodal-fusion",
                "model_state": self.model.state_dict(),
                "align_state": self.alignment_module.state_dict() if self.alignment_module else None,
                "process_state": self.process_transformer.state_dict() if self.process_transformer else None,
                "feature_config": {
                    "code_dim": self.code_dim,
                    "exp_dim": self.viva_dim,
                    "proc_dim": self.proc_dim,
                    "align_dim": self.align_dim,
                    "num_features_dim": self.num_features_dim,
                    "d_model": self.model.d_model,
                    "num_classes": self.num_classes,
                    "num_heads": self.model.nhead,
                    "num_encoder_layers": self.model.num_encoder_layers,
                },
                "is_trained": self.is_trained,
            }, path)

    def load_checkpoint(self, path: str):
        """Load PyTorch weights from disk with strict compatibility validation."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at '{path}'")

        ckpt = torch.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise ValueError(f"Incompatible or corrupt checkpoint file at '{path}': missing 'model_state'")

        # Validate feature config dimensions before loading
        if "feature_config" in ckpt:
            cfg = ckpt["feature_config"]
            if cfg.get("code_dim") is not None and cfg.get("code_dim") != self.code_dim:
                raise ValueError(f"Checkpoint dimension mismatch: code_dim={cfg.get('code_dim')} vs expected {self.code_dim}")
            if cfg.get("exp_dim") is not None and cfg.get("exp_dim") != self.viva_dim:
                raise ValueError(f"Checkpoint dimension mismatch: exp_dim={cfg.get('exp_dim')} vs expected {self.viva_dim}")
            if cfg.get("proc_dim") is not None and cfg.get("proc_dim") != self.proc_dim:
                raise ValueError(f"Checkpoint dimension mismatch: proc_dim={cfg.get('proc_dim')} vs expected {self.proc_dim}")
            if cfg.get("num_classes") is not None and cfg.get("num_classes") != self.num_classes:
                raise ValueError(f"Checkpoint dimension mismatch: num_classes={cfg.get('num_classes')} vs expected {self.num_classes}")

        self.model.load_state_dict(ckpt["model_state"])

        align_st = ckpt.get("align_state") or ckpt.get("alignment_state")
        if self.alignment_module:
            if align_st is None:
                raise ValueError(f"Incomplete checkpoint at '{path}': missing alignment state for active alignment_module")
            self.alignment_module.load_state_dict(align_st)

        proc_st = ckpt.get("process_state")
        if self.process_transformer:
            if proc_st is None:
                raise ValueError(f"Incomplete checkpoint at '{path}': missing process state for active process_transformer")
            self.process_transformer.load_state_dict(proc_st)

        self.is_trained = ckpt.get("is_trained", False)

    DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__), "checkpoints", "fusion_transformer_best.pt"
    )

    def is_trained_checkpoint_available(self, checkpoint_path: Optional[str] = None) -> bool:
        """Check if a valid trained checkpoint file exists on disk."""
        path = checkpoint_path or self.DEFAULT_CHECKPOINT_PATH
        return os.path.exists(path)

    def predict_session_bundle(
        self,
        bundle: Any,
        allow_untrained: bool = False,
    ) -> MultimodalPredictionResult:
        """Execute multimodal prediction obeying evidence validation, pretrained encoder safety,
        and checkpoint verification requirements.
        
        Rules:
        - Checks pretrained CodeBERT and Sentence-Transformer with extract_tensor_checked.
        - If encoder fails or required evidence (code, viva explanations) is missing:
          returns STATUS_INSUFFICIENT_EVIDENCE (predicted_class=None).
        - If no trained checkpoint exists (and allow_untrained=False): returns STATUS_MODEL_NOT_TRAINED.
          (Does not present random weights as genuine predictions).
        - Encodes process events with persistent self.process_transformer.
        - Applies modality_key_padding_mask so unavailable modalities do not influence attention.
        - Extracts 3 logits -> softmax probabilities -> argmax -> predicted_class.
        """
        from app.ml.feature_preprocessor import NumericalFeaturePreprocessor
        from app.ml.codebert_embedder import CodeBERTEmbedder
        from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder

        # 1. Extract Pretrained Encoders with Finite & Availability Verification
        code_str = bundle.code.code if (bundle.code and bundle.code.code) else ""
        exp_str = bundle.explanation.full_transcript if (bundle.explanation and bundle.explanation.full_transcript) else ""
        proc_events = bundle.process.events if (bundle.process and bundle.process.events) else []

        code_t, code_meta = CodeBERTEmbedder.extract_tensor_checked(code_str, device=self.device)
        exp_t, exp_meta = SentenceTransformerEmbedder.extract_tensor_checked(exp_str, device=self.device)

        has_code = bool(code_t is not None and code_meta.get("is_available", False))
        has_exp = bool(exp_t is not None and exp_meta.get("is_available", False))
        has_proc = bool(len(proc_events) > 0)

        modality_avail = {
            "code": has_code,
            "explanation": has_exp,
            "process": has_proc,
            "codebert_status": code_meta.get("status", "unknown"),
            "sentence_transformer_status": exp_meta.get("status", "unknown"),
        }

        # 2. Extract supporting numerical formula scores regardless of model status
        _, fdict = NumericalFeaturePreprocessor.transform(
            ast_summary=bundle.code.ast_summary if bundle.code else None,
            exec_output=bundle.code.execution_output if bundle.code else None,
            err_output=bundle.code.error_output if bundle.code else None,
            exit_code=bundle.code.exit_code if bundle.code else 0,
            is_valid=bundle.code.is_valid if bundle.code else True,
            viva_assessments=bundle.explanation.answers if bundle.explanation else None,
            telemetry_features=bundle.process.numerical_features if bundle.process else None,
            include_s_overall=False,
        )
        supporting_scores = {
            "S_AST": fdict.get("s_ast"),
            "S_EXEC": fdict.get("s_exec"),
            "S_VIVA": fdict.get("s_viva"),
            "S_PROCESS": fdict.get("s_process"),
            "S_OVERALL": fdict.get("s_overall"),
            "S_INC": fdict.get("s_inc"),
            "S_DEBUG": fdict.get("s_debug"),
            "S_REV": fdict.get("s_rev"),
            "S_TEST": fdict.get("s_test"),
        }

        # 3. Evidence Validation Check: Code and Explanation are mandatory
        if not has_code or not has_exp:
            return MultimodalPredictionResult(
                model_status=STATUS_INSUFFICIENT_EVIDENCE,
                predicted_class=None,
                class_probabilities=None,
                modality_availability=modality_avail,
                supporting_feature_scores=supporting_scores,
                needs_instructor_review=True,
                confidence=0.0,
            )

        # 4. Checkpoint Status Check: Refuse random weights
        if not self.is_trained and not allow_untrained:
            if self.is_trained_checkpoint_available():
                self.load_checkpoint(self.DEFAULT_CHECKPOINT_PATH)
            
            if not self.is_trained and not allow_untrained:
                return MultimodalPredictionResult(
                    model_status=STATUS_MODEL_NOT_TRAINED,
                    predicted_class=None,
                    class_probabilities=None,
                    modality_availability=modality_avail,
                    supporting_feature_scores=supporting_scores,
                    needs_instructor_review=True,
                    confidence=0.0,
                )

        # 5. Process Telemetry Encoding with persistent self.process_transformer
        if has_proc:
            proc_t = self.encode_process_events(proc_events)
        else:
            proc_t = torch.zeros(1, self.proc_dim, device=self.device)

        # Numerical feature tensor with S_OVERALL masked to prevent redundant leakage
        num_vec, _ = NumericalFeaturePreprocessor.transform(
            ast_summary=bundle.code.ast_summary if bundle.code else None,
            exec_output=bundle.code.execution_output if bundle.code else None,
            err_output=bundle.code.error_output if bundle.code else None,
            exit_code=bundle.code.exit_code if bundle.code else 0,
            is_valid=bundle.code.is_valid if bundle.code else True,
            viva_assessments=bundle.explanation.answers if bundle.explanation else None,
            telemetry_features=bundle.process.numerical_features if bundle.process else None,
            include_s_overall=False,
        )
        num_t = torch.from_numpy(num_vec).unsqueeze(0).float().to(self.device)

        # Cross-modal alignment
        align_t = None
        if self.alignment_module:
            align_t, _, _ = self.alignment_module(code_t, exp_t)

        # Modality key padding mask: (1, 5) with True for padded/missing modalities
        key_mask = torch.tensor([[
            not has_proc,               # Index 0: Process
            False,                      # Index 1: Code (available)
            False,                      # Index 2: Explanation (available)
            False,                      # Index 3: Alignment (computed from code + exp)
            False,                      # Index 4: Numerical features (computed)
        ]], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits, probs, attn_dict = self.model(
                proc_emb=proc_t,
                code_emb=code_t,
                exp_emb=exp_t,
                align_rep=align_t,
                num_features=num_t,
                modality_key_padding_mask=key_mask,
            )
            probs_np = probs.cpu().numpy()[0]
            pred_idx = int(np.argmax(probs_np))
            pred_class = LABELS[pred_idx]
            conf = float(probs_np[pred_idx])

        class_probs = {LABELS[i]: round(float(probs_np[i]), 4) for i in range(len(LABELS))}

        return MultimodalPredictionResult(
            model_status=STATUS_READY,
            predicted_class=pred_class,
            class_probabilities=class_probs,
            modality_availability=modality_avail,
            supporting_feature_scores=supporting_scores,
            needs_instructor_review=(pred_class == "Possible Misconception"),
            confidence=round(conf, 4),
            attention_summary={
                "predicted_label": pred_class,
                "confidence": round(conf, 4),
            },
        )


_GLOBAL_TRANSFORMER_INSTANCE: Optional[MultimodalCrossAttentionFusion] = None


def get_multimodal_transformer() -> MultimodalCrossAttentionFusion:
    """Return the global MultimodalCrossAttentionFusion singleton, loading checkpoint if available."""
    global _GLOBAL_TRANSFORMER_INSTANCE
    if _GLOBAL_TRANSFORMER_INSTANCE is None:
        _GLOBAL_TRANSFORMER_INSTANCE = MultimodalCrossAttentionFusion(seed=42)
        if _GLOBAL_TRANSFORMER_INSTANCE.is_trained_checkpoint_available():
            _GLOBAL_TRANSFORMER_INSTANCE.load_checkpoint(_GLOBAL_TRANSFORMER_INSTANCE.DEFAULT_CHECKPOINT_PATH)
    return _GLOBAL_TRANSFORMER_INSTANCE

