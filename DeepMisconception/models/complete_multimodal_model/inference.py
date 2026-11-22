"""Production inference service for DeepMisconception Complete Multimodal Model.
Fuses Code (CodeBERT 768-D), Viva (SentenceTransformer 384-D), Process (Process Transformer 64-D),
and 24 Numerical Process Features to predict student misconception status.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["Understood", "Partial Understanding", "Possible Misconception"]

EXACT_24_FEATURES = [
    "n_events", "duration_s", "n_insert", "n_edit", "n_delete", "n_paste",
    "typed_chars", "paste_chars", "paste_char_ratio", "largest_insert_share",
    "n_runs", "n_syntax_errors", "n_runtime_errors", "n_errors", "n_debug_actions",
    "n_tests", "n_test_pass", "n_revisions", "n_errors_resolved", "idle_total_s",
    "max_idle_s", "mean_gap_s", "time_to_first_run_s", "final_state_pass"
]

class DeepMisconceptionMultimodalPredictor:
    def __init__(self, model_dir=None, device="cpu"):
        self.device = torch.device(device)
        repo_root = Path(__file__).resolve().parents[3]
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent

        # 1. Load Normalization & Metadata
        norm_path = repo_root / "process_transformer_results" / "checkpoints" / "normalization_stats.json"
        if not norm_path.exists():
            norm_path = self.model_dir / "normalization_stats.json"
        with open(norm_path) as f:
            self.norm_stats = json.load(f)

        # 2. Load Process Predictor
        import importlib.util
        proc_script_dir = str(repo_root / "process_transformer_results" / "scripts")
        if proc_script_dir not in sys.path:
            sys.path.insert(0, proc_script_dir)
        proc_inf_path = repo_root / "process_transformer_results" / "checkpoints" / "inference.py"
        if proc_inf_path.exists():
            spec = importlib.util.spec_from_file_location("proc_inference", proc_inf_path)
            proc_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(proc_mod)
            self.process_predictor = proc_mod.ProcessTransformerPredictor(model_dir=proc_inf_path.parent, device=self.device)
        else:
            self.process_predictor = None

        # 3. Load Alignment and Fusion Architectures
        deep_scripts_dir = str(repo_root / "DeepMisconception" / "scripts")
        if deep_scripts_dir not in sys.path:
            sys.path.insert(0, deep_scripts_dir)
        from train_cross_modal_alignment import CrossModalAlignmentModule
        from train_multimodal_pipeline import MultimodalFusionTransformer

        self.align_module = CrossModalAlignmentModule().to(self.device)
        self.fusion_model = MultimodalFusionTransformer().to(self.device)

        # 4. Load Trained Weights
        ckpt_path = self.model_dir / "complete_multimodal_model_best.pt"
        if not ckpt_path.exists():
            ckpt_path = repo_root / "DeepMisconception" / "models" / "multimodal_fusion_transformer" / "fusion_transformer_best.pt"

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.fusion_model.load_state_dict(ckpt["model_state"])
        self.align_module.load_state_dict(ckpt["align_state"])

        self.fusion_model.eval()
        self.align_module.eval()
        self.status = "READY"

    def predict(self, code_emb: torch.Tensor, viva_emb: torch.Tensor,
                process_events=None, process_features_24=None,
                proc_emb: Optional[torch.Tensor] = None,
                evidence_status: str = "Sufficient Evidence") -> Dict[str, Any]:
        """
        Runs multimodal inference given:
        - code_emb: [1, 768] or [768] float tensor
        - viva_emb: [1, 384] or [384] float tensor
        - proc_emb: [1, 64] float tensor, or extracted via process_events and process_features_24
        - process_features_24: dict with 24 numerical process features
        """
        if evidence_status == "Insufficient Evidence":
            return {
                "predicted_class": None,
                "class_probabilities": {c: 0.3333 for c in CLASS_NAMES},
                "evidence_status": "Pending / Insufficient Evidence",
                "model_availability_status": self.status,
                "note": "Abstained from prediction: insufficient multimodal evidence"
            }

        if code_emb.dim() == 1:
            code_emb = code_emb.unsqueeze(0)
        if viva_emb.dim() == 1:
            viva_emb = viva_emb.unsqueeze(0)

        code_emb = code_emb.to(self.device)
        viva_emb = viva_emb.to(self.device)

        # Process representation (64-D)
        if proc_emb is None and process_events is not None and process_features_24 is not None:
            p_res = self.process_predictor.extract_process_embedding(process_events, process_features_24)
            proc_emb = p_res["process_embedding"].to(self.device)
        elif proc_emb is not None:
            if proc_emb.dim() == 1:
                proc_emb = proc_emb.unsqueeze(0)
            proc_emb = proc_emb.to(self.device)
        else:
            proc_emb = torch.zeros((1, 64), device=self.device)

        # Numerical process features (24-D)
        if process_features_24 is not None:
            raw_vals = [float(process_features_24.get(f, 0.0)) for f in EXACT_24_FEATURES]
            means = np.array([self.norm_stats["means"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
            stds = np.array([self.norm_stats["stds"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
            norm_num = torch.tensor((np.array(raw_vals, dtype=np.float32) - means) / stds, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            norm_num = torch.zeros((1, 24), device=self.device)

        with torch.no_grad():
            # 1. Cross-modal semantic alignment (Stage A)
            aligned_code, aligned_viva = self.align_module(code_emb, viva_emb)

            # 2. Multimodal fusion transformer (Stage B & C)
            out = self.fusion_model(
                code_emb, viva_emb, proc_emb, norm_num,
                aligned_code=aligned_code, aligned_viva=aligned_viva
            )
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "class_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
            "evidence_status": "Sufficient Evidence",
            "model_availability_status": self.status,
            "fused_embedding": out["fused_embedding"].squeeze(0).cpu().tolist()
        }
