"""Future evaluation and ablation benchmark harness for DeepMisconception.

Prepares comprehensive evaluation metrics and systematic ablation protocols across
all modality permutations. Ready to be executed once the multimodal dataset is
generated and training is performed.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from app.ml.multimodal_transformer import LABELS


# The 9 Systematic Ablation Configurations defined for DeepMisconception
ABLATION_CONFIGURATIONS = [
    {
        "id": "code_only",
        "name": "1. Code Only (AST + CodeBERT)",
        "active_modalities": ["code"],
        "uses_formula_features": False,
        "mask": [True, False, True, True, True],  # Keep index 1 (Code)
    },
    {
        "id": "explanation_only",
        "name": "2. Explanation Only (Sentence-Transformer)",
        "active_modalities": ["explanation"],
        "uses_formula_features": False,
        "mask": [True, True, False, True, True],  # Keep index 2 (Explanation)
    },
    {
        "id": "process_only",
        "name": "3. Process Only (Process Transformer)",
        "active_modalities": ["process"],
        "uses_formula_features": False,
        "mask": [False, True, True, True, True],  # Keep index 0 (Process)
    },
    {
        "id": "code_plus_explanation",
        "name": "4. Code + Explanation (Bimodal)",
        "active_modalities": ["code", "explanation", "alignment"],
        "uses_formula_features": False,
        "mask": [True, False, False, False, True],  # Keep Code, Explanation, Alignment
    },
    {
        "id": "code_plus_process",
        "name": "5. Code + Process (Bimodal)",
        "active_modalities": ["code", "process"],
        "uses_formula_features": False,
        "mask": [False, False, True, True, True],  # Keep Process, Code
    },
    {
        "id": "explanation_plus_process",
        "name": "6. Explanation + Process (Bimodal)",
        "active_modalities": ["explanation", "process"],
        "uses_formula_features": False,
        "mask": [False, True, False, True, True],  # Keep Process, Explanation
    },
    {
        "id": "code_plus_explanation_plus_process",
        "name": "7. Code + Explanation + Process (Trimodal, No Formula Features)",
        "active_modalities": ["code", "explanation", "process", "alignment"],
        "uses_formula_features": False,
        "mask": [False, False, False, False, True],  # Mask only Numerical (index 4)
    },
    {
        "id": "full_multimodal_with_formulas",
        "name": "8. Full Multimodal Model with Formula-Derived Features (Proposed Full)",
        "active_modalities": ["code", "explanation", "process", "alignment", "numerical_formulas"],
        "uses_formula_features": True,
        "mask": [False, False, False, False, False], # All 5 modalities active
    },
    {
        "id": "full_multimodal_without_formulas",
        "name": "9. Full Multimodal Model without Formula-Derived Features",
        "active_modalities": ["code", "explanation", "process", "alignment"],
        "uses_formula_features": False,
        "mask": [False, False, False, False, True], # Numerical features masked out
    },
]


class MultimodalEvaluator:
    """Computes comprehensive multi-class metrics for future trained model evaluations."""

    LABELS = LABELS

    @classmethod
    def compute_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute full suite of classification metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_dict = {cls.LABELS[i]: round(float(per_class_f1[i]), 4) for i in range(min(len(cls.LABELS), len(per_class_f1)))}

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        report = classification_report(y_true, y_pred, target_names=cls.LABELS, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "per_class_f1": per_class_dict,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    @classmethod
    def get_ablation_plan(cls) -> List[Dict[str, Any]]:
        """Return the documented specification of all 9 ablation study experiments."""
        return ABLATION_CONFIGURATIONS
