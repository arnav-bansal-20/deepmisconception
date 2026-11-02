"""Modality Ablation Study Runner evaluating multimodal contributions."""
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import train_test_split
from app.ml.dataset_generator import BenchmarkDatasetGenerator
from app.ml.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    DecisionTreeBaseline,
    MLPBaseline,
)
from app.ml.multimodal_transformer import MultimodalCrossAttentionFusion

MODALITY_SLICES = {
    "code_only": slice(0, 16),
    "explanation_only": slice(16, 32),
    "process_only": slice(32, 40),
    "code_plus_explanation": slice(0, 32),
    "full_multimodal": slice(0, 40),
}


class AblationRunner:
    """Executes systematic ablation studies across modalities and models."""

    @classmethod
    def mask_features(cls, X: np.ndarray, config: str) -> np.ndarray:
        """Zero out modalities not active in the ablation configuration."""
        X_masked = np.zeros_like(X)
        if config == "code_only":
            X_masked[:, 0:16] = X[:, 0:16]
        elif config == "explanation_only":
            X_masked[:, 16:32] = X[:, 16:32]
        elif config == "process_only":
            X_masked[:, 32:40] = X[:, 32:40]
        elif config == "code_plus_explanation":
            X_masked[:, 0:32] = X[:, 0:32]
        elif config == "full_multimodal":
            X_masked = X.copy()
        return X_masked

    @classmethod
    def run_ablation(
        cls,
        n_samples: int = 180,
        test_size: float = 0.25,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run complete ablation matrix across all modality configurations with zero cross-student leakage."""
        X, y, records = BenchmarkDatasetGenerator.generate(n_samples=n_samples, seed=seed)
        X_train, X_test, y_train, y_test, _, _ = BenchmarkDatasetGenerator.split_by_student(
            X, y, records, test_size=test_size, seed=seed
        )

        configs = [
            ("Code Only (AST)", "code_only"),
            ("Explanation Only (Viva)", "explanation_only"),
            ("Process Only (Telemetry)", "process_only"),
            ("Code + Explanation", "code_plus_explanation"),
            ("Full Multimodal Fusion", "full_multimodal"),
        ]

        results = []

        for display_name, config_key in configs:
            X_tr_masked = cls.mask_features(X_train, config_key)
            X_te_masked = cls.mask_features(X_test, config_key)

            # 1. Random Forest Baseline
            rf = RandomForestBaseline(random_state=seed)
            rf.fit(X_tr_masked, y_train)
            rf_eval = rf.evaluate(X_te_masked, y_test)

            # 2. Logistic Regression Baseline
            lr = LogisticRegressionBaseline(random_state=seed)
            lr.fit(X_tr_masked, y_train)
            lr_eval = lr.evaluate(X_te_masked, y_test)

            # 3. Multimodal Cross-Attention Transformer
            transformer = MultimodalCrossAttentionFusion(seed=seed)
            transformer.fit(X_tr_masked, y_train, epochs=25, lr=0.005)
            tf_eval = transformer.evaluate(X_te_masked, y_test)

            results.append({
                "configuration": display_name,
                "config_key": config_key,
                "random_forest": {
                    "accuracy": rf_eval["accuracy"],
                    "macro_f1": rf_eval["macro_f1"],
                },
                "logistic_regression": {
                    "accuracy": lr_eval["accuracy"],
                    "macro_f1": lr_eval["macro_f1"],
                },
                "cross_attention_transformer": {
                    "accuracy": tf_eval["accuracy"],
                    "macro_f1": tf_eval["macro_f1"],
                    "confusion_matrix": tf_eval["confusion_matrix"],
                },
            })

        metadata = BenchmarkDatasetGenerator.get_metadata(n_samples)
        return {
            "is_synthetic": True,
            "dataset_type": metadata["dataset_type"],
            "disclaimer": metadata["disclaimer"],
            "split_method": "student_level_group_split (zero cross-student leakage)",
            "n_samples": n_samples,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "modalities_evaluated": len(configs),
            "results": results,
        }
