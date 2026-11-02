"""Synthetic student benchmark dataset generator for reproducible ML training and ablations.

All records generated here are explicitly flagged with `is_synthetic: True` metadata.
Includes student-level group splitting to prevent cross-session data leakage.
"""
from typing import Tuple, List, Dict, Any
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from app.ml.feature_extractor import MultimodalFeatureExtractor


class BenchmarkDatasetGenerator:
    """Generates balanced, grounded multimodal datasets representing diverse student profiles."""

    IS_SYNTHETIC = True
    DATASET_TYPE = "synthetic_benchmark"
    DISCLAIMER = (
        "Notice: This dataset contains mathematically synthesized student interactions designed for "
        "model shape validation, ablation architecture testing, and training pipeline verification. "
        "It does NOT represent empirical classroom cohort data."
    )

    @classmethod
    def get_metadata(cls, n_samples: int) -> Dict[str, Any]:
        """Return explicit provenance metadata for the generated dataset."""
        return {
            "is_synthetic": cls.IS_SYNTHETIC,
            "dataset_type": cls.DATASET_TYPE,
            "n_samples": n_samples,
            "classes": ["Understood", "Partial Understanding", "Possible Misconception"],
            "disclaimer": cls.DISCLAIMER,
        }

    @classmethod
    def generate(cls, n_samples: int = 150, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Generate balanced student samples with realistic features across 3 classes."""
        rng = np.random.RandomState(seed)
        samples_per_class = n_samples // 3

        X_list = []
        y_list = []
        records = []

        # 1. Class 0: Understood (Skilled Incremental Developers)
        for i in range(samples_per_class):
            code = (
                "def solve(n: int) -> int:\n"
                "    total = 0\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 3 == 0:\n"
                "            continue\n"
                "        total += i * i\n"
                "    return total\n"
            )
            explanation = (
                "The accumulator total starts at 0 and iterates from 1 up to n inclusive. "
                "Because of the modulo conditional branch, numbers divisible by 3 are skipped "
                "using the continue statement before updating the sum of squares."
            )
            proc = {
                "paste_ratio": float(rng.uniform(0.0, 0.15)),
                "burstiness_ratio": float(rng.uniform(0.35, 0.70)),
                "coding_duration_seconds": float(rng.uniform(120, 300)),
                "total_events": int(rng.randint(60, 180)),
                "run_count": int(rng.randint(3, 8)),
                "error_count": int(rng.randint(0, 2)),
                "error_recovery_rate": 1.0,
                "typing_speed_cpm": float(rng.uniform(140, 260)),
            }
            feat = MultimodalFeatureExtractor.extract(code, explanation, proc)
            feat += rng.normal(0, 0.02, size=feat.shape).astype(np.float32)

            X_list.append(feat)
            y_list.append(0)
            records.append({
                "student_id": f"SYN-STU-SKILL-{i+1}",
                "profile": "Skilled Incremental Developer",
                "label": "Understood",
                "is_synthetic": True,
            })

        # 2. Class 1: Partial Understanding (Trial-and-Error / Memorizers)
        for i in range(samples_per_class):
            is_paster = (i % 2 == 0)
            code = (
                "def solve(n: int) -> int:\n"
                "    total = 0\n"
                "    for i in range(n):\n"  # Off-by-one boundary
                "        total += i * i\n"
                "    return total\n"
            )
            explanation = (
                "I used a loop to compute the total square sum, but I am not completely sure "
                "about the exact upper index boundary of range(n)."
            )
            proc = {
                "paste_ratio": float(rng.uniform(0.30, 0.65) if is_paster else rng.uniform(0.0, 0.20)),
                "burstiness_ratio": float(rng.uniform(0.60, 0.85)),
                "coding_duration_seconds": float(rng.uniform(90, 240)),
                "total_events": int(rng.randint(40, 120)),
                "run_count": int(rng.randint(1, 4)),
                "error_count": int(rng.randint(2, 6)),
                "error_recovery_rate": float(rng.uniform(0.40, 0.75)),
                "typing_speed_cpm": float(rng.uniform(100, 200)),
            }
            feat = MultimodalFeatureExtractor.extract(code, explanation, proc)
            feat += rng.normal(0, 0.02, size=feat.shape).astype(np.float32)

            X_list.append(feat)
            y_list.append(1)
            records.append({
                "student_id": f"SYN-STU-PARTIAL-{i+1}",
                "profile": "Trial-and-Error / Memorizer",
                "label": "Partial Understanding",
                "is_synthetic": True,
            })

        # 3. Class 2: Possible Misconception (Flawed Mental Models)
        for i in range(samples_per_class):
            code = (
                "def solve(n: int) -> int:\n"
                "    for i in range(1, n + 1):\n"
                "        total = 0\n"  # Scope reset bug
                "        if i % 3 == 0:\n"
                "            break\n"   # Fatal misconception: break instead of continue
                "        total += i * i\n"
                "    return total\n"
            )
            explanation = (
                "The loop breaks and ends the whole program whenever a multiple of 3 is seen. "
                "Total is initialized to 0 inside the loop so it keeps the sum."
            )
            proc = {
                "paste_ratio": float(rng.uniform(0.40, 0.80)),
                "burstiness_ratio": float(rng.uniform(0.70, 0.95)),
                "coding_duration_seconds": float(rng.uniform(40, 150)),
                "total_events": int(rng.randint(15, 60)),
                "run_count": int(rng.randint(0, 2)),
                "error_count": int(rng.randint(3, 8)),
                "error_recovery_rate": float(rng.uniform(0.10, 0.40)),
                "typing_speed_cpm": float(rng.uniform(80, 160)),
            }
            feat = MultimodalFeatureExtractor.extract(code, explanation, proc)
            feat += rng.normal(0, 0.02, size=feat.shape).astype(np.float32)

            X_list.append(feat)
            y_list.append(2)
            records.append({
                "student_id": f"SYN-STU-MISCON-{i+1}",
                "profile": "Flawed Mental Model",
                "label": "Possible Misconception",
                "is_synthetic": True,
            })

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Shuffle deterministically
        indices = rng.permutation(len(y))
        return X[indices], y[indices], [records[idx] for idx in indices]

    @classmethod
    def split_by_student(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        records: List[Dict[str, Any]],
        test_size: float = 0.25,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset grouped by student_id to strictly prevent cross-session data leakage."""
        student_ids = np.array([r["student_id"] for r in records])
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=student_ids))

        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        return (
            X[train_idx],
            X[test_idx],
            y[train_idx],
            y[test_idx],
            train_records,
            test_records,
        )
