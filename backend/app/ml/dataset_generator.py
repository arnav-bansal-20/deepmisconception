"""Synthetic student benchmark dataset generator for reproducible ML training and ablations."""
from typing import Tuple, List, Dict, Any
import numpy as np
from app.ml.feature_extractor import MultimodalFeatureExtractor


class BenchmarkDatasetGenerator:
    """Generates balanced, grounded multimodal datasets representing diverse student profiles."""

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
            # Add subtle natural variance
            feat += rng.normal(0, 0.02, size=feat.shape).astype(np.float32)

            X_list.append(feat)
            y_list.append(0)
            records.append({
                "student_id": f"STU-SKILL-{i+1}",
                "profile": "Skilled Incremental Developer",
                "label": "Understood",
            })

        # 2. Class 1: Partial Understanding (Trial-and-Error / Memorizers)
        for i in range(samples_per_class):
            is_paster = (i % 2 == 0)
            if is_paster:
                code = (
                    "def binary_search(arr, target):\n"
                    "    low, high = 0, len(arr) - 1\n"
                    "    while low <= high:\n"
                    "        mid = (low + high) // 2\n"
                    "        if arr[mid] == target: return mid\n"
                    "        elif arr[mid] < target: low = mid + 1\n"
                    "        else: high = mid - 1\n"
                    "    return -1\n"
                )
                explanation = "The code does binary search with low and high pointers. It works by checking mid."
                paste_r = float(rng.uniform(0.65, 0.88))
            else:
                code = (
                    "def accumulator(n):\n"
                    "    total = 0\n"
                    "    for i in range(n):\n"
                    "        total += i\n"
                    "    return total\n"
                )
                explanation = "Loops through elements and adds them to total, but I missed the upper bound condition."
                paste_r = float(rng.uniform(0.10, 0.35))

            proc = {
                "paste_ratio": paste_r,
                "burstiness_ratio": float(rng.uniform(0.15, 0.40)),
                "coding_duration_seconds": float(rng.uniform(80, 220)),
                "total_events": int(rng.randint(30, 90)),
                "run_count": int(rng.randint(4, 12)),
                "error_count": int(rng.randint(2, 5)),
                "error_recovery_rate": float(rng.uniform(0.60, 0.85)),
                "typing_speed_cpm": float(rng.uniform(80, 160)),
            }
            feat = MultimodalFeatureExtractor.extract(code, explanation, proc)
            feat += rng.normal(0, 0.02, size=feat.shape).astype(np.float32)

            X_list.append(feat)
            y_list.append(1)
            records.append({
                "student_id": f"STU-PARTIAL-{i+1}",
                "profile": "Partial Understanding (Trial / Paster)",
                "label": "Partial Understanding",
            })

        # 3. Class 2: Possible Misconception (Struggling / Concept Flaws)
        for i in range(samples_per_class):
            code = (
                "def fact(n):\n"
                "    return n * fact(n - 1)\n"
            )
            explanation = (
                "I don't know why it crashed. Maybe the number was too big for memory. "
                "Python functions should automatically stop when n reaches 0."
            )
            proc = {
                "paste_ratio": float(rng.uniform(0.20, 0.50)),
                "burstiness_ratio": float(rng.uniform(0.05, 0.25)),
                "coding_duration_seconds": float(rng.uniform(40, 150)),
                "total_events": int(rng.randint(15, 45)),
                "run_count": int(rng.randint(1, 3)),
                "error_count": int(rng.randint(2, 4)),
                "error_recovery_rate": float(rng.uniform(0.0, 0.30)),
                "typing_speed_cpm": float(rng.uniform(40, 90)),
            }
            feat = MultimodalFeatureExtractor.extract(code, explanation, proc)
            feat += rng.normal(0, 0.02, size=feat.shape).astype(np.float32)

            X_list.append(feat)
            y_list.append(2)
            records.append({
                "student_id": f"STU-MISCON-{i+1}",
                "profile": "Struggling with Critical Misconception",
                "label": "Possible Misconception",
            })

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Shuffle deterministically
        perm = rng.permutation(len(X))
        X = X[perm]
        y = y[perm]
        records = [records[p] for p in perm]

        return X, y, records
