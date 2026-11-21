"""Reusable numerical feature preprocessing and formula normalization module.

Retains existing mathematical formulas (S_PROCESS, S_OVERALL, S_INC, S_DEBUG, S_REV, S_TEST)
strictly as numerical feature inputs for the Multimodal Fusion Transformer, rather than
as decision-making classifiers.
"""
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
from app.config import settings
from app.services.process_telemetry import ProcessTelemetryService
from app.ml.config import NUMERICAL_FEATURE_NAMES, NUM_FEATURES_DIM, FEATURE_CONTRACT

# Compatibility aliases
NUMERICAL_FEATURE_DIM = NUM_FEATURES_DIM


class NumericalFeaturePreprocessor:
    """Extracts, normalizes, and encodes formula-derived scores and evidence into a numerical tensor."""

    FEATURE_NAMES = NUMERICAL_FEATURE_NAMES
    FEATURE_DIM = NUM_FEATURES_DIM

    @classmethod
    def extract_scores(
        cls,
        ast_score: float,
        exec_score: float,
        viva_score: float,
        inc_score: float,
        debug_score: float,
        rev_score: float,
        test_score: float,
    ) -> Tuple[float, float]:
        """Compute formula-based S_process and S_overall purely as numerical features."""
        s_proc = round(
            settings.weight_proc_incremental * inc_score
            + settings.weight_proc_debugging * debug_score
            + settings.weight_proc_revision * rev_score
            + settings.weight_proc_testing * test_score,
            4,
        )
        s_overall = round(
            settings.weight_ast * ast_score
            + settings.weight_exec * exec_score
            + settings.weight_viva * viva_score
            + settings.weight_process * s_proc,
            4,
        )
        return s_proc, s_overall

    @classmethod
    def transform(
        cls,
        ast_summary: Optional[Dict[str, Any]] = None,
        exec_output: Optional[str] = None,
        err_output: Optional[str] = None,
        exit_code: int = 0,
        is_valid: bool = True,
        viva_assessments: Optional[List[Any]] = None,
        telemetry_features: Optional[Any] = None,
        include_s_overall: bool = False,
        include_s_viva: bool = False,
        feature_mode: str = "default",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Transform multimodal inputs into a normalized numerical feature vector (24-D).
        
        Args:
            include_s_overall: If False (default), s_overall is masked to 0.0 in the input vector
                to prevent redundant composite score leakage into the Transformer, while preserving
                s_overall in the returned feature dictionary for baseline reporting.
            include_s_viva: If False (default), s_viva is masked to 0.0 in the input vector
                to prevent evaluator-derived target leakage into the Transformer. Viva text is already
                encoded by Sentence-Transformer; evaluator rubric assessments must not enter the model.
                Only set to True in explicitly designated leakage-diagnostic ablation probes.
            feature_mode: Modular configuration for ablation studies:
                - "default": Individual formula components + AST/telemetry metrics (s_overall and s_viva masked).
                - "no_formula": Zero out all formula-derived scores (indices 0..8).
                - "individual_formula_only": Only individual formula components (telemetry zeroed).
                - "telemetry_only": Only raw telemetry and AST complexity (formulas zeroed).
                - "leakage_probe_with_viva": Explicit ablation probe unmasking s_viva.
                - "full_with_overall": All features including s_overall (for comparison/baseline only).
        
        Returns:
            vector: (24,) float32 numpy array normalized to [0, 1]
            feature_dict: mapping of feature names to their extracted scalar values
        """
        fdict: Dict[str, float] = {}

        # 1. AST Structural Scores
        ast_sum = ast_summary or {}
        complexity = float(ast_sum.get("cyclomatic_complexity", 1) or 1)
        nodes_count = float(ast_sum.get("total_nodes", 10) or 10)
        lines_count = float(ast_sum.get("code_lines", 5) or 5)
        max_depth = float(ast_sum.get("max_depth", 1) or 1)
        has_rec = 1.0 if ast_sum.get("is_recursive") else 0.0

        if is_valid:
            ast_s = 0.75
            if complexity >= 2:
                ast_s += 0.10
            if ast_sum.get("loops") or ast_sum.get("functions"):
                ast_s += 0.15
            ast_s = min(1.0, ast_s)
        else:
            ast_s = 0.40

        # 2. Execution Fidelity Scores
        has_stdout = bool(exec_output and len(exec_output.strip()) > 0)
        has_error = bool(err_output and len(err_output.strip()) > 0)
        if has_stdout:
            exec_s = 0.45 if has_error else 1.0
        elif has_error:
            exec_s = 0.40
        else:
            exec_s = 0.60

        # 3. Viva Reasoning Scores
        if viva_assessments:
            v_scores = []
            for a in viva_assessments:
                lbl = getattr(a, "answer_label", "") if hasattr(a, "answer_label") else (
                    a.get("answer_label", "") if isinstance(a, dict) else ""
                )
                if lbl == "Correct":
                    v_scores.append(1.0)
                elif lbl == "Partially Correct":
                    v_scores.append(0.65)
                elif lbl == "Insufficient Evidence":
                    v_scores.append(0.40)
                else:
                    v_scores.append(0.15)
            viva_s = sum(v_scores) / len(v_scores) if v_scores else 0.50
        else:
            viva_s = 0.50

        # 4. Process Telemetry Component Scores
        get_val = lambda key, dflt: getattr(telemetry_features, key, None) if hasattr(telemetry_features, key) else (
            telemetry_features.get(key, dflt) if isinstance(telemetry_features, dict) else dflt
        )
        inc_s = float(get_val("incremental_score", 0.75) or 0.75)
        debug_s = float(get_val("debugging_score", 0.90) or 0.90)
        rev_s = float(get_val("revision_score", 0.75) or 0.75)
        test_s = float(get_val("testing_score", 0.60) or 0.60)

        # Compute composite formulas
        proc_s, overall_s = cls.extract_scores(ast_s, exec_s, viva_s, inc_s, debug_s, rev_s, test_s)

        # Telemetry metrics
        paste_ratio = float(get_val("paste_ratio", 0.0) or 0.0)
        burst_ratio = float(get_val("burstiness_ratio", 0.0) or 0.0)
        dur_s = float(get_val("coding_duration_seconds", 0.0) or 0.0)
        total_ev = float(get_val("total_events", 0) or 0)
        rec_rate = float(get_val("error_recovery_rate", 1.0) or 1.0)
        cpm = float(get_val("typing_speed_cpm", 0.0) or 0.0)

        # Populate dictionary
        fdict["s_overall"] = min(1.0, max(0.0, overall_s))
        fdict["s_ast"] = min(1.0, max(0.0, ast_s))
        fdict["s_exec"] = min(1.0, max(0.0, exec_s))
        fdict["s_viva"] = min(1.0, max(0.0, viva_s))
        fdict["s_process"] = min(1.0, max(0.0, proc_s))
        fdict["s_inc"] = min(1.0, max(0.0, inc_s))
        fdict["s_debug"] = min(1.0, max(0.0, debug_s))
        fdict["s_rev"] = min(1.0, max(0.0, rev_s))
        fdict["s_test"] = min(1.0, max(0.0, test_s))

        fdict["paste_ratio"] = min(1.0, max(0.0, paste_ratio))
        fdict["burstiness_ratio"] = min(1.0, max(0.0, burst_ratio))
        fdict["coding_duration_norm"] = min(1.0, math.log1p(dur_s) / 7.0)
        fdict["total_events_norm"] = min(1.0, math.log1p(total_ev) / 7.0)
        fdict["error_recovery_rate"] = min(1.0, max(0.0, rec_rate))
        fdict["typing_speed_cpm_norm"] = min(1.0, cpm / 350.0)

        fdict["cyclomatic_complexity_norm"] = min(1.0, complexity / 10.0)
        fdict["total_nodes_norm"] = min(1.0, nodes_count / 100.0)
        fdict["total_lines_norm"] = min(1.0, lines_count / 50.0)
        fdict["max_depth_norm"] = min(1.0, max_depth / 8.0)
        fdict["has_recursion"] = has_rec

        fdict["is_syntax_valid"] = 1.0 if is_valid else 0.0
        fdict["exit_code_zero"] = 1.0 if exit_code == 0 else 0.0
        fdict["has_execution_error"] = 1.0 if has_error else 0.0
        fdict["has_stdout_output"] = 1.0 if has_stdout else 0.0

        # Construct vector in fixed order, applying modular filtering
        raw_vals = [fdict[name] for name in cls.FEATURE_NAMES]
        vec = np.array(raw_vals, dtype=np.float32)

        # Formula scores occupy indices 0 to 8:
        # 0: s_overall, 1: s_ast, 2: s_exec, 3: s_viva, 4: s_process, 5-8: s_inc, s_debug, s_rev, s_test
        # Telemetry and AST metrics occupy indices 9 to 23
        if not include_s_overall and feature_mode != "full_with_overall":
            vec[0] = 0.0  # Mask s_overall to prevent redundant composite leakage

        # CRITICAL LEAKAGE PREVENTION: Mask s_viva (index 3) from primary model input
        # S_VIVA is computed from evaluator rubric annotations (answer_label) and constitutes
        # direct target leakage. The student's verbal reasoning is already encoded via Sentence-Transformer.
        if not include_s_viva and feature_mode != "leakage_probe_with_viva":
            vec[3] = 0.0

        if feature_mode == "no_formula":
            vec[0:9] = 0.0  # Zero out all formula scores
        elif feature_mode == "individual_formula_only":
            vec[0] = 0.0
            vec[3] = 0.0 if not include_s_viva else vec[3]
            vec[9:] = 0.0   # Zero out raw telemetry and AST metrics
        elif feature_mode == "telemetry_only":
            vec[0:9] = 0.0  # Zero out all formula scores

        return vec, fdict
