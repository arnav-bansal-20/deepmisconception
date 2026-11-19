"""Central Canonical Feature and Model Dimension Configuration for DeepMisconception.

Defines the single source of truth for all tensor shapes, feature positions,
and architectural parameters across feature extractors, preprocessors,
pretrained encoders, and multimodal fusion models.
"""
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


# ------------------------------------------------------------------------------
# 1. Canonical Model and Tensor Dimensions
# ------------------------------------------------------------------------------
CODE_DIM: int = 768             # Contextual representation from microsoft/codebert-base
EXP_DIM: int = 384              # Semantic representation from all-MiniLM-L6-v2
PROC_DIM: int = 64              # Learned representation from Process Transformer Encoder
ALIGN_DIM: int = 64             # Cross-modal semantic alignment representation
NUM_FEATURES_DIM: int = 24      # Normalized numerical formula and telemetry features
D_MODEL: int = 128              # Uniform multimodal Transformer latent dimension
NUM_HEADS: int = 4              # Multihead attention heads
NUM_ENCODER_LAYERS: int = 3     # Transformer Fusion Encoder layers
NUM_CLASSES: int = 3            # Exactly 3 comprehension classes


# ------------------------------------------------------------------------------
# 2. Comprehension Class Labels
# ------------------------------------------------------------------------------
LABELS: Tuple[str, str, str] = (
    "Understood",
    "Partial Understanding",
    "Possible Misconception",
)
LABEL_MAP: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(LABELS)}
REV_LABEL_MAP: Dict[int, str] = {idx: lbl for idx, lbl in enumerate(LABELS)}

# Abstention / Evidence Status (NOT a 4th misconception class)
STATUS_INSUFFICIENT_EVIDENCE: str = "INSUFFICIENT_EVIDENCE"
STATUS_MODEL_NOT_TRAINED: str = "MODEL_NOT_TRAINED"
STATUS_READY: str = "READY"


# ------------------------------------------------------------------------------
# 3. Canonical Numerical Feature Names & Order (24 Dimensions)
# ------------------------------------------------------------------------------
# NOTE: Index 0 (s_overall) is masked to 0.0 by default in the Transformer input vector
# to prevent target leakage and redundant composite score feedback, while remaining
# available in supporting dictionaries for reporting and baseline analysis.
NUMERICAL_FEATURE_NAMES: List[str] = [
    # 0–4: High-Level Composite & Modality Formula Scores
    "s_overall",                    # Index 0: masked to 0.0 in model input
    "s_ast",                        # Index 1: AST Structural Score
    "s_exec",                       # Index 2: Sandbox Execution Fidelity Score
    "s_viva",                       # Index 3: Viva Reasoning Score
    "s_process",                    # Index 4: Process Telemetry Behavioral Score
    # 5–8: Process Sub-dimension Component Scores
    "s_inc",                        # Index 5: Incremental Development Score
    "s_debug",                      # Index 6: Error Recovery / Debugging Score
    "s_rev",                        # Index 7: Revision Cycles Score
    "s_test",                       # Index 8: Sandbox Testing Activity Score
    # 9–14: Fine-Grained Editor Behavioral Dynamics
    "paste_ratio",                  # Index 9: Ratio of pasted chars to total chars
    "burstiness_ratio",             # Index 10: Ratio of bursty edits
    "coding_duration_norm",         # Index 11: Log-normalized session duration
    "total_events_norm",            # Index 12: Log-normalized total events
    "error_recovery_rate",          # Index 13: Rate of successful error recovery
    "typing_speed_cpm_norm",        # Index 14: Normalized characters per minute
    # 15–19: AST Structural Complexity Metrics
    "cyclomatic_complexity_norm",   # Index 15: Normalized cyclomatic complexity M/10
    "total_nodes_norm",             # Index 16: Normalized total AST nodes
    "total_lines_norm",             # Index 17: Normalized code lines
    "max_depth_norm",               # Index 18: Normalized AST tree depth
    "has_recursion",                # Index 19: Boolean flag for recursive calls
    # 20–23: Execution Trace & Syntax Validity Indicators
    "is_syntax_valid",              # Index 20: 1.0 if valid Python AST, else 0.0
    "exit_code_zero",               # Index 21: 1.0 if exit code == 0, else 0.0
    "has_execution_error",          # Index 22: 1.0 if stderr/traceback present
    "has_stdout_output",            # Index 23: 1.0 if stdout generated
]

assert len(NUMERICAL_FEATURE_NAMES) == NUM_FEATURES_DIM, (
    f"Mismatch in NUMERICAL_FEATURE_NAMES: expected {NUM_FEATURES_DIM}, got {len(NUMERICAL_FEATURE_NAMES)}"
)


# ------------------------------------------------------------------------------
# 4. Modality Index Mapping in Fusion Sequence
# ------------------------------------------------------------------------------
# In the 5-token multimodal Transformer sequence:
# Index 0: Process representation
# Index 1: Code representation
# Index 2: Explanation representation
# Index 3: Cross-modal alignment representation
# Index 4: Numerical feature representation
MODALITY_NAME_TO_IDX: Dict[str, int] = {
    "process": 0,
    "code": 1,
    "explanation": 2,
    "alignment": 3,
    "numerical": 4,
}
NUM_MODALITIES: int = len(MODALITY_NAME_TO_IDX)  # 5 modalities


@dataclass(frozen=True)
class FeatureDimensionContract:
    """Immutable configuration contract validating all input/output tensor shapes."""
    code_dim: int = CODE_DIM
    exp_dim: int = EXP_DIM
    proc_dim: int = PROC_DIM
    align_dim: int = ALIGN_DIM
    num_features_dim: int = NUM_FEATURES_DIM
    d_model: int = D_MODEL
    num_heads: int = NUM_HEADS
    num_encoder_layers: int = NUM_ENCODER_LAYERS
    num_classes: int = NUM_CLASSES

    def validate_code_shape(self, shape: Tuple[int, ...]):
        if shape[-1] != self.code_dim:
            raise ValueError(f"Code tensor dimension mismatch: expected (*, {self.code_dim}), got {shape}")

    def validate_exp_shape(self, shape: Tuple[int, ...]):
        if shape[-1] != self.exp_dim:
            raise ValueError(f"Explanation tensor dimension mismatch: expected (*, {self.exp_dim}), got {shape}")

    def validate_proc_shape(self, shape: Tuple[int, ...]):
        if shape[-1] != self.proc_dim:
            raise ValueError(f"Process tensor dimension mismatch: expected (*, {self.proc_dim}), got {shape}")

    def validate_align_shape(self, shape: Tuple[int, ...]):
        if shape[-1] != self.align_dim:
            raise ValueError(f"Alignment tensor dimension mismatch: expected (*, {self.align_dim}), got {shape}")

    def validate_num_features_shape(self, shape: Tuple[int, ...]):
        if shape[-1] != self.num_features_dim:
            raise ValueError(f"Numerical features dimension mismatch: expected (*, {self.num_features_dim}), got {shape}")


FEATURE_CONTRACT = FeatureDimensionContract()
