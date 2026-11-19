"""Multimodal student input data representations and pre-processing structures.

Collects and prepares:
1. Coding-Process Data: Ordered chronological event sequences & numerical behavioral features.
2. Submitted Source Code: Source text, AST construct summaries, and sandbox execution traces.
3. Student Explanations: Micro-viva questions, student natural-language explanations, and concept contexts.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from app.ml.process_transformer import normalize_process_event_timestamps


@dataclass
class CodingProcessInput:
    """Represents student coding-process telemetry."""
    # Ordered list of raw/structured telemetry event dictionaries
    events: List[Dict[str, Any]] = field(default_factory=list)
    # Aggregated numerical features (from ProcessTelemetryService)
    numerical_features: Dict[str, float] = field(default_factory=dict)
    session_id: Optional[str] = None

    def get_event_count(self) -> int:
        return len(self.events)


@dataclass
class SubmittedCodeInput:
    """Represents submitted Python source code and structural execution evidence."""
    code: str = ""
    ast_summary: Optional[Dict[str, Any]] = None
    execution_output: Optional[str] = None
    error_output: Optional[str] = None
    exit_code: int = 0
    is_valid: bool = True


@dataclass
class StudentExplanationInput:
    """Represents student verbal or written viva answers paired with question and code context."""
    answers: List[Dict[str, Any]] = field(default_factory=list)
    # Combined textual transcript of explanations
    full_transcript: str = ""
    # Per-question pairs: list of (question_text, student_answer, target_concept)
    question_answer_pairs: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class MultimodalStudentInputBundle:
    """Unified container for all three input modalities of a student session."""
    process: CodingProcessInput
    code: SubmittedCodeInput
    explanation: StudentExplanationInput
    session_id: Optional[str] = None

    @classmethod
    def from_session_data(
        cls,
        session_id: str,
        events: List[Any],
        telemetry_features: Any,
        code: str,
        ast_summary: Optional[Dict[str, Any]] = None,
        exec_output: Optional[str] = None,
        err_output: Optional[str] = None,
        exit_code: int = 0,
        is_valid: bool = True,
        viva_assessments: Optional[List[Any]] = None,
        viva_questions: Optional[List[Any]] = None,
    ) -> "MultimodalStudentInputBundle":
        """Factory method to construct a standard MultimodalStudentInputBundle from DB models/APIs."""
        # 1. Process Event Normalization with robust timestamp conversion
        norm_events: List[Dict[str, Any]] = normalize_process_event_timestamps(events)

        # Process numerical features dict
        num_dict = {}
        if telemetry_features:
            attrs = [
                "edit_count", "paste_count", "run_count", "error_count",
                "total_chars_pasted", "total_chars_edited", "paste_ratio",
                "coding_duration_seconds", "average_edit_interval_seconds",
                "burst_count", "burstiness_ratio", "error_recovery_rate",
                "incremental_score", "debugging_score", "revision_score", "testing_score"
            ]
            for attr in attrs:
                val = getattr(telemetry_features, attr, None) if hasattr(telemetry_features, attr) else (
                    telemetry_features.get(attr) if isinstance(telemetry_features, dict) else 0.0
                )
                num_dict[attr] = float(val or 0.0)

        proc_input = CodingProcessInput(events=norm_events, numerical_features=num_dict, session_id=session_id)

        # 2. Submitted Code Input
        code_input = SubmittedCodeInput(
            code=code,
            ast_summary=ast_summary or {},
            execution_output=exec_output,
            error_output=err_output,
            exit_code=exit_code,
            is_valid=is_valid,
        )

        # 3. Student Explanation Input
        qa_pairs: List[Tuple[str, str, str]] = []
        transcript_parts: List[str] = []
        ans_list: List[Dict[str, Any]] = []

        q_map = {}
        if viva_questions:
            for q in viva_questions:
                q_id = getattr(q, "id", None) if hasattr(q, "id") else q.get("id")
                q_map[q_id] = q

        if viva_assessments:
            for a in viva_assessments:
                q_id = getattr(a, "question_id", None) if hasattr(a, "question_id") else a.get("question_id")
                s_ans = getattr(a, "student_answer", "") if hasattr(a, "student_answer") else a.get("student_answer", "")
                q_obj = q_map.get(q_id)
                q_text = getattr(q_obj, "question_text", "") if q_obj and hasattr(q_obj, "question_text") else (q_obj.get("question_text", "") if isinstance(q_obj, dict) else "")
                q_concept = getattr(q_obj, "concept", "") if q_obj and hasattr(q_obj, "concept") else (q_obj.get("concept", "") if isinstance(q_obj, dict) else "General Logic")

                ans_list.append({
                    "question_id": q_id,
                    "question_text": q_text,
                    "concept": q_concept,
                    "student_answer": s_ans,
                    "answer_label": getattr(a, "answer_label", "") if hasattr(a, "answer_label") else a.get("answer_label", ""),
                })
                if s_ans and s_ans.strip():
                    qa_pairs.append((q_text, s_ans, q_concept))
                    transcript_parts.append(f"Question: {q_text} Explanation: {s_ans}")

        exp_input = StudentExplanationInput(
            answers=ans_list,
            full_transcript=" ".join(transcript_parts),
            question_answer_pairs=qa_pairs,
        )

        return cls(process=proc_input, code=code_input, explanation=exp_input, session_id=session_id)
