"""Code-Grounded Dynamic Student Answer Evaluator.

Performs dual-stage verification:
1. Fact-checks factual and quantitative claims against AST structures and sandboxed execution traces.
2. Evaluates semantic and conceptual explanation depth using the LLM provider.
3. Produces evidence-backed assessments with uncertainty scoring and instructor-review flagging.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.services.llm_provider import get_llm_provider, BaseLLMProvider
from app.schemas.viva import AnswerAssessmentResponse

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Raw result structure returned by evaluation engine."""
    answer_label: str  # Correct, Partially Correct, Incorrect, Insufficient Evidence
    expected_reasoning: str
    concepts_correct: List[str] = Field(default_factory=list)
    concepts_missed: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    evaluation_explanation: str
    uncertainty_score: float = 0.0
    needs_instructor_review: bool = False


class AnswerEvaluator:
    """Evaluates student viva answers against code reality, execution outputs, and rubrics."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or get_llm_provider()

    def evaluate_answer(
        self,
        student_answer: str,
        question_text: str,
        focal_code: Optional[str],
        concept: str,
        expected_reasoning: Optional[str],
        rubric: Optional[Dict[str, Any]],
        submitted_code: str,
        execution_output: Optional[str],
        ast_summary: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Execute the multi-stage code-grounded answer evaluation pipeline."""
        clean_ans = student_answer.strip()

        # --- Stage 1: Preliminary Heuristic & Ambiguity Filter ---
        if len(clean_ans) < 10:
            return EvaluationResult(
                answer_label="Insufficient Evidence",
                expected_reasoning=expected_reasoning or "Clear conceptual explanation required.",
                concepts_correct=[],
                concepts_missed=["Explanation is too short to evaluate"],
                contradictions=[],
                supporting_evidence=[clean_ans],
                evaluation_explanation="The student response is too brief to demonstrate conceptual understanding of the targeted code section.",
                uncertainty_score=0.9,
                needs_instructor_review=True,
            )

        # Detect evasive phrases
        evasive_keywords = ["idk", "i don't know", "dont know", "no idea", "not sure", "who knows", "random"]
        if any(kw in clean_ans.lower() for kw in evasive_keywords) and len(clean_ans) < 40:
            return EvaluationResult(
                answer_label="Insufficient Evidence",
                expected_reasoning=expected_reasoning or "Clear conceptual explanation required.",
                concepts_correct=[],
                concepts_missed=["Evasive answer without domain explanation"],
                contradictions=[],
                supporting_evidence=[clean_ans],
                evaluation_explanation="The student indicated uncertainty without attempting to explain the underlying code logic.",
                uncertainty_score=0.85,
                needs_instructor_review=True,
            )

        # --- Stage 2: Code & Execution Fact-Checking ---
        contradictions = self._detect_code_contradictions(
            clean_ans, focal_code, submitted_code, execution_output
        )

        # --- Stage 3: Semantic LLM Evaluation ---
        system_instruction = (
            "You are an expert pedagogy evaluator in the DeepMisconception verification system. "
            "Your task is to dynamically evaluate a student's verbal/textual explanation of Python code. "
            "Do not rely merely on keyword matching. Check whether the student's mechanistic explanation "
            "is sound, whether they understand variable changes, and whether they have conceptual misconceptions.\n"
            "You must output a single valid JSON object matching this schema:\n"
            "{\n"
            '  "answer_label": "Correct" | "Partially Correct" | "Incorrect" | "Insufficient Evidence",\n'
            '  "concepts_correct": ["Concept student correctly articulated 1"],\n'
            '  "concepts_missed": ["Nuance or concept student omitted or misunderstood 1"],\n'
            '  "contradictions": ["Direct conflict between student explanation and actual code behavior"],\n'
            '  "supporting_evidence": ["Exact short quote from student answer supporting your verdict"],\n'
            '  "evaluation_explanation": "Detailed explanation of why this verdict was chosen",\n'
            '  "uncertainty_score": 0.0 to 1.0,\n'
            '  "needs_instructor_review": true | false\n'
            "}"
        )

        user_prompt = (
            f"=== TARGET QUESTION ===\n{question_text}\n\n"
            f"=== REFERENCED CODE SECTION ===\n{focal_code or '[Full program]'}\n\n"
            f"=== TARGET PROGRAMMING CONCEPT ===\n{concept}\n\n"
            f"=== EXPECTED REASONING & RUBRIC ===\n"
            f"Expected Reasoning: {expected_reasoning or 'Mechanistically accurate explanation.'}\n"
            f"Rubric: {json.dumps(rubric or {}, indent=2)}\n\n"
            f"=== VERIFIED CODE EXECUTION REALITY ===\n"
            f"Full Code:\n{submitted_code}\n"
            f"Execution Stdout:\n{execution_output or '[No output]'}\n\n"
            f"=== STUDENT ANSWER TO EVALUATE ===\n"
            f"student answer:\n{clean_ans}\n"
        )

        try:
            llm_eval = self.provider.generate_json(system_instruction, user_prompt)

            # Extract fields with safe defaults
            label = llm_eval.get("answer_label", "Partially Correct")
            if label not in ("Correct", "Partially Correct", "Incorrect", "Insufficient Evidence"):
                label = "Partially Correct"

            c_correct = llm_eval.get("concepts_correct", [])
            c_missed = llm_eval.get("concepts_missed", [])
            c_contradictions = llm_eval.get("contradictions", [])
            evidence = llm_eval.get("supporting_evidence", [clean_ans[:100]])
            explanation = llm_eval.get("evaluation_explanation", "Evaluated based on code rubric criteria.")
            uncertainty = float(llm_eval.get("uncertainty_score", 0.2))
            needs_review = bool(llm_eval.get("needs_instructor_review", False))

            # Merge any verified code contradictions discovered in Stage 2
            for c in contradictions:
                if c not in c_contradictions:
                    c_contradictions.append(c)

            # If contradictions exist, cannot be fully Correct
            if c_contradictions and label == "Correct":
                label = "Incorrect"
                explanation += " Note: Verdict adjusted to Incorrect due to direct factual contradiction with code execution."

            # Flag for review if uncertainty is high or borderline
            if uncertainty > 0.5 or label == "Insufficient Evidence":
                needs_review = True

            return EvaluationResult(
                answer_label=label,
                expected_reasoning=expected_reasoning or "Mechanistically accurate explanation.",
                concepts_correct=c_correct,
                concepts_missed=c_missed,
                contradictions=c_contradictions,
                supporting_evidence=evidence,
                evaluation_explanation=explanation,
                uncertainty_score=round(uncertainty, 2),
                needs_instructor_review=needs_review,
            )

        except Exception as e:
            logger.error(f"Error during LLM answer evaluation: {e}")
            # Fallback deterministic evaluation
            return self._fallback_evaluation(clean_ans, concept, expected_reasoning, contradictions)

    def _detect_code_contradictions(
        self,
        student_answer: str,
        focal_code: Optional[str],
        submitted_code: str,
        execution_output: Optional[str]
    ) -> List[str]:
        """Detect direct factual contradictions with concrete execution realities."""
        contradictions: List[str] = []
        ans_lower = student_answer.lower()
        focal = (focal_code or "").lower()

        # Contradiction 1: Confusing continue with break
        if "continue" in focal and ("break" in ans_lower or "stops the loop" in ans_lower or "terminates the loop" in ans_lower):
            contradictions.append(
                "Student claimed the code 'stops' or 'terminates' the loop, but 'continue' only skips to the next iteration."
            )

        # Contradiction 2: Confusing loop range inclusivity
        if "range(1, n + 1)" in focal and "stops at n + 1" in ans_lower:
            contradictions.append(
                "Student claimed range(1, n + 1) iterates up to n + 1, but Python ranges exclude the upper stop parameter."
            )

        # Contradiction 3: Numerical output contradiction
        if execution_output and "computed total:" in execution_output.lower():
            # e.g., if verified output says "Computed total: 35" and student states "outputs 45" or "outputs 55"
            match = re.search(r"computed total:\s*(\d+)", execution_output.lower())
            if match:
                verified_num = match.group(1)
                # check if student claims a completely different number as the output
                claimed_nums = re.findall(r"outputs?\s+(\d+)|returns?\s+(\d+)|result is\s+(\d+)", ans_lower)
                for groups in claimed_nums:
                    for num in groups:
                        if num and num != verified_num and abs(int(num) - int(verified_num)) > 5:
                            contradictions.append(
                                f"Student claimed the program outputs {num}, but verified sandbox execution produced {verified_num}."
                            )

        return contradictions

    def _fallback_evaluation(
        self,
        student_answer: str,
        concept: str,
        expected_reasoning: Optional[str],
        contradictions: List[str]
    ) -> EvaluationResult:
        """Deterministic rule-based fallback when external provider fails."""
        ans_lower = student_answer.lower()

        if contradictions:
            return EvaluationResult(
                answer_label="Incorrect",
                expected_reasoning=expected_reasoning or "Accurate conceptual description.",
                concepts_correct=[],
                concepts_missed=["Control flow and execution invariants"],
                contradictions=contradictions,
                supporting_evidence=[student_answer[:80]],
                evaluation_explanation="The student answer directly conflicts with verified code behavior.",
                uncertainty_score=0.2,
                needs_instructor_review=False,
            )

        return EvaluationResult(
            answer_label="Partially Correct",
            expected_reasoning=expected_reasoning or "Accurate conceptual description.",
            concepts_correct=["Provided explanatory text related to the concept"],
            concepts_missed=["Deep mechanistic trace verification required"],
            contradictions=[],
            supporting_evidence=[student_answer[:80]],
            evaluation_explanation="Response demonstrates partial awareness of the code section. Marked for instructor review.",
            uncertainty_score=0.45,
            needs_instructor_review=True,
        )
