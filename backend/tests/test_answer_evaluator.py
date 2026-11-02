"""Unit tests for Code-Grounded Dynamic Student Answer Evaluator."""
import pytest
from app.services.answer_evaluator import AnswerEvaluator
from app.services.llm_provider import MockLLMProvider

FOCAL_CODE = "if i % 3 == 0:\n    continue\ntotal += i * i"
SUBMITTED_CODE = """
def calc(n):
    total = 0
    for i in range(1, n + 1):
        if i % 3 == 0:
            continue
        total += i * i
    return total
"""
EXECUTION_OUTPUT = "Computed total: 35"
QUESTION = "What is the exact condition under which an iteration skips the accumulator step, and why?"
EXPECTED = "The modulo operator % 3 evaluates remainder. When remainder is 0, continue skips adding to total."


@pytest.fixture
def evaluator():
    return AnswerEvaluator(provider=MockLLMProvider())


def test_evaluate_correct_answer(evaluator):
    student_ans = (
        "The condition i % 3 == 0 checks if the current index is divisible by 3 with remainder 0. "
        "When true, the continue statement skips the rest of the current loop body so total is not updated for that multiple."
    )
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={"modulo_check": "explains % 3"},
        submitted_code=SUBMITTED_CODE,
        execution_output=EXECUTION_OUTPUT
    )

    assert result.answer_label == "Correct"
    assert len(result.concepts_correct) > 0
    assert len(result.contradictions) == 0
    assert result.needs_instructor_review is False


def test_evaluate_partially_correct_answer(evaluator):
    student_ans = "The code skips the number when the if condition matches."
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={"modulo_check": "explains % 3"},
        submitted_code=SUBMITTED_CODE,
        execution_output=EXECUTION_OUTPUT
    )

    assert result.answer_label == "Partially Correct"
    assert len(result.concepts_missed) > 0


def test_evaluate_contradictory_misconception_answer(evaluator):
    # Student claims continue stops the loop completely
    student_ans = "When i % 3 == 0, the continue keyword terminates the loop completely and stops iterating."
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={"modulo_check": "explains % 3"},
        submitted_code=SUBMITTED_CODE,
        execution_output=EXECUTION_OUTPUT
    )

    assert result.answer_label == "Incorrect"
    assert len(result.contradictions) > 0
    assert any("stops" in c or "terminates" in c or "continue" in c for c in result.contradictions)


def test_evaluate_insufficient_evidence_evasive(evaluator):
    student_ans = "i dont know honestly"
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={},
        submitted_code=SUBMITTED_CODE,
        execution_output=EXECUTION_OUTPUT
    )

    assert result.answer_label == "Insufficient Evidence"
    assert result.needs_instructor_review is True
    assert result.uncertainty_score > 0.5


def test_evaluate_too_short(evaluator):
    student_ans = "skip"
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={},
        submitted_code=SUBMITTED_CODE,
        execution_output=EXECUTION_OUTPUT
    )

    assert result.answer_label == "Insufficient Evidence"
    assert result.needs_instructor_review is True


def test_evaluate_off_topic_answer(evaluator):
    """Verify off-topic responses are flagged as Insufficient Evidence requiring review."""
    student_ans = "The weather today is really sunny and I want to eat pizza with my friends."
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={},
        submitted_code=SUBMITTED_CODE,
        execution_output=EXECUTION_OUTPUT
    )

    assert result.answer_label == "Insufficient Evidence"
    assert result.needs_instructor_review is True
    assert result.uncertainty_score >= 0.85


def test_evaluate_contradicting_numerical_execution_trace(evaluator):
    """Verify answer claiming incorrect output contradicting sandbox execution is marked Incorrect."""
    student_ans = "The program outputs 80 because it sums everything up to 80."
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={},
        submitted_code=SUBMITTED_CODE,
        execution_output="Computed total: 35"
    )

    assert result.answer_label == "Incorrect"
    assert any("35" in c and "80" in c for c in result.contradictions)


def test_evaluate_conflicting_llm_expectation_and_verified_trace(evaluator):
    """Verify when LLM expected reasoning contradicts verified trace, physical reality overrides and review is flagged."""
    # LLM mistakenly expected 55, but verified execution is 35
    flawed_expected = "The loop calculates and outputs 55 for all elements."
    student_ans = "The program outputs 35 because multiples of 3 are filtered out by the continue statement."

    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=flawed_expected,
        rubric={},
        submitted_code=SUBMITTED_CODE,
        execution_output="Computed total: 35"
    )

    # Discrepancy is flagged, review is triggered
    assert result.needs_instructor_review is True
    assert any("Discrepancy" in ev or "claimed output" in ev for ev in result.supporting_evidence)


def test_evaluate_execution_unavailable_uncertainty(evaluator):
    """Verify missing execution trace produces explicit uncertainty."""
    student_ans = "The continue statement skips the rest of the loop block."
    result = evaluator.evaluate_answer(
        student_answer=student_ans,
        question_text=QUESTION,
        focal_code=FOCAL_CODE,
        concept="modulo-conditional-filtering",
        expected_reasoning=EXPECTED,
        rubric={},
        submitted_code=SUBMITTED_CODE,
        execution_output=None  # No sandbox execution available
    )

    assert result.uncertainty_score >= 0.40
