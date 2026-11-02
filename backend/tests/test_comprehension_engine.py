"""Unit tests for Multimodal Comprehension Classification and Report Synthesis."""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.process_event import ProcessEvent
from app.services.comprehension_engine import ComprehensionEngine

client = TestClient(app)


def test_comprehension_engine_skilled_coder():
    """Verify skilled coder profile is diagnosed as 'Understood' with high confidence."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-SKILLED-01",
            student_name="Elena Rostova",
            title="Binary Search Invariant",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # 1. Submission with valid AST and clean execution
        sub = CodeSubmission(
            session_id=session.id,
            code="def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    return -1\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 3, "functions": [{"name": "binary_search"}]},
            execution_output="Target index found at position 3\n",
        )
        db.add(sub)

        # 2. Viva question & correct assessment
        q = VivaQuestion(
            session_id=session.id,
            submission_id=sub.id,
            question_text="Why is the loop condition low <= high instead of low < high?",
            concept="Loop Boundary Condition",
            difficulty="medium",
        )
        db.add(q)
        db.commit()
        db.refresh(q)

        assessment = AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="Using low <= high ensures that 1-element subarrays when low == high are checked rather than skipped.",
            answer_label="Correct",
            evaluation_explanation="Completely accurate justification of edge element inclusion.",
            concepts_correct=["boundary condition", "single element check"],
        )
        db.add(assessment)

        # 3. Organic process telemetry
        for i in range(15):
            db.add(ProcessEvent(
                session_id=session.id,
                event_type="edit",
                delta_chars=2,
                paste_length=0,
            ))
        db.commit()

        # Run engine synthesis
        report = ComprehensionEngine.synthesize_report(session.id, db)
        assert report.overall_label == "Understood"
        assert report.confidence_score >= 0.75
        assert report.needs_instructor_review is False
        assert "ast_structural_score" in report.multimodal_scores
        assert report.multimodal_scores["ast_structural_score"] >= 0.8
        assert report.multimodal_scores["viva_reasoning_score"] == 1.0
    finally:
        db.close()


def test_comprehension_engine_misconception_diagnosis():
    """Verify flawed reasoning triggers 'Possible Misconception' verdict."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-MISCON-01",
            student_name="Jordan Bell",
            title="Recursive Termination",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def fact(n):\n    return n * fact(n-1)\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 1, "is_recursive": True},
            execution_output="",
            error_output="RecursionError: maximum recursion depth exceeded",
        )
        db.add(sub)

        # 2 incorrect assessments with diagnosed misconceptions
        q1 = VivaQuestion(
            session_id=session.id,
            question_text="What happens when n reaches 0 in your recursion?",
            concept="Recursion Base Case",
            difficulty="medium",
        )
        db.add(q1)
        db.commit()

        a1 = AnswerAssessment(
            question_id=q1.id,
            session_id=session.id,
            student_answer="Python functions stop automatically when numbers reach 0.",
            answer_label="Incorrect",
            contradictions=["Missing base-case termination condition", "Belief in automatic zero termination"],
            evaluation_explanation="Incorrect assumption that Python terminates recursion automatically.",
        )
        db.add(a1)

        q2 = VivaQuestion(
            session_id=session.id,
            question_text="Why did the code crash with RecursionError?",
            concept="Stack Overflow",
            difficulty="medium",
        )
        db.add(q2)
        db.commit()

        a2 = AnswerAssessment(
            question_id=q2.id,
            session_id=session.id,
            student_answer="The computer ran out of memory because the number was too big.",
            answer_label="Incorrect",
            contradictions=["Confusing call-stack recursion depth limit with RAM capacity"],
            evaluation_explanation="Confusing stack frames with physical memory limit.",
        )
        db.add(a2)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        assert report.overall_label == "Possible Misconception"
        assert len(report.detected_misconceptions) >= 2
        assert any("base-case" in m.lower() for m in report.detected_misconceptions)
    finally:
        db.close()


def test_conflicting_signals_flags_instructor_review():
    """Verify high code complexity coupled with low viva explanation flags review."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-PASTE-01",
            student_name="Chris Doe",
            title="Complex Tree Traversal",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Complex AST
        sub = CodeSubmission(
            session_id=session.id,
            code="def complex_tree_traversal():\n    pass\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 7, "functions": [{"name": "complex_tree_traversal"}]},
            execution_output="Tree traversed.\n",
        )
        db.add(sub)

        # High paste telemetry
        db.add(ProcessEvent(
            session_id=session.id,
            event_type="paste",
            paste_length=450,
            delta_chars=450,
            timestamp=datetime.now(timezone.utc),
        ))

        # But evasive viva answer
        q = VivaQuestion(
            session_id=session.id,
            question_text="Explain the traversal invariant.",
            concept="Tree Invariant",
            difficulty="hard",
        )
        db.add(q)
        db.commit()

        a = AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="I don't really know, the code works though.",
            answer_label="Insufficient Evidence",
            evaluation_explanation="Candidate did not articulate any understanding.",
        )
        db.add(a)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # Should flag review because of copypaste disparity
        assert report.needs_instructor_review is True
        assert len(report.evidence_breakdown["review_flags"]) > 0
    finally:
        db.close()


def test_api_report_generate_and_fetch():
    """Verify POST /api/reports/generate and GET /api/reports/session/{id} endpoints."""
    # Create session
    sess_res = client.post("/api/sessions", json={"student_id": "STU-API-REP", "student_name": "API Report Student", "title": "API Test"})
    assert sess_res.status_code == 201
    s_id = sess_res.json()["id"]

    # Generate report via API
    gen_res = client.post("/api/reports/generate", json={"session_id": s_id})
    assert gen_res.status_code == 201
    rep = gen_res.json()
    assert rep["session_id"] == s_id
    assert rep["overall_label"] in ["Understood", "Partial Understanding", "Possible Misconception", "Pending / Insufficient Evidence"]
    assert rep["overall_label"] == "Pending / Insufficient Evidence"
    assert rep["needs_instructor_review"] is True
    assert "evidence_summary" in rep
    assert "multimodal_scores" in rep

    # Retrieve report via API
    get_res = client.get(f"/api/reports/session/{s_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == rep["id"]


def test_student_pasting_code_with_correct_explanation_not_penalized():
    """Requirement: A student who pastes code but explains it correctly is NOT penalized automatically."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-PASTE-ACCURATE",
            student_name="Alex Chen",
            title="Fast Modular Exponentiation",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Student pasted large block of valid code
        sub = CodeSubmission(
            session_id=session.id,
            code="def power(base, exp):\n    res = 1\n    while exp > 0:\n        if exp % 2 == 1: res *= base\n        base *= base\n        exp //= 2\n    return res\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 3, "functions": [{"name": "power"}]},
            execution_output="power(2, 10) = 1024\n",
        )
        db.add(sub)

        # High paste telemetry (paste ratio > 80%)
        db.add(ProcessEvent(
            session_id=session.id,
            event_type="paste",
            paste_length=220,
            delta_chars=220,
            timestamp=datetime.now(timezone.utc),
        ))
        db.commit()

        # But student provided perfect viva explanation demonstrating understanding
        q = VivaQuestion(
            session_id=session.id,
            submission_id=sub.id,
            question_text="Why does base square each iteration while exp divides by 2?",
            concept="Binary Exponentiation",
            difficulty="medium",
        )
        db.add(q)
        db.commit()

        a = AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="Squaring the base reduces the exponent by half each step, achieving O(log n) time complexity. When exp is odd, the extra factor is accumulated in res.",
            answer_label="Correct",
            evaluation_explanation="Flawless explanation of binary exponentiation mechanics.",
            concepts_correct=["O(log n) complexity", "bit decomposition", "accumulator preservation"],
        )
        db.add(a)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # Even with high paste ratio, student is classified as 'Understood' because they explained it correctly!
        assert report.overall_label == "Understood"
        assert report.confidence_score >= 0.72
        assert report.multimodal_scores["process_authenticity_score"] >= 0.80
    finally:
        db.close()


def test_student_typing_manually_without_understanding_not_classified_as_understood():
    """Requirement: A student who types manually but cannot explain it is NOT classified as understood."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-TYPE-NOUNDERSTAND",
            student_name="Sam Miller",
            title="Manual Typist Without Understanding",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def check(x):\n    return x > 0\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 1},
            execution_output="True\n",
        )
        db.add(sub)

        # 30 incremental manual keystroke edits (paste ratio = 0)
        for i in range(30):
            db.add(ProcessEvent(
                session_id=session.id,
                event_type="edit",
                delta_chars=1,
                paste_length=0,
            ))
        db.commit()

        # But viva answer is completely evasive / insufficient
        q = VivaQuestion(
            session_id=session.id,
            question_text="Explain the condition x > 0.",
            concept="Comparison",
        )
        db.add(q)
        db.commit()

        a = AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="I just typed it because the instructions said so, I have no idea.",
            answer_label="Insufficient Evidence",
            evaluation_explanation="No conceptual understanding provided.",
        )
        db.add(a)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # MUST NOT be classified as 'Understood' - explicit pending review
        assert report.overall_label == "Pending / Insufficient Evidence"
        assert report.overall_label != "Understood"
        assert report.needs_instructor_review is True
    finally:
        db.close()


def test_single_incorrect_answer_does_not_trigger_misconception():
    """Requirement: An individual incorrect answer must NOT automatically determine a misconception."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-SINGLE-ERROR",
            student_name="Taylor Swift",
            title="Single Error Tolerant Test",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def sum_positive(nums):\n    total = 0\n    for n in nums:\n        if n > 0: total += n\n    return total\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 2},
            execution_output="15\n",
        )
        db.add(sub)

        # Question 1: Answered Correctly
        q1 = VivaQuestion(session_id=session.id, question_text="What does total += n do?", concept="Accumulator")
        db.add(q1)
        db.commit()
        db.add(AnswerAssessment(
            question_id=q1.id,
            session_id=session.id,
            student_answer="It adds positive numbers to the running sum total.",
            answer_label="Correct",
            evaluation_explanation="Accurate accumulator explanation.",
        ))

        # Question 2: Answered Incorrectly (single isolated error on edge case)
        q2 = VivaQuestion(session_id=session.id, question_text="What happens with empty list?", concept="Empty List Edge Case")
        db.add(q2)
        db.commit()
        db.add(AnswerAssessment(
            question_id=q2.id,
            session_id=session.id,
            student_answer="It throws an exception immediately.",
            answer_label="Incorrect",
            evaluation_explanation="Incorrect assumption about empty list traversal.",
        ))
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # One incorrect answer should yield 'Partial Understanding', NOT 'Possible Misconception'
        assert report.overall_label == "Partial Understanding"
    finally:
        db.close()


def test_missing_viva_answers_produce_uncertainty_review_state():
    """Requirement: Missing viva answers produce an explicit Pending / Insufficient Evidence status."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-NO-VIVA",
            student_name="Unanswered Student",
            title="Unanswered Viva Session",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def quick_code():\n    return 42\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 1},
            execution_output="42\n",
        )
        db.add(sub)
        db.commit()

        # Zero viva questions answered -> Explicit Pending / Insufficient Evidence
        report = ComprehensionEngine.synthesize_report(session.id, db)
        assert report.overall_label == "Pending / Insufficient Evidence"
        assert report.needs_instructor_review is True
        assert report.confidence_score <= 0.50
        assert any("Insufficient viva evidence" in flag for flag in report.evidence_breakdown["review_flags"])
    finally:
        db.close()


def test_empty_viva_answers_produce_pending_insufficient_evidence():
    """Requirement: Empty viva answers produce Pending / Insufficient Evidence."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-EMPTY-VIVA",
            student_name="Empty Answer Student",
            title="Empty Viva Session",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def compute(): return 10\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 1},
            execution_output="10\n",
        )
        db.add(sub)
        q = VivaQuestion(session_id=session.id, question_text="Explain return value.", concept="Return Value")
        db.add(q)
        db.commit()

        # Answer is empty whitespace
        db.add(AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="   ",
            answer_label="Insufficient Evidence",
            evaluation_explanation="Empty response submitted.",
        ))
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        assert report.overall_label == "Pending / Insufficient Evidence"
        assert report.needs_instructor_review is True
    finally:
        db.close()


def test_syntax_error_without_viva_does_not_trigger_misconception():
    """Requirement: Syntax errors alone must NOT classify a student as having a misconception."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-SYNTAX-ERR",
            student_name="Active Debugger",
            title="Syntax Error Session",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Broken syntax during active development
        sub = CodeSubmission(
            session_id=session.id,
            code="def broken(\n",
            is_valid=False,
            error_output="SyntaxError: unexpected EOF while parsing",
        )
        db.add(sub)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # Must be Pending / Insufficient Evidence, NEVER Possible Misconception
        assert report.overall_label == "Pending / Insufficient Evidence"
        assert report.overall_label != "Possible Misconception"
    finally:
        db.close()


def test_runtime_error_without_viva_does_not_trigger_misconception():
    """Requirement: Runtime crashes alone must NOT classify a student as having a misconception."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-RUNTIME-ERR",
            student_name="Runtime Crash Student",
            title="Runtime Error Session",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="x = 1 / 0\n",
            is_valid=True,
            error_output="ZeroDivisionError: division by zero",
        )
        db.add(sub)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # Must be Pending / Insufficient Evidence, NEVER Possible Misconception
        assert report.overall_label == "Pending / Insufficient Evidence"
        assert report.overall_label != "Possible Misconception"
    finally:
        db.close()


def test_paste_telemetry_does_not_penalize_comprehension_verdict():
    """Requirement: Pasting code does NOT reduce score or prevent 'Understood' when explained correctly."""
    db = SessionLocal()
    try:
        session = CodingSession(
            student_id="STU-PASTE-EXPLAIN",
            student_name="Pasting Master",
            title="Paste and Explain Session",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n",
            is_valid=True,
            ast_summary={"cyclomatic_complexity": 2, "functions": [{"name": "fib"}]},
            execution_output="55\n",
        )
        db.add(sub)

        # 95% paste telemetry (single massive paste event)
        db.add(ProcessEvent(
            session_id=session.id,
            event_type="paste",
            paste_length=150,
            delta_chars=150,
        ))
        db.add(ProcessEvent(
            session_id=session.id,
            event_type="run",
            delta_chars=0,
            paste_length=0,
        ))
        db.commit()

        # Viva correctly justified
        q = VivaQuestion(
            session_id=session.id,
            question_text="Explain base cases n <= 1.",
            concept="Recursion Base Case",
        )
        db.add(q)
        db.commit()
        db.add(AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="When n is 0 or 1, it returns n directly preventing infinite recursion stack overflow.",
            answer_label="Correct",
            evaluation_explanation="Flawless justification of base case termination.",
        ))
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)
        # Candidate receives 'Understood' despite heavy pasting
        assert report.overall_label == "Understood"
        assert report.multimodal_scores["process_rhythm_score"] >= 0.80
        assert report.needs_instructor_review is False
    finally:
        db.close()

