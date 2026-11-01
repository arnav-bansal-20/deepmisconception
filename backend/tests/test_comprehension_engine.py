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
    assert rep["overall_label"] in ["Understood", "Partial Understanding", "Possible Misconception"]
    assert "evidence_summary" in rep
    assert "multimodal_scores" in rep

    # Retrieve report via API
    get_res = client.get(f"/api/reports/session/{s_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == rep["id"]
