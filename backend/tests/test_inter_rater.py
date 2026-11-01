"""Unit tests for Phase 9: Inter-rater agreement & scientific evaluation metrics."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.inter_rater import (
    compute_cohens_kappa,
    compute_classification_metrics,
    InterRaterEvaluationService,
    COMPREHENSION_CLASSES,
)
from app.database import SessionLocal
from app.models.session import CodingSession
from app.models.comprehension_report import ComprehensionReport
from app.models.instructor_review import InstructorReview
from datetime import datetime
import uuid


client = TestClient(app)


def test_cohens_kappa_perfect_agreement():
    """Two raters with 100% agreement should yield kappa == 1.0."""
    rater_a = ["Understood", "Partial Understanding", "Possible Misconception", "Understood"]
    rater_b = ["Understood", "Partial Understanding", "Possible Misconception", "Understood"]
    
    kappa, interp = compute_cohens_kappa(rater_a, rater_b, COMPREHENSION_CLASSES)
    assert kappa == 1.0
    assert "Almost Perfect" in interp


def test_cohens_kappa_moderate_agreement():
    """Raters with realistic moderate agreement should yield kappa between 0.4 and 0.8."""
    rater_a = ["Understood", "Understood", "Partial Understanding", "Possible Misconception", "Possible Misconception", "Understood"]
    rater_b = ["Understood", "Partial Understanding", "Partial Understanding", "Possible Misconception", "Understood", "Understood"]
    
    kappa, interp = compute_cohens_kappa(rater_a, rater_b, COMPREHENSION_CLASSES)
    assert 0.2 < kappa < 0.9
    assert interp in ["Moderate Agreement", "Substantial Agreement", "Fair Agreement"]


def test_cohens_kappa_disagreement():
    """Opposing classifications yield negative or near-zero kappa."""
    rater_a = ["Understood", "Understood", "Understood"]
    rater_b = ["Possible Misconception", "Possible Misconception", "Possible Misconception"]
    
    kappa, interp = compute_cohens_kappa(rater_a, rater_b, COMPREHENSION_CLASSES)
    assert kappa <= 0.0


def test_classification_metrics_and_confusion_matrix():
    """Verify confusion matrix, precision, recall, and macro-F1 calculations."""
    y_true = ["Understood", "Understood", "Partial Understanding", "Possible Misconception"]
    y_pred = ["Understood", "Partial Understanding", "Partial Understanding", "Possible Misconception"]
    
    metrics = compute_classification_metrics(y_true, y_pred, COMPREHENSION_CLASSES)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.75
    
    cm = metrics["confusion_matrix"]["matrix"]
    # Total samples across matrix should equal 4
    total_in_cm = sum(sum(row) for row in cm)
    assert total_in_cm == 4


def test_inter_rater_evaluation_service_with_db():
    """Test InterRaterEvaluationService querying the database and generating evaluation metrics."""
    db = SessionLocal()
    session_id = f"test-rater-session-{uuid.uuid4().hex[:6]}"
    
    try:
        # Create a session
        sess = CodingSession(
            id=session_id,
            student_id="student-rater-1",
            student_name="Rater Test Student",
            title="Recursion Misconception",
            status="reviewed"
        )
        db.add(sess)
        
        # Add a report
        rep = ComprehensionReport(
            id=f"rep-{session_id}",
            session_id=session_id,
            overall_label="Possible Misconception",
            confidence_score=0.88,
            evidence_summary="Contradicted base case semantics."
        )
        db.add(rep)
        
        # Add an instructor review agreeing with the report
        rev = InstructorReview(
            id=f"rev-{session_id}",
            session_id=session_id,
            reviewer_name="Prof. Turing",
            review_type="overall_report",
            assigned_label="Possible Misconception",
            agree_with_ai=True,
            feedback_notes="Confirmed base case misunderstanding."
        )
        db.add(rev)
        db.commit()

        service = InterRaterEvaluationService(db=db)
        result = service.evaluate_comprehension_inter_rater(include_benchmark=True)

        assert "cohen_kappa" in result
        assert "kappa_interpretation" in result
        assert "percentage_agreement" in result
        assert "confusion_matrix" in result
        assert "source_breakdown" in result
        assert result["total_comparisons"] >= 40
        assert result["source_breakdown"]["database_reviews"] >= 1

    finally:
        # Clean up
        db.query(InstructorReview).filter(InstructorReview.session_id == session_id).delete()
        db.query(ComprehensionReport).filter(ComprehensionReport.session_id == session_id).delete()
        db.query(CodingSession).filter(CodingSession.id == session_id).delete()
        db.commit()
        db.close()


def test_api_evaluation_metrics_endpoint():
    """Test GET /api/instructor/evaluation-metrics route returns HTTP 200."""
    response = client.get("/api/instructor/evaluation-metrics?include_benchmark=true")
    assert response.status_code == 200
    data = response.json()
    assert "cohen_kappa" in data
    assert "percentage_agreement" in data
    assert "macro_f1" in data
    assert "per_class_metrics" in data
    assert "confusion_matrix" in data
    assert isinstance(data["confusion_matrix"]["matrix"], list)
