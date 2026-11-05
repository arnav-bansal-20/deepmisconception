"""Instructor review and annotation routes."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.comprehension_report import ComprehensionReport
from app.models.instructor_review import InstructorReview
from app.models.user import Faculty
from app.schemas.instructor import InstructorReviewCreate, InstructorReviewResponse

router = APIRouter(prefix="/instructor", tags=["Instructor"])


@router.get("/overview")
def get_instructor_overview(
    classroom_id: Optional[str] = Query(None, description="Filter overview by classroom ID"),
    teacher_id: Optional[str] = Query(None, description="Filter overview by guiding teacher ID"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get aggregate statistics across sessions for the instructor dashboard, optionally filtered."""
    session_query = db.query(CodingSession.id)
    if classroom_id:
        session_query = session_query.filter(CodingSession.classroom_id == classroom_id)
    if teacher_id:
        fac = db.query(Faculty).filter(Faculty.faculty_id == teacher_id).first()
        if fac:
            session_query = session_query.filter(or_(CodingSession.teacher_id == teacher_id, CodingSession.teacher_name.ilike(fac.name)))
        else:
            session_query = session_query.filter(CodingSession.teacher_id == teacher_id)

    matched_session_ids = [s_id for (s_id,) in session_query.all()]

    if not matched_session_ids:
        # If no matching sessions, return zeroed counters
        return {
            "total_sessions": 0,
            "total_submissions": 0,
            "total_questions": 0,
            "total_evaluations": 0,
            "flagged_for_review": 0,
            "comprehension_breakdown": {
                "Understood": 0,
                "Partial Understanding": 0,
                "Possible Misconception": 0,
            },
        }

    total_sessions = len(matched_session_ids)
    total_submissions = (
        db.query(func.count(CodeSubmission.id))
        .filter(CodeSubmission.session_id.in_(matched_session_ids))
        .scalar()
        or 0
    )
    total_questions = (
        db.query(func.count(VivaQuestion.id))
        .filter(VivaQuestion.session_id.in_(matched_session_ids))
        .scalar()
        or 0
    )
    total_evaluations = (
        db.query(func.count(AnswerAssessment.id))
        .filter(AnswerAssessment.session_id.in_(matched_session_ids))
        .scalar()
        or 0
    )

    # Label breakdown
    reports = (
        db.query(ComprehensionReport.overall_label)
        .filter(ComprehensionReport.session_id.in_(matched_session_ids))
        .all()
    )
    label_counts = {
        "Understood": 0,
        "Partial Understanding": 0,
        "Possible Misconception": 0,
    }
    for (lbl,) in reports:
        if lbl in label_counts:
            label_counts[lbl] += 1

    # Needs review count
    flagged_reviews = (
        db.query(func.count(AnswerAssessment.id))
        .filter(
            AnswerAssessment.session_id.in_(matched_session_ids),
            AnswerAssessment.needs_instructor_review == True,
        )
        .scalar()
        or 0
    )

    return {
        "total_sessions": total_sessions,
        "total_submissions": total_submissions,
        "total_questions": total_questions,
        "total_evaluations": total_evaluations,
        "flagged_for_review": flagged_reviews,
        "comprehension_breakdown": label_counts,
    }


@router.post("/reviews", response_model=InstructorReviewResponse, status_code=status.HTTP_201_CREATED)
def submit_instructor_review(review_in: InstructorReviewCreate, db: Session = Depends(get_db)):
    """Save an instructor review/annotation for a session or answer."""
    session = db.query(CodingSession).filter(CodingSession.id == review_in.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    review = InstructorReview(
        session_id=review_in.session_id,
        reviewer_name=review_in.reviewer_name,
        review_type=review_in.review_type,
        target_id=review_in.target_id,
        assigned_label=review_in.assigned_label,
        agree_with_ai=review_in.agree_with_ai,
        feedback_notes=review_in.feedback_notes
    )
    db.add(review)
    
    # Mark session as reviewed
    session.status = "reviewed"
    
    # If reviewing an answer specifically, update the assessment instructor_status
    if review_in.target_id:
        assessment = db.query(AnswerAssessment).filter(AnswerAssessment.id == review_in.target_id).first()
        if assessment:
            assessment.instructor_status = "confirmed" if review_in.agree_with_ai else "overridden"
            assessment.needs_instructor_review = False

    # Update existing ComprehensionReport with instructor review outcome
    report = db.query(ComprehensionReport).filter(ComprehensionReport.session_id == review_in.session_id).first()
    if report:
        if review_in.assigned_label and review_in.assigned_label not in ("agree", ""):
            report.overall_label = review_in.assigned_label
        report.needs_instructor_review = False
        eb = dict(report.evidence_breakdown or {})
        eb["instructor_review"] = {
            "has_review": True,
            "comments": review_in.feedback_notes or "Instructor verified and signed off.",
            "reviewed_by": review_in.reviewer_name or (session.teacher_name or "Faculty Advisor"),
            "review_date": datetime.now(timezone.utc).strftime("%d %b %Y"),
            "final_decision": f"Instructor Review: {review_in.assigned_label}",
            "agree_with_ai": review_in.agree_with_ai,
        }
        report.evidence_breakdown = eb
        db.add(report)

    db.commit()
    db.refresh(review)
    return review


@router.get("/reviews/session/{session_id}", response_model=List[InstructorReviewResponse])
def get_session_reviews(session_id: str, db: Session = Depends(get_db)):
    """Get all instructor reviews for a session."""
    return (
        db.query(InstructorReview)
        .filter(InstructorReview.session_id == session_id)
        .order_by(InstructorReview.created_at.desc())
        .all()
    )


@router.get("/evaluation-metrics")
def get_evaluation_metrics(
    classroom_id: Optional[str] = Query(None, description="Filter evaluations by classroom ID"),
    teacher_id: Optional[str] = Query(None, description="Filter evaluations by teacher ID"),
    include_benchmark: bool = Query(True, description="Supplement with synthetic benchmark dataset"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Compute Cohen's Kappa, confusion matrix, and accuracy metrics between AI and instructors."""
    from app.services.inter_rater import InterRaterEvaluationService
    service = InterRaterEvaluationService(db=db)
    return service.evaluate_comprehension_inter_rater(
        classroom_id=classroom_id,
        teacher_id=teacher_id,
        include_benchmark=include_benchmark,
    )

