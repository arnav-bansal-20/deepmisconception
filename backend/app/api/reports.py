"""Comprehension report generation and retrieval routes."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.answer_assessment import AnswerAssessment
from app.models.comprehension_report import ComprehensionReport
from app.schemas.report import ComprehensionReportResponse, ReportGenerateRequest
from app.services.comprehension_engine import ComprehensionEngine

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/generate", response_model=ComprehensionReportResponse, status_code=status.HTTP_201_CREATED)
def generate_session_report(payload: ReportGenerateRequest, db: Session = Depends(get_db)):
    """Synthesize a multimodal comprehension report fusing code, viva, and telemetry."""
    session = db.query(CodingSession).filter(CodingSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    try:
        report = ComprehensionEngine.synthesize_report(payload.session_id, db)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to synthesize comprehension report: {str(e)}")


@router.get("/session/{session_id}", response_model=ComprehensionReportResponse)
def get_session_report(
    session_id: str,
    auto_generate: bool = True,
    db: Session = Depends(get_db),
):
    """Retrieve the synthesized comprehension report for a session (auto-synthesizes if ready)."""
    session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    report = (
        db.query(ComprehensionReport)
        .filter(ComprehensionReport.session_id == session_id)
        .order_by(ComprehensionReport.created_at.desc())
        .first()
    )

    ans_count = db.query(AnswerAssessment).filter(AnswerAssessment.session_id == session_id).count()
    rep_answered = (report.rubric_summary or {}).get("answered_questions", 0) if report else 0

    if auto_generate:
        # Check if session needs initial synthesis OR re-synthesis because new viva answers were submitted
        sub_count = db.query(CodeSubmission).filter(CodeSubmission.session_id == session_id).count()
        needs_synth = (not report and sub_count > 0) or (
            report is not None and (
                ans_count > rep_answered or 
                (report.overall_label in ("Insufficient Evidence", "Pending / Insufficient Evidence") and ans_count > 0)
            )
        )
        if needs_synth:
            try:
                report = ComprehensionEngine.synthesize_report(session_id, db)
            except Exception:
                pass

    if not report:
        raise HTTPException(
            status_code=404,
            detail="No comprehension report generated yet for this session. Submit code and answer viva questions to generate report."
        )

    return report
