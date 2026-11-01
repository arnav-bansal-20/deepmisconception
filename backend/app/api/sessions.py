"""Session management API routes."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.comprehension_report import ComprehensionReport
from app.schemas.session import SessionCreate, SessionUpdate, SessionSummary, SessionDetail

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionDetail, status_code=status.HTTP_201_CREATED)
def create_session(session_in: SessionCreate, db: Session = Depends(get_db)):
    """Initialize a new coding and evaluation session under teacher/classroom guidance."""
    session = CodingSession(
        student_id=session_in.student_id,
        student_name=session_in.student_name,
        classroom_id=session_in.classroom_id,
        classroom_name=session_in.classroom_name,
        teacher_id=session_in.teacher_id,
        teacher_name=session_in.teacher_name,
        title=session_in.title,
        language=session_in.language,
        status="active"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=List[SessionSummary])
def list_sessions(
    classroom_id: Optional[str] = Query(None, description="Filter sessions by classroom ID"),
    teacher_id: Optional[str] = Query(None, description="Filter sessions by guiding teacher ID"),
    student_id: Optional[str] = Query(None, description="Filter sessions by student ID"),
    db: Session = Depends(get_db),
):
    """List all student sessions with summary metadata, optionally filtered by classroom/teacher/student."""
    query = db.query(CodingSession)
    if classroom_id:
        query = query.filter(CodingSession.classroom_id == classroom_id)
    if teacher_id:
        query = query.filter(CodingSession.teacher_id == teacher_id)
    if student_id:
        query = query.filter(CodingSession.student_id == student_id)

    sessions = query.order_by(CodingSession.created_at.desc()).all()
    results = []
    
    for s in sessions:
        sub_count = db.query(func.count(CodeSubmission.id)).filter(CodeSubmission.session_id == s.id).scalar() or 0
        q_count = db.query(func.count(VivaQuestion.id)).filter(VivaQuestion.session_id == s.id).scalar() or 0
        latest_report = db.query(ComprehensionReport).filter(ComprehensionReport.session_id == s.id).order_by(ComprehensionReport.created_at.desc()).first()
        
        results.append(SessionSummary(
            id=s.id,
            student_id=s.student_id,
            student_name=s.student_name,
            classroom_id=s.classroom_id,
            classroom_name=s.classroom_name,
            teacher_id=s.teacher_id,
            teacher_name=s.teacher_name,
            title=s.title,
            language=s.language,
            status=s.status,
            created_at=s.created_at,
            updated_at=s.updated_at,
            submission_count=sub_count,
            question_count=q_count,
            has_report=latest_report is not None,
            latest_verdict=latest_report.overall_label if latest_report else None
        ))
    return results


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get detailed information for a specific session."""
    session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")
    return session


@router.patch("/{session_id}", response_model=SessionDetail)
def update_session(session_id: str, session_in: SessionUpdate, db: Session = Depends(get_db)):
    """Update title or status of a session."""
    session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")
    
    if session_in.title is not None:
        session.title = session_in.title
    if session_in.status is not None:
        session.status = session_in.status
    
    db.commit()
    db.refresh(session)
    return session
