"""Session management API routes."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.comprehension_report import ComprehensionReport
from app.models.user import Student, Faculty
from app.schemas.session import SessionCreate, SessionUpdate, SessionSummary, SessionDetail

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionDetail, status_code=status.HTTP_201_CREATED)
def create_session(session_in: SessionCreate, db: Session = Depends(get_db)):
    """Initialize a new coding and evaluation session under teacher/classroom guidance."""
    # Ensure student is recorded in unique students registry if not already present
    if session_in.student_id:
        clean_sid = session_in.student_id.strip()
        existing_s = db.query(Student).filter(Student.student_id == clean_sid).first()
        if not existing_s:
            try:
                db.add(Student(
                    student_id=clean_sid,
                    name=session_in.student_name.strip(),
                    classroom_id=session_in.classroom_id,
                    classroom_name=session_in.classroom_name,
                ))
                db.commit()
            except Exception:
                db.rollback()

    # Ensure faculty is recorded in unique faculty registry if present
    if session_in.teacher_id:
        clean_fid = session_in.teacher_id.strip()
        existing_f = db.query(Faculty).filter(Faculty.faculty_id == clean_fid).first()
        if not existing_f:
            try:
                db.add(Faculty(
                    faculty_id=clean_fid,
                    name=session_in.teacher_name.strip() if session_in.teacher_name else "Faculty Instructor",
                    department="Computer Science & Engineering",
                ))
                db.commit()
            except Exception:
                db.rollback()

    # Prevent duplicate problem solving: check if student has already completed this problem
    if session_in.student_id and session_in.title:
        clean_sid = session_in.student_id.strip()
        clean_title = session_in.title.strip()
        existing_completed = (
            db.query(CodingSession)
            .filter(
                CodingSession.student_id == clean_sid,
                func.lower(CodingSession.title) == clean_title.lower(),
                CodingSession.status.in_(["completed", "reviewed"]),
            )
            .first()
        )
        if existing_completed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"You have already completed problem '{session_in.title}'. Each coding problem can only be solved once.",
            )

        # Return existing active session if one is already in progress
        existing_active = (
            db.query(CodingSession)
            .filter(
                CodingSession.student_id == clean_sid,
                func.lower(CodingSession.title) == clean_title.lower(),
                CodingSession.status == "active",
            )
            .first()
        )
        if existing_active:
            return existing_active

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
        fac = db.query(Faculty).filter(Faculty.faculty_id == teacher_id).first()
        if fac:
            query = query.filter(or_(CodingSession.teacher_id == teacher_id, CodingSession.teacher_name.ilike(fac.name)))
        else:
            query = query.filter(CodingSession.teacher_id == teacher_id)
    if student_id:
        query = query.filter(CodingSession.student_id == student_id)

    sessions = query.order_by(CodingSession.created_at.desc()).all()
    results = []
    
    for s in sessions:
        sub_count = db.query(func.count(CodeSubmission.id)).filter(CodeSubmission.session_id == s.id).scalar() or 0
        q_count = db.query(func.count(VivaQuestion.id)).filter(VivaQuestion.session_id == s.id).scalar() or 0
        ans_count = db.query(func.count(AnswerAssessment.id)).filter(AnswerAssessment.session_id == s.id).scalar() or 0
        latest_report = db.query(ComprehensionReport).filter(ComprehensionReport.session_id == s.id).order_by(ComprehensionReport.created_at.desc()).first()
        
        # Self-healing: if student completed problem and answered viva (or report exists), mark status as completed
        current_status = s.status
        if current_status == "active" and (latest_report is not None or (q_count > 0 and ans_count >= q_count)):
            s.status = "completed"
            current_status = "completed"
            db.add(s)
            db.commit()

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
            status=current_status,
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

    # Mark completed if student answered viva or report exists
    if session.status == "active":
        q_count = db.query(func.count(VivaQuestion.id)).filter(VivaQuestion.session_id == session.id).scalar() or 0
        ans_count = db.query(func.count(AnswerAssessment.id)).filter(AnswerAssessment.session_id == session.id).scalar() or 0
        has_report = db.query(ComprehensionReport).filter(ComprehensionReport.session_id == session.id).first() is not None
        if has_report or (q_count > 0 and ans_count >= q_count):
            session.status = "completed"
            db.add(session)
            db.commit()
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
