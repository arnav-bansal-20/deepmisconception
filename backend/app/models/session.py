"""CodingSession SQLAlchemy model."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class CodingSession(Base):
    """Represents an interactive student coding and evaluation session."""
    __tablename__ = "coding_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(64), nullable=False, index=True)
    student_name = Column(String(128), nullable=False)
    classroom_id = Column(String(36), ForeignKey("classrooms.id", ondelete="SET NULL"), nullable=True, index=True)
    classroom_name = Column(String(256), nullable=True)
    teacher_id = Column(String(64), nullable=True, index=True)
    teacher_name = Column(String(128), nullable=True)
    title = Column(String(256), nullable=False, default="Untitled Coding Task")
    language = Column(String(32), nullable=False, default="python")
    status = Column(String(32), nullable=False, default="active")  # active, completed, reviewed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    submissions = relationship("CodeSubmission", back_populates="session", cascade="all, delete-orphan", order_by="CodeSubmission.created_at.desc()")
    process_events = relationship("ProcessEvent", back_populates="session", cascade="all, delete-orphan", order_by="ProcessEvent.timestamp.asc()")
    viva_questions = relationship("VivaQuestion", back_populates="session", cascade="all, delete-orphan", order_by="VivaQuestion.created_at.asc()")
    answer_assessments = relationship("AnswerAssessment", back_populates="session", cascade="all, delete-orphan")
    comprehension_reports = relationship("ComprehensionReport", back_populates="session", cascade="all, delete-orphan", order_by="ComprehensionReport.created_at.desc()")
    instructor_reviews = relationship("InstructorReview", back_populates="session", cascade="all, delete-orphan")
