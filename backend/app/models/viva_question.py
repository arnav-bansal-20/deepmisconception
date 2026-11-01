"""VivaQuestion SQLAlchemy model."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class VivaQuestion(Base):
    """Represents a dynamically generated viva question grounded in the student's code."""
    __tablename__ = "viva_questions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("coding_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    submission_id = Column(String(36), ForeignKey("code_submissions.id", ondelete="CASCADE"), nullable=True)
    question_text = Column(Text, nullable=False)
    focal_code = Column(Text, nullable=True)
    concept = Column(String(128), nullable=False)  # e.g., 'loop-termination', 'variable-scope'
    difficulty = Column(String(32), default="medium", nullable=False)  # easy, medium, hard
    expected_reasoning = Column(Text, nullable=True)
    rubric = Column(JSON, nullable=True)
    misconception_indicators = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    session = relationship("CodingSession", back_populates="viva_questions")
    submission = relationship("CodeSubmission", back_populates="viva_questions")
    assessment = relationship("AnswerAssessment", back_populates="question", uselist=False, cascade="all, delete-orphan")
