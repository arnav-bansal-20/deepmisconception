"""AnswerAssessment SQLAlchemy model."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class AnswerAssessment(Base):
    """Represents the dynamic evaluation of a student's answer to a viva question."""
    __tablename__ = "answer_assessments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    question_id = Column(String(36), ForeignKey("viva_questions.id", ondelete="CASCADE"), nullable=False, unique=True)
    session_id = Column(String(36), ForeignKey("coding_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    student_answer = Column(Text, nullable=False)
    
    # Label: Correct, Partially Correct, Incorrect, Insufficient Evidence
    answer_label = Column(String(32), nullable=False)
    
    expected_reasoning = Column(Text, nullable=True)
    concepts_correct = Column(JSON, nullable=True)   # list of concepts correctly articulated
    concepts_missed = Column(JSON, nullable=True)    # list of concepts omitted or confused
    contradictions = Column(JSON, nullable=True)     # list of direct conflicts with code/execution
    supporting_evidence = Column(JSON, nullable=True)# quotes/evidence from student explanation & code
    evaluation_explanation = Column(Text, nullable=False)
    uncertainty_score = Column(Float, default=0.0, nullable=False)
    needs_instructor_review = Column(Boolean, default=False, nullable=False)
    instructor_status = Column(String(32), default="pending", nullable=False)  # pending, confirmed, overridden
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    question = relationship("VivaQuestion", back_populates="assessment")
    session = relationship("CodingSession", back_populates="answer_assessments")
