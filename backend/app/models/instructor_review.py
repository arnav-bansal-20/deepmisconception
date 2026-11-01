"""InstructorReview SQLAlchemy model."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class InstructorReview(Base):
    """Represents an instructor or researcher manual annotation and review."""
    __tablename__ = "instructor_reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("coding_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_name = Column(String(128), nullable=False)
    review_type = Column(String(32), default="overall", nullable=False)  # overall, answer, flagging
    target_id = Column(String(36), nullable=True)  # optional ID of specific answer being reviewed
    
    assigned_label = Column(String(32), nullable=False)  # Understood, Partial Understanding, Possible Misconception
    agree_with_ai = Column(Boolean, nullable=True)
    feedback_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    session = relationship("CodingSession", back_populates="instructor_reviews")
