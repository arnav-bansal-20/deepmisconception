"""CodeSubmission SQLAlchemy model."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class CodeSubmission(Base):
    """Represents a snapshot of student code submitted for analysis."""
    __tablename__ = "code_submissions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("coding_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    code = Column(Text, nullable=False)
    ast_summary = Column(JSON, nullable=True)
    execution_output = Column(Text, nullable=True)
    error_output = Column(Text, nullable=True)
    is_valid = Column(Boolean, default=True, nullable=False)
    execution_time_ms = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    session = relationship("CodingSession", back_populates="submissions")
    viva_questions = relationship("VivaQuestion", back_populates="submission", cascade="all, delete-orphan")
