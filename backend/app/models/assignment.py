"""Assignment SQLAlchemy model for teacher-created coding problems."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from app.database import Base


class Assignment(Base):
    """Represents a coding problem created by an instructor for students to solve."""
    __tablename__ = "assignments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    classroom_id = Column(String(36), ForeignKey("classrooms.id", ondelete="SET NULL"), nullable=True, index=True)
    classroom_name = Column(String(256), nullable=True)
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=False)
    starter_code = Column(Text, nullable=False)
    difficulty = Column(String(32), default="medium", nullable=False)  # easy, medium, hard
    concept_tags = Column(JSON, nullable=True)  # e.g., ["loops", "modulo", "accumulator"]
    created_by = Column(String(128), default="Instructor", nullable=False)
    teacher_id = Column(String(64), nullable=True, index=True)
    sample_input = Column(Text, nullable=True)
    sample_output = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
