"""Classroom / Cohort SQLAlchemy model for grouping students under a teacher."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.orm import relationship
from app.database import Base


class Classroom(Base):
    """Represents an academic cohort/section guided by a specific instructor."""
    __tablename__ = "classrooms"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256), nullable=False)
    code = Column(String(64), unique=True, index=True, nullable=False)  # e.g., "CS101-ALPHA"
    teacher_id = Column(String(64), nullable=False, index=True)
    teacher_name = Column(String(128), nullable=False)
    department = Column(String(128), default="Computer Science & Engineering", nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
