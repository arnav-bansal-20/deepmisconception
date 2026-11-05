"""Student and Faculty SQLAlchemy models enforcing unique academic identifiers."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey
from app.database import Base


class Student(Base):
    """Represents an enrolled student with a guaranteed unique student ID / roll number."""
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    classroom_id = Column(String(36), ForeignKey("classrooms.id", ondelete="SET NULL"), nullable=True, index=True)
    classroom_name = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class Faculty(Base):
    """Represents a faculty member/instructor with a guaranteed unique faculty ID."""
    __tablename__ = "faculty"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    faculty_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    department = Column(String(128), default="Computer Science & Engineering", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

