"""ProcessEvent SQLAlchemy model for telemetry."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class ProcessEvent(Base):
    """Represents a fine-grained telemetry event emitted from the student editor."""
    __tablename__ = "process_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("coding_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)  # edit, paste, run, submit, idle
    cursor_line = Column(Integer, nullable=True)
    cursor_column = Column(Integer, nullable=True)
    delta_chars = Column(Integer, default=0, nullable=False)
    paste_length = Column(Integer, default=0, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    event_metadata = Column(JSON, nullable=True)

    # Relationships
    session = relationship("CodingSession", back_populates="process_events")
