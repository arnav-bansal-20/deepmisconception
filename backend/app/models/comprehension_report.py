"""ComprehensionReport SQLAlchemy model."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from app.database import Base


class ComprehensionReport(Base):
    """Represents the synthesized student-level comprehension classification and evidence."""
    __tablename__ = "comprehension_reports"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("coding_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Overall label: Understood, Partial Understanding, Possible Misconception
    overall_label = Column(String(32), nullable=False)
    confidence_score = Column(Float, nullable=False, default=0.0)
    
    # Synthesized evidence
    evidence_summary = Column(Text, nullable=False)
    evidence_breakdown = Column(JSON, nullable=True)     # code, viva, process contributions
    detected_misconceptions = Column(JSON, nullable=True)# list of specific diagnosed misconceptions
    process_features = Column(JSON, nullable=True)       # paste ratio, burst count, etc.
    rubric_summary = Column(JSON, nullable=True)         # question-by-question metrics
    multimodal_scores = Column(JSON, nullable=True)      # ast, exec, viva, process breakdown
    recommended_interventions = Column(JSON, nullable=True) # pedagogical suggestions
    needs_instructor_review = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    session = relationship("CodingSession", back_populates="comprehension_reports")
