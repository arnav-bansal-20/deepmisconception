"""Pydantic schemas for Comprehension Reports."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ReportGenerateRequest(BaseModel):
    """Request payload to synthesize a comprehension report for a session."""
    session_id: str = Field(..., description="Target session UUID")


class ComprehensionReportResponse(BaseModel):
    """Schema for final comprehension and misconception synthesis report."""
    id: str
    session_id: str
    overall_label: str  # Understood, Partial Understanding, Possible Misconception, Insufficient Evidence
    confidence_score: float
    evidence_summary: str
    evidence_breakdown: Optional[Dict[str, Any]] = None
    detected_misconceptions: Optional[List[str]] = None
    process_features: Optional[Dict[str, Any]] = None
    rubric_summary: Optional[Dict[str, Any]] = None
    needs_instructor_review: bool = False
    recommended_interventions: Optional[List[str]] = None
    multimodal_scores: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
