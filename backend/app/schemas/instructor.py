"""Pydantic schemas for Instructor Reviews and Annotations."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class InstructorReviewCreate(BaseModel):
    """Schema for submitting an instructor review or annotation."""
    session_id: str
    reviewer_name: str = Field(..., min_length=2, max_length=128)
    review_type: str = Field(default="overall", description="'overall' or 'answer'")
    target_id: Optional[str] = None
    assigned_label: str = Field(..., description="Understood, Partial Understanding, or Possible Misconception")
    agree_with_ai: Optional[bool] = None
    feedback_notes: Optional[str] = None


class InstructorReviewResponse(BaseModel):
    """Schema for an existing instructor review."""
    id: str
    session_id: str
    reviewer_name: str
    review_type: str
    target_id: Optional[str]
    assigned_label: str
    agree_with_ai: Optional[bool]
    feedback_notes: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
