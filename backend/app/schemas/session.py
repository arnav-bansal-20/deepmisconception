"""Pydantic schemas for Coding Sessions."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class SessionCreate(BaseModel):
    """Schema for initiating a new student session."""
    student_id: str = Field(..., min_length=1, max_length=64, description="Student identifier or roll number")
    student_name: str = Field(..., min_length=1, max_length=128, description="Student full name")
    classroom_id: Optional[str] = Field(default=None, description="Enrolled classroom/cohort id")
    classroom_name: Optional[str] = Field(default=None, description="Enrolled classroom title")
    teacher_id: Optional[str] = Field(default=None, description="Guiding teacher identifier")
    teacher_name: Optional[str] = Field(default=None, description="Guiding teacher name")
    title: str = Field(default="Python Coding Assessment", max_length=256)
    language: str = Field(default="python", max_length=32)


class SessionUpdate(BaseModel):
    """Schema for updating session status or title."""
    title: Optional[str] = None
    status: Optional[str] = None


class SessionSummary(BaseModel):
    """Compact summary of a coding session."""
    id: str
    student_id: str
    student_name: str
    classroom_id: Optional[str] = None
    classroom_name: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    title: str
    language: str
    status: str
    created_at: datetime
    updated_at: datetime
    submission_count: int = 0
    question_count: int = 0
    has_report: bool = False
    latest_verdict: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SessionDetail(BaseModel):
    """Detailed view of a coding session including relationships."""
    id: str
    student_id: str
    student_name: str
    classroom_id: Optional[str] = None
    classroom_name: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    title: str
    language: str
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
