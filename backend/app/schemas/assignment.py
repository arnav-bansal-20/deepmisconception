"""Pydantic schemas for Teacher-created coding assignments."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class AssignmentCreate(BaseModel):
    """Schema for a teacher creating a new coding question."""
    title: str = Field(..., min_length=3, max_length=256)
    description: str = Field(..., min_length=10)
    starter_code: str = Field(..., min_length=5)
    difficulty: str = Field(default="medium", description="easy, medium, hard")
    concept_tags: List[str] = Field(default_factory=list)
    created_by: str = Field(default="Faculty Instructor")
    teacher_id: Optional[str] = Field(default=None, description="Faculty instructor identifier")
    classroom_id: Optional[str] = Field(default=None, description="Optional target classroom/cohort id")
    classroom_name: Optional[str] = Field(default=None, description="Optional target classroom name")
    sample_input: Optional[str] = Field(default=None, description="Optional sample test case input")
    sample_output: Optional[str] = Field(default=None, description="Optional sample test case expected output")


class AssignmentResponse(BaseModel):
    """Schema returned for a coding assignment."""
    id: str
    classroom_id: Optional[str] = None
    classroom_name: Optional[str] = None
    title: str
    description: str
    starter_code: str
    difficulty: str
    concept_tags: Optional[List[str]] = None
    created_by: str
    teacher_id: Optional[str] = None
    sample_input: Optional[str] = None
    sample_output: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
