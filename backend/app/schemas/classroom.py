"""Pydantic schemas for Classroom / Cohort management."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ClassroomCreate(BaseModel):
    """Schema for a teacher creating a new student classroom group."""
    name: str = Field(..., min_length=3, max_length=256, description="Classroom title e.g. CS101 Section A")
    code: str = Field(..., min_length=2, max_length=64, description="Unique enrollment code e.g. CS101-ALPHA")
    teacher_id: str = Field(..., min_length=1, max_length=64)
    teacher_name: str = Field(..., min_length=1, max_length=128)
    department: str = Field(default="Computer Science & Engineering")
    description: Optional[str] = None


class ClassroomResponse(BaseModel):
    """Schema returned for a classroom group."""
    id: str
    name: str
    code: str
    teacher_id: str
    teacher_name: str
    department: str
    description: Optional[str] = None
    created_at: datetime
    active_students_count: int = 0
    assigned_problems_count: int = 0

    model_config = ConfigDict(from_attributes=True)
