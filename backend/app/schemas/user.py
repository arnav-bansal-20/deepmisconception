"""Pydantic schemas for Student and Faculty management."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class StudentCreate(BaseModel):
    """Schema for registering or validating a student with unique student ID."""
    student_id: str = Field(..., min_length=2, max_length=64, description="Unique student ID / roll number")
    name: str = Field(..., min_length=1, max_length=128, description="Student full name")
    classroom_id: Optional[str] = None
    classroom_name: Optional[str] = None


class StudentResponse(BaseModel):
    """Schema returned for a student."""
    id: str
    student_id: str
    name: str
    classroom_id: Optional[str] = None
    classroom_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FacultyCreate(BaseModel):
    """Schema for registering or validating faculty with unique faculty ID."""
    faculty_id: str = Field(..., min_length=2, max_length=64, description="Unique faculty / instructor ID")
    name: str = Field(..., min_length=1, max_length=128, description="Faculty full name")
    department: str = Field(default="Computer Science & Engineering", max_length=128)


class FacultyResponse(BaseModel):
    """Schema returned for a faculty member."""
    id: str
    faculty_id: str
    name: str
    department: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
