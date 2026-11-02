"""Schemas and annotation guidelines for real instructor-reviewed empirical data.

Separates answer-level labels from overall student-level diagnoses, establishes
inter-rater protocols, and prevents data leakage across student cohorts.
"""
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AnswerAnnotationCreate(BaseModel):
    """Schema for ground-truth instructor review of a single student viva explanation."""
    question_id: str
    session_id: str
    instructor_id: str
    instructor_name: str
    annotated_label: str = Field(..., description="One of: Correct, Partially Correct, Incorrect, Insufficient Evidence")
    agrees_with_ai: bool
    identified_misconceptions: List[str] = Field(default_factory=list)
    pedagogical_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StudentSessionAnnotationCreate(BaseModel):
    """Schema for ground-truth instructor review of overall student comprehension."""
    session_id: str
    student_id: str
    instructor_id: str
    instructor_name: str
    annotated_overall_label: str = Field(..., description="One of: Understood, Partial Understanding, Possible Misconception")
    agrees_with_ai: bool
    confidence_in_verdict: float = Field(default=1.0, ge=0.0, le=1.0)
    annotated_misconception_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmpiricalDatasetMetadata(BaseModel):
    """Metadata describing an empirical or synthetic dataset."""
    dataset_name: str
    is_synthetic: bool
    dataset_type: str  # "synthetic_benchmark" or "real_instructor_annotated"
    num_students: int
    num_sessions: int
    num_answers: int
    split_method: str = "student_level_group_split"  # Grouped by student_id to prevent leakage
    class_distribution: Dict[str, int]
    disclaimer: str

