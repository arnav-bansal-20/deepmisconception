"""Pydantic schemas for Dynamic Viva and Answer Assessment."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class VivaGenerateRequest(BaseModel):
    """Schema for requesting dynamic question generation."""
    session_id: str
    submission_id: Optional[str] = None
    count: int = Field(default=3, ge=1, le=5)


class VivaQuestionResponse(BaseModel):
    """Schema for a dynamically generated viva question."""
    id: str
    session_id: str
    submission_id: Optional[str]
    question_text: str
    focal_code: Optional[str]
    concept: str
    difficulty: str
    expected_reasoning: Optional[str]
    rubric: Optional[Dict[str, Any]]
    misconception_indicators: Optional[List[str]]
    created_at: datetime
    has_answer: bool = False
    assessment: Optional["AnswerAssessmentResponse"] = None

    model_config = ConfigDict(from_attributes=True)


class VivaAnswerSubmit(BaseModel):
    """Schema for student submitting an answer to a viva question."""
    question_id: str
    student_answer: str = Field(..., min_length=2, description="Student explanation or reasoning")


class AnswerAssessmentResponse(BaseModel):
    """Schema for an evaluated student answer."""
    id: str
    question_id: str
    session_id: str
    student_answer: str
    answer_label: str  # Correct, Partially Correct, Incorrect, Insufficient Evidence
    expected_reasoning: Optional[str]
    concepts_correct: Optional[List[str]]
    concepts_missed: Optional[List[str]]
    contradictions: Optional[List[str]]
    supporting_evidence: Optional[List[str]]
    evaluation_explanation: str
    uncertainty_score: float
    needs_instructor_review: bool
    instructor_status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


VivaQuestionResponse.model_rebuild()
