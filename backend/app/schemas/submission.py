"""Pydantic schemas for Code Submissions."""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CodeSubmissionCreate(BaseModel):
    """Schema for submitting code for analysis."""
    session_id: str
    code: str = Field(..., min_length=1, description="Student Python source code")


class CodeRunRequest(BaseModel):
    """Schema for requesting isolated execution of code."""
    code: str = Field(..., min_length=1)
    timeout_seconds: Optional[float] = None


class CodeRunResponse(BaseModel):
    """Schema for code execution results."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    sandbox_used: bool = True
    error_type: Optional[str] = None


class CodeSubmissionResponse(BaseModel):
    """Schema returned after a code submission is processed."""
    id: str
    session_id: str
    code: str
    ast_summary: Optional[Dict[str, Any]] = None
    execution_output: Optional[str] = None
    error_output: Optional[str] = None
    is_valid: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
