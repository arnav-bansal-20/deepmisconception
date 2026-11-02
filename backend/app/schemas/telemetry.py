"""Pydantic schemas for Process Telemetry."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ProcessEventItem(BaseModel):
    """Single telemetry event item."""
    event_type: str = Field(..., description="edit, paste, run, submit, idle, error")
    cursor_line: Optional[int] = None
    cursor_column: Optional[int] = None
    delta_chars: int = 0
    paste_length: int = 0
    timestamp: Optional[datetime] = None
    event_metadata: Optional[Dict[str, Any]] = None


class ProcessBatchCreate(BaseModel):
    """Batch of telemetry events sent by editor."""
    session_id: str
    events: List[ProcessEventItem]


class ProcessEventResponse(BaseModel):
    """Persisted event schema."""
    id: str
    session_id: str
    event_type: str
    cursor_line: Optional[int]
    cursor_column: Optional[int]
    delta_chars: int
    paste_length: int
    timestamp: datetime
    event_metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class TimelineEventSummary(BaseModel):
    """Chronological milestone in student coding process."""
    offset_seconds: float
    event_type: str
    label: str
    detail: Optional[str] = None


class ProcessFeaturesResponse(BaseModel):
    """Derived statistical features from coding process."""
    session_id: str
    total_events: int
    edit_count: int
    paste_count: int
    run_count: int
    error_count: int = 0
    total_chars_pasted: int
    total_chars_edited: int = 0
    paste_ratio: float
    coding_duration_seconds: float
    average_edit_interval_seconds: float
    burst_count: int
    burstiness_ratio: float = 0.0
    time_to_first_run_seconds: Optional[float] = None
    error_recovery_rate: float = 1.0
    typing_speed_cpm: float = 0.0
    process_profile: str = "Incremental Developer"
    # Interpretable process evaluation sub-scores
    incremental_score: float = 0.75
    debugging_score: float = 0.90
    revision_score: float = 0.75
    testing_score: float = 0.60
    process_score: float = 0.75
    has_process_evidence: bool = True
    ethical_disclaimer: str = (
        "Process telemetry serves as supporting contextual evidence of problem-solving habits "
        "and is never used in isolation to accuse students of misconduct or grade penalties."
    )
    disclosure_notice: str = (
        "Telemetry tracks timestamps, edit volumes, execution attempts, and error recoveries within this editor session "
        "to assess iterative development and debugging habits. No personal files or external keystrokes are recorded."
    )
    timeline_events: List[TimelineEventSummary] = []
