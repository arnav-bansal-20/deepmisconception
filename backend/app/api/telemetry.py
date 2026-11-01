"""Coding-process telemetry routes."""
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.session import CodingSession
from app.models.process_event import ProcessEvent
from app.schemas.telemetry import ProcessBatchCreate, ProcessEventResponse, ProcessFeaturesResponse
from app.services.process_telemetry import ProcessTelemetryService

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def record_telemetry_batch(batch_in: ProcessBatchCreate, db: Session = Depends(get_db)):
    """Ingest a batch of editor telemetry events."""
    session = db.query(CodingSession).filter(CodingSession.id == batch_in.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    events_to_create = []
    for item in batch_in.events:
        ts = item.timestamp or datetime.now(timezone.utc)
        events_to_create.append(
            ProcessEvent(
                session_id=batch_in.session_id,
                event_type=item.event_type,
                cursor_line=item.cursor_line,
                cursor_column=item.cursor_column,
                delta_chars=item.delta_chars,
                paste_length=item.paste_length,
                timestamp=ts,
                event_metadata=item.event_metadata or {}
            )
        )
    if events_to_create:
        db.bulk_save_objects(events_to_create)
        db.commit()

    return {"status": "ok", "recorded_events": len(events_to_create)}


@router.get("/session/{session_id}", response_model=List[ProcessEventResponse])
def get_session_telemetry(session_id: str, db: Session = Depends(get_db)):
    """Retrieve raw process events for a session timeline."""
    events = (
        db.query(ProcessEvent)
        .filter(ProcessEvent.session_id == session_id)
        .order_by(ProcessEvent.timestamp.asc())
        .all()
    )
    return events


@router.get("/features/{session_id}", response_model=ProcessFeaturesResponse)
def get_process_features(session_id: str, db: Session = Depends(get_db)):
    """Compute rich summary process telemetry features and profile."""
    events = (
        db.query(ProcessEvent)
        .filter(ProcessEvent.session_id == session_id)
        .order_by(ProcessEvent.timestamp.asc())
        .all()
    )
    return ProcessTelemetryService.compute_features(session_id, events)
