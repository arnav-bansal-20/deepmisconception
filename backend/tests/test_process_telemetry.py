"""Unit tests for Process Telemetry feature extraction and behavioral profiling."""
from datetime import datetime, timezone, timedelta
import pytest
from app.models.process_event import ProcessEvent
from app.services.process_telemetry import ProcessTelemetryService
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_empty_events_telemetry():
    """Verify empty events list computes zeroed baseline without error."""
    features = ProcessTelemetryService.compute_features("test-sess-0", [])
    assert features.total_events == 0
    assert features.edit_count == 0
    assert features.paste_ratio == 0.0
    assert features.time_to_first_run_seconds is None
    assert features.error_recovery_rate == 1.0


def test_organic_typing_telemetry():
    """Verify organic keystrokes and burst calculation."""
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        ProcessEvent(session_id="s1", event_type="edit", delta_chars=1, paste_length=0, timestamp=t0),
        # fast interval (< 0.5s) -> burst!
        ProcessEvent(session_id="s1", event_type="edit", delta_chars=1, paste_length=0, timestamp=t0 + timedelta(milliseconds=200)),
        ProcessEvent(session_id="s1", event_type="edit", delta_chars=1, paste_length=0, timestamp=t0 + timedelta(milliseconds=400)),
        # slower pause
        ProcessEvent(session_id="s1", event_type="edit", delta_chars=1, paste_length=0, timestamp=t0 + timedelta(seconds=2)),
        ProcessEvent(session_id="s1", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=5)),
    ]

    features = ProcessTelemetryService.compute_features("s1", events)
    assert features.total_events == 5
    assert features.edit_count == 4
    assert features.run_count == 1
    assert features.paste_count == 0
    assert features.paste_ratio == 0.0
    assert features.burst_count >= 2
    assert features.time_to_first_run_seconds == 5.0
    assert "supporting contextual evidence" in features.ethical_disclaimer


def test_block_paster_profile():
    """Verify high paste ratio generates block paster profile."""
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        ProcessEvent(session_id="s2", event_type="paste", delta_chars=350, paste_length=350, timestamp=t0),
        ProcessEvent(session_id="s2", event_type="edit", delta_chars=5, paste_length=0, timestamp=t0 + timedelta(seconds=2)),
        ProcessEvent(session_id="s2", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=4)),
    ]

    features = ProcessTelemetryService.compute_features("s2", events)
    assert features.paste_count == 1
    assert features.total_chars_pasted == 350
    assert features.paste_ratio > 0.9
    assert features.process_profile == "Rapid Block Importer"
    assert len(features.timeline_events) >= 2


def test_error_recovery_and_iterative_debugging():
    """Verify error recovery rate and iterative debugging profile."""
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        ProcessEvent(session_id="s3", event_type="edit", delta_chars=20, paste_length=0, timestamp=t0),
        ProcessEvent(session_id="s3", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=3), event_metadata={"is_error": True, "message": "SyntaxError"}),
        # Student fixes error
        ProcessEvent(session_id="s3", event_type="edit", delta_chars=15, paste_length=0, timestamp=t0 + timedelta(seconds=6)),
        ProcessEvent(session_id="s3", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=8), event_metadata={"is_error": False}),
        ProcessEvent(session_id="s3", event_type="edit", delta_chars=10, paste_length=0, timestamp=t0 + timedelta(seconds=10)),
        ProcessEvent(session_id="s3", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=12)),
        ProcessEvent(session_id="s3", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=14)),
        ProcessEvent(session_id="s3", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=16)),
    ]

    features = ProcessTelemetryService.compute_features("s3", events)
    assert features.error_count == 1
    assert features.error_recovery_rate == 1.0
    assert features.run_count >= 5
    assert features.process_profile == "Iterative Trial-and-Error Debugger"


def test_telemetry_features_api_endpoint():
    """Verify /api/telemetry/features/{session_id} endpoint returns enhanced schema."""
    # Create session first
    sess_resp = client.post("/api/sessions", json={"student_id": "STU-TEL-1", "student_name": "Telemetry Tester", "title": "Telemetry Test"})
    assert sess_resp.status_code == 201
    s_id = sess_resp.json()["id"]

    # Ingest batch
    batch_payload = {
        "session_id": s_id,
        "events": [
            {"event_type": "edit", "delta_chars": 12, "paste_length": 0, "cursor_line": 1},
            {"event_type": "paste", "delta_chars": 80, "paste_length": 80, "cursor_line": 2},
            {"event_type": "run", "delta_chars": 0, "paste_length": 0, "cursor_line": 3, "event_metadata": {"is_error": False}},
        ]
    }
    ingest_resp = client.post("/api/telemetry/batch", json=batch_payload)
    assert ingest_resp.status_code == 201

    # Fetch features
    feat_resp = client.get(f"/api/telemetry/features/{s_id}")
    assert feat_resp.status_code == 200
    data = feat_resp.json()
    assert data["session_id"] == s_id
    assert data["total_events"] == 3
    assert data["paste_count"] == 1
    assert data["run_count"] == 1
    assert "ethical_disclaimer" in data
    assert len(data["timeline_events"]) >= 2
