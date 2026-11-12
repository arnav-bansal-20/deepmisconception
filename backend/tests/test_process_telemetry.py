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


def test_incremental_score_paste_ratio_zero():
    """Requirement 1: Existing incremental score remains unchanged when paste ratio is 0."""
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.0) == 1.00
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=10, paste_ratio=0.0) == 0.92
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=5, paste_ratio=0.0) == 0.85
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=1, paste_ratio=0.0) == 0.80


def test_incremental_score_paste_ratio_threshold_30():
    """Requirement 2: Existing incremental score remains unchanged when paste ratio is 0.30."""
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.30) == 1.00
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=10, paste_ratio=0.30) == 0.92
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=5, paste_ratio=0.30) == 0.85
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=1, paste_ratio=0.30) == 0.80

    # Intermediate value below 0.30
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.15) == 1.00


def test_incremental_score_paste_ratio_50():
    """Requirement 3: Incremental score decreases when paste ratio is 0.50."""
    base_score = ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.0)
    score_50 = ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.50)

    # Expected factor: (1 - 0.50) / 0.70 = 0.50 / 0.70 ≈ 0.7142857
    expected = 1.00 * ((1.0 - 0.50) / 0.70)
    assert score_50 < base_score
    assert abs(score_50 - expected) < 1e-6


def test_incremental_score_paste_ratio_70():
    """Requirement 4: Incremental score decreases when paste ratio is 0.70."""
    score_50 = ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.50)
    score_70 = ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=0.70)

    # Expected factor: (1 - 0.70) / 0.70 = 0.30 / 0.70 ≈ 0.4285714
    expected = 1.00 * ((1.0 - 0.70) / 0.70)
    assert score_70 < score_50
    assert abs(score_70 - expected) < 1e-6


def test_incremental_score_paste_ratio_100_percent():
    """Requirement 5: Incremental score becomes exactly 0 when paste ratio is 1.00."""
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=20, paste_ratio=1.00) == 0.0
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=10, paste_ratio=1.00) == 0.0
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=1, paste_ratio=1.00) == 0.0

    # Also test via compute_features with telemetry events where all content is pasted
    events = [
        ProcessEvent(session_id="s_full_paste", event_type="paste", delta_chars=200, paste_length=200),
    ]
    features = ProcessTelemetryService.compute_features("s_full_paste", events)
    assert features.paste_ratio == 1.0
    assert features.incremental_score == 0.0


def test_missing_telemetry_does_not_produce_zero_score():
    """Requirement 6: Missing telemetry preserves existing no-telemetry behaviour and does not produce zero."""
    features = ProcessTelemetryService.compute_features("s_empty", [])
    assert features.incremental_score == 0.75
    assert features.process_score == 0.75
    assert features.debugging_score == 0.90
    assert features.revision_score == 0.75
    assert features.testing_score == 0.60


def test_missing_or_invalid_paste_ratio_fallback():
    """Requirement 7: Missing or invalid paste ratio uses the safe fallback (no accidental zero)."""
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=15, paste_ratio=None) == 1.00
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=15, paste_ratio=float("nan")) == 1.00
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=15, paste_ratio=-0.25) == 1.00
    assert ProcessTelemetryService.calculate_incremental_score(edit_count=15, paste_ratio=1.5) == 0.0


def test_debugging_revision_testing_scores_remain_unchanged_by_paste_ratio():
    """Requirement 8: Debugging, revision, and testing scores remain unchanged regardless of paste ratio."""
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Session 1: No paste (paste_ratio = 0.0)
    events_no_paste = [
        ProcessEvent(session_id="s_norm", event_type="edit", delta_chars=20, paste_length=0, timestamp=t0),
        ProcessEvent(session_id="s_norm", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=2), event_metadata={"is_error": True}),
        ProcessEvent(session_id="s_norm", event_type="edit", delta_chars=10, paste_length=0, timestamp=t0 + timedelta(seconds=4)),
        ProcessEvent(session_id="s_norm", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=6), event_metadata={"is_error": False}),
    ]
    feat_no_paste = ProcessTelemetryService.compute_features("s_norm", events_no_paste)

    # Session 2: Same actions plus a paste event (paste_ratio > 0.50)
    events_with_paste = [
        ProcessEvent(session_id="s_paste", event_type="paste", delta_chars=100, paste_length=100, timestamp=t0),
        ProcessEvent(session_id="s_paste", event_type="edit", delta_chars=20, paste_length=0, timestamp=t0 + timedelta(seconds=1)),
        ProcessEvent(session_id="s_paste", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=2), event_metadata={"is_error": True}),
        ProcessEvent(session_id="s_paste", event_type="edit", delta_chars=10, paste_length=0, timestamp=t0 + timedelta(seconds=4)),
        ProcessEvent(session_id="s_paste", event_type="run", delta_chars=0, paste_length=0, timestamp=t0 + timedelta(seconds=6), event_metadata={"is_error": False}),
    ]
    feat_with_paste = ProcessTelemetryService.compute_features("s_paste", events_with_paste)

    # S_DEBUG, S_REV, S_TEST must be completely identical between the two sessions
    assert feat_with_paste.debugging_score == feat_no_paste.debugging_score
    assert feat_with_paste.revision_score == feat_no_paste.revision_score
    assert feat_with_paste.testing_score == feat_no_paste.testing_score

    # Only S_INC is reduced by the paste ratio
    assert feat_with_paste.incremental_score < feat_no_paste.incremental_score

    # Final Process Score formula remains: 0.30*S_INC + 0.30*S_DEBUG + 0.25*S_REV + 0.15*S_TEST
    expected_process_score = round(
        0.30 * feat_with_paste.incremental_score
        + 0.30 * feat_with_paste.debugging_score
        + 0.25 * feat_with_paste.revision_score
        + 0.15 * feat_with_paste.testing_score,
        3
    )
    assert abs(feat_with_paste.process_score - expected_process_score) < 1e-4
