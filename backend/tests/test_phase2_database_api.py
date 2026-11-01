"""Integration tests for Phase 2: Database and Core API endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app

# In-memory SQLite database for isolated test runs
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_session_lifecycle_and_listing(client):
    """Test creating, reading, updating, and listing sessions."""
    # 1. Create a session
    payload = {
        "student_id": "STUDENT_001",
        "student_name": "Alice Turing",
        "title": "Factorial & Recursion",
        "language": "python"
    }
    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["student_id"] == "STUDENT_001"
    assert data["status"] == "active"
    session_id = data["id"]

    # 2. Get specific session
    res_get = client.get(f"/api/sessions/{session_id}")
    assert res_get.status_code == 200
    assert res_get.json()["student_name"] == "Alice Turing"

    # 3. List sessions
    res_list = client.get("/api/sessions")
    assert res_list.status_code == 200
    sessions = res_list.json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    assert sessions[0]["submission_count"] == 0

    # 4. Update session
    res_patch = client.patch(f"/api/sessions/{session_id}", json={"title": "Updated Title"})
    assert res_patch.status_code == 200
    assert res_patch.json()["title"] == "Updated Title"


def test_submission_and_telemetry(client):
    """Test submitting code and recording telemetry events."""
    # Create session
    s_res = client.post("/api/sessions", json={
        "student_id": "STUDENT_002",
        "student_name": "Bob Babbage",
        "title": "Loop Accumulator"
    })
    session_id = s_res.json()["id"]

    # Submit code
    sub_res = client.post("/api/submissions", json={
        "session_id": session_id,
        "code": "def total(n):\n    return sum(range(n))\n"
    })
    assert sub_res.status_code == 201
    assert sub_res.json()["is_valid"] is True

    # Check session submissions
    list_subs = client.get(f"/api/submissions/session/{session_id}")
    assert list_subs.status_code == 200
    assert len(list_subs.json()) == 1

    # Record telemetry batch
    telemetry_payload = {
        "session_id": session_id,
        "events": [
            {"event_type": "edit", "delta_chars": 15, "cursor_line": 1, "cursor_column": 10},
            {"event_type": "edit", "delta_chars": 20, "cursor_line": 2, "cursor_column": 12},
            {"event_type": "paste", "paste_length": 35, "delta_chars": 35, "cursor_line": 2, "cursor_column": 25},
            {"event_type": "run", "delta_chars": 0}
        ]
    }
    telem_res = client.post("/api/telemetry/batch", json=telemetry_payload)
    assert telem_res.status_code == 201
    assert telem_res.json()["recorded_events"] == 4

    # Check process features
    feat_res = client.get(f"/api/telemetry/features/{session_id}")
    assert feat_res.status_code == 200
    features = feat_res.json()
    assert features["total_events"] == 4
    assert features["paste_count"] == 1
    assert features["run_count"] == 1
    assert features["total_chars_pasted"] == 35


def test_instructor_overview_and_reviews(client):
    """Test instructor metrics and annotation flow."""
    s_res = client.post("/api/sessions", json={
        "student_id": "STUDENT_003",
        "student_name": "Carol Lovelace",
        "title": "Condition Logic"
    })
    session_id = s_res.json()["id"]

    # Submit review
    review_payload = {
        "session_id": session_id,
        "reviewer_name": "Prof. Dijkstra",
        "review_type": "overall",
        "assigned_label": "Understood",
        "agree_with_ai": True,
        "feedback_notes": "Student showed solid conceptual grasp of conditional flow."
    }
    rev_res = client.post("/api/instructor/reviews", json=review_payload)
    assert rev_res.status_code == 201
    assert rev_res.json()["assigned_label"] == "Understood"

    # Overview endpoint
    overview_res = client.get("/api/instructor/overview")
    assert overview_res.status_code == 200
    ov = overview_res.json()
    assert ov["total_sessions"] >= 1
