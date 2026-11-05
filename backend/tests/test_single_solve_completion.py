"""Integration tests for single-solve enforcement, problem completion, and re-attempt blocking."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.session import CodingSession
from app.models.viva_question import VivaQuestion
from app.models.submission import CodeSubmission

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


def test_cannot_resolve_completed_problem(client):
    """Test that once a session is completed, student cannot create a new session for the same problem."""
    # 1. Create session
    payload = {
        "student_id": "STUDENT_TEST_1",
        "student_name": "Test Student",
        "title": "Recursive Factorial",
        "language": "python"
    }
    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 201
    session_id = res.json()["id"]

    # 2. Mark session as completed (as happens upon viva completion)
    db = TestingSessionLocal()
    s = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    s.status = "completed"
    db.commit()
    db.close()

    # 3. Attempting to start the same problem again should be blocked with 409 Conflict
    dup_res = client.post("/api/sessions", json=payload)
    assert dup_res.status_code == 409
    assert "already completed" in dup_res.json()["detail"]


def test_cannot_submit_code_to_completed_problem(client):
    """Test that code cannot be modified or submitted to a completed session."""
    payload = {
        "student_id": "STUDENT_TEST_2",
        "student_name": "Test Student",
        "title": "Binary Search",
        "language": "python"
    }
    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 201
    session_id = res.json()["id"]

    # Mark completed
    db = TestingSessionLocal()
    s = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    s.status = "completed"
    db.commit()
    db.close()

    # Attempt to submit code
    sub_res = client.post("/api/submissions", json={
        "session_id": session_id,
        "code": "print('hello')"
    })
    assert sub_res.status_code == 400
    assert "already marked as completed" in sub_res.json()["detail"]


def test_viva_submission_automatically_marks_session_completed(client):
    """Test that answering a viva question automatically marks the session as completed."""
    # 1. Create active session
    res = client.post("/api/sessions", json={
        "student_id": "STUDENT_TEST_3",
        "student_name": "Test Student",
        "title": "Accumulator Pattern",
        "language": "python"
    })
    session_id = res.json()["id"]
    assert res.json()["status"] == "active"

    # 2. Submit code
    client.post("/api/submissions", json={
        "session_id": session_id,
        "code": "total = 0\nfor i in range(5):\n    total += i\n"
    })

    # 3. Add viva question
    db = TestingSessionLocal()
    q = VivaQuestion(
        session_id=session_id,
        question_text="Explain the accumulator variable total.",
        concept="accumulator-loop-invariants",
        focal_code="total = 0"
    )
    db.add(q)
    db.commit()
    q_id = q.id
    db.close()

    # 4. Submit viva answer
    ans_res = client.post("/api/viva/evaluate", json={
        "question_id": q_id,
        "student_answer": "The total variable accumulates the running sum across loop iterations."
    })
    assert ans_res.status_code == 201

    # 5. Check session status is now 'completed'
    db = TestingSessionLocal()
    s = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    assert s.status == "completed"
    db.close()
