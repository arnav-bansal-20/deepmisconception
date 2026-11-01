"""Integration tests for Viva Generation and Evaluation API endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app

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


def test_viva_generation_and_evaluation_workflow(client):
    # 1. Create a session
    s_res = client.post("/api/sessions", json={
        "student_id": "STU_VIVA_001",
        "student_name": "Ada Lovelace",
        "title": "Accumulator Viva Verification"
    })
    session_id = s_res.json()["id"]

    # 2. Submit code
    code = """
def sum_squares(n):
    total = 0
    for i in range(1, n + 1):
        if i % 3 == 0:
            continue
        total += i * i
    return total
"""
    sub_res = client.post("/api/submissions", json={
        "session_id": session_id,
        "code": code
    })
    assert sub_res.status_code == 201

    # 3. Retrieve viva questions (auto-generated on submission)
    viva_list_res = client.get(f"/api/viva/session/{session_id}")
    assert viva_list_res.status_code == 200
    questions = viva_list_res.json()
    assert len(questions) >= 1
    target_q = questions[0]
    question_id = target_q["id"]

    # 4. Student submits correct answer
    ans_res = client.post("/api/viva/evaluate", json={
        "question_id": question_id,
        "student_answer": (
            "The modulo condition i % 3 == 0 checks for multiples of 3. "
            "When true, the continue statement skips adding i * i to total for that cycle."
        )
    })
    assert ans_res.status_code == 201
    assessment = ans_res.json()
    assert assessment["answer_label"] in ("Correct", "Partially Correct")
    assert assessment["question_id"] == question_id

    # 5. Check session questions again to verify assessment is attached
    recheck_res = client.get(f"/api/viva/session/{session_id}")
    assert recheck_res.status_code == 200
    updated_qs = recheck_res.json()
    matched = [q for q in updated_qs if q["id"] == question_id][0]
    assert matched["has_answer"] is True
    assert matched["assessment"] is not None
    assert matched["assessment"]["answer_label"] == assessment["answer_label"]
