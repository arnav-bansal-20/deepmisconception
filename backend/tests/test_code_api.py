"""Integration tests for Code Analysis and Execution API routes."""
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


def test_api_analyze_code(client):
    code = "def add(a, b):\n    return a + b\n"
    res = client.post("/api/code/analyze", json={"code": code})
    assert res.status_code == 200
    data = res.json()
    assert data["is_syntax_valid"] is True
    assert len(data["functions"]) == 1
    assert data["functions"][0]["name"] == "add"


def test_api_run_code(client):
    code = "print('Hello DeepMisconception')\n"
    res = client.post("/api/code/run", json={"code": code})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "Hello DeepMisconception" in data["stdout"]
    assert data["sandbox_used"] is True


def test_submission_automatic_ast_and_execution(client):
    # Create session
    s_res = client.post("/api/sessions", json={
        "student_id": "STU_AUTO_01",
        "student_name": "Dave Developer",
        "title": "Accumulator Task"
    })
    session_id = s_res.json()["id"]

    code = "x = 42\nprint(f'Val: {x}')\n"
    sub_res = client.post("/api/submissions", json={
        "session_id": session_id,
        "code": code
    })
    assert sub_res.status_code == 201
    data = sub_res.json()
    assert data["is_valid"] is True
    assert data["ast_summary"] is not None
    assert data["ast_summary"]["is_syntax_valid"] is True
    assert "Val: 42" in data["execution_output"]
