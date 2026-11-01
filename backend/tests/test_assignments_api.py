"""Integration tests for Instructor Assignment Authoring API."""
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


def test_list_and_create_assignments(client):
    # 1. List initial seeded defaults
    res = client.get("/api/assignments")
    assert res.status_code == 200
    assignments = res.json()
    assert len(assignments) >= 3

    # 2. Teacher creates a custom coding question
    new_q = {
        "title": "Prime Number Sieve with Break Logic",
        "description": "Write a function is_prime(n) that checks if an integer is prime.",
        "starter_code": "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
        "difficulty": "medium",
        "concept_tags": ["primes", "loops", "early-return"],
        "created_by": "Dr. Ada Smith",
    }
    create_res = client.post("/api/assignments", json=new_q)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["title"] == new_q["title"]
    assert created["created_by"] == "Dr. Ada Smith"

    # 3. Fetch specific assignment
    get_res = client.get(f"/api/assignments/{created['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["difficulty"] == "medium"
