"""Unit and integration tests for unique student and faculty ID enforcement."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError

from app.database import Base, get_db
from app.models.user import Student, Faculty
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


def test_database_student_id_unique_constraint():
    """Test that SQLAlchemy model enforces unique student_id in the database table."""
    db = TestingSessionLocal()
    try:
        s1 = Student(student_id="1RV24CS051", name="Arnav Bansal")
        db.add(s1)
        db.commit()

        # Attempt to add another student with identical student_id
        s2 = Student(student_id="1RV24CS051", name="Duplicate Student")
        db.add(s2)
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()


def test_database_faculty_id_unique_constraint():
    """Test that SQLAlchemy model enforces unique faculty_id in the database table."""
    db = TestingSessionLocal()
    try:
        f1 = Faculty(faculty_id="FAC-CS-804", name="Dr. Ada Smith", department="Computer Science")
        db.add(f1)
        db.commit()

        # Attempt to add another faculty member with identical faculty_id
        f2 = Faculty(faculty_id="FAC-CS-804", name="Duplicate Faculty", department="Information Science")
        db.add(f2)
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()


def test_api_student_registration_and_unique_enforcement(client):
    """Test API endpoint enforces unique student ID and handles duplicate conflict."""
    # 1. Register student
    res = client.post("/api/students", json={
        "student_id": "1RV24CS051",
        "name": "Arnav Bansal",
        "classroom_id": "class-cse-101",
        "classroom_name": "CS101: Python Foundations - Group Alpha"
    })
    assert res.status_code == 201
    data = res.json()
    assert data["student_id"] == "1RV24CS051"
    assert data["name"] == "Arnav Bansal"

    # 2. Idempotent login with same ID and name succeeds
    res_same = client.post("/api/students", json={
        "student_id": "1RV24CS051",
        "name": "Arnav Bansal"
    })
    assert res_same.status_code == 201
    assert res_same.json()["student_id"] == "1RV24CS051"

    # 3. Conflict: someone else tries to use the same student_id
    res_conflict = client.post("/api/students", json={
        "student_id": "1RV24CS051",
        "name": "Impostor Candidate"
    })
    assert res_conflict.status_code == 409
    assert "already registered" in res_conflict.json()["detail"]


def test_api_faculty_registration_and_unique_enforcement(client):
    """Test API endpoint enforces unique faculty ID and handles duplicate conflict."""
    # 1. Register faculty
    res = client.post("/api/faculty", json={
        "faculty_id": "FAC-CS-804",
        "name": "Dr. Ada Smith",
        "department": "Computer Science & Engineering"
    })
    assert res.status_code == 201
    data = res.json()
    assert data["faculty_id"] == "FAC-CS-804"
    assert data["name"] == "Dr. Ada Smith"

    # 2. Idempotent login with same ID and name succeeds
    res_same = client.post("/api/faculty", json={
        "faculty_id": "FAC-CS-804",
        "name": "Dr. Ada Smith"
    })
    assert res_same.status_code == 201

    # 3. Conflict: someone else tries to register the same faculty_id
    res_conflict = client.post("/api/faculty", json={
        "faculty_id": "FAC-CS-804",
        "name": "Prof. Impostor"
    })
    assert res_conflict.status_code == 409
    assert "already registered" in res_conflict.json()["detail"]


def test_api_list_and_get_student_and_faculty(client):
    """Test retrieving students and faculty by unique ID."""
    client.post("/api/students", json={"student_id": "STU-001", "name": "Student One"})
    client.post("/api/students", json={"student_id": "STU-002", "name": "Student Two"})

    res_list = client.get("/api/students")
    assert res_list.status_code == 200
    assert len(res_list.json()) == 2

    res_get = client.get("/api/students/STU-001")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "Student One"

    res_not_found = client.get("/api/students/NONEXISTENT")
    assert res_not_found.status_code == 404

    client.post("/api/faculty", json={"faculty_id": "FAC-100", "name": "Prof. One"})
    res_fac = client.get("/api/faculty/FAC-100")
    assert res_fac.status_code == 200
    assert res_fac.json()["name"] == "Prof. One"
