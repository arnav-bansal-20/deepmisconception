"""Unit tests for Classroom / Cohort Guidance and Targeted Assignments."""
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_list_and_seed_classrooms():
    """Verify default classrooms are seeded and queryable."""
    response = client.get("/api/classrooms")
    assert response.status_code == 200
    classrooms = response.json()
    assert len(classrooms) >= 2

    codes = [c["code"] for c in classrooms]
    assert "CS101-ALPHA" in codes
    assert "CS201-BETA" in codes

    alpha = next(c for c in classrooms if c["code"] == "CS101-ALPHA")
    assert alpha["teacher_name"] == "Dr. Ada Smith"
    assert alpha["teacher_id"] == "FAC-CS-804"


def test_create_new_classroom_and_lookup_by_code():
    """Verify a teacher can create a new student group with a custom code."""
    unique_suffix = uuid.uuid4().hex[:6].upper()
    code = f"CS305-GAMMA-{unique_suffix}"
    payload = {
        "name": f"CS305: Software Engineering - Lab Gamma {unique_suffix}",
        "code": code,
        "teacher_id": "FAC-CS-999",
        "teacher_name": "Dr. Grace Hopper",
        "department": "Computer Science & Engineering",
        "description": "Advanced lab focused on unit testing and AST static verification."
    }
    create_resp = client.post("/api/classrooms", json=payload)
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["code"] == code
    assert created["teacher_name"] == "Dr. Grace Hopper"

    # Test lookup by code (case-insensitive)
    lookup_resp = client.get(f"/api/classrooms/by-code/{code.lower()}")
    assert lookup_resp.status_code == 200
    assert lookup_resp.json()["id"] == created["id"]

    # Test duplicate code rejection
    dup_resp = client.post("/api/classrooms", json=payload)
    assert dup_resp.status_code == 400
    assert "already in use" in dup_resp.json()["detail"]


def test_targeted_assignment_visibility():
    """Verify coding problems authored for a classroom are filtered to that classroom."""
    unique_suffix = uuid.uuid4().hex[:6].upper()
    code = f"CS401-DELTA-{unique_suffix}"
    class_resp = client.post("/api/classrooms", json={
        "name": f"CS401: Delta Testing {unique_suffix}",
        "code": code,
        "teacher_id": "FAC-CS-888",
        "teacher_name": "Dr. Turing",
        "department": "CS",
        "description": "Testing cohort"
    })
    delta_id = class_resp.json()["id"]

    assign_payload = {
        "title": f"Delta Invariant Verifier {unique_suffix}",
        "description": "Analyze whether the loop maintains invariant P throughout all iterations.",
        "starter_code": "def check_invariant(x: int) -> bool:\n    return x > 0\n",
        "difficulty": "medium",
        "concept_tags": ["invariants", "loop"],
        "created_by": "Dr. Turing",
        "classroom_id": delta_id,
        "classroom_name": f"CS401: Delta Testing {unique_suffix}"
    }
    assign_resp = client.post("/api/assignments", json=assign_payload)
    assert assign_resp.status_code == 201
    created_assign = assign_resp.json()
    assert created_assign["classroom_id"] == delta_id

    # Filter assignments for Delta classroom: must include the new Delta assignment
    delta_list_resp = client.get(f"/api/assignments?classroom_id={delta_id}")
    assert delta_list_resp.status_code == 200
    delta_titles = [a["title"] for a in delta_list_resp.json()]
    assert assign_payload["title"] in delta_titles

    # Filter assignments for Alpha classroom: should NOT include Delta Invariant Verifier
    alpha_list_resp = client.get("/api/assignments?classroom_id=class-alpha-101")
    assert alpha_list_resp.status_code == 200
    alpha_titles = [a["title"] for a in alpha_list_resp.json()]
    assert assign_payload["title"] not in alpha_titles


def test_student_session_under_teacher_guidance():
    """Verify student sessions record teacher and classroom association and can be filtered."""
    session_payload = {
        "student_id": "STU-GAMMA-01",
        "student_name": "Maya Lin",
        "classroom_id": "class-alpha-101",
        "classroom_name": "CS101: Python Foundations - Group Alpha",
        "teacher_id": "FAC-CS-804",
        "teacher_name": "Dr. Ada Smith",
        "title": "Accumulator Task"
    }
    sess_resp = client.post("/api/sessions", json=session_payload)
    assert sess_resp.status_code == 201
    session_data = sess_resp.json()
    assert session_data["classroom_id"] == "class-alpha-101"
    assert session_data["teacher_name"] == "Dr. Ada Smith"

    # Verify session query filtering by classroom
    filtered_resp = client.get("/api/sessions?classroom_id=class-alpha-101")
    assert filtered_resp.status_code == 200
    matching_students = [s["student_name"] for s in filtered_resp.json()]
    assert "Maya Lin" in matching_students
