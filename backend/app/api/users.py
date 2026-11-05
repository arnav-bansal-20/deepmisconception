"""API routes for Student and Faculty user registration and verification with unique ID enforcement."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database import get_db
from app.models.user import Student, Faculty
from app.schemas.user import (
    StudentCreate,
    StudentResponse,
    FacultyCreate,
    FacultyResponse,
)

router = APIRouter(tags=["Academic Users"])


@router.post(
    "/students",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register or verify a student with unique student ID",
)
def register_or_verify_student(payload: StudentCreate, db: Session = Depends(get_db)):
    """Enforces that student_id is unique in the database."""
    clean_id = payload.student_id.strip()
    clean_name = payload.name.strip()

    existing = db.query(Student).filter(Student.student_id == clean_id).first()
    if existing:
        # If student exists with the same ID, check if name matches (idempotent login)
        if existing.name.lower() != clean_name.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Student ID '{clean_id}' is already registered to '{existing.name}'. Student IDs must be globally unique.",
            )
        # Update classroom if changed
        if payload.classroom_id and payload.classroom_id != existing.classroom_id:
            existing.classroom_id = payload.classroom_id
            existing.classroom_name = payload.classroom_name
            db.commit()
            db.refresh(existing)
        return existing

    student = Student(
        student_id=clean_id,
        name=clean_name,
        classroom_id=payload.classroom_id,
        classroom_name=payload.classroom_name,
    )
    try:
        db.add(student)
        db.commit()
        db.refresh(student)
        return student
    except IntegrityError as err:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Database constraint violation: Student ID '{clean_id}' must be unique.",
        ) from err


@router.get("/students", response_model=List[StudentResponse], summary="List enrolled students")
def list_students(
    classroom_id: Optional[str] = Query(None, description="Filter by classroom ID"),
    db: Session = Depends(get_db),
):
    """Retrieve enrolled students."""
    query = db.query(Student)
    if classroom_id:
        query = query.filter(Student.classroom_id == classroom_id)
    return query.order_by(Student.student_id.asc()).all()


@router.get("/students/{student_id}", response_model=StudentResponse, summary="Get student by ID")
def get_student(student_id: str, db: Session = Depends(get_db)):
    """Retrieve single student by unique student ID."""
    student = db.query(Student).filter(Student.student_id == student_id.strip()).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID '{student_id}' not found.",
        )
    return student


@router.post(
    "/faculty",
    response_model=FacultyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register or verify a faculty member with unique faculty ID",
)
def register_or_verify_faculty(payload: FacultyCreate, db: Session = Depends(get_db)):
    """Enforces that faculty_id is unique in the database."""
    clean_id = payload.faculty_id.strip()
    clean_name = payload.name.strip()
    clean_dept = payload.department.strip() or "Computer Science & Engineering"

    existing = db.query(Faculty).filter(Faculty.faculty_id == clean_id).first()
    if existing:
        if existing.name.lower() != clean_name.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Faculty ID '{clean_id}' is already registered to '{existing.name}'. Faculty IDs must be globally unique.",
            )
        return existing

    faculty = Faculty(
        faculty_id=clean_id,
        name=clean_name,
        department=clean_dept,
    )
    try:
        db.add(faculty)
        db.commit()
        db.refresh(faculty)
        return faculty
    except IntegrityError as err:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Database constraint violation: Faculty ID '{clean_id}' must be unique.",
        ) from err


@router.get("/faculty", response_model=List[FacultyResponse], summary="List faculty instructors")
def list_faculty(db: Session = Depends(get_db)):
    """Retrieve all faculty members."""
    return db.query(Faculty).order_by(Faculty.faculty_id.asc()).all()


@router.get("/faculty/{faculty_id}", response_model=FacultyResponse, summary="Get faculty by ID")
def get_faculty(faculty_id: str, db: Session = Depends(get_db)):
    """Retrieve single faculty member by unique faculty ID."""
    fac = db.query(Faculty).filter(Faculty.faculty_id == faculty_id.strip()).first()
    if not fac:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Faculty with ID '{faculty_id}' not found.",
        )
    return fac

