"""Classroom and Cohort API routes for teacher-guided student groups."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, func
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db
from app.models.classroom import Classroom
from app.models.session import CodingSession
from app.models.assignment import Assignment
from app.models.user import Faculty
from app.schemas.classroom import ClassroomCreate, ClassroomResponse

router = APIRouter(prefix="/classrooms", tags=["Classrooms"])


def seed_default_classrooms_if_empty(db: Session):
    """Seed initial academic classrooms if auto-seed is enabled and table is empty."""
    if settings.auto_seed_defaults and db.query(Classroom).count() == 0:
        from app.seed_real_data import seed_real_data
        seed_real_data(db=db, clean=False)


@router.get("", response_model=List[ClassroomResponse])
def list_classrooms(
    teacher_id: Optional[str] = Query(None, description="Filter by teacher identifier"),
    db: Session = Depends(get_db),
):
    """List classrooms/groups with student and assignment counts."""
    seed_default_classrooms_if_empty(db)

    query = db.query(Classroom)
    if teacher_id:
        fac = db.query(Faculty).filter(Faculty.faculty_id == teacher_id).first()
        if fac:
            query = query.filter(or_(Classroom.teacher_id == teacher_id, Classroom.teacher_name.ilike(fac.name)))
        else:
            query = query.filter(Classroom.teacher_id == teacher_id)

    classrooms = query.order_by(Classroom.created_at.asc()).all()

    result = []
    for c in classrooms:
        students_count = (
            db.query(func.count(func.distinct(CodingSession.student_id)))
            .filter(CodingSession.classroom_id == c.id)
            .scalar()
            or 0
        )
        problems_count = db.query(Assignment).filter(Assignment.classroom_id == c.id).count()
        result.append(
            ClassroomResponse(
                id=c.id,
                name=c.name,
                code=c.code,
                teacher_id=c.teacher_id,
                teacher_name=c.teacher_name,
                department=c.department,
                description=c.description,
                created_at=c.created_at,
                active_students_count=students_count,
                assigned_problems_count=problems_count,
            )
        )
    return result


@router.post("", response_model=ClassroomResponse, status_code=status.HTTP_201_CREATED)
def create_classroom(classroom_in: ClassroomCreate, db: Session = Depends(get_db)):
    """Allow a teacher to create a new guided classroom group."""
    code_normalized = classroom_in.code.strip().upper()
    existing = db.query(Classroom).filter(Classroom.code == code_normalized).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Classroom code '{code_normalized}' is already in use. Please choose a unique code."
        )

    classroom = Classroom(
        name=classroom_in.name.strip(),
        code=code_normalized,
        teacher_id=classroom_in.teacher_id.strip(),
        teacher_name=classroom_in.teacher_name.strip(),
        department=classroom_in.department.strip(),
        description=classroom_in.description.strip() if classroom_in.description else None,
    )
    db.add(classroom)

    # Ensure teacher is registered in unique faculty registry
    existing_fac = db.query(Faculty).filter(Faculty.faculty_id == classroom_in.teacher_id.strip()).first()
    if not existing_fac:
        db.add(Faculty(
            faculty_id=classroom_in.teacher_id.strip(),
            name=classroom_in.teacher_name.strip(),
            department=classroom_in.department.strip() or "Computer Science & Engineering",
        ))

    db.commit()
    db.refresh(classroom)

    return ClassroomResponse(
        id=classroom.id,
        name=classroom.name,
        code=classroom.code,
        teacher_id=classroom.teacher_id,
        teacher_name=classroom.teacher_name,
        department=classroom.department,
        description=classroom.description,
        created_at=classroom.created_at,
        active_students_count=0,
        assigned_problems_count=0,
    )


@router.get("/by-code/{code}", response_model=ClassroomResponse)
def get_classroom_by_code(code: str, db: Session = Depends(get_db)):
    """Retrieve classroom details by enrollment code (case-insensitive)."""
    seed_default_classrooms_if_empty(db)
    code_normalized = code.strip().upper()
    classroom = db.query(Classroom).filter(Classroom.code == code_normalized).first()
    if not classroom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No classroom found with code '{code_normalized}'."
        )

    students_count = (
        db.query(func.count(func.distinct(CodingSession.student_id)))
        .filter(CodingSession.classroom_id == classroom.id)
        .scalar()
        or 0
    )
    problems_count = db.query(Assignment).filter(Assignment.classroom_id == classroom.id).count()

    return ClassroomResponse(
        id=classroom.id,
        name=classroom.name,
        code=classroom.code,
        teacher_id=classroom.teacher_id,
        teacher_name=classroom.teacher_name,
        department=classroom.department,
        description=classroom.description,
        created_at=classroom.created_at,
        active_students_count=students_count,
        assigned_problems_count=problems_count,
    )


@router.get("/{classroom_id}", response_model=ClassroomResponse)
def get_classroom(classroom_id: str, db: Session = Depends(get_db)):
    """Retrieve classroom by ID."""
    seed_default_classrooms_if_empty(db)
    classroom = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not classroom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Classroom '{classroom_id}' not found."
        )

    students_count = (
        db.query(func.count(func.distinct(CodingSession.student_id)))
        .filter(CodingSession.classroom_id == classroom.id)
        .scalar()
        or 0
    )
    problems_count = db.query(Assignment).filter(Assignment.classroom_id == classroom.id).count()

    return ClassroomResponse(
        id=classroom.id,
        name=classroom.name,
        code=classroom.code,
        teacher_id=classroom.teacher_id,
        teacher_name=classroom.teacher_name,
        department=classroom.department,
        description=classroom.description,
        created_at=classroom.created_at,
        active_students_count=students_count,
        assigned_problems_count=problems_count,
    )
