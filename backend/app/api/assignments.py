"""Assignment management API routes for instructors to author coding problems."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.assignment import Assignment
from app.models.classroom import Classroom
from app.schemas.assignment import AssignmentCreate, AssignmentResponse

router = APIRouter(prefix="/assignments", tags=["Assignments"])


def _seed_defaults_if_empty(db: Session):
    count = db.query(Assignment).count()
    if count == 0:
        from app.seed_real_data import seed_real_data
        seed_real_data(db=db, clean=False)


@router.get("", response_model=List[AssignmentResponse])
def list_assignments(
    classroom_id: Optional[str] = Query(None, description="Filter assignments by target classroom ID"),
    teacher_id: Optional[str] = Query(None, description="Filter assignments authored by this teacher identifier"),
    created_by: Optional[str] = Query(None, description="Filter assignments authored by this teacher name"),
    db: Session = Depends(get_db),
):
    """List teacher-assigned coding problems, optionally filtered by student's classroom or authoring teacher."""
    _seed_defaults_if_empty(db)
    query = db.query(Assignment)

    # Filter by specific instructor if requested (so instructors only see questions published by themselves)
    if teacher_id or created_by:
        conditions = []
        if teacher_id:
            conditions.append(Assignment.teacher_id == teacher_id)
            conditions.append(Assignment.created_by == teacher_id)
        if created_by:
            conditions.append(Assignment.created_by == created_by)
            conditions.append(Assignment.created_by.ilike(f"%{created_by}%"))
        query = query.filter(or_(*conditions))

    if classroom_id:
        # Include assignments specifically created for this classroom OR open/global assignments (classroom_id is None)
        query = query.filter(
            or_(
                Assignment.classroom_id == classroom_id,
                Assignment.classroom_id.is_(None),
            )
        )
    return query.order_by(Assignment.created_at.desc()).all()


@router.post("", response_model=AssignmentResponse, status_code=status.HTTP_201_CREATED)
def create_assignment(payload: AssignmentCreate, db: Session = Depends(get_db)):
    """Create a new coding assignment authored by a teacher for a classroom group."""
    classroom_name = payload.classroom_name
    if payload.classroom_id and not classroom_name:
        classroom = db.query(Classroom).filter(Classroom.id == payload.classroom_id).first()
        if classroom:
            classroom_name = classroom.name

    assignment = Assignment(
        title=payload.title,
        description=payload.description,
        starter_code=payload.starter_code,
        difficulty=payload.difficulty,
        concept_tags=payload.concept_tags,
        created_by=payload.created_by,
        teacher_id=payload.teacher_id,
        classroom_id=payload.classroom_id,
        classroom_name=classroom_name,
        sample_input=payload.sample_input,
        sample_output=payload.sample_output,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


@router.get("/{assignment_id}", response_model=AssignmentResponse)
def get_assignment(assignment_id: str, db: Session = Depends(get_db)):
    """Retrieve single assignment details."""
    item = db.query(Assignment).filter(Assignment.id == assignment_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return item
