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

DEFAULT_ASSIGNMENTS = [
    {
        "id": "asgn-acc-modulo",
        "title": "Accumulator with Modulo Filtering",
        "description": "Write a function `sum_of_squares_skipping_threes(n)` that calculates the sum of squares of numbers from 1 to n (inclusive), but skips numbers that are multiples of 3 using a conditional control-flow statement.",
        "starter_code": (
            "def sum_of_squares_skipping_threes(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 3 == 0:\n"
            "            continue\n"
            "        total += i * i\n"
            "    return total\n\n"
            "# Test with n = 5\n"
            "result = sum_of_squares_skipping_threes(5)\n"
            "print(f'Computed total: {result}')\n"
        ),
        "difficulty": "medium",
        "concept_tags": ["loops", "modulo", "accumulator", "continue"],
        "created_by": "Dr. Ada Smith",
        "teacher_id": "FAC-CS-804",
        "sample_input": "n = 5",
        "sample_output": "Computed total: 26",
        "classroom_id": "class-cse-101",
        "classroom_name": "CS101: Python Foundations - Group Alpha",
    },
    {
        "id": "asgn-palindrome-ptr",
        "title": "Two-Pointer Palindrome Verification",
        "description": "Write a function `is_palindrome_phrase(s: str) -> bool` that verifies whether a string is a palindrome, ignoring non-alphanumeric characters and case, using converging symmetric pointers.",
        "starter_code": (
            "def is_palindrome_phrase(s: str) -> bool:\n"
            "    left = 0\n"
            "    right = len(s) - 1\n"
            "    while left < right:\n"
            "        while left < right and not s[left].isalnum():\n"
            "            left += 1\n"
            "        while left < right and not s[right].isalnum():\n"
            "            right -= 1\n"
            "        if s[left].lower() != s[right].lower():\n"
            "            return False\n"
            "        left += 1\n"
            "        right -= 1\n"
            "    return True\n\n"
            "test_phrase = 'A man, a plan, a canal: Panama'\n"
            "print(f'Is palindrome: {is_palindrome_phrase(test_phrase)}')\n"
        ),
        "difficulty": "easy",
        "concept_tags": ["two-pointers", "string-manipulation", "invariants", "boundary-check"],
        "created_by": "Dr. Ada Smith",
        "teacher_id": "FAC-CS-804",
        "sample_input": "s = 'A man, a plan, a canal: Panama'",
        "sample_output": "Is palindrome: True",
        "classroom_id": "class-cse-101",
        "classroom_name": "CS101: Python Foundations - Group Alpha",
    },
    {
        "id": "asgn-word-frequency",
        "title": "Word Frequency Mapping via Dictionary",
        "description": "Implement `word_frequency_map(text: str) -> dict[str, int]` that counts occurrences of each word in a string using dictionary key lookup.",
        "starter_code": (
            "def word_frequency_map(text: str) -> dict[str, int]:\n"
            "    counts = {}\n"
            "    words = text.lower().split()\n"
            "    for w in words:\n"
            "        counts[w] = counts.get(w, 0) + 1\n"
            "    return counts\n\n"
            "sample = 'apple banana apple orange banana apple'\n"
            "print(f'Frequency map: {word_frequency_map(sample)}')\n"
        ),
        "difficulty": "easy",
        "concept_tags": ["dictionaries", "strings", "hash-map", "counting"],
        "created_by": "Dr. Ada Smith",
        "teacher_id": "FAC-CS-804",
        "sample_input": "text = 'apple banana apple orange banana apple'",
        "sample_output": "Frequency map: {'apple': 3, 'banana': 2, 'orange': 1}",
        "classroom_id": "class-cse-101",
        "classroom_name": "CS101: Python Foundations - Group Alpha",
    },
    {
        "id": "asgn-factorial-bounds",
        "title": "Recursive Factorial & Call Stack Bounds",
        "description": "Implement a recursive function `factorial(n: int) -> int` that computes n!. Ensure you define a correct base case for n <= 1 to avoid a RecursionError.",
        "starter_code": (
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('Factorial is not defined for negative numbers')\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n\n"
            "print(f'Factorial of 5: {factorial(5)}')\n"
        ),
        "difficulty": "medium",
        "concept_tags": ["recursion", "base-case", "call-stack", "invariants"],
        "created_by": "Prof. Rajesh Kumar",
        "teacher_id": "FAC-CSE-205",
        "sample_input": "n = 5",
        "sample_output": "Factorial of 5: 120",
        "classroom_id": "class-cse-201",
        "classroom_name": "CS201: Algorithms & Data Structures - Group Beta",
    },
    {
        "id": "asgn-binary-search",
        "title": "Binary Search on Sorted Sequence",
        "description": "Implement `binary_search(arr: list[int], target: int) -> int` on a sorted list. Correctly manage low and high pointers without infinite looping or missing edge elements.",
        "starter_code": (
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    low = 0\n"
            "    high = len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n\n"
            "data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]\n"
            "print(f'Index of 23: {binary_search(data, 23)}')\n"
        ),
        "difficulty": "medium",
        "concept_tags": ["binary-search", "pointers", "loop-bounds", "invariants"],
        "created_by": "Prof. Rajesh Kumar",
        "teacher_id": "FAC-CSE-205",
        "sample_input": "arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91], target = 23",
        "sample_output": "Index of 23: 5",
        "classroom_id": "class-cse-201",
        "classroom_name": "CS201: Algorithms & Data Structures - Group Beta",
    },
    {
        "id": "asgn-matrix-spiral",
        "title": "Matrix Boundary Spiral Traversal",
        "description": "Given an M x N 2D matrix, return all elements in clockwise spiral order maintaining 4 strict boundary limits.",
        "starter_code": (
            "def spiral_order(matrix: list[list[int]]) -> list[int]:\n"
            "    if not matrix or not matrix[0]:\n"
            "        return []\n"
            "    res = []\n"
            "    top, bottom = 0, len(matrix) - 1\n"
            "    left, right = 0, len(matrix[0]) - 1\n"
            "    while top <= bottom and left <= right:\n"
            "        for col in range(left, right + 1):\n"
            "            res.append(matrix[top][col])\n"
            "        top += 1\n"
            "        for row in range(top, bottom + 1):\n"
            "            res.append(matrix[row][right])\n"
            "        right -= 1\n"
            "        if top <= bottom:\n"
            "            for col in range(right, left - 1, -1):\n"
            "                res.append(matrix[bottom][col])\n"
            "            bottom -= 1\n"
            "        if left <= right:\n"
            "            for row in range(bottom, top - 1, -1):\n"
            "                res.append(matrix[row][left])\n"
            "            left += 1\n"
            "    return res\n\n"
            "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"
            "print(f'Spiral traversal: {spiral_order(grid)}')\n"
        ),
        "difficulty": "hard",
        "concept_tags": ["matrix", "2d-arrays", "spiral-order", "boundary-invariants"],
        "created_by": "Dr. Michael Chang",
        "teacher_id": "FAC-CSE-301",
        "sample_input": "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
        "sample_output": "Spiral traversal: [1, 2, 3, 6, 9, 8, 7, 4, 5]",
        "classroom_id": "class-ise-204",
        "classroom_name": "CS204: Object-Oriented Software Design",
    },
]


def _seed_defaults_if_empty(db: Session):
    count = db.query(Assignment).count()
    if count == 0:
        for item in DEFAULT_ASSIGNMENTS:
            db.add(Assignment(**item))
        db.commit()


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
