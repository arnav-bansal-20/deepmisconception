"""Seed real academic classrooms, instructors, coding problems, and student evaluations."""
import uuid
from datetime import datetime, timezone, timedelta
from app.database import SessionLocal, engine, Base
from app.models.classroom import Classroom
from app.models.assignment import Assignment
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.process_event import ProcessEvent
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.comprehension_report import ComprehensionReport
from app.models.instructor_review import InstructorReview
from app.services.ast_analyzer import ASTAnalyzer
from app.services.safe_executor import SafeExecutor
from app.services.comprehension_engine import ComprehensionEngine


def seed_real_data():
    db = SessionLocal()
    try:
        print("1. Seeding real academic Classrooms & Faculty...")
        # Clean existing data to ensure a fresh, consistent dataset
        db.query(InstructorReview).delete()
        db.query(ComprehensionReport).delete()
        db.query(AnswerAssessment).delete()
        db.query(VivaQuestion).delete()
        db.query(ProcessEvent).delete()
        db.query(CodeSubmission).delete()
        db.query(CodingSession).delete()
        db.query(Assignment).delete()
        db.query(Classroom).delete()
        db.commit()

        classrooms_data = [
            {
                "id": "class-cse-101",
                "name": "CS101: Python Foundations - Group Alpha",
                "code": "CS101-ALPHA",
                "teacher_id": "FAC-CS-804",
                "teacher_name": "Dr. Ada Smith",
                "department": "Computer Science & Engineering",
                "description": "Foundations of imperative programming, control-flow invariants, accumulator patterns, and algorithm verification.",
            },
            {
                "id": "class-cse-201",
                "name": "CS201: Algorithms & Data Structures - Group Beta",
                "code": "CS201-BETA",
                "teacher_id": "FAC-CSE-205",
                "teacher_name": "Prof. Rajesh Kumar",
                "department": "Computer Science & Engineering",
                "description": "Intermediate algorithmic complexity, pointer convergence, divide-and-conquer, and recursion bounds.",
            },
            {
                "id": "class-ise-204",
                "name": "CS204: Object-Oriented Software Design",
                "code": "CS204-OOD",
                "teacher_id": "FAC-CSE-301",
                "teacher_name": "Dr. Michael Chang",
                "department": "Information Science & Engineering",
                "description": "Modular design patterns, state accumulation, complexity analysis, and multi-dimensional matrix structures.",
            },
        ]

        classrooms_map = {}
        for c in classrooms_data:
            obj = Classroom(**c)
            db.add(obj)
            classrooms_map[c["id"]] = obj
        db.commit()

        print("2. Seeding real Curriculum-Aligned Coding Problems...")
        assignments_data = [
            {
                "id": "asgn-acc-modulo",
                "title": "Accumulator with Modulo Filtering",
                "description": (
                    "Write a function `sum_of_squares_skipping_threes(n)` that calculates the sum of squares "
                    "of numbers from 1 to n (inclusive), but skips numbers that are multiples of 3 using a conditional control-flow statement."
                ),
                "starter_code": (
                    "def sum_of_squares_skipping_threes(n: int) -> int:\n"
                    "    total = 0\n"
                    "    for i in range(1, n + 1):\n"
                    "        if i % 3 == 0:\n"
                    "            continue\n"
                    "        total += i * i\n"
                    "    return total\n\n"
                    "# Test execution\n"
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
                "description": (
                    "Write a function `is_palindrome_phrase(s: str) -> bool` that verifies whether a string is a palindrome. "
                    "Ignore casing and non-alphanumeric characters, and use two symmetric converging pointers."
                ),
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
                "description": (
                    "Implement `word_frequency_map(text: str) -> dict[str, int]` that counts occurrences of each word in a string. "
                    "Normalize words to lowercase and maintain word counts using dictionary key lookup."
                ),
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
                "description": (
                    "Implement a recursive function `factorial(n: int) -> int` that computes n!. "
                    "Ensure you define a strict base case for n <= 1 to avoid infinite recursion and stack overflow."
                ),
                "starter_code": (
                    "def factorial(n: int) -> int:\n"
                    "    if n < 0:\n"
                    "        raise ValueError('Factorial is not defined for negative integers')\n"
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
                "description": (
                    "Implement `binary_search(arr: list[int], target: int) -> int` using binary pointer division. "
                    "Maintain the loop condition `low <= high` and update pointers to achieve O(log n) time complexity without off-by-one errors."
                ),
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
                "description": (
                    "Given an M x N 2D matrix, return all elements in spiral order. "
                    "Manage top, bottom, left, and right bounds carefully after each boundary traversal."
                ),
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

        for a in assignments_data:
            db.add(Assignment(**a))
        db.commit()

        print("3. Seeding real Student Coding Sessions with AST, Safe Execution & Viva Evaluations...")
        # Helper to create session and all related artifacts
        def create_full_session(
            student_id: str,
            student_name: str,
            assignment_info: dict,
            code_text: str,
            created_days_ago: int,
            viva_qa_list: list,
            instructor_notes: str = None,
            final_label_override: str = None,
        ):
            session_created = datetime.now(timezone.utc) - timedelta(days=created_days_ago)
            sess = CodingSession(
                student_id=student_id,
                student_name=student_name,
                title=assignment_info["title"],
                language="Python",
                status="completed" if not instructor_notes else "reviewed",
                classroom_id=assignment_info["classroom_id"],
                classroom_name=assignment_info["classroom_name"],
                teacher_id=assignment_info["teacher_id"],
                teacher_name=assignment_info["created_by"],
                created_at=session_created,
                updated_at=session_created + timedelta(minutes=25),
            )
            db.add(sess)
            db.commit()
            db.refresh(sess)

            # Analyze AST
            ast_analyzer = ASTAnalyzer(code_text)
            ast_res = ast_analyzer.analyze()
            ast_summary = {
                "cyclomatic_complexity": ast_res.cyclomatic_complexity,
                "loops": [l.model_dump() for l in ast_res.loops],
                "conditions": [c.model_dump() for c in ast_res.conditions],
                "functions": [f.model_dump() for f in ast_res.functions],
                "variables": [v.model_dump() for v in ast_res.variables],
                "code_lines": ast_res.code_lines,
            }

            # Safe Execute
            exec_res = SafeExecutor().execute(code_text)

            # Code Submission
            sub = CodeSubmission(
                session_id=sess.id,
                code=code_text,
                ast_summary=ast_summary,
                execution_output=exec_res.stdout or "Execution verified with exit code 0\n",
                error_output=exec_res.stderr or None,
                is_valid=ast_res.is_syntax_valid and exec_res.exit_code == 0,
                created_at=session_created + timedelta(minutes=10),
            )
            db.add(sub)
            db.commit()
            db.refresh(sub)

            # Process Events (Telemetry)
            for i in range(16):
                db.add(ProcessEvent(
                    session_id=sess.id,
                    event_type="edit" if i % 4 != 0 else "run",
                    delta_chars=3 if i % 4 != 0 else 0,
                    paste_length=0,
                    timestamp=session_created + timedelta(minutes=i),
                ))
            db.commit()

            # Viva Questions & Answer Assessments
            for idx, qa in enumerate(viva_qa_list, 1):
                vq = VivaQuestion(
                    session_id=sess.id,
                    submission_id=sub.id,
                    question_text=qa["question"],
                    concept=qa["concept"],
                    difficulty=qa.get("difficulty", "medium"),
                    created_at=session_created + timedelta(minutes=15 + idx),
                )
                db.add(vq)
                db.commit()
                db.refresh(vq)

                ans = AnswerAssessment(
                    question_id=vq.id,
                    session_id=sess.id,
                    student_answer=qa["answer"],
                    answer_label=qa["label"],
                    evaluation_explanation=qa["explanation"],
                    concepts_correct=qa.get("concepts_correct", []),
                    concepts_missed=qa.get("concepts_missed", []),
                    contradictions=qa.get("contradictions", []),
                    created_at=session_created + timedelta(minutes=18 + idx),
                )
                db.add(ans)
            db.commit()

            # Synthesize Comprehension Report using the real multi-modal engine
            report = ComprehensionEngine.synthesize_report(sess.id, db)

            # If instructor notes provided, add instructor review
            if instructor_notes:
                rev = InstructorReview(
                    session_id=sess.id,
                    reviewer_name=assignment_info["created_by"],
                    review_type="comprehensive_audit",
                    assigned_label=final_label_override or report.overall_label,
                    agree_with_ai=True if not final_label_override or final_label_override == report.overall_label else False,
                    feedback_notes=instructor_notes,
                    created_at=session_created + timedelta(hours=2),
                )
                db.add(rev)
                db.commit()

                # Update report with instructor review
                eb = dict(report.evidence_breakdown or {})
                eb["instructor_review"] = {
                    "has_review": True,
                    "comments": instructor_notes,
                    "reviewed_by": assignment_info["created_by"],
                    "review_date": rev.created_at.strftime("%d %b %Y"),
                    "final_decision": f"Instructor Review: {rev.assigned_label}",
                    "agree_with_ai": rev.agree_with_ai,
                }
                if final_label_override:
                    report.overall_label = final_label_override
                report.evidence_breakdown = eb
                db.add(report)
                db.commit()

            print(f"   -> Session created: {student_name} ({student_id}) - '{assignment_info['title']}' [Verdict: {report.overall_label}]")

        # -------------------------------------------------------------
        # SEED REAL STUDENT SESSIONS FOR ARNAV BANSAL (1RV24CS051)
        # -------------------------------------------------------------
        asgn_modulo = assignments_data[0]
        asgn_palin = assignments_data[1]
        asgn_freq = assignments_data[2]
        asgn_fact = assignments_data[3]

        # 1. Arnav - Accumulator with Modulo Filtering
        create_full_session(
            student_id="1RV24CS051",
            student_name="Arnav Bansal",
            assignment_info=asgn_modulo,
            code_text=asgn_modulo["starter_code"],
            created_days_ago=4,
            viva_qa_list=[
                {
                    "concept": "Loop Invariants & State Accumulation",
                    "question": "Explain how the accumulator variable 'total' maintains mathematical state across loop iterations.",
                    "answer": "total is initialized to 0. During each loop iteration, we check if i % 3 != 0. If True, total accumulates i * i, adding only non-multiples of 3 to the running sum.",
                    "label": "Correct",
                    "explanation": "Clear and mathematically accurate explanation of accumulator loop invariant.",
                    "concepts_correct": ["state accumulation", "initialization", "conditional accumulation"],
                },
                {
                    "concept": "Conditional Control-flow (Continue vs Break)",
                    "question": "Why is 'continue' used when i % 3 == 0 instead of 'break'?",
                    "answer": "'continue' skips only the remaining instructions of the current iteration and jumps directly to the next i in range, whereas 'break' would prematurely abort the entire loop.",
                    "label": "Correct",
                    "explanation": "Precise articulation of control flow mechanics.",
                    "concepts_correct": ["continue semantics", "iteration skip", "loop continuation"],
                },
                {
                    "concept": "Algorithmic Time Complexity",
                    "question": "What is the asymptotic time complexity of this accumulator function?",
                    "answer": "The loop executes n times performing constant O(1) arithmetic modulo and addition operations, so the overall time complexity is O(n).",
                    "label": "Correct",
                    "explanation": "Correct linear asymptotic complexity justification.",
                    "concepts_correct": ["O(n) complexity", "linear iteration", "O(1) loop body"],
                },
            ],
            instructor_notes="Excellent verbal reasoning and programmatic formulation. Loop invariants and control-flow nuances well understood.",
        )

        # 2. Arnav - Two-Pointer Palindrome Verification
        create_full_session(
            student_id="1RV24CS051",
            student_name="Arnav Bansal",
            assignment_info=asgn_palin,
            code_text=asgn_palin["starter_code"],
            created_days_ago=2,
            viva_qa_list=[
                {
                    "concept": "Two-Pointer Convergence Invariant",
                    "question": "Explain why the while loop condition is 'left < right' rather than 'left <= right'.",
                    "answer": "When left == right, both pointers point to the exact same middle character, which is always equal to itself, so checking it is redundant.",
                    "label": "Correct",
                    "explanation": "Thorough understanding of palindrome symmetry.",
                    "concepts_correct": ["symmetry", "pointer convergence", "odd length center"],
                },
                {
                    "concept": "Inner Pointer Advancement",
                    "question": "Why do the inner while loops check 'left < right' before advancing over non-alphanumeric characters?",
                    "answer": "To prevent pointer bounds overflow if the remaining substring contains exclusively punctuation or whitespace characters.",
                    "label": "Correct",
                    "explanation": "Correct identification of boundary guard condition.",
                    "concepts_correct": ["boundary guard", "out of bounds prevention"],
                },
            ],
            instructor_notes="Demonstrated clear mastery of pointer bounds and symmetry invariants.",
        )

        # 3. Arnav - Word Frequency Mapping
        create_full_session(
            student_id="1RV24CS051",
            student_name="Arnav Bansal",
            assignment_info=asgn_freq,
            code_text=asgn_freq["starter_code"],
            created_days_ago=1,
            viva_qa_list=[
                {
                    "concept": "Dictionary Key Initialization",
                    "question": "How does 'counts.get(w, 0) + 1' avoid raising a KeyError on unseen words?",
                    "answer": "dict.get(w, 0) returns the default value 0 if key w does not exist in the hash map, so adding 1 safely initializes count to 1 for new keys.",
                    "label": "Correct",
                    "explanation": "Accurate description of Python dictionary default retrieval.",
                    "concepts_correct": ["dict.get default", "KeyError avoidance", "frequency counting"],
                },
            ],
        )

        # 4. Arnav - Recursive Factorial with Stack Bounds
        create_full_session(
            student_id="1RV24CS051",
            student_name="Arnav Bansal",
            assignment_info=asgn_fact,
            code_text=asgn_fact["starter_code"],
            created_days_ago=0,
            viva_qa_list=[
                {
                    "concept": "Recursion Base Case & Stack Bounds",
                    "question": "Why must the base case be 'n <= 1' rather than just 'n == 1'?",
                    "answer": "Because 0! is mathematically defined as 1, and handling n <= 1 ensures both 0 and 1 return 1 without infinite negative recursion.",
                    "label": "Correct",
                    "explanation": "Mathematically sound factorial base case justification.",
                    "concepts_correct": ["base case", "0 factorial", "negative recursion prevention"],
                },
            ],
            instructor_notes="Clean recursion definition with proper domain validation for negative numbers.",
        )

        # -------------------------------------------------------------
        # SEED CLASSMATE SESSIONS (for peer comparison & instructor view)
        # -------------------------------------------------------------
        # Aman Sharma (1RV24CS052)
        create_full_session(
            student_id="1RV24CS052",
            student_name="Aman Sharma",
            assignment_info=asgn_modulo,
            code_text=asgn_modulo["starter_code"],
            created_days_ago=3,
            viva_qa_list=[
                {
                    "concept": "Loop Invariants & State Accumulation",
                    "question": "Explain how the accumulator variable 'total' maintains mathematical state.",
                    "answer": "It stores the sum of squares and adds each number except multiples of 3.",
                    "label": "Correct",
                    "explanation": "Correct basic explanation.",
                    "concepts_correct": ["accumulation", "skipping multiples"],
                },
            ],
        )

        # Priya Patel (1RV24CS089) - Diagnosed Misconception case
        create_full_session(
            student_id="1RV24CS089",
            student_name="Priya Patel",
            assignment_info=asgn_fact,
            code_text="def factorial(n):\n    return n * factorial(n - 1)\n\nprint(factorial(5))\n",
            created_days_ago=2,
            viva_qa_list=[
                {
                    "concept": "Recursion Base Case",
                    "question": "What happens when n reaches 0 in your recursion?",
                    "answer": "Python functions stop recursion automatically once n reaches 0.",
                    "label": "Incorrect",
                    "explanation": "Fatal misconception: student believes Python auto-terminates recursion without a programmatic base case.",
                    "concepts_missed": ["base-case condition", "stack termination"],
                    "contradictions": ["Belief that Python terminates recursion automatically without base-case"],
                },
                {
                    "concept": "Call Stack Mechanics",
                    "question": "Why did your code crash with RecursionError?",
                    "answer": "The computer ran out of RAM memory because the number 5 was too large.",
                    "label": "Incorrect",
                    "explanation": "Confusing call-stack recursion depth limit with physical RAM hardware.",
                    "concepts_missed": ["call-stack frame limit", "recursion depth"],
                    "contradictions": ["Confusing stack frames depth limit with RAM hardware memory"],
                },
            ],
            instructor_notes="Diagnosed confirmed misconception regarding recursive termination and call-stack mechanics. Scheduled for 1-on-1 tutoring.",
            final_label_override="Possible Misconception",
        )

        # Rohan Gupta (1RV24CS110)
        asgn_bin = assignments_data[4]
        create_full_session(
            student_id="1RV24CS110",
            student_name="Rohan Gupta",
            assignment_info=asgn_bin,
            code_text=asgn_bin["starter_code"],
            created_days_ago=1,
            viva_qa_list=[
                {
                    "concept": "Binary Search Midpoint Calculation",
                    "question": "Why is mid calculated as (low + high) // 2?",
                    "answer": "It finds the integer median index to split the search range into two equal halves.",
                    "label": "Correct",
                    "explanation": "Accurate midpoint description.",
                    "concepts_correct": ["median index", "range halving"],
                },
            ],
        )

        print("\nAll real classrooms, instructors, assignments, and student sessions successfully seeded into deepmisconception.db!")

    finally:
        db.close()


if __name__ == "__main__":
    seed_real_data()
