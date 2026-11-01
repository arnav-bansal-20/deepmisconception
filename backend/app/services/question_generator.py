"""Dynamic Viva Question Generation Service.

Synthesizes student source code, AST inspection results, and verified execution traces
to dynamically formulate targeted conceptual viva questions with rubrics and misconception indicators.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.services.llm_provider import get_llm_provider, BaseLLMProvider
from app.schemas.ast_analysis import ASTAnalysisResult

logger = logging.getLogger(__name__)


class GeneratedVivaQuestion(BaseModel):
    """Schema for a dynamically generated viva question."""
    question_text: str
    focal_code: Optional[str] = None
    concept: str
    difficulty: str = "medium"
    expected_reasoning: str
    rubric: Dict[str, Any] = Field(default_factory=dict)
    misconception_indicators: List[str] = Field(default_factory=list)


class QuestionGenerator:
    """Generates code-grounded viva questions dynamically using an LLM provider."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or get_llm_provider()

    def generate_questions(
        self,
        code: str,
        ast_result: ASTAnalysisResult,
        execution_output: Optional[str] = None,
        count: int = 3
    ) -> List[GeneratedVivaQuestion]:
        """Generate structured viva questions from student code and execution trace."""
        system_instruction = (
            "You are an expert computer science oral examiner and pedagogy researcher in the DeepMisconception system. "
            "Your task is to generate diagnostic viva questions directly grounded in the student's actual Python source code. "
            "Never ask generic trivia. Every question must test whether the student genuinely understands what happens during "
            "execution, how variables change, why loops and conditions behave the way they do, and whether they can detect "
            "subtle bugs or off-by-one boundaries. "
            "You must output valid JSON with a single key 'questions' containing a list of question objects matching this schema:\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "question_text": "Clear, direct question asking student to explain the mechanism",\n'
            '      "focal_code": "Exact 1-4 line snippet from student code referenced",\n'
            '      "concept": "concept-slug (e.g. variable-accumulator, loop-invariants, recursion-base-case)",\n'
            '      "difficulty": "easy" | "medium" | "hard",\n'
            '      "expected_reasoning": "Detailed, correct mechanistic explanation",\n'
            '      "rubric": {"criterion_1": "expectation", "criterion_2": "expectation"},\n'
            '      "misconception_indicators": ["Common misconception 1", "Common misconception 2"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        # Summarize AST constructs for LLM prompt
        ast_summary = {
            "functions": [f.name for f in ast_result.functions],
            "loops": [f"{l.loop_type} loop on line {l.line_number} (range: {l.iterable_expression})" for l in ast_result.loops],
            "conditions": [c.test_expression for c in ast_result.conditions],
            "candidate_concepts": [c.concept_name for c in ast_result.candidate_concepts],
            "cyclomatic_complexity": ast_result.cyclomatic_complexity,
        }

        user_prompt = (
            f"Generate {count} distinct viva questions for the following student Python submission.\n\n"
            f"=== STUDENT SOURCE CODE ===\n{code}\n\n"
            f"=== VERIFIED EXECUTION OUTPUT ===\n{execution_output or '[No output / not executed]'}\n\n"
            f"=== AST STRUCTURAL ANALYSIS ===\n{json.dumps(ast_summary, indent=2)}\n\n"
            "Formulate questions probing:\n"
            "1. Output logic and state mutations across iterations\n"
            "2. Control flow decisions (conditions, break/continue)\n"
            "3. Boundary invariants or edge cases\n"
        )

        try:
            response_json = self.provider.generate_json(system_instruction, user_prompt)
            raw_questions = response_json.get("questions", [])
            questions: List[GeneratedVivaQuestion] = []

            for q in raw_questions:
                try:
                    questions.append(GeneratedVivaQuestion(**q))
                except Exception as val_err:
                    logger.warning(f"Skipping malformed question object: {val_err}")

            if questions:
                return questions[:count]

        except Exception as e:
            logger.error(f"Error during LLM question generation: {e}")

        # Fallback to AST candidate concepts if LLM produced no valid questions
        return self._generate_ast_fallback_questions(code, ast_result, count)

    def _generate_ast_fallback_questions(
        self,
        code: str,
        ast_result: ASTAnalysisResult,
        count: int
    ) -> List[GeneratedVivaQuestion]:
        """Construct deterministic questions directly from AST candidate concepts."""
        questions: List[GeneratedVivaQuestion] = []

        for candidate in ast_result.candidate_concepts[:count]:
            q_text = (
                f"In your code at line {candidate.line_number}, explain how the construct "
                f"'{candidate.concept_name}' operates during execution and what impact it has on program state."
            )
            questions.append(
                GeneratedVivaQuestion(
                    question_text=q_text,
                    focal_code=candidate.focal_code,
                    concept=candidate.concept_key,
                    difficulty=candidate.suggested_difficulty,
                    expected_reasoning=f"Student should describe how {candidate.concept_name} behaves and affects state.",
                    rubric={
                        "identifies_mechanism": "Explains the underlying syntactic and runtime mechanism",
                        "describes_state_change": "Traces how values change across execution passes"
                    },
                    misconception_indicators=[
                        f"Misunderstanding the role of {candidate.concept_name}",
                        "Inverting condition or skipping logic"
                    ]
                )
            )

        if not questions:
            # Fallback for minimal/blank code
            questions.append(
                GeneratedVivaQuestion(
                    question_text="Walk through the execution flow of your program line by line and explain how the variables evolve.",
                    focal_code=code[:100],
                    concept="general-execution-flow",
                    difficulty="easy",
                    expected_reasoning="Step-by-step description of line executions and variable state updates.",
                    rubric={"step_by_step_trace": "Mentions variables and statement order"},
                    misconception_indicators=["Confusing declaration with execution time"]
                )
            )

        return questions
