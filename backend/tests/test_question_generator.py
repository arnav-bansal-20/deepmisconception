"""Unit tests for Dynamic AI Viva Question Generator."""
import pytest
from app.services.ast_analyzer import ASTAnalyzer
from app.services.question_generator import QuestionGenerator, GeneratedVivaQuestion
from app.services.llm_provider import MockLLMProvider


def test_question_generation_from_ast_and_code():
    code = """
def sum_of_squares_skipping_threes(n: int) -> int:
    total = 0
    for i in range(1, n + 1):
        if i % 3 == 0:
            continue
        total += i * i
    return total
"""
    analyzer = ASTAnalyzer(code)
    ast_result = analyzer.analyze()

    generator = QuestionGenerator(provider=MockLLMProvider())
    questions = generator.generate_questions(
        code=code,
        ast_result=ast_result,
        execution_output="Computed total: 35",
        count=3
    )

    assert len(questions) >= 1
    for q in questions:
        assert isinstance(q, GeneratedVivaQuestion)
        assert len(q.question_text) > 10
        assert q.concept != ""
        assert q.difficulty in ("easy", "medium", "hard")
        assert len(q.expected_reasoning) > 10
        assert isinstance(q.rubric, dict)
        assert isinstance(q.misconception_indicators, list)


def test_question_generator_fallback():
    # Code with a while loop
    code = """
count = 10
while count > 0:
    count -= 1
"""
    analyzer = ASTAnalyzer(code)
    ast_result = analyzer.analyze()

    # Pass an empty provider to trigger AST fallback
    class FailingProvider(MockLLMProvider):
        def generate_json(self, s, u):
            raise RuntimeError("Simulated API failure")

    generator = QuestionGenerator(provider=FailingProvider())
    questions = generator.generate_questions(
        code=code,
        ast_result=ast_result,
        count=2
    )

    assert len(questions) >= 1
    assert questions[0].question_text != ""


def test_dynamic_code_grounded_questions_for_arbitrary_code():
    """Verify that different student programs dynamically produce questions tailored to their constructs."""
    code_prime = """def check_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
"""
    analyzer = ASTAnalyzer(code_prime)
    ast_result = analyzer.analyze()
    generator = QuestionGenerator(provider=MockLLMProvider())
    questions = generator.generate_questions(
        code=code_prime,
        ast_result=ast_result,
        execution_output="check_prime(17) -> True",
        count=3
    )

    assert len(questions) >= 1
    # Ensure questions are NOT hard-coded accumulator questions
    all_texts = " ".join(q.question_text.lower() for q in questions)
    assert "sum_of_squares" not in all_texts
    # Must reference the student's actual function or constructs
    assert any("check_prime" in q.question_text or "loop" in q.question_text or "line" in q.question_text for q in questions)
    assert any(q.focal_code and ("check_prime" in q.focal_code or "range" in q.focal_code or "num" in q.focal_code) for q in questions)

