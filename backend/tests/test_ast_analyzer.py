"""Unit tests for Python AST Analysis Engine."""
import pytest
from app.services.ast_analyzer import ASTAnalyzer


def test_valid_program_ast_extraction():
    code = """
def calculate_evens(n: int) -> int:
    total = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            total += i
        else:
            continue
    return total

result = calculate_evens(10)
print(result)
"""
    analyzer = ASTAnalyzer(code)
    result = analyzer.analyze()

    assert result.is_syntax_valid is True
    assert result.syntax_error_details is None
    assert len(result.functions) == 1
    assert result.functions[0].name == "calculate_evens"
    assert result.functions[0].arguments == ["n"]
    assert result.functions[0].has_return is True
    assert result.functions[0].is_recursive is False

    # Check loop
    assert len(result.loops) == 1
    loop = result.loops[0]
    assert loop.loop_type == "for"
    assert loop.target_variable == "i"
    assert "range" in (loop.iterable_expression or "")
    assert loop.has_continue is True

    # Check condition
    assert len(result.conditions) >= 1
    cond = result.conditions[0]
    assert "%" in cond.test_expression
    assert cond.has_else is True

    # Check candidate concepts
    concept_keys = [c.concept_key for c in result.candidate_concepts]
    assert "loop-bounds-and-range-invariants" in concept_keys
    assert "modulo-conditional-filtering" in concept_keys
    assert "variable-accumulator-invariants" in concept_keys

    # Check complexity
    assert result.cyclomatic_complexity >= 3


def test_recursion_detection():
    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    analyzer = ASTAnalyzer(code)
    result = analyzer.analyze()

    assert result.is_syntax_valid is True
    assert len(result.functions) == 1
    fn = result.functions[0]
    assert fn.is_recursive is True

    concept_keys = [c.concept_key for c in result.candidate_concepts]
    assert "recursion-base-case-and-stack" in concept_keys


def test_syntax_error_diagnostic():
    # Missing colon after if
    code = """
def test_fn():
    if True
        print("oops")
"""
    analyzer = ASTAnalyzer(code)
    result = analyzer.analyze()

    assert result.is_syntax_valid is False
    assert result.syntax_error_details is not None
    assert "line" in result.syntax_error_details
    assert "diagnostic" in result.syntax_error_details
    assert result.syntax_error_details["line"] == 3


def test_unclosed_parenthesis_syntax_error():
    code = "print(1 + (2 * 3"
    analyzer = ASTAnalyzer(code)
    result = analyzer.analyze()

    assert result.is_syntax_valid is False
    assert result.syntax_error_details is not None
