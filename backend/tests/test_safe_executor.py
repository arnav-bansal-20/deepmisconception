"""Unit tests for Isolated Safe Execution Sandbox."""
import pytest
from app.services.safe_executor import SafeExecutor


def test_safe_execution_valid_code():
    code = """
total = sum([1, 2, 3, 4, 5])
print(f"Result: {total}")
"""
    executor = SafeExecutor(timeout_seconds=2.0)
    res = executor.execute(code)

    assert res.success is True
    assert "Result: 15" in res.stdout
    assert res.exit_code == 0
    assert res.sandbox_used is True
    assert res.execution_time_ms > 0.0


def test_safe_execution_runtime_error():
    code = """
def divide_by_zero():
    return 10 / 0

divide_by_zero()
"""
    executor = SafeExecutor(timeout_seconds=2.0)
    res = executor.execute(code)

    assert res.success is False
    assert "ZeroDivisionError" in res.stderr
    assert res.error_type == "ZeroDivisionError"


def test_safe_execution_infinite_loop_timeout():
    code = """
while True:
    pass
"""
    executor = SafeExecutor(timeout_seconds=1.0)
    res = executor.execute(code)

    assert res.success is False
    assert res.error_type == "TimeoutError"
    assert "timed out" in res.stderr
    assert res.exit_code == 124


def test_safe_execution_security_guard_blocked_import():
    code = """
import subprocess
subprocess.run(["ls"])
"""
    executor = SafeExecutor(timeout_seconds=2.0)
    res = executor.execute(code)

    assert res.success is False
    assert "PermissionError" in res.stderr or "restricted in the student sandbox" in res.stderr
    assert res.error_type == "SecurityPermissionError"


def test_safe_execution_security_guard_blocked_os_system():
    code = """
import os
os.system("echo hacked")
"""
    executor = SafeExecutor(timeout_seconds=2.0)
    res = executor.execute(code)

    assert res.success is False
    assert "PermissionError" in res.stderr or "Security Error" in res.stderr


def test_safe_execution_environment_sanitization(monkeypatch):
    """Verify student process is isolated from host API keys, secrets, and credentials."""
    monkeypatch.setenv("SECRET_API_TOKEN_12345", "top_secret_credentials")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_mock_api_key_abc")

    code = """
import os
secret = os.environ.get("SECRET_API_TOKEN_12345", "NOT_FOUND")
gemini = os.environ.get("GEMINI_API_KEY", "NOT_FOUND")
print(f"Secret: {secret}, Gemini: {gemini}")
"""
    executor = SafeExecutor(timeout_seconds=2.0)
    res = executor.execute(code)

    assert res.success is True
    assert "Secret: NOT_FOUND" in res.stdout
    assert "Gemini: NOT_FOUND" in res.stdout
    assert "top_secret_credentials" not in res.stdout


def test_safe_execution_output_truncation():
    """Verify output buffer is capped at 64KB preventing runaway stdout memory leaks."""
    code = """
print("X" * 100000)
"""
    executor = SafeExecutor(timeout_seconds=2.0, max_output_bytes=1024)
    res = executor.execute(code)

    assert len(res.stdout) <= 2048
    assert "Output truncated" in res.stdout


def test_safe_execution_disabled_mode(monkeypatch):
    """Verify execution can be disabled by security policy while reporting static analysis mode."""
    executor = SafeExecutor(timeout_seconds=2.0)
    executor.sandbox_enabled = False

    res = executor.execute("print('hello')")
    assert res.success is False
    assert res.sandbox_used is False
    assert res.static_analysis_only is True
    assert res.error_type == "SandboxDisabled"
    assert "disabled by server security policy" in res.stderr


def test_ast_analysis_works_when_execution_disabled():
    """Verify AST analysis functions independently even when execution sandbox is disabled."""
    from app.services.ast_analyzer import ASTAnalyzer

    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    # 1. AST analysis succeeds
    analyzer = ASTAnalyzer(code)
    ast_res = analyzer.analyze()
    assert ast_res.is_syntax_valid is True
    assert ast_res.cyclomatic_complexity >= 2
    assert len(ast_res.functions) == 1
    assert ast_res.functions[0].name == "factorial"

    # 2. Execution disabled
    executor = SafeExecutor(timeout_seconds=2.0)
    executor.sandbox_enabled = False
    run_res = executor.execute(code)
    assert run_res.sandbox_used is False
    assert run_res.static_analysis_only is True


def test_process_group_termination_handles_clean_exit():
    """Verify termination handles normal exits cleanly without dangling exception."""
    executor = SafeExecutor(timeout_seconds=2.0)
    res = executor.execute("import sys\nsys.exit(0)")
    assert res.exit_code == 0
    assert res.success is True
