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
