"""Isolated and resource-bounded Python code execution sandbox.

Executes student code strictly outside the main API process using isolated
subprocesses with strict CPU limits, memory limits, output caps, and dangerous
API restrictions.
"""
import sys
import os
import time
import subprocess
import tempfile
import resource
from typing import Optional, Dict, Any
from app.config import settings
from app.schemas.submission import CodeRunResponse


class SafeExecutor:
    """Executes untrusted student code within an isolated, sandboxed subprocess."""

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        max_memory_mb: Optional[int] = None,
        max_output_bytes: int = 65536  # 64 KB
    ):
        self.timeout_seconds = timeout_seconds or settings.execution_timeout_seconds
        self.max_memory_mb = max_memory_mb or settings.execution_max_memory_mb
        self.max_output_bytes = max_output_bytes
        self.sandbox_enabled = settings.safe_sandbox_enabled

    def execute(self, code: str) -> CodeRunResponse:
        """Run student code in a sandboxed child process and capture results."""
        if not self.sandbox_enabled:
            return CodeRunResponse(
                success=False,
                stdout="",
                stderr="Code execution is disabled by server security policy.",
                exit_code=-1,
                execution_time_ms=0.0,
                sandbox_used=False,
                error_type="SandboxDisabled"
            )

        start_time = time.perf_counter()
        temp_file = None

        # Build runner wrapper that installs sandbox guards before executing user code
        sandboxed_code = self._wrap_code(code)

        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(sandboxed_code)
                temp_file = f.name

            # Pre-exec resource limiter for Unix/macOS child process
            def set_limits():
                # CPU time limit in seconds
                cpu_limit = max(1, int(self.timeout_seconds) + 1)
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
                except Exception:
                    pass

                # Virtual address space (memory) limit
                if hasattr(resource, "RLIMIT_AS"):
                    bytes_limit = self.max_memory_mb * 1024 * 1024
                    try:
                        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
                    except Exception:
                        pass

            # Launch Python subprocess with -I (isolated mode, ignores PYTHONPATH/user site)
            cmd = [sys.executable, "-I", "-u", temp_file]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=set_limits if os.name != "nt" else None
            )

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                # Truncate runaway output
                if len(stdout) > self.max_output_bytes:
                    stdout = stdout[:self.max_output_bytes] + "\n... [Output truncated: buffer exceeded 64KB]"
                if len(stderr) > self.max_output_bytes:
                    stderr = stderr[:self.max_output_bytes] + "\n... [Stderr truncated: buffer exceeded 64KB]"

                # Clean internal wrapper frames from stderr if present
                clean_stderr = self._sanitize_traceback(stderr, temp_file)

                is_success = proc.returncode == 0 and not clean_stderr
                error_type = None if is_success else self._classify_error(clean_stderr, proc.returncode)

                return CodeRunResponse(
                    success=is_success,
                    stdout=stdout,
                    stderr=clean_stderr,
                    exit_code=proc.returncode,
                    execution_time_ms=round(elapsed_ms, 2),
                    sandbox_used=True,
                    error_type=error_type
                )

            except subprocess.TimeoutExpired:
                # Forcefully terminate child process immediately
                proc.kill()
                try:
                    proc.communicate(timeout=0.5)
                except Exception:
                    pass
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return CodeRunResponse(
                    success=False,
                    stdout="",
                    stderr=(
                        f"Execution timed out after {self.timeout_seconds:.1f}s.\n"
                        "This typically indicates an infinite loop (e.g. while condition never becomes False) "
                        "or exponential computation."
                    ),
                    exit_code=124,
                    execution_time_ms=round(elapsed_ms, 2),
                    sandbox_used=True,
                    error_type="TimeoutError"
                )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CodeRunResponse(
                success=False,
                stdout="",
                stderr=f"Sandbox execution error: {str(e)}",
                exit_code=-1,
                execution_time_ms=round(elapsed_ms, 2),
                sandbox_used=True,
                error_type="SystemError"
            )

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    def _wrap_code(self, student_code: str) -> str:
        """Inject security preambles to disable high-risk standard library hooks."""
        preamble = (
            "# --- DeepMisconception Isolated Sandbox Preamble ---\n"
            "import sys\n"
            "\n"
            "# Guard against dangerous module imports\n"
            "BLOCKED_MODULES = {'subprocess', 'socket', 'shutil', 'pty', 'posix', 'ctypes'}\n"
            "orig_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__\n"
            "\n"
            "def safe_import(name, *args, **kwargs):\n"
            "    top_pkg = name.split('.')[0]\n"
            "    if top_pkg in BLOCKED_MODULES:\n"
            "        raise PermissionError(f'Security Error: module \"{name}\" is restricted in the student sandbox.')\n"
            "    return orig_import(name, *args, **kwargs)\n"
            "\n"
            "if isinstance(__builtins__, dict):\n"
            "    __builtins__['__import__'] = safe_import\n"
            "else:\n"
            "    __builtins__.__import__ = safe_import\n"
            "\n"
            "# Disable dangerous OS methods while keeping benign ones (os.path)\n"
            "try:\n"
            "    import os\n"
            "    for dangerous_attr in ['system', 'popen', 'spawn', 'fork', 'kill', 'remove', 'rmdir']:\n"
            "        if hasattr(os, dangerous_attr):\n"
            "            def blocked(*args, **kwargs):\n"
            "                raise PermissionError('Security Error: OS mutation functions are disabled in the student sandbox.')\n"
            "            setattr(os, dangerous_attr, blocked)\n"
            "except Exception:\n"
            "    pass\n"
            "\n"
            "# --- Student Source Code Follows ---\n"
        )
        return preamble + student_code

    def _sanitize_traceback(self, stderr: str, temp_file_path: str) -> str:
        """Clean out temp file paths and wrapper preamble lines from traceback."""
        if not stderr:
            return ""
        # Replace temp file name with <student_code.py>
        cleaned = stderr.replace(temp_file_path, "<student_code.py>")
        # Filter out lines referring to safe_import or preamble if any
        lines = cleaned.splitlines()
        filtered = []
        skip = False
        for line in lines:
            if "safe_import" in line or "orig_import" in line:
                continue
            filtered.append(line)
        return "\n".join(filtered)

    def _classify_error(self, stderr: str, returncode: int) -> str:
        """Categorize runtime error for pedagogical feedback."""
        if "RecursionError" in stderr:
            return "RecursionError"
        if "ZeroDivisionError" in stderr:
            return "ZeroDivisionError"
        if "IndexError" in stderr:
            return "IndexError"
        if "KeyError" in stderr:
            return "KeyError"
        if "TypeError" in stderr:
            return "TypeError"
        if "ValueError" in stderr:
            return "ValueError"
        if "NameError" in stderr:
            return "NameError"
        if "PermissionError" in stderr:
            return "SecurityPermissionError"
        if "SyntaxError" in stderr:
            return "SyntaxError"
        if returncode != 0:
            return f"RuntimeError (code {returncode})"
        return "UnknownError"
