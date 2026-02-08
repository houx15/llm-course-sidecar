"""User code runner for per-session Python execution."""

import os
import sys
import time
import subprocess
from threading import Event
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import resource
except Exception:  # pragma: no cover - resource not available on some platforms
    resource = None


class UserCodeRunner:
    """Execute user-provided Python code inside a session workspace."""

    def __init__(self, sessions_root: Path, max_output_chars: int = 120_000):
        self.sessions_root = sessions_root
        self.max_output_chars = max_output_chars

    def _workspace_dir(self, session_id: str) -> Path:
        workspace = self.sessions_root / session_id / "user_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _clip_output(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        keep = self.max_output_chars
        return text[:keep] + "\n...[output truncated]..."

    def _build_preexec(self, memory_limit_mb: Optional[int]):
        if os.name == "nt" or resource is None or not memory_limit_mb:
            return None
        memory_bytes = int(memory_limit_mb) * 1024 * 1024

        def _limit_memory():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        return _limit_memory

    def run_python(
        self,
        session_id: str,
        code: str,
        timeout_seconds: int = 20,
        memory_limit_mb: Optional[int] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        timeout_seconds = max(1, min(int(timeout_seconds or 20), 120))
        workspace = self._workspace_dir(session_id)
        timestamp = int(time.time() * 1000)
        code_file = workspace / f"_run_{timestamp}.py"

        start = time.time()
        code_file.write_text(code, encoding="utf-8")
        try:
            process = subprocess.Popen(
                [sys.executable, str(code_file)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                preexec_fn=self._build_preexec(memory_limit_mb),
            )
            timed_out = False
            cancelled = False

            while True:
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    process.kill()
                    break
                if process.poll() is not None:
                    break
                if (time.time() - start) > timeout_seconds:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(0.05)

            stdout, stderr = process.communicate()
            if cancelled:
                return {
                    "success": False,
                    "stdout": self._clip_output(stdout or ""),
                    "stderr": "Execution cancelled",
                    "returncode": -2,
                    "execution_time_ms": int((time.time() - start) * 1000),
                    "cancelled": True,
                }
            if timed_out:
                return {
                    "success": False,
                    "stdout": self._clip_output(stdout or ""),
                    "stderr": f"Execution timeout after {timeout_seconds} seconds",
                    "returncode": -1,
                    "execution_time_ms": int((time.time() - start) * 1000),
                    "cancelled": False,
                }
            return {
                "success": process.returncode == 0,
                "stdout": self._clip_output(stdout or ""),
                "stderr": self._clip_output(stderr or ""),
                "returncode": int(process.returncode),
                "execution_time_ms": int((time.time() - start) * 1000),
                "cancelled": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout after {timeout_seconds} seconds",
                "returncode": -1,
                "execution_time_ms": timeout_seconds * 1000,
                "cancelled": False,
            }
        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {exc}",
                "returncode": -3,
                "execution_time_ms": int((time.time() - start) * 1000),
                "cancelled": False,
            }
        finally:
            code_file.unlink(missing_ok=True)
