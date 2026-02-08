"""User code runner for per-session Python execution."""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any


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

    def run_python(self, session_id: str, code: str, timeout_seconds: int = 20) -> Dict[str, Any]:
        timeout_seconds = max(1, min(int(timeout_seconds or 20), 120))
        workspace = self._workspace_dir(session_id)
        timestamp = int(time.time() * 1000)
        code_file = workspace / f"_run_{timestamp}.py"

        start = time.time()
        code_file.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return {
                "success": result.returncode == 0,
                "stdout": self._clip_output(result.stdout or ""),
                "stderr": self._clip_output(result.stderr or ""),
                "returncode": int(result.returncode),
                "execution_time_ms": int((time.time() - start) * 1000),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout after {timeout_seconds} seconds",
                "returncode": -1,
                "execution_time_ms": timeout_seconds * 1000,
            }
        finally:
            code_file.unlink(missing_ok=True)
