"""User code runner for per-session Python execution."""

import json
import logging
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

logger = logging.getLogger(__name__)


class UserCodeRunner:
    """Execute user-provided Python code inside a session workspace."""

    def __init__(self, sessions_root: Path, max_output_chars: int = 120_000):
        self.sessions_root = sessions_root
        self.max_output_chars = max_output_chars

    def _workspace_dir(self, session_id: str) -> Path:
        workspace = self.sessions_root / session_id / "user_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _history_dir(self, session_id: str) -> Path:
        history_dir = self.sessions_root / session_id / "code_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir

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

    @staticmethod
    def _trim_history_output(text: str, limit: int = 2000) -> str:
        text = text or ""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]..."

    @staticmethod
    def _trim_history_code(text: str, limit: int = 1200) -> str:
        text = text or ""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n# ...[code truncated]..."

    def _persist_execution_record(
        self,
        session_id: str,
        timestamp_ms: int,
        code: str,
        result: Dict[str, Any],
    ) -> None:
        """Persist a lightweight code execution history record."""
        try:
            history_dir = self._history_dir(session_id)
            record_path = history_dir / f"run_{timestamp_ms}.json"
            payload = {
                "timestamp": timestamp_ms / 1000.0,
                "code": self._trim_history_code(code),
                "stdout": self._trim_history_output(result.get("stdout", "")),
                "stderr": self._trim_history_output(result.get("stderr", "")),
                "exit_code": int(result.get("returncode", 0)),
                "success": bool(result.get("success", False)),
                "cancelled": bool(result.get("cancelled", False)),
                "execution_time_ms": int(result.get("execution_time_ms", 0)),
            }
            record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # Keep only the latest 10 records.
            history_files = sorted(history_dir.glob("run_*.json"), reverse=True)
            for old_file in history_files[10:]:
                old_file.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Failed to persist code history for session {session_id}: {exc}")

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
        result: Dict[str, Any]

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
                result = {
                    "success": False,
                    "stdout": self._clip_output(stdout or ""),
                    "stderr": "Execution cancelled",
                    "returncode": -2,
                    "execution_time_ms": int((time.time() - start) * 1000),
                    "cancelled": True,
                }
                self._persist_execution_record(session_id, timestamp, code, result)
                return result
            if timed_out:
                result = {
                    "success": False,
                    "stdout": self._clip_output(stdout or ""),
                    "stderr": f"Execution timeout after {timeout_seconds} seconds",
                    "returncode": -1,
                    "execution_time_ms": int((time.time() - start) * 1000),
                    "cancelled": False,
                }
                self._persist_execution_record(session_id, timestamp, code, result)
                return result
            result = {
                "success": process.returncode == 0,
                "stdout": self._clip_output(stdout or ""),
                "stderr": self._clip_output(stderr or ""),
                "returncode": int(process.returncode),
                "execution_time_ms": int((time.time() - start) * 1000),
                "cancelled": False,
            }
            self._persist_execution_record(session_id, timestamp, code, result)
            return result
        except subprocess.TimeoutExpired:
            result = {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout after {timeout_seconds} seconds",
                "returncode": -1,
                "execution_time_ms": timeout_seconds * 1000,
                "cancelled": False,
            }
            self._persist_execution_record(session_id, timestamp, code, result)
            return result
        except Exception as exc:
            result = {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {exc}",
                "returncode": -3,
                "execution_time_ms": int((time.time() - start) * 1000),
                "cancelled": False,
            }
            self._persist_execution_record(session_id, timestamp, code, result)
            return result
        finally:
            code_file.unlink(missing_ok=True)
