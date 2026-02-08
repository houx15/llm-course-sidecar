"""Minimal stateful notebook manager for per-session cell execution."""

import ast
import io
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any


class NotebookManager:
    """In-process notebook-like execution with per-session persistent namespace."""

    def __init__(self, max_output_chars: int = 120_000):
        self._namespaces: Dict[str, Dict[str, Any]] = {}
        self.max_output_chars = max_output_chars

    def _namespace(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._namespaces:
            self._namespaces[session_id] = {"__name__": "__main__"}
        return self._namespaces[session_id]

    def _clip_output(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n...[output truncated]..."

    def execute_cell(self, session_id: str, code: str) -> Dict[str, Any]:
        namespace = self._namespace(session_id)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_repr = ""
        success = True
        start = time.time()

        try:
            module = ast.parse(code, mode="exec")
            last_expr = None
            if module.body and isinstance(module.body[-1], ast.Expr):
                last_expr = module.body.pop().value
        except Exception:
            return {
                "success": False,
                "stdout": "",
                "stderr": self._clip_output(traceback.format_exc()),
                "result_repr": "",
                "execution_time_ms": int((time.time() - start) * 1000),
            }

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                if module.body:
                    exec(compile(module, "<notebook-cell>", "exec"), namespace, namespace)
                if last_expr is not None:
                    result = eval(
                        compile(ast.Expression(last_expr), "<notebook-cell-result>", "eval"),
                        namespace,
                        namespace,
                    )
                    if result is not None:
                        result_repr = repr(result)
        except Exception:
            success = False
            stderr_buf.write(traceback.format_exc())

        return {
            "success": success,
            "stdout": self._clip_output(stdout_buf.getvalue()),
            "stderr": self._clip_output(stderr_buf.getvalue()),
            "result_repr": self._clip_output(result_repr),
            "execution_time_ms": int((time.time() - start) * 1000),
        }

    def reset_session(self, session_id: str) -> None:
        self._namespaces.pop(session_id, None)
