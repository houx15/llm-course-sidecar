"""Async code execution job manager with cancellation support."""

import time
import uuid
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

from .user_code_runner import UserCodeRunner


@dataclass
class CodeExecutionJob:
    job_id: str
    session_id: str
    code: str
    timeout_seconds: int
    memory_limit_mb: Optional[int]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class CodeExecutionManager:
    """Manage background code execution jobs and cancellation."""

    def __init__(self, runner: UserCodeRunner, max_workers: int = 4):
        self.runner = runner
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sidecar-code")
        self.jobs: Dict[str, CodeExecutionJob] = {}
        self.lock = threading.Lock()

    def create_job(
        self,
        session_id: str,
        code: str,
        timeout_seconds: int = 20,
        memory_limit_mb: Optional[int] = None,
    ) -> CodeExecutionJob:
        job_id = str(uuid.uuid4())
        job = CodeExecutionJob(
            job_id=job_id,
            session_id=session_id,
            code=code,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run_job, job_id)
        return job

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.status = "running"
            job.started_at = time.time()

        result = self.runner.run_python(
            session_id=job.session_id,
            code=job.code,
            timeout_seconds=job.timeout_seconds,
            memory_limit_mb=job.memory_limit_mb,
            cancel_event=job.cancel_event,
        )

        with self.lock:
            current = self.jobs.get(job_id)
            if not current:
                return
            current.result = result
            current.ended_at = time.time()
            if result.get("cancelled"):
                current.status = "cancelled"
            elif result.get("success"):
                current.status = "succeeded"
            else:
                current.status = "failed"

    def get_job(self, job_id: str) -> Optional[CodeExecutionJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> Optional[CodeExecutionJob]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            job.cancel_event.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.ended_at = time.time()
                job.result = {
                    "success": False,
                    "stdout": "",
                    "stderr": "Execution cancelled before start",
                    "returncode": -2,
                    "execution_time_ms": 0,
                    "cancelled": True,
                }
            return job
