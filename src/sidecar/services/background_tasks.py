"""Background tasks manager for async agent execution."""

import asyncio
import logging
from typing import Dict, Optional, Any, Callable, Coroutine
from datetime import datetime

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manager for background task execution."""

    def __init__(self):
        """Initialize background task manager."""
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, Exception] = {}

    def create_task(
        self,
        task_id: str,
        coroutine: Coroutine,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> asyncio.Task:
        """
        Create and track a background task.

        Args:
            task_id: Unique identifier for the task
            coroutine: Coroutine to execute
            on_complete: Optional callback when task completes successfully
            on_error: Optional callback when task fails

        Returns:
            The created asyncio.Task
        """
        if task_id in self.tasks:
            logger.warning(f"Task {task_id} already exists, cancelling old task")
            self.tasks[task_id].cancel()

        async def wrapped_coroutine():
            """Wrapper to handle completion and errors."""
            try:
                logger.info(f"Background task started: {task_id}")
                result = await coroutine
                self.task_results[task_id] = result
                logger.info(f"Background task completed: {task_id}")

                if on_complete:
                    try:
                        on_complete(result)
                    except Exception as e:
                        logger.error(f"Error in on_complete callback for {task_id}: {e}")

                return result

            except asyncio.CancelledError:
                logger.info(f"Background task cancelled: {task_id}")
                raise

            except Exception as e:
                logger.error(f"Background task failed: {task_id}: {e}")
                self.task_errors[task_id] = e

                if on_error:
                    try:
                        on_error(e)
                    except Exception as callback_error:
                        logger.error(f"Error in on_error callback for {task_id}: {callback_error}")

                raise

            finally:
                # Clean up task reference
                if task_id in self.tasks:
                    del self.tasks[task_id]

        task = asyncio.create_task(wrapped_coroutine())
        self.tasks[task_id] = task
        return task

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a background task to complete.

        Args:
            task_id: Task identifier
            timeout: Optional timeout in seconds

        Returns:
            Task result

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            KeyError: If task doesn't exist
            Exception: If task failed
        """
        if task_id not in self.tasks:
            # Check if result is already available
            if task_id in self.task_results:
                return self.task_results[task_id]
            if task_id in self.task_errors:
                raise self.task_errors[task_id]
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        if timeout:
            await asyncio.wait_for(task, timeout=timeout)
        else:
            await task

        # Return result or raise error
        if task_id in self.task_results:
            return self.task_results[task_id]
        if task_id in self.task_errors:
            raise self.task_errors[task_id]

        raise RuntimeError(f"Task {task_id} completed but no result or error found")

    def is_task_running(self, task_id: str) -> bool:
        """
        Check if a task is currently running.

        Args:
            task_id: Task identifier

        Returns:
            True if task is running
        """
        return task_id in self.tasks and not self.tasks[task_id].done()

    def is_task_complete(self, task_id: str) -> bool:
        """
        Check if a task has completed (successfully or with error).

        Args:
            task_id: Task identifier

        Returns:
            True if task is complete
        """
        return task_id in self.task_results or task_id in self.task_errors

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get result of a completed task.

        Args:
            task_id: Task identifier

        Returns:
            Task result or None if not available
        """
        return self.task_results.get(task_id)

    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """
        Get error from a failed task.

        Args:
            task_id: Task identifier

        Returns:
            Task error or None if not available
        """
        return self.task_errors.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled
        """
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            logger.info(f"Cancelled task: {task_id}")
            return True
        return False

    def cancel_all_tasks(self) -> int:
        """
        Cancel all running tasks.

        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task_id in list(self.tasks.keys()):
            if self.cancel_task(task_id):
                count += 1
        return count

    def cleanup_completed_tasks(self) -> int:
        """
        Clean up results and errors from completed tasks.

        Returns:
            Number of tasks cleaned up
        """
        count = 0

        # Clean up results
        for task_id in list(self.task_results.keys()):
            if task_id not in self.tasks:
                del self.task_results[task_id]
                count += 1

        # Clean up errors
        for task_id in list(self.task_errors.keys()):
            if task_id not in self.tasks:
                del self.task_errors[task_id]
                count += 1

        if count > 0:
            logger.debug(f"Cleaned up {count} completed tasks")

        return count

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status information for a task.

        Args:
            task_id: Task identifier

        Returns:
            Status dictionary
        """
        status = {
            "task_id": task_id,
            "exists": False,
            "running": False,
            "complete": False,
            "has_result": False,
            "has_error": False,
        }

        if task_id in self.tasks:
            status["exists"] = True
            status["running"] = not self.tasks[task_id].done()

        if task_id in self.task_results:
            status["complete"] = True
            status["has_result"] = True

        if task_id in self.task_errors:
            status["complete"] = True
            status["has_error"] = True

        return status


# Global instance
_manager = BackgroundTaskManager()


def get_background_task_manager() -> BackgroundTaskManager:
    """Get the global background task manager instance."""
    return _manager
