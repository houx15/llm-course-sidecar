"""Performance tracking service for monitoring agent operations."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)
SESSIONS_ROOT = Path(os.getenv("SESSIONS_DIR", "sessions"))


class PerformanceTracker:
    """Tracks performance metrics for agent operations."""

    def __init__(self):
        """Initialize performance tracker."""
        self.current_turn_metrics: Optional[Dict[str, Any]] = None
        self.operation_stack: List[Dict[str, Any]] = []

    def start_turn(self, session_id: str, turn_index: int) -> None:
        """
        Start tracking a new turn.

        Args:
            session_id: Session identifier
            turn_index: Turn index
        """
        self.current_turn_metrics = {
            "session_id": session_id,
            "turn_index": turn_index,
            "start_time": time.time(),
            "operations": [],
            "total_duration_ms": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_tokens": 0,
        }
        self.operation_stack = []
        logger.info(f"Started tracking turn {turn_index} for session {session_id}")

    def start_operation(self, operation_name: str, metadata: Optional[Dict] = None) -> None:
        """
        Start tracking an operation.

        Args:
            operation_name: Name of the operation (e.g., "run_companion", "run_roadmap_manager")
            metadata: Optional metadata for the operation
        """
        if self.current_turn_metrics is None:
            logger.warning(f"Cannot start operation {operation_name}: no turn in progress")
            return

        operation = {
            "name": operation_name,
            "start_time": time.time(),
            "metadata": metadata or {},
            "duration_ms": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "tokens_total": 0,
            "success": True,
            "error": None,
        }
        self.operation_stack.append(operation)
        logger.debug(f"Started operation: {operation_name}")

    def end_operation(
        self,
        success: bool = True,
        error: Optional[str] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        """
        End tracking the current operation.

        Args:
            success: Whether the operation succeeded
            error: Error message if operation failed
            tokens_input: Number of input tokens used
            tokens_output: Number of output tokens used
        """
        if not self.operation_stack:
            logger.warning("Cannot end operation: no operation in progress")
            return

        operation = self.operation_stack.pop()
        operation["duration_ms"] = int((time.time() - operation["start_time"]) * 1000)
        operation["success"] = success
        operation["error"] = error
        operation["tokens_input"] = tokens_input
        operation["tokens_output"] = tokens_output
        operation["tokens_total"] = tokens_input + tokens_output

        # Remove start_time from final record
        del operation["start_time"]

        if self.current_turn_metrics:
            self.current_turn_metrics["operations"].append(operation)
            self.current_turn_metrics["total_tokens_input"] += tokens_input
            self.current_turn_metrics["total_tokens_output"] += tokens_output
            self.current_turn_metrics["total_tokens"] += tokens_input + tokens_output

        logger.debug(
            f"Ended operation: {operation['name']} "
            f"(duration: {operation['duration_ms']}ms, "
            f"tokens: {operation['tokens_total']})"
        )

    @contextmanager
    def track_operation(
        self,
        operation_name: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Context manager for tracking an operation.

        Usage:
            with tracker.track_operation("run_companion"):
                result = await agent_runner.run_companion(...)

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata
        """
        self.start_operation(operation_name, metadata)
        try:
            yield
            self.end_operation(success=True)
        except Exception as e:
            self.end_operation(success=False, error=str(e))
            raise

    def end_turn(self, session_id: str) -> Dict[str, Any]:
        """
        End tracking the current turn and save metrics.

        Args:
            session_id: Session identifier

        Returns:
            Turn metrics dictionary
        """
        if self.current_turn_metrics is None:
            logger.warning("Cannot end turn: no turn in progress")
            return {}

        # Calculate total duration
        self.current_turn_metrics["total_duration_ms"] = int(
            (time.time() - self.current_turn_metrics["start_time"]) * 1000
        )

        # Remove start_time from final record
        del self.current_turn_metrics["start_time"]

        # Add timestamp
        self.current_turn_metrics["timestamp"] = datetime.now().isoformat()

        # Save to file
        self._save_metrics(session_id, self.current_turn_metrics)

        # Log summary
        self._log_summary(self.current_turn_metrics)

        metrics = self.current_turn_metrics
        self.current_turn_metrics = None
        self.operation_stack = []

        return metrics

    def _save_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to JSON file.

        Args:
            session_id: Session identifier
            metrics: Metrics dictionary
        """
        try:
            metrics_dir = SESSIONS_ROOT / session_id / "performance_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            turn_index = metrics["turn_index"]
            metrics_path = metrics_dir / f"turn_{turn_index:03d}.json"

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved performance metrics: {metrics_path}")

        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")

    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance summary.

        Args:
            metrics: Metrics dictionary
        """
        turn_index = metrics["turn_index"]
        total_duration = metrics["total_duration_ms"]
        total_tokens = metrics["total_tokens"]

        logger.info(
            f"Turn {turn_index} performance summary: "
            f"duration={total_duration}ms, tokens={total_tokens}"
        )

        # Log operation breakdown
        operations = metrics["operations"]
        if operations:
            logger.info(f"  Operations ({len(operations)}):")
            for op in operations:
                status = "✓" if op["success"] else "✗"
                logger.info(
                    f"    {status} {op['name']}: {op['duration_ms']}ms, "
                    f"{op['tokens_total']} tokens"
                )

        # Identify bottlenecks (operations taking >30% of total time)
        if total_duration > 0:
            bottlenecks = [
                op for op in operations
                if op["duration_ms"] / total_duration > 0.3
            ]
            if bottlenecks:
                logger.warning("  Bottlenecks detected:")
                for op in bottlenecks:
                    percentage = (op["duration_ms"] / total_duration) * 100
                    logger.warning(
                        f"    - {op['name']}: {op['duration_ms']}ms ({percentage:.1f}%)"
                    )

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get performance summary for entire session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary dictionary
        """
        try:
            metrics_dir = SESSIONS_ROOT / session_id / "performance_metrics"
            if not metrics_dir.exists():
                return {}

            # Load all turn metrics
            turn_files = sorted(metrics_dir.glob("turn_*.json"))
            all_metrics = []

            for turn_file in turn_files:
                with open(turn_file, "r", encoding="utf-8") as f:
                    all_metrics.append(json.load(f))

            if not all_metrics:
                return {}

            # Calculate aggregates
            total_turns = len(all_metrics)
            total_duration_ms = sum(m["total_duration_ms"] for m in all_metrics)
            total_tokens = sum(m["total_tokens"] for m in all_metrics)
            avg_duration_ms = total_duration_ms / total_turns
            avg_tokens = total_tokens / total_turns

            # Operation breakdown
            operation_stats: Dict[str, Dict[str, Any]] = {}
            for metrics in all_metrics:
                for op in metrics["operations"]:
                    op_name = op["name"]
                    if op_name not in operation_stats:
                        operation_stats[op_name] = {
                            "count": 0,
                            "total_duration_ms": 0,
                            "total_tokens": 0,
                            "failures": 0,
                        }
                    operation_stats[op_name]["count"] += 1
                    operation_stats[op_name]["total_duration_ms"] += op["duration_ms"]
                    operation_stats[op_name]["total_tokens"] += op["tokens_total"]
                    if not op["success"]:
                        operation_stats[op_name]["failures"] += 1

            # Calculate averages
            for op_name, stats in operation_stats.items():
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["count"]

            summary = {
                "session_id": session_id,
                "total_turns": total_turns,
                "total_duration_ms": total_duration_ms,
                "total_tokens": total_tokens,
                "avg_duration_ms": avg_duration_ms,
                "avg_tokens": avg_tokens,
                "operation_stats": operation_stats,
                "timestamp": datetime.now().isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return {}


# Global instance
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    return _tracker
