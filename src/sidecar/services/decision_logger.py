"""Decision logger for expert skills - logs decisions to CSV for analysis."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Logger for expert skill decisions."""

    def __init__(self, session_id: str, expert_id: str, consultation_id: str):
        """
        Initialize decision logger.

        Args:
            session_id: Session identifier
            expert_id: Expert identifier
            consultation_id: Consultation identifier
        """
        self.session_id = session_id
        self.expert_id = expert_id
        self.consultation_id = consultation_id

        # Create log directory
        self.log_dir = (
            Path("sessions") / session_id / "working_files" / "expert_logs" / consultation_id
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_decision(
        self,
        skill_name: str,
        decision_point: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Log a decision to CSV file.

        Args:
            skill_name: Name of the skill (e.g., "check_suitability")
            decision_point: Description of the decision point (e.g., "column_check")
            input_data: Input data for the decision
            output_data: Output data from the decision
            reasoning: Reasoning for the decision
            confidence: Confidence score (0.0 to 1.0)
        """
        try:
            csv_path = self.log_dir / f"{skill_name}_decisions.csv"

            # Check if file exists to determine if we need to write header
            file_exists = csv_path.exists()

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write header if file is new
                if not file_exists:
                    writer.writerow([
                        "timestamp",
                        "skill_name",
                        "decision_point",
                        "input_data",
                        "output_data",
                        "reasoning",
                        "confidence",
                    ])

                # Write decision row
                writer.writerow([
                    datetime.now().isoformat(),
                    skill_name,
                    decision_point,
                    str(input_data),
                    str(output_data),
                    reasoning,
                    confidence,
                ])

            logger.debug(f"Logged decision: {skill_name}/{decision_point}")

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            # Don't raise - logging failure shouldn't break skill execution


def log_skill_decision(decision_point: str):
    """
    Decorator for logging skill decisions.

    Usage:
        @log_skill_decision("column_check")
        def check_columns(self, ...):
            ...

    Args:
        decision_point: Description of the decision point
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the function
            result = func(self, *args, **kwargs)

            # Log the decision if logger is available
            if hasattr(self, "_decision_logger") and self._decision_logger:
                try:
                    # Extract skill name from function name
                    skill_name = func.__name__

                    # Prepare input data (combine args and kwargs)
                    input_data = {
                        "args": args,
                        "kwargs": kwargs,
                    }

                    # Prepare output data
                    output_data = {"result": result}

                    # Extract reasoning from result if available
                    reasoning = "Decision executed successfully"
                    if isinstance(result, dict):
                        reasoning = result.get("reasoning", reasoning)

                    # Log the decision
                    self._decision_logger.log_decision(
                        skill_name=skill_name,
                        decision_point=decision_point,
                        input_data=input_data,
                        output_data=output_data,
                        reasoning=reasoning,
                        confidence=1.0,
                    )

                except Exception as e:
                    logger.error(f"Failed to log decision in decorator: {e}")

            return result

        return wrapper
    return decorator


class DecisionLoggerMixin:
    """Mixin class for adding decision logging to skill classes."""

    def _init_decision_logger(
        self,
        session_id: str,
        expert_id: str,
        consultation_id: str,
    ) -> None:
        """
        Initialize decision logger for this skill instance.

        Args:
            session_id: Session identifier
            expert_id: Expert identifier
            consultation_id: Consultation identifier
        """
        self._decision_logger = DecisionLogger(
            session_id=session_id,
            expert_id=expert_id,
            consultation_id=consultation_id,
        )

    def _log_decision(
        self,
        skill_name: str,
        decision_point: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Log a decision (convenience method).

        Args:
            skill_name: Name of the skill
            decision_point: Description of the decision point
            input_data: Input data for the decision
            output_data: Output data from the decision
            reasoning: Reasoning for the decision
            confidence: Confidence score (0.0 to 1.0)
        """
        if hasattr(self, "_decision_logger") and self._decision_logger:
            self._decision_logger.log_decision(
                skill_name=skill_name,
                decision_point=decision_point,
                input_data=input_data,
                output_data=output_data,
                reasoning=reasoning,
                confidence=confidence,
            )
