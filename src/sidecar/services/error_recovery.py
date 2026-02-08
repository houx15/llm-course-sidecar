"""Error recovery service for handling agent failures gracefully."""

import logging
import re
from enum import Enum
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in the system."""
    RMA_TIMEOUT = "rma_timeout"
    RMA_INVALID_OUTPUT = "rma_invalid_output"
    EXPERT_TIMEOUT = "expert_timeout"
    EXPERT_SKILL_FAILURE = "expert_skill_failure"
    EXPERT_NO_RESPONSE = "expert_no_response"
    CA_TIMEOUT = "ca_timeout"
    CA_INVALID_OUTPUT = "ca_invalid_output"
    MA_TIMEOUT = "ma_timeout"
    STORAGE_ERROR = "storage_error"
    LLM_API_ERROR = "llm_api_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    FALLBACK_TO_CA_DIRECT = "fallback_to_ca_direct"
    USE_FALLBACK_RESPONSE = "use_fallback_response"
    SKIP_MA_UPDATE = "skip_ma_update"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CONTINUE_WITH_DEFAULT = "continue_with_default"


class ErrorRecovery:
    """Service for classifying errors and determining recovery strategies."""

    # Error classification patterns
    ERROR_PATTERNS = {
        ErrorType.RMA_TIMEOUT: [
            r"roadmap.*timeout",
            r"rma.*timeout",
            r"timeout.*roadmap",
        ],
        ErrorType.RMA_INVALID_OUTPUT: [
            r"roadmap.*invalid",
            r"rma.*validation.*failed",
            r"instruction.*packet.*invalid",
        ],
        ErrorType.EXPERT_TIMEOUT: [
            r"expert.*timeout",
            r"consultation.*timeout",
        ],
        ErrorType.EXPERT_SKILL_FAILURE: [
            r"skill.*failed",
            r"expert.*skill.*error",
            r"DataInspectorSkillsError",
        ],
        ErrorType.EXPERT_NO_RESPONSE: [
            r"expert.*no.*response",
            r"expert.*empty.*output",
        ],
        ErrorType.CA_TIMEOUT: [
            r"companion.*timeout",
            r"ca.*timeout",
        ],
        ErrorType.CA_INVALID_OUTPUT: [
            r"companion.*invalid",
            r"ca.*validation.*failed",
            r"turn.*outcome.*invalid",
        ],
        ErrorType.MA_TIMEOUT: [
            r"memo.*timeout",
            r"ma.*timeout",
        ],
        ErrorType.STORAGE_ERROR: [
            r"storage.*error",
            r"failed.*to.*save",
            r"failed.*to.*load",
            r"file.*not.*found",
        ],
        ErrorType.LLM_API_ERROR: [
            r"api.*error",
            r"llm.*error",
            r"rate.*limit",
            r"connection.*error",
        ],
    }

    # Recovery strategies for each error type
    RECOVERY_STRATEGIES = {
        ErrorType.RMA_TIMEOUT: RecoveryStrategy.FALLBACK_TO_CA_DIRECT,
        ErrorType.RMA_INVALID_OUTPUT: RecoveryStrategy.CONTINUE_WITH_DEFAULT,
        ErrorType.EXPERT_TIMEOUT: RecoveryStrategy.FALLBACK_TO_CA_DIRECT,
        ErrorType.EXPERT_SKILL_FAILURE: RecoveryStrategy.FALLBACK_TO_CA_DIRECT,
        ErrorType.EXPERT_NO_RESPONSE: RecoveryStrategy.FALLBACK_TO_CA_DIRECT,
        ErrorType.CA_TIMEOUT: RecoveryStrategy.USE_FALLBACK_RESPONSE,
        ErrorType.CA_INVALID_OUTPUT: RecoveryStrategy.RETRY_WITH_BACKOFF,
        ErrorType.MA_TIMEOUT: RecoveryStrategy.SKIP_MA_UPDATE,
        ErrorType.STORAGE_ERROR: RecoveryStrategy.USE_FALLBACK_RESPONSE,
        ErrorType.LLM_API_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
        ErrorType.UNKNOWN_ERROR: RecoveryStrategy.USE_FALLBACK_RESPONSE,
    }

    # User-friendly messages for each error type
    USER_MESSAGES = {
        ErrorType.RMA_TIMEOUT: "系统正在处理您的请求，但响应时间较长。我将直接为您提供帮助。",
        ErrorType.RMA_INVALID_OUTPUT: "系统遇到了一些技术问题，但我会继续为您提供指导。",
        ErrorType.EXPERT_TIMEOUT: "专家系统响应超时。我将基于现有信息为您提供指导。",
        ErrorType.EXPERT_SKILL_FAILURE: "专家工具执行失败。我将使用其他方式为您提供帮助。",
        ErrorType.EXPERT_NO_RESPONSE: "专家系统暂时无法响应。我将基于现有信息为您提供指导。",
        ErrorType.CA_TIMEOUT: "抱歉，系统响应较慢。请稍后再试。",
        ErrorType.CA_INVALID_OUTPUT: "系统正在重新处理您的请求，请稍候。",
        ErrorType.MA_TIMEOUT: "",  # Silent failure - no user message
        ErrorType.STORAGE_ERROR: "抱歉，系统遇到了存储问题。请稍后再试。",
        ErrorType.LLM_API_ERROR: "抱歉，AI服务暂时不可用。请稍后再试。",
        ErrorType.UNKNOWN_ERROR: "抱歉，系统遇到了未知错误。请稍后再试。",
    }

    @classmethod
    def classify_error(cls, exception: Exception) -> ErrorType:
        """
        Classify an exception into an ErrorType.

        Args:
            exception: The exception to classify

        Returns:
            ErrorType enum value
        """
        error_message = str(exception).lower()

        # Try to match against patterns
        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    logger.info(f"Classified error as {error_type.value}: {error_message[:100]}")
                    return error_type

        # Default to unknown error
        logger.warning(f"Could not classify error: {error_message[:100]}")
        return ErrorType.UNKNOWN_ERROR

    @classmethod
    def get_recovery_strategy(cls, error_type: ErrorType) -> RecoveryStrategy:
        """
        Get the recovery strategy for an error type.

        Args:
            error_type: The error type

        Returns:
            RecoveryStrategy enum value
        """
        return cls.RECOVERY_STRATEGIES.get(error_type, RecoveryStrategy.USE_FALLBACK_RESPONSE)

    @classmethod
    def get_user_message(cls, error_type: ErrorType) -> str:
        """
        Get user-friendly message for an error type.

        Args:
            error_type: The error type

        Returns:
            User-friendly error message
        """
        return cls.USER_MESSAGES.get(error_type, cls.USER_MESSAGES[ErrorType.UNKNOWN_ERROR])

    @classmethod
    def should_retry(cls, error_type: ErrorType) -> bool:
        """
        Determine if an error should trigger a retry.

        Args:
            error_type: The error type

        Returns:
            True if should retry
        """
        retry_types = {
            ErrorType.CA_INVALID_OUTPUT,
            ErrorType.RMA_INVALID_OUTPUT,
            ErrorType.LLM_API_ERROR,
        }
        return error_type in retry_types

    @classmethod
    def should_notify_user(cls, error_type: ErrorType) -> bool:
        """
        Determine if user should be notified of an error.

        Args:
            error_type: The error type

        Returns:
            True if user should be notified
        """
        # MA timeout is silent - don't notify user
        return error_type != ErrorType.MA_TIMEOUT

    @classmethod
    def get_fallback_ca_response(cls, error_type: ErrorType, user_message: str) -> str:
        """
        Generate a fallback CA response for an error.

        Args:
            error_type: The error type
            user_message: The user's message

        Returns:
            Fallback response text
        """
        user_msg = cls.get_user_message(error_type)

        # Add context-aware guidance
        if "数据" in user_message or "dataset" in user_message.lower():
            guidance = "如果您上传了数据集，我可以帮您检查数据的基本信息。"
        elif "代码" in user_message or "code" in user_message.lower():
            guidance = "我可以帮您理解代码的逻辑，或者指导您如何实现功能。"
        elif "错误" in user_message or "error" in user_message.lower():
            guidance = "请告诉我您遇到的具体错误信息，我会帮您分析原因。"
        else:
            guidance = "请告诉我您想做什么，我会尽力帮助您。"

        return f"{user_msg}\n\n{guidance}"

    @classmethod
    def log_error_recovery(
        cls,
        session_id: str,
        turn_index: int,
        error_type: ErrorType,
        recovery_strategy: RecoveryStrategy,
        original_error: str,
    ) -> None:
        """
        Log error recovery information.

        Args:
            session_id: Session identifier
            turn_index: Turn index
            error_type: The error type
            recovery_strategy: The recovery strategy used
            original_error: Original error message
        """
        logger.warning(
            f"Error recovery triggered for session {session_id}, turn {turn_index}:\n"
            f"  Error type: {error_type.value}\n"
            f"  Recovery strategy: {recovery_strategy.value}\n"
            f"  Original error: {original_error[:200]}"
        )

        # TODO: Save to error recovery log file for analysis
        # This could be useful for identifying patterns and improving the system


class ErrorRecoveryContext:
    """Context for error recovery operations."""

    def __init__(
        self,
        session_id: str,
        turn_index: int,
        user_message: str,
        error: Exception,
    ):
        """
        Initialize error recovery context.

        Args:
            session_id: Session identifier
            turn_index: Turn index
            user_message: User's message
            error: The exception that occurred
        """
        self.session_id = session_id
        self.turn_index = turn_index
        self.user_message = user_message
        self.error = error
        self.error_type = ErrorRecovery.classify_error(error)
        self.recovery_strategy = ErrorRecovery.get_recovery_strategy(self.error_type)
        self.user_message_text = ErrorRecovery.get_user_message(self.error_type)

        # Log the error recovery
        ErrorRecovery.log_error_recovery(
            session_id=session_id,
            turn_index=turn_index,
            error_type=self.error_type,
            recovery_strategy=self.recovery_strategy,
            original_error=str(error),
        )

    def get_fallback_response(self) -> str:
        """
        Get fallback response for the error.

        Returns:
            Fallback response text
        """
        return ErrorRecovery.get_fallback_ca_response(self.error_type, self.user_message)

    def should_retry(self) -> bool:
        """
        Check if operation should be retried.

        Returns:
            True if should retry
        """
        return ErrorRecovery.should_retry(self.error_type)

    def should_notify_user(self) -> bool:
        """
        Check if user should be notified.

        Returns:
            True if user should be notified
        """
        return ErrorRecovery.should_notify_user(self.error_type)
