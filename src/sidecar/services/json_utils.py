"""JSON validation utilities for LLM responses."""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
from pathlib import Path
import jsonschema
from pydantic import BaseModel, ValidationError

from ..models.schemas import (
    InstructionPacket,
    TurnOutcome,
    MemoDigest,
    SessionState,
)


T = TypeVar("T", bound=BaseModel)


class JSONValidationError(Exception):
    """Custom exception for JSON validation errors."""
    pass


def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response text

    Returns:
        Extracted JSON string or None if not found
    """
    # Try to find JSON in markdown code blocks (```json ... ```)
    json_block_pattern = r"```json\s*\n(.*?)\n```"
    matches = re.findall(json_block_pattern, response, re.DOTALL | re.IGNORECASE)

    if matches:
        # Return the last match (most likely the final output)
        return matches[-1].strip()

    # Try to find JSON in generic code blocks (``` ... ```)
    code_block_pattern = r"```\s*\n(.*?)\n```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        # Check if it looks like JSON
        for match in reversed(matches):  # Check from last to first
            match = match.strip()
            if match.startswith("{") or match.startswith("["):
                return match

    # Try to find raw JSON with balanced braces
    # This pattern looks for JSON objects with proper nesting
    try:
        # Find the first { and try to extract a balanced JSON object
        start_idx = response.find("{")
        if start_idx != -1:
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(response)):
                char = response[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            return response[start_idx:i+1].strip()
    except Exception:
        pass

    # Fallback: try to find any JSON-like pattern
    json_pattern = r"(\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]+\})"
    matches = re.findall(json_pattern, response, re.DOTALL)

    if matches:
        # Return the longest match (likely the most complete)
        return max(matches, key=len)

    return None


def validate_with_schema(data: Dict[str, Any], schema_path: Path) -> None:
    """
    Validate data against a JSON schema.

    Args:
        data: Data to validate
        schema_path: Path to JSON schema file

    Raises:
        JSONValidationError: If validation fails
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise JSONValidationError(f"Schema validation failed: {e.message}")
    except FileNotFoundError:
        raise JSONValidationError(f"Schema file not found: {schema_path}")


def parse_and_validate(
    response: str,
    model_class: Type[T],
    schema_path: Optional[Path] = None,
) -> T:
    """
    Parse JSON from LLM response and validate with Pydantic model.

    Args:
        response: Raw LLM response text
        model_class: Pydantic model class to validate against
        schema_path: Optional JSON schema file for additional validation

    Returns:
        Validated Pydantic model instance

    Raises:
        JSONValidationError: If parsing or validation fails
    """
    # Extract JSON
    json_str = extract_json_from_response(response)
    if not json_str:
        raise JSONValidationError("No JSON found in response")

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common issues: single quotes instead of double quotes
        try:
            # Replace single quotes with double quotes (carefully)
            # This is a simple heuristic and may not work for all cases
            fixed_json = json_str.replace("'", '"')
            data = json.loads(fixed_json)
        except json.JSONDecodeError:
            # If that doesn't work, raise the original error
            raise JSONValidationError(f"Invalid JSON: {e}")

    # Validate with JSON schema if provided
    if schema_path:
        validate_with_schema(data, schema_path)

    # Validate with Pydantic model
    try:
        return model_class(**data)
    except ValidationError as e:
        raise JSONValidationError(f"Pydantic validation failed: {e}")


def get_default_instruction_packet() -> InstructionPacket:
    """Get default instruction packet for error recovery."""
    return InstructionPacket(
        current_focus="继续当前任务",
        guidance_for_ca="继续引导学习者完成当前任务",
        must_check=["检查学习者是否理解当前概念"],
        nice_check=[],
        instruction_version=1,
        lock_until="checkpoint_reached",
        allow_setup_helper_code=False,
        setup_helper_scope="none",
        task_type="core"
    )


def get_default_turn_outcome() -> TurnOutcome:
    """Get default turn outcome for error recovery."""
    return TurnOutcome(
        what_user_attempted="用户尝试继续学习",
        what_user_observed="用户正在探索",
        ca_teaching_mode="socratic",
        ca_next_suggestion="继续探索当前主题",
        checkpoint_reached=False,
        blocker_type="none",
        student_sentiment="engaged"
    )


def get_default_memo_digest() -> MemoDigest:
    """Get default memo digest for error recovery."""
    return MemoDigest(
        key_observations=["学习者正在进行中"],
        student_struggles=[],
        student_strengths=[],
        student_sentiment="engaged",
        blocker_type="none",
        progress_delta="none",
        diagnostic_log=[]
    )


def get_default_state_update() -> Dict[str, Any]:
    """Get default state update for error recovery."""
    return {}


def validate_with_retry(
    llm_call_func,
    model_class: Type[T],
    schema_path: Optional[Path] = None,
    max_retries: int = 1,
) -> T:
    """
    Call LLM and validate response with single retry on validation failure.

    Args:
        llm_call_func: Function that calls LLM and returns response text
        model_class: Pydantic model class to validate against
        schema_path: Optional JSON schema file for additional validation
        max_retries: Maximum number of retries (default: 1)

    Returns:
        Validated Pydantic model instance

    Raises:
        JSONValidationError: If validation fails after all retries
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Call LLM
            if attempt == 0:
                response = llm_call_func()
            else:
                # Retry with error feedback
                error_msg = f"上一次的JSON输出无效。错误: {last_error}。请重新生成有效的JSON。"
                response = llm_call_func(retry_message=error_msg)

            # Parse and validate
            return parse_and_validate(response, model_class, schema_path)

        except JSONValidationError as e:
            last_error = str(e)
            if attempt == max_retries:
                # Final attempt failed, raise error
                raise JSONValidationError(
                    f"Validation failed after {max_retries + 1} attempts: {last_error}"
                )

    # Should never reach here
    raise JSONValidationError("Unexpected error in validate_with_retry")


def format_validation_error_for_llm(error: JSONValidationError) -> str:
    """
    Format validation error message for LLM retry.

    Args:
        error: Validation error

    Returns:
        Formatted error message in Chinese
    """
    error_str = str(error)

    # Translate common error messages to Chinese
    translations = {
        "No JSON found in response": "响应中未找到JSON",
        "Invalid JSON": "JSON格式无效",
        "Schema validation failed": "JSON模式验证失败",
        "Pydantic validation failed": "数据验证失败",
    }

    for eng, chn in translations.items():
        if eng in error_str:
            error_str = error_str.replace(eng, chn)

    return f"JSON验证错误: {error_str}\n\n请确保输出有效的JSON格式，并包含所有必需字段。"
