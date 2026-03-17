"""Streaming orchestrator for real-time agent responses."""

import logging
import asyncio
from typing import AsyncGenerator, Dict, Optional, Any, Tuple

from .storage import Storage
from .memory_manager import MemoryManager
from .agent_runner import AgentRunner
from .llm_client import LLMError
from ..models.schemas import InstructionPacket

logger = logging.getLogger(__name__)

# Fake-streaming parameters (1 char at a time, matching previous behavior)
_FAKE_CHUNK_SIZE = 1  # characters per chunk
_FAKE_CHUNK_DELAY = 0.02  # seconds between chunks


class TurnCancelled(Exception):
    """Raised when the user cancels the current turn mid-stream."""
    pass


def _check_cancelled(cancel_event: Optional[asyncio.Event]) -> None:
    """Raise TurnCancelled if the cancel event has been set."""
    if cancel_event is not None and cancel_event.is_set():
        raise TurnCancelled()


async def _run_memo_with_usage(
    session_id: str,
    current_turn_index: int,
    user_message: str,
    ca_response: str,
    turn_outcome,
    dynamic_report: str,
    state,
    templates: Dict[str, str],
    storage: Storage,
    agent_runner: AgentRunner,
) -> Dict[str, int]:
    """Run Memo Agent and return token usage. Saves report/digest to storage."""
    try:
        memo_result, ma_usage = await agent_runner.run_memo(
            user_message=user_message,
            companion_response=ca_response,
            turn_outcome=turn_outcome,
            current_report=dynamic_report,
            session_state=state,
            dynamic_report_template=templates.get("dynamic_report_template", ""),
            student_error_summary_template=templates.get("student_error_summary_template", ""),
            final_learning_report_template=templates.get("final_learning_report_template", ""),
        )
        storage.save_dynamic_report(session_id, memo_result.updated_report)
        storage.save_memo_digest(session_id, memo_result.digest)

        if state.turn_index > 0 and (current_turn_index % state.constraints.batch_error_log_every_n_turns == 0):
            if memo_result.error_entries:
                storage.append_student_error_summary(session_id, memo_result.error_entries)

        logger.info(
            f"MA complete for session {session_id} turn {current_turn_index} "
            f"(tokens: {ma_usage.get('input_tokens', 0)}+{ma_usage.get('output_tokens', 0)})"
        )
        return ma_usage
    except Exception as e:
        logger.warning(f"MA failed for session {session_id}: {e}")
        return {"input_tokens": 0, "output_tokens": 0}


async def process_turn_stream(
    session_id: str,
    user_message: str,
    storage: Storage,
    agent_runner: AgentRunner,
    memory_manager: MemoryManager,
    chapter_content: Dict[str, str],
    templates: Dict[str, str],
    available_experts_info: str = "",
    uploaded_files_info_text: str = "",
    recent_code_executions_text: str = "",
    consultation_guide_text: str = "",
    uploaded_files_info: Optional[Dict[str, Any]] = None,
    consultation_engine: Optional[Any] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Process a turn with fake-streamed output.

    Execution flow:
    1. CA runs (non-streaming, exact token counts)
    2. Check unlock → if yes, run RMA (+ expert), then CA step 2
    3. Start MA concurrently, fake-stream CA response to user
    4. After fake-stream, await MA, emit MA token usage

    Yields events:
    - {'type': 'start'}
    - {'type': 'consultation_start', 'expert_id': '...', 'title': '...'}
    - {'type': 'consultation_complete', 'expert_id': '...', 'consultation_id': '...'}
    - {'type': 'consultation_error', 'expert_id': '...', 'error': '...'}
    - {'type': 'companion_start'}
    - {'type': 'companion_chunk', 'content': '...'}
    - {'type': 'companion_complete'}
    - {'type': 'token_usage', 'agent': '...', ...}
    """
    # Load session state
    state = storage.load_state(session_id)
    instruction_packet = storage.load_instruction_packet(session_id)
    dynamic_report = storage.load_dynamic_report(session_id)
    memory_sections = memory_manager.build_memory_sections(
        session_id=session_id,
        user_message_length=len(user_message),
    )

    yield {"type": "start"}

    try:
        if uploaded_files_info is None:
            uploaded_files_info = {"has_uploaded_files": False, "file_count": 0, "files": []}

        memo_digest = storage.load_memo_digest(session_id) or {
            "key_observations": [],
            "student_struggles": [],
            "student_strengths": [],
            "student_sentiment": "engaged",
            "blocker_type": "none",
            "progress_delta": "none"
        }

        current_turn_index = state.turn_index

        # Accumulate token usage across all agent calls in this turn
        turn_token_usage: Dict[str, Dict[str, int]] = {}

        # Check cancellation before starting expensive LLM calls
        _check_cancelled(cancel_event)

        # ── STEP 1: Run CA (non-streaming → exact token counts) ──────────
        try:
            ca_response, turn_outcome, ca_usage = await agent_runner.run_companion(
                user_message=user_message,
                instruction_packet=instruction_packet,
                dynamic_report=dynamic_report,
                session_state=state,
                chapter_context=chapter_content.get("chapter_context", ""),
                task_list=chapter_content.get("task_list", ""),
                task_completion_principles=chapter_content.get("task_completion_principles", ""),
                interaction_protocol=chapter_content.get("interaction_protocol", ""),
                socratic_vs_direct=chapter_content.get("socratic_vs_direct", ""),
                memory_long_term=memory_sections["long_term"],
                memory_mid_term=memory_sections["mid_term"],
                memory_recent_turns=memory_sections["recent_turns"],
                available_experts_info=available_experts_info,
                uploaded_files_info=uploaded_files_info_text,
                recent_code_executions=recent_code_executions_text,
            )
        except LLMError as e:
            logger.error(f"LLM error in CA step 1: {e}")
            yield {"type": "llm_error", "error": str(e)}
            return

        ca_input = ca_usage.get("input_tokens", 0)
        ca_output = ca_usage.get("output_tokens", 0)
        turn_token_usage.setdefault("CA", {"input_tokens": 0, "output_tokens": 0})
        turn_token_usage["CA"]["input_tokens"] += ca_input
        turn_token_usage["CA"]["output_tokens"] += ca_output

        # ── STEP 2: Check unlock ─────────────────────────────────────────
        should_unlock, unlock_reason = _should_unlock_instruction(
            state=state,
            turn_outcome=turn_outcome,
            current_instruction=instruction_packet,
            memo_digest=memo_digest,
            uploaded_files_info=uploaded_files_info,
        )

        if not should_unlock and unlock_reason.startswith("lock_until_"):
            previous_blocker_type = "none"
            if hasattr(memo_digest, "blocker_type"):
                previous_blocker_type = memo_digest.blocker_type
            elif isinstance(memo_digest, dict):
                previous_blocker_type = memo_digest.get("blocker_type", "none")
            storage.append_system_event(
                session_id,
                "instruction_unlock_blocked",
                "Unlock blocked by lock_until condition.",
                {
                    "turn_index": current_turn_index,
                    "lock_until": instruction_packet.lock_until,
                    "reason": unlock_reason,
                    "checkpoint_reached": turn_outcome.checkpoint_reached,
                    "attempts_since_last_progress": state.attempts_since_last_progress,
                    "max_attempts_before_unlock": state.constraints.max_attempts_before_unlock,
                    "blocker_type": turn_outcome.blocker_type,
                    "previous_blocker_type": previous_blocker_type,
                    "expert_consultation_needed": getattr(turn_outcome, "expert_consultation_needed", False),
                    "expert_consultation_reason": getattr(turn_outcome, "expert_consultation_reason", ""),
                },
            )

        rma_final_result = None
        new_instruction = instruction_packet

        _check_cancelled(cancel_event)

        # ── STEP 3: Run RMA (and expert) only if unlock ──────────────────
        if should_unlock:
            rma_result, rma_usage = await agent_runner.run_roadmap_manager(
                dynamic_report=dynamic_report,
                memo_digest=memo_digest,
                session_state=state,
                turn_outcome=turn_outcome,
                chapter_context=chapter_content.get("chapter_context", ""),
                task_list=chapter_content.get("task_list", ""),
                task_completion_principles=chapter_content.get("task_completion_principles", ""),
                consultation_guide=consultation_guide_text,
                available_experts=available_experts_info,
                uploaded_files_info=uploaded_files_info,
            )
            rma_input = rma_usage.get("input_tokens", 0)
            rma_output = rma_usage.get("output_tokens", 0)
            yield {
                "type": "token_usage",
                "agent": "RMA",
                "turn_index": current_turn_index,
                "input_tokens": rma_input,
                "output_tokens": rma_output,
            }
            turn_token_usage.setdefault("RMA", {"input_tokens": 0, "output_tokens": 0})
            turn_token_usage["RMA"]["input_tokens"] += rma_input
            turn_token_usage["RMA"]["output_tokens"] += rma_output

            if not rma_result.consultation_request:
                storage.append_system_event(
                    session_id,
                    "rma_consultation_not_requested",
                    "RMA did not request expert consultation for this turn.",
                    {
                        "turn_index": current_turn_index,
                        "should_unlock": should_unlock,
                    },
                )

            logger.info(f"Unlocking instruction: updating to version {state.current_instruction_version + 1}")
            new_instruction = rma_result.instruction_packet
            new_instruction.instruction_version = state.current_instruction_version + 1
            storage.save_instruction_packet(session_id, new_instruction)

            state.update(rma_result.state_update)
            state.current_instruction_version = new_instruction.instruction_version

            if rma_result.consultation_request and consultation_engine:
                logger.info(
                    f"RMA requested consultation with {rma_result.consultation_request.expert_id}"
                )
                logger.info(f"Reasoning: {rma_result.consultation_request.reasoning}")

                yield {
                    "type": "consultation_start",
                    "expert_id": rma_result.consultation_request.expert_id,
                    "title": rma_result.consultation_request.consulting_letter_title,
                }

                try:
                    consultation_result = await consultation_engine.consult_expert(
                        expert_id=rma_result.consultation_request.expert_id,
                        question=rma_result.consultation_request.question,
                        context=rma_result.consultation_request.context,
                        expected_output_type=rma_result.consultation_request.expected_output_type,
                        session_id=session_id,
                        session_state=state,
                        scenario_id=rma_result.consultation_request.scenario_id,
                    )

                    if consultation_result.get("instruction_updates"):
                        logger.info("Applying instruction updates from binding rules")
                        for key, value in consultation_result["instruction_updates"].items():
                            if hasattr(new_instruction, key):
                                setattr(new_instruction, key, value)
                                logger.info(f"  Updated {key} = {value}")
                        storage.save_instruction_packet(session_id, new_instruction)

                    logger.info(
                        f"Consultation completed: {consultation_result['consultation_id']}"
                    )
                    if consultation_result.get("binding_rules_triggered"):
                        logger.info(
                            "Binding rules triggered: "
                            f"{[r['rule_name'] for r in consultation_result['binding_rules_triggered']]}"
                        )

                    logger.info("Running RMA Phase 2: Interpreting expert output")

                    expert_id = rma_result.consultation_request.expert_id
                    expert_description = _get_expert_description(expert_id)

                    rma_final_result, rma_phase2_usage = await agent_runner.run_roadmap_manager_phase2(
                        phase1_result=rma_result,
                        expert_output=consultation_result.get("expert_output", {}),
                        expert_description=expert_description,
                        session_state=state,
                    )
                    rma2_input = rma_phase2_usage.get("input_tokens", 0)
                    rma2_output = rma_phase2_usage.get("output_tokens", 0)
                    yield {
                        "type": "token_usage",
                        "agent": "RMA",
                        "turn_index": current_turn_index,
                        "input_tokens": rma2_input,
                        "output_tokens": rma2_output,
                    }
                    turn_token_usage.setdefault("RMA", {"input_tokens": 0, "output_tokens": 0})
                    turn_token_usage["RMA"]["input_tokens"] += rma2_input
                    turn_token_usage["RMA"]["output_tokens"] += rma2_output

                    new_instruction = rma_final_result.instruction_packet
                    storage.save_instruction_packet(session_id, new_instruction)

                    state.update(rma_final_result.state_update)

                    yield {
                        "type": "consultation_complete",
                        "expert_id": rma_result.consultation_request.expert_id,
                        "consultation_id": consultation_result['consultation_id'],
                    }

                except Exception as e:
                    logger.error(f"Failed to execute consultation: {e}")
                    storage.append_system_event(
                        session_id,
                        "expert_consultation_failed",
                        f"专家咨询失败: {e}",
                        {
                            "expert_id": rma_result.consultation_request.expert_id,
                            "scenario_id": rma_result.consultation_request.scenario_id,
                        },
                    )
                    yield {
                        "type": "consultation_error",
                        "expert_id": rma_result.consultation_request.expert_id,
                        "error": str(e),
                    }

            rma_guidance = ""
            if rma_final_result:
                rma_guidance = f"{rma_final_result.expert_consultation_summary}\n\n{rma_final_result.guidance_for_ca}"

            # Run CA step 2 with updated instruction (non-streaming)
            _check_cancelled(cancel_event)
            try:
                ca_response, turn_outcome, ca2_usage = await agent_runner.run_companion(
                    user_message=user_message,
                    instruction_packet=new_instruction,
                    dynamic_report=dynamic_report,
                    session_state=state,
                    chapter_context=chapter_content.get("chapter_context", ""),
                    task_list=chapter_content.get("task_list", ""),
                    task_completion_principles=chapter_content.get("task_completion_principles", ""),
                    interaction_protocol=chapter_content.get("interaction_protocol", ""),
                    socratic_vs_direct=chapter_content.get("socratic_vs_direct", ""),
                    memory_long_term=memory_sections["long_term"],
                    memory_mid_term=memory_sections["mid_term"],
                    memory_recent_turns=memory_sections["recent_turns"],
                    available_experts_info=available_experts_info,
                    uploaded_files_info=uploaded_files_info_text,
                    recent_code_executions=recent_code_executions_text,
                    expert_output_summary=rma_guidance,
                )
            except LLMError as e:
                logger.error(f"LLM error in CA step 2: {e}")
                yield {"type": "llm_error", "error": str(e)}
                return

            ca2_input = ca2_usage.get("input_tokens", 0)
            ca2_output = ca2_usage.get("output_tokens", 0)
            turn_token_usage["CA"]["input_tokens"] += ca2_input
            turn_token_usage["CA"]["output_tokens"] += ca2_output

        # ── STEP 4: Emit CA token usage, then fake-stream + MA in parallel ─
        yield {
            "type": "token_usage",
            "agent": "CA",
            "turn_index": current_turn_index,
            "input_tokens": turn_token_usage["CA"]["input_tokens"],
            "output_tokens": turn_token_usage["CA"]["output_tokens"],
        }

        # Update state
        if should_unlock:
            if turn_outcome.checkpoint_reached:
                state.attempts_since_last_progress = 0
                state.last_progress_turn = current_turn_index
            else:
                state.attempts_since_last_progress = 0
        else:
            logger.info(f"Keeping instruction locked at version {instruction_packet.instruction_version}")
            if turn_outcome.checkpoint_reached:
                state.attempts_since_last_progress = 0
                state.last_progress_turn = current_turn_index
            else:
                state.attempts_since_last_progress += 1

        state.turn_index = current_turn_index + 1
        storage.save_state(session_id, state)

        # Save turn data
        storage.save_turn(
            session_id=session_id,
            turn_index=current_turn_index,
            user_message=user_message,
            companion_response=ca_response,
            turn_outcome=turn_outcome.model_dump(),
            token_usage=turn_token_usage if turn_token_usage else None,
        )

        # ── Start MA concurrently, then fake-stream CA response ──────────
        ma_task = asyncio.create_task(_run_memo_with_usage(
            session_id=session_id,
            current_turn_index=current_turn_index,
            user_message=user_message,
            ca_response=ca_response,
            turn_outcome=turn_outcome,
            dynamic_report=dynamic_report,
            state=state,
            templates=templates,
            storage=storage,
            agent_runner=agent_runner,
        ))

        # Fake-stream the CA response to the user
        yield {"type": "companion_start"}
        for i in range(0, len(ca_response), _FAKE_CHUNK_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                ma_task.cancel()
                yield {"type": "companion_complete"}
                raise TurnCancelled()
            yield {"type": "companion_chunk", "content": ca_response[i:i + _FAKE_CHUNK_SIZE]}
            await asyncio.sleep(_FAKE_CHUNK_DELAY)
        yield {"type": "companion_complete"}

        # ── Await MA, emit its token usage ───────────────────────────────
        ma_usage = await ma_task
        ma_input = ma_usage.get("input_tokens", 0)
        ma_output = ma_usage.get("output_tokens", 0)
        if ma_input > 0 or ma_output > 0:
            yield {
                "type": "token_usage",
                "agent": "MA",
                "turn_index": current_turn_index,
                "input_tokens": ma_input,
                "output_tokens": ma_output,
            }
            turn_token_usage["MA"] = {"input_tokens": ma_input, "output_tokens": ma_output}

    except TurnCancelled:
        logger.info(f"Turn cancelled for session {session_id}, discarding turn {current_turn_index}")
        raise  # propagate to caller — no turn saved
    except Exception as e:
        logger.error(f"Error in streaming turn: {e}", exc_info=True)
        yield {"type": "error", "message": str(e)}


def _should_unlock_instruction(
    state,
    turn_outcome,
    current_instruction: InstructionPacket,
    memo_digest=None,
    uploaded_files_info: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """
    Determine if instruction packet should be unlocked.

    Unlock conditions (based on CA's judgment):
    1. checkpoint_reached = True (CA判定达成检查点)
    2. attempts_exceeded (stuck detection)
    """
    # Unlock if CA signals expert consultation needed
    if getattr(turn_outcome, "expert_consultation_needed", False):
        reason = getattr(turn_outcome, "expert_consultation_reason", "unspecified")
        logger.info(f"Unlocking: expert consultation needed (reason: {reason})")
        return True, "expert_consultation_needed"

    lock_until = current_instruction.lock_until

    if lock_until == "checkpoint_reached":
        if turn_outcome.checkpoint_reached:
            logger.info("Unlocking: checkpoint reached (CA判定)")
            return True, "checkpoint_reached"
        return False, "lock_until_checkpoint_not_reached"
    elif lock_until == "attempts_exceeded":
        if state.attempts_since_last_progress >= state.constraints.max_attempts_before_unlock:
            logger.info(
                "Unlocking: stuck detected "
                f"({state.attempts_since_last_progress} >= {state.constraints.max_attempts_before_unlock})"
            )
            return True, "attempts_exceeded"
        return False, "lock_until_attempts_not_exceeded"
    elif lock_until == "new_error_type":
        previous_blocker_type = "none"
        if memo_digest is not None:
            if hasattr(memo_digest, "blocker_type"):
                previous_blocker_type = memo_digest.blocker_type
            elif isinstance(memo_digest, dict):
                previous_blocker_type = memo_digest.get("blocker_type", "none")
        if turn_outcome.blocker_type != "none" and turn_outcome.blocker_type != previous_blocker_type:
            logger.info(f"Unlocking: new error type detected ({turn_outcome.blocker_type})")
            return True, "new_error_type"
        return False, "lock_until_new_error_type_not_met"
    elif lock_until == "user_uploads_suitable_dataset_or_uses_example":
        has_new_uploads = bool(uploaded_files_info and uploaded_files_info.get("has_new_uploads"))
        if has_new_uploads:
            logger.info("Unlocking: new dataset uploaded after unsuitability")
            return True, "user_uploads_suitable_dataset_or_uses_example"
        return False, "lock_until_new_upload_not_found"

    # Otherwise, keep locked
    return False, "lock_until_not_met"


def _build_rma_decision_message(rma_result) -> str:
    """
    Build a natural-language summary of RMA's decision for the user.

    Note: We don't expose internal reasoning to users.
    The consultation progress is shown via the thinking-style progress bar.
    """
    return ""


def _get_expert_description(expert_id: str) -> str:
    """
    Get expert description from expert registry.

    Args:
        expert_id: The expert identifier

    Returns:
        Expert description text, or a default message if not found
    """
    from pathlib import Path
    import json
    import os

    try:
        candidates = []
        explicit = os.getenv("EXPERT_YELLOW_PAGE_PATH", "").strip()
        if explicit:
            candidates.append(Path(explicit))

        experts_dir = os.getenv("EXPERTS_DIR", "").strip()
        if experts_dir:
            root = Path(experts_dir)
            candidates.extend(
                [
                    root / "yellow_page.generated.json",
                    root / ".metadata" / "experts" / "yellow_page.generated.json",
                    root.parent / "yellow_page.generated.json",
                    root.parent / ".metadata" / "experts" / "yellow_page.generated.json",
                ]
            )

        candidates.append(Path(".metadata/experts/yellow_page.generated.json"))
        registry_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if not registry_path:
            logger.warning("Expert registry not found")
            return f"Expert {expert_id} (description not available)"

        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        for expert in registry.get("experts", []):
            if expert.get("expert_id") == expert_id:
                return expert.get("description", f"Expert {expert_id}")

        logger.warning(f"Expert {expert_id} not found in registry")
        return f"Expert {expert_id} (description not available)"

    except Exception as e:
        logger.error(f"Failed to load expert description: {e}")
        return f"Expert {expert_id} (description not available)"
