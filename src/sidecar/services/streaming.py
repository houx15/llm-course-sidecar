"""Streaming orchestrator for real-time agent responses."""

import re
import logging
import asyncio
from typing import AsyncGenerator, Dict, Optional, Any

from .storage import Storage
from .memory_manager import MemoryManager
from .agent_runner import AgentRunner
from ..models.schemas import InstructionPacket

logger = logging.getLogger(__name__)


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
    consultation_guide_text: str = "",
    uploaded_files_info: Optional[Dict[str, Any]] = None,
    consultation_engine: Optional[Any] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Process a turn with streaming output.

    Execution flow:
    1. CA runs first to judge the current turn
    2. If unlock conditions met, run RMA (and expert consultation if requested)
    3. CA generates response after any RMA updates
    4. Stream CA response to user immediately
    5. MA runs in background to update report

    Yields events:
    - {'type': 'start'}
    - {'type': 'skip_task_confirm', 'current_task_id': '...', 'next_task_id': '...'}
    - {'type': 'consultation_start', 'expert_id': '...', 'title': '...'}
    - {'type': 'consultation_complete', 'expert_id': '...', 'consultation_id': '...'}
    - {'type': 'consultation_error', 'expert_id': '...', 'error': '...'}
    - {'type': 'companion_start'}
    - {'type': 'companion_chunk', 'content': '字'}
    - {'type': 'companion_complete'}
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

        # STEP 1: Run CA first to judge the current turn
        ca_response, turn_outcome = await agent_runner.run_companion(
            user_message=user_message,
            instruction_packet=instruction_packet,
            dynamic_report=dynamic_report,
            session_state=state,
            chapter_context=chapter_content.get("chapter_context", ""),
            task_list=chapter_content.get("task_list", ""),
            task_completion_principles=chapter_content.get("task_completion_principles", ""),
            interaction_protocol=chapter_content.get("interaction_protocol", ""),
            socratic_vs_direct=chapter_content.get("socratic_vs_direct", ""),
            memory_recent_turns=memory_sections["recent_turns"],
            available_experts_info=available_experts_info,
            uploaded_files_info=uploaded_files_info_text,
        )

        should_unlock, unlock_reason = _should_unlock_instruction(
            state=state,
            turn_outcome=turn_outcome,
            current_instruction=instruction_packet,
            memo_digest=memo_digest,
            uploaded_files_info=uploaded_files_info,
        )

        # v3.4.0: Diagnostic logging for skip detection
        skip_flag = getattr(turn_outcome, "student_wants_to_skip", None)
        logger.info(
            f"[SKIP-DIAG] turn_outcome.student_wants_to_skip={skip_flag}, "
            f"should_unlock={should_unlock}, unlock_reason={unlock_reason}"
        )

        # v3.4.0: Skip task confirmation — intercept before RMA
        if unlock_reason == "student_wants_to_skip":
            current_task_id = _extract_task_id(instruction_packet.current_focus)
            next_task_id = _find_next_task_id(state.subtask_status, current_task_id)
            logger.info(f"Skip task requested: current={current_task_id}, next={next_task_id}")

            # Yield confirmation event as a dialogue option box in chat
            yield {
                "type": "chat",
                "role": "assistant",
                "content": f"看起来你希望跳过当前任务 ({current_task_id or 'unknown'})，是否确定跳过并进入下一环节 ({next_task_id or 'none'})？",
                "options": [
                    {"label": "确认跳过", "action": "skip_task"},
                    {"label": "暂不跳过", "action": "cancel"}
                ]
            }

            # Stream CA's response (CA already acknowledged the skip intent)
            yield {"type": "companion_start"}
            for char in ca_response:
                yield {"type": "companion_chunk", "content": char}
                await asyncio.sleep(0.02)
            yield {"type": "companion_complete"}

            # Save turn data without running RMA — RMA deferred to /skip_task API
            state.turn_index = current_turn_index + 1
            storage.save_state(session_id, state)
            storage.save_turn(
                session_id=session_id,
                turn_index=current_turn_index,
                user_message=user_message,
                companion_response=ca_response,
                turn_outcome=turn_outcome.model_dump(),
            )

            # Run MA to update report even for skip turns
            memo_result = await agent_runner.run_memo(
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

            return  # Exit early — RMA will run when user confirms skip via API

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

        # STEP 2: Run RMA (and expert) only if current-turn judgment unlocks
        if should_unlock:
            rma_result = await agent_runner.run_roadmap_manager(
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

                    rma_final_result = await agent_runner.run_roadmap_manager_phase2(
                        phase1_result=rma_result,
                        expert_output=consultation_result.get("expert_output", {}),
                        expert_description=expert_description,
                        session_state=state,
                    )

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

            ca_response, turn_outcome = await agent_runner.run_companion(
                user_message=user_message,
                instruction_packet=new_instruction,
                dynamic_report=dynamic_report,
                session_state=state,
                chapter_context=chapter_content.get("chapter_context", ""),
                task_list=chapter_content.get("task_list", ""),
                task_completion_principles=chapter_content.get("task_completion_principles", ""),
                interaction_protocol=chapter_content.get("interaction_protocol", ""),
                socratic_vs_direct=chapter_content.get("socratic_vs_direct", ""),
                memory_recent_turns=memory_sections["recent_turns"],
                available_experts_info=available_experts_info,
                uploaded_files_info=uploaded_files_info_text,
                expert_output_summary=rma_guidance,
            )

            # v3.4.1: Stage 2 skip re-check — catch skip intent missed by Stage 1
            if getattr(turn_outcome, "student_wants_to_skip", False):
                current_task_id = _extract_task_id(new_instruction.current_focus)
                next_task_id = _find_next_task_id(state.subtask_status, current_task_id)
                logger.info(
                    f"[SKIP-DIAG] Stage 2 detected skip (Stage 1 missed): "
                    f"current={current_task_id}, next={next_task_id}"
                )

                yield {
                    "type": "chat",
                    "role": "assistant",
                    "content": f"看起来你希望跳过当前任务 ({current_task_id or 'unknown'})，是否确定跳过并进入下一环节 ({next_task_id or 'none'})？",
                    "options": [
                        {"label": "确认跳过", "action": "skip_task"},
                        {"label": "暂不跳过", "action": "cancel"}
                    ]
                }

                yield {"type": "companion_start"}
                for char in ca_response:
                    yield {"type": "companion_chunk", "content": char}
                    await asyncio.sleep(0.02)
                yield {"type": "companion_complete"}

                state.turn_index = current_turn_index + 1
                storage.save_state(session_id, state)
                storage.save_turn(
                    session_id=session_id,
                    turn_index=current_turn_index,
                    user_message=user_message,
                    companion_response=ca_response,
                    turn_outcome=turn_outcome.model_dump(),
                )

                memo_result = await agent_runner.run_memo(
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

                return  # Exit early — skip confirmation deferred to /skip_task API

        # STEP 4: Stream CA response to user immediately
        yield {"type": "companion_start"}
        for char in ca_response:
            yield {"type": "companion_chunk", "content": char}
            await asyncio.sleep(0.02)

        yield {"type": "companion_complete"}

        # STEP 5: Run MA in background (non-blocking for user)
        memo_result = await agent_runner.run_memo(
            user_message=user_message,
            companion_response=ca_response,
            turn_outcome=turn_outcome,
            current_report=dynamic_report,
            session_state=state,
            dynamic_report_template=templates.get("dynamic_report_template", ""),
            student_error_summary_template=templates.get("student_error_summary_template", ""),
            final_learning_report_template=templates.get("final_learning_report_template", ""),
        )

        # Update state based on unlock and progress
        if should_unlock:
            if turn_outcome.checkpoint_reached:
                state.attempts_since_last_progress = 0
                state.last_progress_turn = current_turn_index
            else:
                state.attempts_since_last_progress = 0
        else:
            logger.info(f"Keeping instruction locked at version {instruction_packet.instruction_version}")
            if turn_outcome.checkpoint_reached or memo_result.digest.progress_delta == "evidence_added":
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
        )

        # Save updated report
        storage.save_dynamic_report(session_id, memo_result.updated_report)
        storage.save_memo_digest(session_id, memo_result.digest)

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
    # v3.4.0: Unlock if student wants to skip current task (must be checked first!)
    if getattr(turn_outcome, "student_wants_to_skip", False):
        logger.info("Unlocking: student wants to skip current task")
        return True, "student_wants_to_skip"

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

    try:
        registry_path = Path(".metadata/experts/yellow_page.generated.json")
        if not registry_path.exists():
            logger.warning(f"Expert registry not found: {registry_path}")
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


def _extract_task_id(current_focus: str) -> str:
    """
    Extract task_id from InstructionPacket.current_focus.

    Patterns matched:
    - "AI研究范式转变地图（ai_research_paradigm_shift_map）" → "ai_research_paradigm_shift_map"
    - "task: ai_risk_identification" → "ai_risk_identification"
    - "(task_id_here)" → "task_id_here"
    """
    # Try Chinese parentheses first
    m = re.search(r'[（(]([a-zA-Z0-9_]+)[）)]', current_focus)
    if m:
        return m.group(1)
    # Try extracting after colon/equals
    m = re.search(r'(?:task|任务)[:\s=]+([a-zA-Z0-9_]+)', current_focus, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: find any snake_case identifier
    m = re.search(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', current_focus)
    if m:
        return m.group(1)
    return ""


def _find_next_task_id(subtask_status: dict, current_task_id: str) -> str:
    """
    Find the next unfinished task after current_task_id.

    Iterates subtask_status in insertion order and returns the first
    task with status 'not_started' or 'in_progress' that isn't the current one.
    """
    found_current = False
    for task_id, status_obj in subtask_status.items():
        if task_id == current_task_id:
            found_current = True
            continue
        if found_current:
            st = status_obj.status if hasattr(status_obj, 'status') else status_obj.get('status', '')
            if st in ('not_started', 'in_progress'):
                return task_id
    # If current not found or no next task after it, scan from beginning
    for task_id, status_obj in subtask_status.items():
        if task_id == current_task_id:
            continue
        st = status_obj.status if hasattr(status_obj, 'status') else status_obj.get('status', '')
        if st in ('not_started', 'in_progress'):
            return task_id
    return ""
