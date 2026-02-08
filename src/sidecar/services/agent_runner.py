"""Agent runner for executing agents with prompt injection and JSON validation."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from .llm_client import get_llm_client, LLMError
from .json_utils import (
    parse_and_validate,
    JSONValidationError,
    get_default_instruction_packet,
    get_default_turn_outcome,
    get_default_memo_digest,
    get_default_state_update,
)
from ..models.schemas import (
    InstructionPacket,
    TurnOutcome,
    MemoDigest,
    MemoResult,
    RoadmapManagerResult,
    SessionState,
    StudentErrorEntry,
)

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runner for executing agents with prompt injection and validation."""

    def __init__(self, agents_dir: str = "app/server/agents"):
        """
        Initialize agent runner.

        Args:
            agents_dir: Directory containing agent prompt files
        """
        self.agents_dir = Path(agents_dir)
        self.llm_client = get_llm_client()

    def _load_prompt_template(self, agent_name: str) -> str:
        """Load agent prompt template from file."""
        prompt_file = self.agents_dir / f"{agent_name}.prompt.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()

    def _load_tool_schema(self, tool_name: str) -> str:
        """Load tool schema from tools directory."""
        tool_path = self.agents_dir / "tools" / f"{tool_name}.json"
        if not tool_path.exists():
            logger.warning(f"Tool schema file not found: {tool_path}")
            return ""

        with open(tool_path, "r", encoding="utf-8") as f:
            return f.read()

    def _inject_context(self, template: str, context: Dict[str, str]) -> str:
        """
        Inject context into prompt template.

        Args:
            template: Prompt template with {{PLACEHOLDER}} markers
            context: Dictionary of placeholder values

        Returns:
            Prompt with placeholders replaced
        """
        prompt = template
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

    async def _call_llm_with_retry(
        self,
        prompt: str,
        model_class,
        max_retries: int = 1,
    ):
        """
        Call LLM with retry logic for JSON validation.

        Args:
            prompt: Full prompt to send to LLM
            model_class: Pydantic model class for validation
            max_retries: Maximum number of retries

        Returns:
            Validated model instance
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Add retry message if this is a retry
                if attempt > 0:
                    retry_prompt = f"{prompt}\n\n---\n\n上一次的JSON输出无效。错误: {last_error}\n\n请重新生成有效的JSON，确保：\n1. JSON格式正确\n2. 包含所有必需字段\n3. 字段类型正确"
                    response = await self.llm_client.generate(retry_prompt)
                else:
                    response = await self.llm_client.generate(prompt)

                # Parse and validate
                return parse_and_validate(response, model_class)

            except JSONValidationError as e:
                last_error = str(e)
                logger.warning(f"JSON validation failed (attempt {attempt + 1}): {last_error}")

                if attempt == max_retries:
                    logger.error(f"All retry attempts failed: {last_error}")
                    raise

        raise JSONValidationError("Unexpected error in retry logic")

    async def _recover_turn_outcome(
        self,
        user_message: str,
        companion_response: str,
        instruction_packet: InstructionPacket,
        session_state: SessionState,
        max_retries: int = 1,
    ) -> TurnOutcome:
        """
        Recover TurnOutcome JSON when CA response lacks valid JSON.

        This keeps the original companion response while regenerating only the JSON.
        """
        prompt = (
            "你是JSON生成器。请根据上下文输出TurnOutcome的JSON。\n"
            "要求：只输出JSON，不要任何额外文本或解释。\n"
            "字段必须齐全，字符串字段使用简体中文。\n\n"
            "TurnOutcome字段：\n"
            "- what_user_attempted\n"
            "- what_user_observed\n"
            "- ca_teaching_mode: socratic|direct\n"
            "- ca_next_suggestion\n"
            "- checkpoint_reached: true/false\n"
            "- blocker_type: none|scaffolding|core_concept|core_implementation|external_resource_needed\n"
            "- student_sentiment: engaged|confused|frustrated|fatigued\n"
            "- evidence_for_subtasks: 数组，可为空\n\n"
            "会话状态：\n"
            f"{session_state.model_dump_json(indent=2)}\n\n"
            "指令包：\n"
            f"{instruction_packet.model_dump_json(indent=2)}\n\n"
            "用户消息：\n"
            f"{user_message}\n\n"
            "Companion回复：\n"
            f"{companion_response}\n"
        )

        return await self._call_llm_with_retry(prompt, TurnOutcome, max_retries=max_retries)

    async def run_companion(
        self,
        user_message: str,
        instruction_packet: InstructionPacket,
        dynamic_report: str,
        session_state: SessionState,
        chapter_context: str,
        task_list: str,
        task_completion_principles: str,
        interaction_protocol: str,
        socratic_vs_direct: str,
        memory_long_term: str,
        memory_mid_term: str,
        memory_recent_turns: str,
        available_experts_info: str = "",  # v3.2.0: Expert information
        uploaded_files_info: str = "",  # v3.2.0: Uploaded files information
        expert_output_summary: Optional[str] = None,  # v3.2.0: Expert consultation results
    ) -> Tuple[str, TurnOutcome]:
        """
        Run Companion Agent.

        Returns:
            Tuple of (companion_response, turn_outcome)
        """
        # Load template
        template = self._load_prompt_template("companion")

        # Inject context
        context = {
            "CHAPTER_CONTEXT": chapter_context,
            "TASK_LIST": task_list,
            "TASK_COMPLETION_PRINCIPLES": task_completion_principles,
            "INTERACTION_PROTOCOL": interaction_protocol,
            "SOCRATIC_VS_DIRECT": socratic_vs_direct,
            "SESSION_STATE_JSON": session_state.model_dump_json(indent=2),
            "INSTRUCTION_PACKET_JSON": instruction_packet.model_dump_json(indent=2),
            "DYNAMIC_REPORT": dynamic_report,
            "MEMORY_LONG_TERM": memory_long_term,
            "MEMORY_MID_TERM": memory_mid_term,
            "MEMORY_RECENT_TURNS": memory_recent_turns,
            "USER_MESSAGE": user_message,
            "AVAILABLE_EXPERTS_INFO": available_experts_info,  # v3.2.0
            "UPLOADED_FILES_INFO": uploaded_files_info,  # v3.2.0
            "EXPERT_OUTPUT_SUMMARY": expert_output_summary or "",  # v3.2.0
        }
        prompt = self._inject_context(template, context)

        # Call LLM with retry
        try:
            response = await self.llm_client.generate(prompt)

            # Extract companion response (text before JSON block)
            companion_response = response
            if "```json" in response:
                companion_response = response.split("```json")[0].strip()

            # Parse turn outcome from JSON block
            try:
                turn_outcome = parse_and_validate(response, TurnOutcome)
            except JSONValidationError as e:
                logger.warning(f"Failed to parse turn outcome: {e}")
                try:
                    turn_outcome = await self._recover_turn_outcome(
                        user_message=user_message,
                        companion_response=companion_response,
                        instruction_packet=instruction_packet,
                        session_state=session_state,
                        max_retries=1,
                    )
                except (LLMError, JSONValidationError) as recover_error:
                    logger.error(f"Failed to recover turn outcome: {recover_error}")
                    turn_outcome = get_default_turn_outcome()

            return companion_response, turn_outcome

        except (LLMError, JSONValidationError) as e:
            logger.error(f"Companion agent failed: {e}")
            # Return fallback response
            return (
                "抱歉，我遇到了一些技术问题。请再试一次，或者告诉我你想做什么，我会尽力帮助你。",
                get_default_turn_outcome()
            )

    async def run_memo(
        self,
        user_message: str,
        companion_response: str,
        turn_outcome: TurnOutcome,
        current_report: str,
        session_state: SessionState,
        dynamic_report_template: str,
        student_error_summary_template: str,
        final_learning_report_template: str,
    ) -> MemoResult:
        """
        Run Memo Agent.

        Returns:
            MemoResult with updated report, digest, and error entries
        """
        # Load template
        template = self._load_prompt_template("memo")

        # Inject context
        context = {
            "DYNAMIC_REPORT_TEMPLATE": dynamic_report_template,
            "STUDENT_ERROR_SUMMARY_TEMPLATE": student_error_summary_template,
            "FINAL_LEARNING_REPORT_TEMPLATE": final_learning_report_template,
            "SESSION_STATE_JSON": session_state.model_dump_json(indent=2),
            "CURRENT_DYNAMIC_REPORT": current_report,
            "USER_MESSAGE": user_message,
            "COMPANION_RESPONSE": companion_response,
            "TURN_OUTCOME_JSON": turn_outcome.model_dump_json(indent=2),
        }
        prompt = self._inject_context(template, context)

        # Call LLM with retry
        try:
            result = await self._call_llm_with_retry(prompt, MemoResult, max_retries=1)
            return result

        except (LLMError, JSONValidationError) as e:
            logger.error(f"Memo agent failed: {e}")
            # Return fallback result
            return MemoResult(
                updated_report=current_report,
                digest=get_default_memo_digest(),
                error_entries=[]
            )

    async def run_roadmap_manager(
        self,
        dynamic_report: str,
        memo_digest: MemoDigest,
        session_state: SessionState,
        chapter_context: str,
        task_list: str,
        task_completion_principles: str,
        consultation_guide: str = "",
        available_experts: str = "",
        uploaded_files_info: dict = None,  # v3.2.0: Add uploaded files info
        turn_outcome: Optional[TurnOutcome] = None,
    ) -> RoadmapManagerResult:
        """
        Run Roadmap Manager Agent.

        Args:
            consultation_guide: v3.1 consultation guide text (Markdown) for RMA prompt
            available_experts: v3.1 available experts info (Markdown) for RMA prompt
            uploaded_files_info: v3.2.0 uploaded files information
            turn_outcome: CA's current-turn judgment output (optional)

        Returns:
            RoadmapManagerResult with instruction packet and state update
        """
        # Load template
        template = self._load_prompt_template("roadmap_manager")

        # Load consult_expert tool schema
        tool_schema = self._load_tool_schema("consult_expert_tool")

        # v3.2.0: Format uploaded files info
        if uploaded_files_info is None:
            uploaded_files_info = {"has_uploaded_files": False, "file_count": 0, "files": []}

        uploaded_files_text = "## 已上传文件\n\n"
        if uploaded_files_info["has_uploaded_files"]:
            new_files = [f for f in uploaded_files_info["files"] if f.get("is_new")]
            existing_files = [f for f in uploaded_files_info["files"] if not f.get("is_new")]

            if new_files:
                uploaded_files_text += "### 🆕 本回合新上传的文件\n\n"
                for file in new_files:
                    uploaded_files_text += (
                        f"- **{file['filename']}** ({file['file_type']}, {file['size']} bytes) 🆕\n"
                    )
                uploaded_files_text += "\n"

            if existing_files:
                uploaded_files_text += "### 已存在的文件\n\n"
                for file in existing_files:
                    uploaded_files_text += (
                        f"- **{file['filename']}** ({file['file_type']}, {file['size']} bytes)\n"
                    )
                uploaded_files_text += "\n"

            uploaded_files_text += (
                "**注意**：如本回合出现新上传文件，应结合consultation_guide的must_consult规则判断是否立即咨询专家。\n"
            )
        else:
            uploaded_files_text += "用户尚未上传任何文件。\n"

        # Inject context
        context = {
            "CHAPTER_CONTEXT": chapter_context,
            "TASK_LIST": task_list,
            "TASK_COMPLETION_PRINCIPLES": task_completion_principles,
            "SESSION_STATE_JSON": session_state.model_dump_json(indent=2),
            "DYNAMIC_REPORT": dynamic_report,
            "MEMO_DIGEST_JSON": memo_digest.model_dump_json(indent=2),
            "TURN_OUTCOME_JSON": (
                turn_outcome.model_dump_json(indent=2) if turn_outcome else "{}"
            ),
            "CONSULTATION_GUIDE": consultation_guide if consultation_guide else "(No consultation guide available)",
            "AVAILABLE_EXPERTS": available_experts if available_experts else "(No expert information available)",
            "CONSULT_EXPERT_TOOL_SCHEMA": tool_schema if tool_schema else "(Tool schema not available)",
            "UPLOADED_FILES_INFO": uploaded_files_text,  # v3.2.0: Add uploaded files info
        }
        prompt = self._inject_context(template, context)

        # Call LLM with retry
        try:
            result = await self._call_llm_with_retry(prompt, RoadmapManagerResult, max_retries=1)
            return result

        except (LLMError, JSONValidationError) as e:
            logger.error(f"Roadmap manager failed: {e}")
            # Return fallback result
            return RoadmapManagerResult(
                instruction_packet=get_default_instruction_packet(),
                state_update=get_default_state_update()
            )

    async def run_roadmap_manager_phase2(
        self,
        phase1_result: RoadmapManagerResult,
        expert_output: Dict,
        expert_description: str,
        session_state: SessionState,
    ):
        """
        Run RMA Phase 2: Interpret expert output and provide guidance to CA.

        Args:
            phase1_result: The initial RMA result with consultation_request
            expert_output: Raw output from the expert
            expert_description: The expert's description/capabilities
            session_state: Current session state

        Returns:
            RoadmapManagerFinalResult with expert summary and CA guidance
        """
        from ..models.schemas import RoadmapManagerFinalResult

        # Load phase 2 template
        template = self._load_prompt_template("roadmap_manager_phase2")

        # Inject context
        context = {
            "INSTRUCTION_PACKET_JSON": phase1_result.instruction_packet.model_dump_json(indent=2),
            "STATE_UPDATE_JSON": str(phase1_result.state_update),
            "SESSION_STATE_JSON": session_state.model_dump_json(indent=2),
            "EXPERT_ID": phase1_result.consultation_request.expert_id,
            "EXPERT_DESCRIPTION": expert_description,
            "CONSULTATION_QUESTION": phase1_result.consultation_request.question,
            "CONSULTATION_CONTEXT": str(phase1_result.consultation_request.context),
            "EXPERT_OUTPUT_JSON": str(expert_output),
        }
        prompt = self._inject_context(template, context)

        # Call LLM with retry
        try:
            result = await self._call_llm_with_retry(prompt, RoadmapManagerFinalResult, max_retries=1)
            return result

        except (LLMError, JSONValidationError) as e:
            logger.error(f"Roadmap manager phase 2 failed: {e}")
            # Return fallback result with empty summaries
            return RoadmapManagerFinalResult(
                instruction_packet=phase1_result.instruction_packet,
                state_update=phase1_result.state_update,
                expert_consultation_summary="",
                guidance_for_ca=""
            )
