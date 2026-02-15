"""Orchestrator service for coordinating multi-agent interactions."""

import logging
import uuid
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Optional, List, Any

from .storage import Storage, StorageError
from .agent_runner import AgentRunner
from .memory_manager import MemoryManager
from .consultation_engine import ConsultationEngine
from .expert_report_generator import ExpertReportGenerator
from .performance_tracker import get_tracker
from .error_recovery import ErrorRecovery, ErrorRecoveryContext, RecoveryStrategy
from ..models.schemas import (
    SessionState,
    SessionConstraints,
    SubtaskStatus,
    InstructionPacket,
    ConsultationContext,
)
from ..config import settings

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Custom exception for orchestrator errors."""
    pass


class Orchestrator:
    """Main orchestration service for multi-agent tutor system."""

    def __init__(
        self,
        storage: Optional[Storage] = None,
        agent_runner: Optional[AgentRunner] = None,
        memory_manager: Optional[MemoryManager] = None,
        curriculum_dir: str = "curriculum",
        experts_dir: str = "experts",
        main_agents_dir: Optional[str] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            storage: Storage instance (creates new if None)
            agent_runner: Agent runner instance (creates new if None)
            memory_manager: Memory manager instance (creates new if None)
            curriculum_dir: Directory containing curriculum content
            experts_dir: Directory containing expert definitions
        """
        self.storage = storage or Storage()
        self.agent_runner = agent_runner or AgentRunner()
        self.memory_manager = memory_manager or MemoryManager(self.storage)
        self.curriculum_dir = Path(curriculum_dir)
        self.experts_dir = Path(experts_dir)
        self.main_agents_dir = Path(main_agents_dir) if main_agents_dir else None

        # v3.0: Initialize consultation engine and report generator
        try:
            self.consultation_engine = ConsultationEngine(
                experts_root=self.experts_dir,
                curriculum_root=self.curriculum_dir,
            )
            self.expert_report_generator = ExpertReportGenerator()
            logger.info("Consultation engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize consultation engine: {e}")
            self.consultation_engine = None
            self.expert_report_generator = None

        # Session-level locks to prevent concurrent turn processing
        self._session_locks: Dict[str, asyncio.Lock] = {}

    def _get_global_prompt_candidates(self, file_name: str) -> List[Path]:
        """
        Build deterministic fallback candidates for chapter-global prompt files.
        """
        roots: List[Path] = []

        env_main_agents = os.getenv("MAIN_AGENTS_DIR", "").strip()
        if env_main_agents:
            roots.append(Path(env_main_agents))

        if self.main_agents_dir:
            roots.append(self.main_agents_dir)

        # Repo-level fallback used in local development.
        repo_root = Path(__file__).resolve().parents[3]
        roots.append(repo_root / "content" / "agents")

        unique_roots = []
        seen = set()
        for root in roots:
            marker = str(root.resolve()) if root.exists() else str(root)
            if marker not in seen:
                seen.add(marker)
                unique_roots.append(root)

        candidates: List[Path] = []
        for root in unique_roots:
            candidates.extend(
                [
                    root / file_name,
                    root / "shared" / file_name,
                    root / "companion" / file_name,
                    root / "global" / file_name,
                ]
            )

        return candidates

    def _load_global_prompt_fallback(self, file_name: str) -> str:
        """
        Load chapter-global prompt markdown using deterministic fallback order.
        """
        for candidate in self._get_global_prompt_candidates(file_name):
            try:
                if candidate.exists():
                    logger.info(f"Using global prompt fallback for {file_name}: {candidate}")
                    return candidate.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed reading fallback candidate {candidate}: {e}")

        logger.warning(f"No global prompt fallback found for {file_name}")
        return ""

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    def _load_chapter_content(self, chapter_id: str) -> Dict[str, str]:
        """
        Load all content files for a chapter.

        Args:
            chapter_id: Can be either "course_id/chapter_name" (new format)
                       or "chapter_name" (legacy format)

        Returns:
            Dictionary with content for each file
        """
        # Support both new format (course_id/chapter_name) and legacy format (chapter_name)
        if "/" in chapter_id:
            # New format: course_id/chapter_name
            course_id, chapter_name = chapter_id.split("/", 1)
            chapter_dir = self.curriculum_dir / "courses" / course_id / "chapters" / chapter_name
        else:
            # Legacy format: chapter_name (look in old chapters directory)
            chapter_dir = self.curriculum_dir / "chapters" / chapter_id

        if not chapter_dir.exists():
            raise OrchestratorError(f"Chapter not found: {chapter_id}")

        content = {}
        files = [
            "chapter_context.md",
            "task_list.md",
            "task_completion_principles.md",
            "interaction_protocol.md",
            "socratic_vs_direct.md",
        ]

        for file_name in files:
            file_path = chapter_dir / file_name
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    key = file_name.replace(".md", "")
                    content[key] = f.read()
            else:
                key = file_name.replace(".md", "")
                if file_name in ("interaction_protocol.md", "socratic_vs_direct.md"):
                    content[key] = self._load_global_prompt_fallback(file_name)
                    if content[key]:
                        continue
                logger.warning(f"Chapter file not found: {file_path}")
                content[key] = ""

        return content

    def _has_consultation_assets(self, chapter_id: str) -> bool:
        """
        Check if a chapter has consultation configuration assets.

        Returns:
            True if any consultation config/guide file exists, False otherwise.
        """
        if "/" in chapter_id:
            course_id, chapter_name = chapter_id.split("/", 1)
            chapter_dir = self.curriculum_dir / "courses" / course_id / "chapters" / chapter_name
        else:
            chapter_dir = self.curriculum_dir / "chapters" / chapter_id

        if not chapter_dir.exists():
            return False

        return any(
            [
                (chapter_dir / "consultation_config.yaml").exists(),
                (chapter_dir / "consultation_guide.md").exists(),
                (chapter_dir / "consultation_guide.json").exists(),
            ]
        )

    def _load_available_experts_info(self, chapter_id: str) -> str:
        """
        Load available experts information for a chapter (v3.1).

        Reads consultation_config.yaml to get available_experts list,
        then fetches each expert's description from yellow_page.

        Args:
            chapter_id: Chapter identifier

        Returns:
            Formatted string with expert information for RMA prompt
        """
        if not self.consultation_engine:
            return ""

        if not self._has_consultation_assets(chapter_id):
            return ""

        try:
            consultation_context = self.consultation_engine.load_chapter_consultation_context(chapter_id)

            if not consultation_context:
                return ""

            # Get available experts list
            if isinstance(consultation_context, ConsultationContext):
                # v3.1: Get from config
                available_expert_ids = consultation_context.config.available_experts
            else:
                # v3.0: Get from guide
                available_expert_ids = consultation_context.available_experts_list

            if not available_expert_ids:
                return ""

            # Load yellow page
            from .yellow_page_generator import load_yellow_page, get_expert_by_id
            yellow_page_path = self._resolve_yellow_page_path()
            yellow_page = load_yellow_page(yellow_page_path) if yellow_page_path else None

            if not yellow_page:
                logger.warning("Yellow page not found, cannot load expert info")
                return ""

            # Build expert info string
            expert_info_lines = ["# 本章可用专家列表\n"]

            for expert_id in available_expert_ids:
                expert = get_expert_by_id(yellow_page, expert_id)
                if expert:
                    expert_info_lines.append(f"## {expert.expert_id}")
                    expert_info_lines.append(f"**描述**: {expert.description}")
                    if expert.tags:
                        expert_info_lines.append(f"**标签**: {', '.join(expert.tags)}")
                    if expert.output_modes:
                        expert_info_lines.append(f"**输出模式**: {', '.join(expert.output_modes)}")
                    expert_info_lines.append("")  # Empty line
                else:
                    logger.warning(f"Expert {expert_id} not found in yellow page")

            return "\n".join(expert_info_lines)

        except Exception as e:
            logger.warning(f"Failed to load available experts info for {chapter_id}: {e}")
            return ""

    def _resolve_yellow_page_path(self) -> Optional[Path]:
        """
        Resolve yellow page path for expert metadata.

        Priority:
        1. EXPERT_YELLOW_PAGE_PATH env
        2. Paths near current experts root
        3. Legacy cwd-relative fallback
        """
        env_path = os.getenv("EXPERT_YELLOW_PAGE_PATH", "").strip()
        candidates: List[Path] = []
        if env_path:
            candidates.append(Path(env_path))

        experts_root = self.experts_dir
        candidates.extend(
            [
                experts_root / "yellow_page.generated.json",
                experts_root / ".metadata" / "experts" / "yellow_page.generated.json",
                experts_root.parent / "yellow_page.generated.json",
                experts_root.parent / ".metadata" / "experts" / "yellow_page.generated.json",
                Path(".metadata/experts/yellow_page.generated.json"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _format_uploaded_files_for_ca(
        self,
        files_metadata: List[Dict[str, Any]],
        last_turn_time: Optional[float] = None,
    ) -> str:
        """
        Format uploaded files information for CA prompt injection (v3.2.0).

        Args:
            files_metadata: List of file metadata dictionaries
            last_turn_time: Timestamp of the most recent saved user turn

        Returns:
            Formatted string with uploaded files information
        """
        if not files_metadata:
            return ""

        lines = ["# 已上传文件\n"]

        # Separate new files from existing files
        new_files = []
        existing_files = []

        for file_info in files_metadata:
            upload_time = file_info.get("upload_time", 0)
            # Consider files uploaded after the last user turn as "new"
            if last_turn_time is None or upload_time > last_turn_time:
                new_files.append(file_info)
            else:
                existing_files.append(file_info)

        # Show new files first with special marking
        if new_files:
            lines.append("## 🆕 本回合新上传的文件\n")
            for file_info in new_files:
                filename = file_info.get("name", file_info.get("filename", "unknown"))
                size_kb = file_info.get("size", 0) / 1024
                file_type = file_info.get("file_type", "unknown")
                lines.append(f"- **{filename}** ({size_kb:.1f} KB, {file_type}) 🆕")
            lines.append("")

        # Show existing files
        if existing_files:
            lines.append("## 已存在的文件\n")
            for file_info in existing_files:
                filename = file_info.get("name", file_info.get("filename", "unknown"))
                size_kb = file_info.get("size", 0) / 1024
                file_type = file_info.get("file_type", "unknown")
                lines.append(f"- **{filename}** ({size_kb:.1f} KB, {file_type})")
            lines.append("")

        return "\n".join(lines)

    def _format_recent_code_executions_for_ca(self, executions: List[Dict[str, Any]]) -> str:
        """Format recent code executions for CA prompt injection."""
        if not executions:
            return ""

        def _clip_text(value: Any, limit: int) -> str:
            text = str(value or "")
            if len(text) <= limit:
                return text
            return text[:limit] + "\n...[truncated]..."

        now = time.time()
        lines = ["## 学生最近的代码执行", ""]
        for idx, record in enumerate(executions, 1):
            try:
                timestamp = float(record.get("timestamp", now))
            except (TypeError, ValueError):
                logger.warning("Skipping malformed code history record with invalid timestamp: %s", record)
                continue
            age_seconds = max(0, int(now - timestamp))
            if age_seconds < 60:
                age_label = f"{age_seconds} 秒前"
            else:
                age_label = f"{max(1, age_seconds // 60)} 分钟前"

            try:
                exit_code = int(record.get("exit_code", 0))
            except (TypeError, ValueError):
                logger.warning("Skipping malformed code history record with invalid exit_code: %s", record)
                continue

            code = _clip_text(record.get("code"), 1000).strip()
            stdout = _clip_text(record.get("stdout"), 800).strip()
            stderr = _clip_text(record.get("stderr"), 800).strip()
            status = "成功" if exit_code == 0 else "失败"
            output = stdout if stdout else stderr if stderr else "(无输出)"

            lines.append(f"### 执行 {idx} ({age_label})")
            lines.append("```python")
            lines.append(code or "# 无代码")
            lines.append("```")
            lines.append(f"**输出:** ({status})")
            lines.append("```")
            lines.append(output)
            lines.append("```")
            lines.append("")

        return "\n".join(lines) if len(lines) > 2 else ""

    def _load_consultation_guide_text(self, chapter_id: str) -> str:
        """
        Load consultation guide text for RMA prompt injection (v3.1).

        Args:
            chapter_id: Chapter identifier

        Returns:
            Consultation guide text (Markdown) or empty string if not available
        """
        if not self.consultation_engine:
            return ""

        if not self._has_consultation_assets(chapter_id):
            return ""

        try:
            consultation_context = self.consultation_engine.load_chapter_consultation_context(chapter_id)

            if isinstance(consultation_context, ConsultationContext):
                # v3.2: Inject binding rules into guide text
                guide_text = consultation_context.guide_text
                binding_rules_text = self._format_binding_rules(consultation_context.binding_rules)

                # Append binding rules to guide text
                full_guide = f"{guide_text}\n\n---\n\n## Binding Rules (系统强制规则)\n\n{binding_rules_text}"
                return full_guide
            elif consultation_context:
                # v3.0: Generate a summary from JSON (fallback)
                return "# Consultation Guide\n\n(Legacy v3.0 format - no detailed guide available)"

            return ""

        except Exception as e:
            logger.warning(f"Failed to load consultation guide for {chapter_id}: {e}")
            return ""

    def _format_binding_rules(self, binding_rules) -> str:
        """
        Format binding rules for RMA prompt injection.

        Args:
            binding_rules: BindingRules object

        Returns:
            Formatted Markdown text
        """
        lines = []

        # Must consult rules
        if binding_rules.must_consult:
            lines.append("### 必须咨询专家 (MUST Consult)")
            lines.append("")
            lines.append("以下情况下，RMA **必须**咨询相应专家：")
            lines.append("")
            for i, rule in enumerate(binding_rules.must_consult, 1):
                lines.append(f"{i}. {rule}")
            lines.append("")

        # Must not consult rules
        if binding_rules.must_not_consult:
            lines.append("### 禁止咨询专家 (MUST NOT Consult)")
            lines.append("")
            lines.append("以下情况下，RMA **禁止**咨询专家：")
            lines.append("")
            for i, rule in enumerate(binding_rules.must_not_consult, 1):
                lines.append(f"{i}. {rule}")
            lines.append("")

        # Expert judgment binding rules
        if binding_rules.expert_judgment_binding:
            lines.append("### 专家判断的强制执行 (Expert Judgment Binding)")
            lines.append("")
            lines.append("当专家给出以下判断时，RMA **必须**按照enforcement规则更新instruction_packet：")
            lines.append("")
            for rule_name, rule in binding_rules.expert_judgment_binding.items():
                lines.append(f"#### {rule_name}")
                lines.append(f"- **触发条件**: {rule.condition}")
                lines.append(f"- **强制执行**: {rule.enforcement}")
                lines.append("")

        return "\n".join(lines)

    def _load_templates(self) -> Dict[str, str]:
        """Load report templates."""
        templates_dir = self.curriculum_dir / "templates"
        fallback_dir = self.curriculum_dir / "_templates"
        if not templates_dir.exists() and fallback_dir.exists():
            templates_dir = fallback_dir

        templates = {}
        files = [
            "dynamic_report_template.md",
            "student_error_summary_template.md",
            "final_learning_report_template.md",
        ]

        for file_name in files:
            file_path = templates_dir / file_name
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    key = file_name.replace(".md", "")
                    templates[key] = f.read()
            else:
                logger.warning(f"Template file not found: {file_path}")
                templates[file_name.replace(".md", "")] = ""

        return templates

    def _parse_task_list(self, task_list_content: str) -> Dict[str, SubtaskStatus]:
        """
        Parse task list to extract subtask IDs.

        This is a simple parser that looks for task IDs in the format:
        - **task_id**: Task description

        Returns:
            Dictionary of subtask_id -> SubtaskStatus
        """
        subtasks = {}

        for line in task_list_content.split("\n"):
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                # Extract task ID
                task_id = line.split("**")[1]
                subtasks[task_id] = SubtaskStatus(
                    status="not_started",
                    evidence=[]
                )

        return subtasks

    def _should_unlock_instruction_packet(
        self,
        state: SessionState,
        turn_outcome,
        memo_digest,
        current_instruction: InstructionPacket,
        uploaded_files_info: dict = None  # v3.2.0: Add uploaded files info
    ) -> tuple[bool, str]:
        """
        Determine if instruction packet should be unlocked and updated.

        Unlock conditions:
        1. checkpoint_reached = True
        2. attempts_exceeded (attempts_since_last_progress >= max_attempts_before_unlock)
        3. new_error_type detected (blocker_type changed from previous)
        4. expert_consultation_needed = True (v3.2.0 - CA signals need for expert)

        Returns:
            (should_unlock, reason) tuple
        """
        # v3.2.0: Unlock if CA signals expert consultation needed
        if hasattr(turn_outcome, 'expert_consultation_needed') and turn_outcome.expert_consultation_needed:
            reason = getattr(turn_outcome, 'expert_consultation_reason', 'unspecified')
            logger.info(f"Unlocking instruction: CA signals expert consultation needed (reason: {reason})")
            return True, "expert_consultation_needed"

        lock_until = current_instruction.lock_until

        if lock_until == "checkpoint_reached":
            if turn_outcome.checkpoint_reached:
                logger.info("Unlocking instruction: checkpoint reached")
                return True, "checkpoint_reached"
            return False, "lock_until_checkpoint_not_reached"
        elif lock_until == "attempts_exceeded":
            if state.attempts_since_last_progress >= state.constraints.max_attempts_before_unlock:
                logger.info(
                    "Unlocking instruction: attempts exceeded "
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
                logger.info(f"Unlocking instruction: new error type detected ({turn_outcome.blocker_type})")
                return True, "new_error_type"
            return False, "lock_until_new_error_type_not_met"
        elif lock_until == "user_uploads_suitable_dataset_or_uses_example":
            has_new_uploads = bool(uploaded_files_info and uploaded_files_info.get("has_new_uploads"))
            if has_new_uploads:
                logger.info("Unlocking instruction: new dataset uploaded after unsuitability")
                return True, "user_uploads_suitable_dataset_or_uses_example"
            return False, "lock_until_new_upload_not_found"

        # Otherwise, keep locked
        logger.info(f"Keeping instruction locked (version {current_instruction.instruction_version})")
        return False, "lock_until_not_met"

    async def create_session(self, chapter_id: str) -> str:
        """
        Create new session and initialize with 5-step flow.

        Steps:
        1. Load chapter config
        2. Create session folder and initial state
        3. Call RMA for initial instruction packet
        4. Call CA for first message
        5. Call MA to initialize reports

        Returns:
            Session ID
        """
        try:
            # Step 1: Load chapter content
            logger.info(f"Loading chapter content for: {chapter_id}")
            chapter_content = self._load_chapter_content(chapter_id)
            templates = self._load_templates()

            # Step 2: Create session and initial state
            session_id = str(uuid.uuid4())
            logger.info(f"Creating session: {session_id}")

            # Parse task list to get subtasks
            subtasks = self._parse_task_list(chapter_content.get("task_list", ""))

            initial_state = SessionState(
                session_id=session_id,
                chapter_id=chapter_id,
                turn_index=0,
                subtask_status=subtasks,
                end_suggested=False,
                end_confirmed=False,
                constraints=SessionConstraints(
                    max_input_length=settings.max_input_length,
                    batch_error_log_every_n_turns=settings.batch_error_log_every_n_turns,
                    max_attempts_before_unlock=settings.max_attempts_before_unlock,
                ),
                current_instruction_version=1,
                attempts_since_last_progress=0,
                last_progress_turn=0,
            )

            self.storage.create_session(session_id, chapter_id, initial_state)
            self.memory_manager.initialize_session(session_id)

            # Step 3: Call RMA for initial instruction packet
            logger.info("Calling RMA for initial instruction")
            initial_digest = {
                "key_observations": ["会话刚开始"],
                "student_struggles": [],
                "student_strengths": [],
                "student_sentiment": "engaged",
                "blocker_type": "none",
                "progress_delta": "none",
                "diagnostic_log": []
            }

            # Load consultation guide for RMA (v3.1)
            consultation_guide_text = self._load_consultation_guide_text(chapter_id)
            available_experts_info = self._load_available_experts_info(chapter_id)

            from ..models.schemas import MemoDigest
            rma_result = await self.agent_runner.run_roadmap_manager(
                dynamic_report="",
                memo_digest=MemoDigest(**initial_digest),
                session_state=initial_state,
                turn_outcome=None,
                chapter_context=chapter_content.get("chapter_context", ""),
                task_list=chapter_content.get("task_list", ""),
                task_completion_principles=chapter_content.get("task_completion_principles", ""),
                consultation_guide=consultation_guide_text,
                available_experts=available_experts_info,
            )

            self.storage.save_instruction_packet(session_id, rma_result.instruction_packet)

            # Update state with RMA's initial update
            if rma_result.state_update:
                initial_state.update(rma_result.state_update)
                self.storage.save_state(session_id, initial_state)

            # Step 4: Call CA for first message (greeting + task overview)
            logger.info("Calling CA for initial greeting")
            initial_user_message = "[系统：这是会话的开始，请向学习者打招呼并介绍任务]"
            memory_sections = self.memory_manager.build_memory_sections(
                session_id=session_id,
                user_message_length=len(initial_user_message),
            )

            ca_response, turn_outcome = await self.agent_runner.run_companion(
                user_message=initial_user_message,
                instruction_packet=rma_result.instruction_packet,
                dynamic_report="",
                session_state=initial_state,
                chapter_context=chapter_content.get("chapter_context", ""),
                task_list=chapter_content.get("task_list", ""),
                task_completion_principles=chapter_content.get("task_completion_principles", ""),
                interaction_protocol=chapter_content.get("interaction_protocol", ""),
                socratic_vs_direct=chapter_content.get("socratic_vs_direct", ""),
                memory_long_term=memory_sections["long_term"],
                memory_mid_term=memory_sections["mid_term"],
                memory_recent_turns=memory_sections["recent_turns"],
            )

            # Save initial turn
            self.storage.save_turn(
                session_id,
                0,
                initial_user_message,
                ca_response,
                turn_outcome.model_dump()
            )

            # Step 5: Call MA to initialize reports
            logger.info("Calling MA to initialize reports")
            memo_result = await self.agent_runner.run_memo(
                user_message=initial_user_message,
                companion_response=ca_response,
                turn_outcome=turn_outcome,
                current_report="",
                session_state=initial_state,
                dynamic_report_template=templates.get("dynamic_report_template", ""),
                student_error_summary_template=templates.get("student_error_summary_template", ""),
                final_learning_report_template=templates.get("final_learning_report_template", ""),
            )

            self.storage.save_dynamic_report(session_id, memo_result.updated_report)
            self.storage.save_memo_digest(session_id, memo_result.digest)

            await self.memory_manager.update_after_turn(
                session_id=session_id,
                turn_index=0,
                user_message=initial_user_message,
                companion_response=ca_response,
                turn_outcome=turn_outcome,
            )

            # Increment turn index
            initial_state.turn_index = 1
            self.storage.save_state(session_id, initial_state)

            logger.info(f"Session created successfully: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise OrchestratorError(f"Failed to create session: {e}")

    async def process_turn(self, session_id: str, user_message: str) -> str:
        """
        Process a single turn in the conversation.

        Args:
            session_id: Session identifier
            user_message: User's message

        Returns:
            Companion's response
        """
        # Initialize performance tracker
        perf_tracker = get_tracker()
        perf_tracker.start_turn(session_id, -1)  # Will update turn_index later

        try:
            # 1. Validate input
            if len(user_message) > settings.max_input_length:
                raise OrchestratorError(
                    f"输入超过最大长度限制（{settings.max_input_length}字符）"
                )

            # 2. Load session state and content
            logger.info(f"Processing turn for session: {session_id}")
            state = self.storage.load_state(session_id)

            # Update tracker with correct turn index
            perf_tracker.current_turn_metrics["turn_index"] = state.turn_index

            instruction_packet = self.storage.load_instruction_packet(session_id)
            dynamic_report = self.storage.load_dynamic_report(session_id)
            memory_sections = self.memory_manager.build_memory_sections(
                session_id=session_id,
                user_message_length=len(user_message),
            )
            recent_code_executions = self.storage.get_recent_code_executions(session_id, limit=3)
            recent_code_executions_text = self._format_recent_code_executions_for_ca(
                recent_code_executions
            )

            chapter_content = self._load_chapter_content(state.chapter_id)
            templates = self._load_templates()

            # v3.2.0: Load expert information for CA
            available_experts_info = self._load_available_experts_info(state.chapter_id)

            # v3.2.0: Get uploaded files information for CA
            last_turn_time = self.storage.get_last_turn_timestamp(session_id)
            uploaded_files_metadata = self.storage.get_uploaded_files_metadata(session_id)
            uploaded_files_info_text = self._format_uploaded_files_for_ca(
                uploaded_files_metadata,
                last_turn_time,
            )
            uploaded_files_info = {
                "has_uploaded_files": len(uploaded_files_metadata) > 0,
                "file_count": len(uploaded_files_metadata),
                "has_new_uploads": False,
                "new_file_count": 0,
                "files": [],
            }
            for file_info in uploaded_files_metadata:
                is_new = last_turn_time is None or file_info.get("upload_time", 0) > last_turn_time
                uploaded_files_info["files"].append(
                    {
                        "filename": file_info["filename"],
                        "file_type": file_info["file_type"],
                        "size": file_info["size"],
                        "upload_time": file_info.get("upload_time", 0),
                        "is_new": is_new,
                    }
                )
                if is_new:
                    uploaded_files_info["has_new_uploads"] = True
                    uploaded_files_info["new_file_count"] += 1

            # 3. Call Companion Agent for current-turn judgment (with tracking and error recovery)
            logger.info("Calling Companion Agent")
            perf_tracker.start_operation("run_companion", {"phase": "initial"})

            try:
                ca_response, turn_outcome = await self.agent_runner.run_companion(
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
                    available_experts_info=available_experts_info,  # v3.2.0: Pass expert info
                    uploaded_files_info=uploaded_files_info_text,  # v3.2.0: Pass uploaded files info
                    recent_code_executions=recent_code_executions_text,  # v3.3.0
                )
                perf_tracker.end_operation(success=True)

            except Exception as e:
                perf_tracker.end_operation(success=False, error=str(e))

                # Error recovery
                recovery_ctx = ErrorRecoveryContext(
                    session_id=session_id,
                    turn_index=state.turn_index,
                    user_message=user_message,
                    error=e,
                )

                if recovery_ctx.recovery_strategy == RecoveryStrategy.USE_FALLBACK_RESPONSE:
                    ca_response = recovery_ctx.get_fallback_response()
                    from .json_utils import get_default_turn_outcome
                    turn_outcome = get_default_turn_outcome()
                else:
                    raise

            previous_memo_digest = self.storage.load_memo_digest(session_id)
            if not previous_memo_digest:
                from .json_utils import get_default_memo_digest
                previous_memo_digest = get_default_memo_digest()

            # 4. Determine if instruction packet should be updated (current turn)
            should_unlock, unlock_reason = self._should_unlock_instruction_packet(
                state=state,
                turn_outcome=turn_outcome,
                memo_digest=previous_memo_digest,
                current_instruction=instruction_packet,
                uploaded_files_info=uploaded_files_info  # v3.2.0: Pass uploaded files info
            )

            if not should_unlock and unlock_reason.startswith("lock_until_"):
                previous_blocker_type = "none"
                if hasattr(previous_memo_digest, "blocker_type"):
                    previous_blocker_type = previous_memo_digest.blocker_type
                elif isinstance(previous_memo_digest, dict):
                    previous_blocker_type = previous_memo_digest.get("blocker_type", "none")
                self.storage.append_system_event(
                    session_id,
                    "instruction_unlock_blocked",
                    "Unlock blocked by lock_until condition.",
                    {
                        "turn_index": state.turn_index,
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

            final_instruction = instruction_packet
            final_response = ca_response

            if should_unlock:
                # Call Roadmap Manager Agent to get new instruction packet
                logger.info("Calling Roadmap Manager Agent for new instruction")

                consultation_guide_text = self._load_consultation_guide_text(state.chapter_id)
                available_experts_info = self._load_available_experts_info(state.chapter_id)

                perf_tracker.start_operation("run_roadmap_manager")
                try:
                    rma_result = await self.agent_runner.run_roadmap_manager(
                        dynamic_report=dynamic_report,
                        memo_digest=previous_memo_digest,
                        session_state=state,
                        turn_outcome=turn_outcome,
                        chapter_context=chapter_content.get("chapter_context", ""),
                        task_list=chapter_content.get("task_list", ""),
                        task_completion_principles=chapter_content.get("task_completion_principles", ""),
                        consultation_guide=consultation_guide_text,
                        available_experts=available_experts_info,
                        uploaded_files_info=uploaded_files_info,  # v3.2.0: Pass uploaded files info
                    )
                    perf_tracker.end_operation(success=True)

                except Exception as e:
                    perf_tracker.end_operation(success=False, error=str(e))

                    # Error recovery for RMA
                    recovery_ctx = ErrorRecoveryContext(
                        session_id=session_id,
                        turn_index=state.turn_index,
                        user_message=user_message,
                        error=e,
                    )

                    if recovery_ctx.recovery_strategy == RecoveryStrategy.FALLBACK_TO_CA_DIRECT:
                        logger.warning("RMA failed, continuing with existing instruction packet")
                        # Continue with existing instruction packet
                        rma_result = None
                    else:
                        raise

                if rma_result:
                    # Update instruction packet with incremented version
                    new_instruction = rma_result.instruction_packet
                    new_instruction.instruction_version = state.current_instruction_version + 1
                    self.storage.save_instruction_packet(session_id, new_instruction)
                    final_instruction = new_instruction

                    # Update state with RMA's state update
                    state.update(rma_result.state_update)
                    state.current_instruction_version = new_instruction.instruction_version

                    # v3.1: Handle RMA's consultation request (if any)
                    if rma_result.consultation_request and self.consultation_engine:
                        logger.info(f"RMA requested consultation with {rma_result.consultation_request.expert_id}")
                        logger.info(f"Reasoning: {rma_result.consultation_request.reasoning}")

                        perf_tracker.start_operation("consult_expert", {"expert_id": rma_result.consultation_request.expert_id})
                        try:
                            consultation_result = await self.consultation_engine.consult_expert(
                                expert_id=rma_result.consultation_request.expert_id,
                                question=rma_result.consultation_request.question,
                                context=rma_result.consultation_request.context,
                                expected_output_type=rma_result.consultation_request.expected_output_type,
                                session_id=session_id,
                                session_state=state,
                                scenario_id=rma_result.consultation_request.scenario_id,
                            )
                            perf_tracker.end_operation(success=True)

                            # Apply instruction updates from binding rules
                            if consultation_result.get("instruction_updates"):
                                logger.info("Applying instruction updates from binding rules")
                                for key, value in consultation_result["instruction_updates"].items():
                                    if hasattr(new_instruction, key):
                                        setattr(new_instruction, key, value)
                                        logger.info(f"  Updated {key} = {value}")

                                # Save updated instruction packet
                                self.storage.save_instruction_packet(session_id, new_instruction)
                                final_instruction = new_instruction

                            # Log consultation result
                            logger.info(f"Consultation completed: {consultation_result['consultation_id']}")
                            if consultation_result.get("binding_rules_triggered"):
                                logger.info(f"Binding rules triggered: {[r['rule_name'] for r in consultation_result['binding_rules_triggered']]}")

                        except Exception as e:
                            perf_tracker.end_operation(success=False, error=str(e))
                            logger.error(f"Failed to execute consultation: {e}")
                            self.storage.append_system_event(
                                session_id,
                                "expert_consultation_failed",
                                f"专家咨询失败: {e}",
                                {
                                    "expert_id": rma_result.consultation_request.expert_id,
                                    "scenario_id": rma_result.consultation_request.scenario_id,
                                },
                            )
                            # Continue execution even if consultation fails

                    # Re-run Companion Agent after RMA update
                    logger.info("Re-running Companion Agent after RMA update")
                    perf_tracker.start_operation("run_companion", {"phase": "after_rma"})
                    try:
                        final_response, final_turn_outcome = await self.agent_runner.run_companion(
                            user_message=user_message,
                            instruction_packet=final_instruction,
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
                        turn_outcome = final_turn_outcome
                        perf_tracker.end_operation(success=True)

                    except Exception as e:
                        perf_tracker.end_operation(success=False, error=str(e))
                        # If CA re-run fails, use original response
                        logger.warning("CA re-run failed, using original response")
                        final_response = ca_response
                        turn_outcome = turn_outcome  # Keep original turn_outcome

                # v3.0: Check for consultation triggers after RMA update (deprecated)
                if self.consultation_engine and self.expert_report_generator:
                    await self._handle_consultations(
                        session_id=session_id,
                        state=state,
                        turn_outcome=turn_outcome,
                        memo_digest=previous_memo_digest,
                        new_instruction=final_instruction,
                    )

            # 5. Save turn artifacts (after final response)
            self.storage.save_turn(
                session_id,
                state.turn_index,
                user_message,
                final_response,
                turn_outcome.model_dump()
            )

            # 6. Call Memo Agent (with tracking and error recovery)
            logger.info("Calling Memo Agent")
            perf_tracker.start_operation("run_memo")
            try:
                memo_result = await self.agent_runner.run_memo(
                    user_message=user_message,
                    companion_response=final_response,
                    turn_outcome=turn_outcome,
                    current_report=dynamic_report,
                    session_state=state,
                    dynamic_report_template=templates.get("dynamic_report_template", ""),
                    student_error_summary_template=templates.get("student_error_summary_template", ""),
                    final_learning_report_template=templates.get("final_learning_report_template", ""),
                )
                perf_tracker.end_operation(success=True)

            except Exception as e:
                perf_tracker.end_operation(success=False, error=str(e))

                # Error recovery for MA
                recovery_ctx = ErrorRecoveryContext(
                    session_id=session_id,
                    turn_index=state.turn_index,
                    user_message=user_message,
                    error=e,
                )

                if recovery_ctx.recovery_strategy == RecoveryStrategy.SKIP_MA_UPDATE:
                    logger.warning("MA failed, skipping memo update")
                    # Use previous report
                    from ..models.schemas import MemoResult
                    from .json_utils import get_default_memo_digest
                    memo_result = MemoResult(
                        updated_report=dynamic_report,
                        digest=get_default_memo_digest(),
                        error_entries=[]
                    )
                else:
                    raise

            # 7. Update dynamic report and memo digest
            self.storage.save_dynamic_report(session_id, memo_result.updated_report)
            self.storage.save_memo_digest(session_id, memo_result.digest)

            # 8. Batch student error summary if needed
            if state.turn_index % state.constraints.batch_error_log_every_n_turns == 0:
                if memo_result.error_entries:
                    logger.info(f"Appending {len(memo_result.error_entries)} error entries")
                    self.storage.append_student_error_summary(session_id, memo_result.error_entries)

            # 8.5 Update layered memory
            await self.memory_manager.update_after_turn(
                session_id=session_id,
                turn_index=state.turn_index,
                user_message=user_message,
                companion_response=final_response,
                turn_outcome=turn_outcome,
            )

            # 9. Update progress tracking
            if should_unlock:
                if turn_outcome.checkpoint_reached:
                    state.attempts_since_last_progress = 0
                    state.last_progress_turn = state.turn_index
                else:
                    state.attempts_since_last_progress = 0
            else:
                logger.info(f"Reusing instruction packet version {instruction_packet.instruction_version}")
                if turn_outcome.checkpoint_reached or memo_result.digest.progress_delta == "evidence_added":
                    state.attempts_since_last_progress = 0
                    state.last_progress_turn = state.turn_index
                else:
                    state.attempts_since_last_progress += 1

            # 9. Update session state
            state.turn_index += 1
            self.storage.save_state(session_id, state)

            # 10. Check end condition
            if state.end_suggested and not state.end_confirmed:
                logger.info("End suggested, waiting for user confirmation")

            # End performance tracking
            perf_tracker.end_turn(session_id)

            logger.info(f"Turn processed successfully: {state.turn_index}")
            return final_response

        except StorageError as e:
            # End performance tracking on error
            if perf_tracker.current_turn_metrics:
                perf_tracker.end_turn(session_id)

            logger.error(f"Storage error: {e}")
            raise OrchestratorError(f"存储错误: {e}")

        except Exception as e:
            # End performance tracking on error
            if perf_tracker.current_turn_metrics:
                perf_tracker.end_turn(session_id)

            logger.error(f"Failed to process turn: {e}")
            raise OrchestratorError(f"处理回合失败: {e}")

    async def _handle_consultations(
        self,
        session_id: str,
        state: SessionState,
        turn_outcome,
        memo_digest,
        new_instruction: InstructionPacket,
    ) -> None:
        """
        Handle expert consultations after RMA update (v3.0).

        Args:
            session_id: Session identifier
            state: Current session state
            turn_outcome: Turn outcome from CA
            memo_digest: Memo digest
            new_instruction: New instruction packet from RMA
        """
        try:
            # v3.1: Disable v3.0 automatic consultation by default
            import os
            if not os.getenv("ENABLE_V3_0_AUTO_CONSULTATION", "false").lower() == "true":
                logger.info("v3.0 automatic consultation disabled (use v3.1 RMA-initiated consultation)")
                return

            logger.warning("Using deprecated v3.0 automatic consultation system. "
                          "Consider migrating to v3.1 RMA-initiated consultation.")

            logger.info("Checking for consultation triggers")

            # Check if any scenarios are triggered
            triggered_scenarios = self.consultation_engine.check_triggers(
                session_state=state,
                turn_outcome=turn_outcome,
                memo_digest=memo_digest,
            )

            if not triggered_scenarios:
                logger.info("No consultation scenarios triggered")
                return

            logger.info(f"Triggered {len(triggered_scenarios)} consultation scenario(s)")

            # Load consultation guide
            guide = self.consultation_engine._load_consultation_guide(state.chapter_id)
            if not guide:
                logger.warning("No consultation guide found, skipping consultations")
                return

            # Run consultations for each triggered scenario
            for scenario in triggered_scenarios:
                logger.info(f"Running consultation for scenario: {scenario.scenario_id}")

                # Build context for consultation
                context = self._build_consultation_context(
                    scenario=scenario,
                    state=state,
                    turn_outcome=turn_outcome,
                )

                # Run consultation
                consultation_result = await self.consultation_engine.run_consultation(
                    session_id=session_id,
                    session_state=state,
                    scenario=scenario,
                    guide=guide,
                    context=context,
                )

                # Skip if no experts were selected
                if consultation_result.get("skipped"):
                    logger.info(f"Consultation skipped: {consultation_result.get('reason')}")
                    continue

                # Generate expert report
                expert_report = self.expert_report_generator.generate_report(
                    session_id=session_id,
                    consultation_result=consultation_result,
                    guide=guide,
                )

                # Enforce binding rules
                enforcement = self.expert_report_generator.enforce_binding_rules(
                    consultation_result=consultation_result,
                    scenario=scenario,
                )

                # Apply binding constraints to instruction packet
                if enforcement.get("binding_enforced") and enforcement.get("actions"):
                    for action in enforcement["actions"]:
                        instruction_update = action.get("instruction_packet_update")
                        if instruction_update:
                            logger.info(f"Applying binding rule: {action['rule']}")
                            # Update instruction packet fields
                            for key, value in instruction_update.items():
                                if hasattr(new_instruction, key):
                                    setattr(new_instruction, key, value)

                            # Save updated instruction packet
                            self.storage.save_instruction_packet(session_id, new_instruction)

                logger.info(f"Consultation completed: {consultation_result['consultation_id']}")

        except Exception as e:
            logger.error(f"Error handling consultations: {e}")
            # Don't raise - consultation failure shouldn't break the main flow

    def _build_consultation_context(
        self,
        scenario,
        state: SessionState,
        turn_outcome,
    ) -> Dict:
        """
        Build context dictionary for consultation envelope.

        Args:
            scenario: Consultation scenario
            state: Session state
            turn_outcome: Turn outcome

        Returns:
            Context dictionary with placeholders filled
        """
        context = {
            "chapter_id": state.chapter_id,
            "task_id": "current_task",  # TODO: Extract from instruction packet
            "file_name": "uploaded_file.csv",  # TODO: Track uploaded files
            "task_requirements": {},  # TODO: Extract from task list
            "concept_name": "",  # TODO: Extract from turn outcome
            "student_confusion": "",  # TODO: Extract from memo digest
            "student_level": "beginner",  # TODO: Track student level
            "error_message": "",  # TODO: Extract from turn outcome
            "error_type": turn_outcome.blocker_type,
            "student_code_snippet": "",  # TODO: Extract from turn
            "student_evidence": "",  # TODO: Extract from state
            "learning_objectives": "",  # TODO: Extract from task list
        }

        return context

    async def end_session(self, session_id: str) -> str:
        """
        End session and generate final report.

        Returns:
            Final learning report
        """
        try:
            logger.info(f"Ending session: {session_id}")

            # Load state
            state = self.storage.load_state(session_id)
            dynamic_report = self.storage.load_dynamic_report(session_id)
            error_summary = self.storage.load_student_error_summary(session_id)

            # Mark as confirmed
            state.end_confirmed = True
            self.storage.save_state(session_id, state)

            # Generate final report (for now, use dynamic report + error summary)
            # In a full implementation, MA would generate a comprehensive final report
            final_report = f"""# 最终学习报告

## 学习概览
- 会话ID: {session_id}
- 章节: {state.chapter_id}
- 总回合数: {state.turn_index}

## 任务完成情况
"""
            for task_id, status in state.subtask_status.items():
                status_icon = "✅" if status.status == "completed" else "🔄" if status.status == "in_progress" else "⏸️"
                final_report += f"- {status_icon} {task_id}: {status.status}\n"

            final_report += f"\n## 学习进度\n\n{dynamic_report}\n"

            if error_summary:
                final_report += f"\n## 学习过程中的错误模式\n\n{error_summary}\n"

            final_report += "\n## 总结\n\n感谢你的学习！继续保持好奇心和探索精神。\n"

            self.storage.save_final_learning_report(session_id, final_report)

            logger.info("Session ended successfully")
            return final_report

        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            raise OrchestratorError(f"结束会话失败: {e}")

    async def process_turn_stream(self, session_id: str, user_message: str):
        """
        Process a turn with streaming output.

        Yields events for real-time UI updates.
        """
        from .streaming import process_turn_stream

        state = self.storage.load_state(session_id)
        chapter_content = self._load_chapter_content(state.chapter_id)
        templates = self._load_templates()
        available_experts_info = self._load_available_experts_info(state.chapter_id)
        consultation_guide_text = self._load_consultation_guide_text(state.chapter_id)
        recent_code_executions = self.storage.get_recent_code_executions(session_id, limit=3)
        recent_code_executions_text = self._format_recent_code_executions_for_ca(
            recent_code_executions
        )
        last_turn_time = self.storage.get_last_turn_timestamp(session_id)
        uploaded_files_metadata = self.storage.get_uploaded_files_metadata(session_id)
        uploaded_files_info_text = self._format_uploaded_files_for_ca(
            uploaded_files_metadata,
            last_turn_time,
        )
        uploaded_files_info = {
            "has_uploaded_files": len(uploaded_files_metadata) > 0,
            "file_count": len(uploaded_files_metadata),
            "has_new_uploads": False,
            "new_file_count": 0,
            "files": [],
        }
        for file_info in uploaded_files_metadata:
            is_new = last_turn_time is None or file_info.get("upload_time", 0) > last_turn_time
            uploaded_files_info["files"].append(
                {
                    "filename": file_info["filename"],
                    "file_type": file_info["file_type"],
                    "size": file_info["size"],
                    "upload_time": file_info.get("upload_time", 0),
                    "is_new": is_new,
                }
            )
            if is_new:
                uploaded_files_info["has_new_uploads"] = True
                uploaded_files_info["new_file_count"] += 1

        async for event in process_turn_stream(
            session_id=session_id,
            user_message=user_message,
            storage=self.storage,
            agent_runner=self.agent_runner,
            memory_manager=self.memory_manager,
            chapter_content=chapter_content,
            templates=templates,
            available_experts_info=available_experts_info,
            uploaded_files_info_text=uploaded_files_info_text,
            recent_code_executions_text=recent_code_executions_text,
            consultation_guide_text=consultation_guide_text,
            uploaded_files_info=uploaded_files_info,
            consultation_engine=self.consultation_engine,
        ):
            yield event
