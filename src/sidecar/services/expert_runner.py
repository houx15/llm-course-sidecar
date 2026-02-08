"""Expert runner service for executing expert consultations with skill invocation."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..models.schemas import (
    ConsultingEnvelope,
    ExpertOutput,
    SuitabilityJudgment,
    ConceptExplanation,
    DatasetOverviewReport,
    SkillCallLog,
    SkillInputs,
    SkillOutputs,
)
from .agent_runner import AgentRunner
from .data_inspector_skills import DataInspectorSkills

logger = logging.getLogger(__name__)


class ExpertRunnerError(Exception):
    """Custom exception for expert runner errors."""
    pass


class ExpertRunner:
    """Service for running expert consultations with skill execution capabilities."""

    def __init__(self, experts_root: Path = Path("experts")):
        """
        Initialize expert runner.

        Args:
            experts_root: Root directory containing expert definitions
        """
        self.experts_root = experts_root
        self.agent_runner = AgentRunner()

    def _load_expert_files(self, expert_id: str) -> Dict[str, str]:
        """
        Load expert's principles, knowledge, and skills.

        Args:
            expert_id: Expert identifier

        Returns:
            Dictionary with file contents
        """
        expert_dir = self.experts_root / expert_id
        if not expert_dir.exists():
            raise ExpertRunnerError(f"Expert directory not found: {expert_id}")

        files = {}

        # Load principles (required)
        principles_path = expert_dir / "principles.md"
        if not principles_path.exists():
            raise ExpertRunnerError(f"Principles file not found for expert: {expert_id}")
        files["principles"] = principles_path.read_text(encoding="utf-8")

        # Load knowledge files (optional)
        knowledge_dir = expert_dir / "knowledge"
        if knowledge_dir.exists():
            knowledge_files = []
            for knowledge_file in knowledge_dir.glob("*.md"):
                knowledge_files.append(f"## {knowledge_file.name}\n\n{knowledge_file.read_text(encoding='utf-8')}")
            files["knowledge"] = "\n\n---\n\n".join(knowledge_files) if knowledge_files else ""
        else:
            files["knowledge"] = ""

        # Load skill files (optional)
        skills_dir = expert_dir / "skills"
        if skills_dir.exists():
            skill_files = []
            for skill_file in skills_dir.glob("*.md"):
                skill_files.append(f"## {skill_file.name}\n\n{skill_file.read_text(encoding='utf-8')}")
            files["skills"] = "\n\n---\n\n".join(skill_files) if skill_files else ""
        else:
            files["skills"] = ""

        return files

    def _build_expert_prompt(
        self,
        expert_id: str,
        envelope: ConsultingEnvelope,
        expert_files: Dict[str, str],
        consultation_history: List[Dict] = None,
    ) -> str:
        """
        Build the prompt for expert consultation.

        Args:
            expert_id: Expert identifier
            envelope: Consulting envelope from RMA
            expert_files: Loaded expert files (principles, knowledge, skills)
            consultation_history: Previous rounds in this consultation

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"# Expert Consultation: {expert_id}",
            "",
            "## Your Role and Principles",
            expert_files["principles"],
            "",
        ]

        if expert_files["knowledge"]:
            prompt_parts.extend([
                "## Knowledge Base",
                expert_files["knowledge"],
                "",
            ])

        if expert_files["skills"]:
            prompt_parts.extend([
                "## Available Skills",
                expert_files["skills"],
                "",
            ])

        prompt_parts.extend([
            "## Consultation Request",
            f"**Consultation ID**: {envelope.consultation_id}",
            f"**Scenario**: {envelope.scenario_id}",
            f"**Round**: {envelope.round_index} of {envelope.round_index + envelope.rounds_remaining_after_this}",
            f"**Title**: {envelope.consulting_letter_title}",
            "",
            "### Question",
            envelope.question,
            "",
        ])

        if envelope.context:
            prompt_parts.extend([
                "### Context",
                "```json",
                json.dumps(envelope.context, ensure_ascii=False, indent=2),
                "```",
                "",
            ])

        prompt_parts.extend([
            "### Session Scope",
            f"**Allowed working directory**: `{envelope.session_scope.allowed_root}`",
            "",
            "**IMPORTANT**: You may ONLY access files within this directory. Any file operations must be relative to this root.",
            "",
        ])

        prompt_parts.extend([
            "### Expected Output",
            "```json",
            json.dumps(envelope.expected_output_template, ensure_ascii=False, indent=2),
            "```",
            "",
        ])

        if consultation_history:
            prompt_parts.extend([
                "## Previous Rounds in This Consultation",
                "",
            ])
            for i, round_data in enumerate(consultation_history, 1):
                prompt_parts.extend([
                    f"### Round {i}",
                    f"**Request**: {round_data.get('request', 'N/A')}",
                    f"**Response**: {round_data.get('response', 'N/A')}",
                    "",
                ])

        prompt_parts.extend([
            "## Instructions",
            "",
            "1. **Read the question carefully** and understand what RMA is asking",
            "2. **Check your principles** to ensure you can answer this question",
            "3. **If you need to invoke a skill**:",
            "   - Verify the skill is in your whitelist",
            "   - Ensure preconditions are met",
            "   - Call the skill and wait for results",
            "   - Log the skill invocation",
            "4. **Provide your answer** in the expected output format",
            "5. **Include evidence** from your analysis",
            "6. **Signal termination** if appropriate (no new information, binding judgment delivered, etc.)",
            "",
            "## Output Format",
            "",
            "Provide your response as a JSON object matching the expected_output_template.",
            "Include a `knowledge_sources` field listing any knowledge documents you referenced.",
            "",
            "If you need to signal termination, include a `termination_signal` field:",
            '```json',
            '{',
            '  "termination_signal": true,',
            '  "termination_reason": "binding_judgment_delivered|no_new_information|drift_from_question"',
            '}',
            '```',
        ])

        return "\n".join(prompt_parts)

    async def run_consultation_round(
        self,
        expert_id: str,
        envelope: ConsultingEnvelope,
        session_id: str,
        consultation_history: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run a single round of expert consultation.

        Args:
            expert_id: Expert identifier
            envelope: Consulting envelope from RMA
            session_id: Session identifier for file access
            consultation_history: Previous rounds in this consultation

        Returns:
            Dictionary containing expert output and metadata
        """
        try:
            # Load expert files
            expert_files = self._load_expert_files(expert_id)

            # Check if this expert has executable skills
            if expert_id == "data_inspector":
                # Use skill-based execution for data_inspector
                result = await self._run_data_inspector_consultation(
                    expert_id=expert_id,
                    envelope=envelope,
                    session_id=session_id,
                    expert_files=expert_files,
                    consultation_history=consultation_history,
                )
            else:
                # Use LLM-based execution for other experts
                result = await self._run_llm_consultation(
                    expert_id=expert_id,
                    envelope=envelope,
                    session_id=session_id,
                    expert_files=expert_files,
                    consultation_history=consultation_history,
                )

            logger.info(f"Expert {expert_id} completed round {envelope.round_index}")
            return result

        except Exception as e:
            logger.error(f"Error running expert {expert_id}: {e}")
            raise ExpertRunnerError(f"Failed to run expert {expert_id}: {e}")

    async def _run_data_inspector_consultation(
        self,
        expert_id: str,
        envelope: ConsultingEnvelope,
        session_id: str,
        expert_files: Dict[str, str],
        consultation_history: List[Dict],
    ) -> Dict[str, Any]:
        """
        Run data_inspector consultation using actual skill execution.

        Args:
            expert_id: Expert identifier
            envelope: Consulting envelope
            session_id: Session identifier
            expert_files: Expert files (principles, knowledge, skills)
            consultation_history: Previous rounds

        Returns:
            Consultation result dictionary
        """
        try:
            # Initialize skills with decision logging
            working_root = Path(envelope.session_scope.allowed_root)
            skills = DataInspectorSkills(
                working_root=working_root,
                session_id=session_id,
                consultation_id=envelope.consultation_id,
            )

            # Determine which skill to call based on scenario
            scenario_id = envelope.scenario_id
            expected_output_type = envelope.expected_output_template.output_type

            expert_output = None

            if scenario_id == "dataset_suitability_check" and expected_output_type == "suitability_judgment":
                # Extract parameters from context
                # Debug: Check context type
                logger.info(f"Context type: {type(envelope.context)}, value: {envelope.context}")

                if not isinstance(envelope.context, dict):
                    raise ExpertRunnerError(f"Expected context to be dict, got {type(envelope.context)}: {envelope.context}")

                file_name = envelope.context.get("file_name", "uploaded_data.csv")
                task_requirements = envelope.context.get("task_requirements", {})

                # Handle task_requirements: can be dict or string
                if isinstance(task_requirements, str):
                    # If it's a string, use default values
                    logger.info(f"task_requirements is a string, using defaults: {task_requirements}")
                    required_columns = []
                    minimum_rows = 100
                    maximum_missing_rate = 0.3
                    expected_dtypes = None
                elif isinstance(task_requirements, dict):
                    # If it's a dict, extract values
                    required_columns = task_requirements.get("required_columns", [])
                    minimum_rows = task_requirements.get("minimum_rows", 100)
                    maximum_missing_rate = task_requirements.get("maximum_missing_rate", 0.3)
                    expected_dtypes = task_requirements.get("expected_dtypes", None)
                else:
                    # Fallback to defaults
                    logger.warning(f"Unexpected task_requirements type: {type(task_requirements)}")
                    required_columns = []
                    minimum_rows = 100
                    maximum_missing_rate = 0.3
                    expected_dtypes = None

                # Log skill invocation
                skill_log = SkillCallLog(
                    timestamp=datetime.now().isoformat(),
                    expert_id=expert_id,
                    consultation_id=envelope.consultation_id,
                    user_turn_index=envelope.user_turn_index,
                    round_index=envelope.round_index,
                    skill_name="check_suitability",
                    cwd=str(working_root),
                    inputs=SkillInputs(
                        files_read=[file_name],
                        parameters={
                            "file_name": file_name,
                            "required_columns": required_columns,
                            "minimum_rows": minimum_rows,
                            "maximum_missing_rate": maximum_missing_rate,
                            "expected_dtypes": expected_dtypes,
                        },
                    ),
                    outputs=SkillOutputs(),
                    stdout_stderr_paths={"stdout": None, "stderr": None},
                    success=True,
                )

                try:
                    # Execute skill
                    expert_output = skills.check_suitability(
                        file_name=file_name,
                        required_columns=required_columns,
                        minimum_rows=minimum_rows,
                        maximum_missing_rate=maximum_missing_rate,
                        expected_dtypes=expected_dtypes,
                    )

                    skill_log.outputs = SkillOutputs(
                        files_written=["suitability_judgment.json", "dataset_profile.json"]
                    )
                    skill_log.success = True

                except Exception as e:
                    skill_log.success = False
                    skill_log.error_message = str(e)
                    raise

                finally:
                    # Log skill invocation
                    self.log_skill_invocation(session_id, expert_id, envelope.consultation_id, skill_log)

            elif expected_output_type == "dataset_overview_report":
                # Profile dataset
                file_name = envelope.context.get("file_name", "uploaded_data.csv")

                skill_log = SkillCallLog(
                    timestamp=datetime.now().isoformat(),
                    expert_id=expert_id,
                    consultation_id=envelope.consultation_id,
                    user_turn_index=envelope.user_turn_index,
                    round_index=envelope.round_index,
                    skill_name="profile_dataset",
                    cwd=str(working_root),
                    inputs=SkillInputs(
                        files_read=[file_name],
                        parameters={"file_name": file_name},
                    ),
                    outputs=SkillOutputs(),
                    stdout_stderr_paths={"stdout": None, "stderr": None},
                    success=True,
                )

                try:
                    expert_output = skills.profile_dataset(file_name=file_name)
                    skill_log.outputs = SkillOutputs(
                        files_written=["dataset_detailed_profile.json"]
                    )
                    skill_log.success = True

                except Exception as e:
                    skill_log.success = False
                    skill_log.error_message = str(e)
                    raise

                finally:
                    self.log_skill_invocation(session_id, expert_id, envelope.consultation_id, skill_log)

            else:
                # Fallback to LLM-based consultation
                return await self._run_llm_consultation(
                    expert_id, envelope, session_id, expert_files, consultation_history
                )

            # Check for termination
            termination_signal = False
            termination_reason = None

            if expert_output and expert_output.get("output_type") == "suitability_judgment":
                # Binding judgment delivered
                termination_signal = True
                termination_reason = "binding_judgment_delivered"

            result = {
                "expert_id": expert_id,
                "consultation_id": envelope.consultation_id,
                "round_index": envelope.round_index,
                "expert_output": expert_output,
                "termination_signal": termination_signal,
                "termination_reason": termination_reason,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Error in data_inspector consultation: {e}")
            raise ExpertRunnerError(f"Data inspector consultation failed: {e}")

    async def _run_llm_consultation(
        self,
        expert_id: str,
        envelope: ConsultingEnvelope,
        session_id: str,
        expert_files: Dict[str, str],
        consultation_history: List[Dict],
    ) -> Dict[str, Any]:
        """
        Run consultation using LLM (for experts without executable skills).

        Args:
            expert_id: Expert identifier
            envelope: Consulting envelope
            session_id: Session identifier
            expert_files: Expert files
            consultation_history: Previous rounds

        Returns:
            Consultation result dictionary
        """
        # Build prompt
        prompt = self._build_expert_prompt(
            expert_id,
            envelope,
            expert_files,
            consultation_history or []
        )

        # Run expert via agent runner
        logger.info(f"Running expert {expert_id} for consultation {envelope.consultation_id}, round {envelope.round_index}")

        response = await self.agent_runner.run_agent(
            agent_type="expert",
            prompt=prompt,
            model="claude-sonnet-4-5-20250929",
        )

        # Parse expert output
        expert_output = self._parse_expert_output(response, envelope.expected_output_template)

        # Check for termination signal
        termination_signal = expert_output.get("termination_signal", False)
        termination_reason = expert_output.get("termination_reason", None)

        result = {
            "expert_id": expert_id,
            "consultation_id": envelope.consultation_id,
            "round_index": envelope.round_index,
            "expert_output": expert_output,
            "termination_signal": termination_signal,
            "termination_reason": termination_reason,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def _parse_expert_output(self, response: str, expected_template: Dict) -> Dict:
        """
        Parse expert response into structured output.

        Args:
            response: Raw response from expert
            expected_template: Expected output template

        Returns:
            Parsed expert output dictionary
        """
        try:
            # Try to extract JSON from response
            # Look for JSON code block or raw JSON
            import re

            # Try to find JSON code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in expert response")

            output = json.loads(json_str)

            # Validate required fields
            output_type = expected_template.get("output_type")
            required_fields = expected_template.get("required_fields", [])

            if output_type and output.get("output_type") != output_type:
                logger.warning(f"Output type mismatch: expected {output_type}, got {output.get('output_type')}")

            for field in required_fields:
                if field not in output:
                    logger.warning(f"Missing required field in expert output: {field}")

            return output

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse expert output as JSON: {e}")
            # Return a fallback output
            return {
                "output_type": expected_template.get("output_type", "error"),
                "error": "Failed to parse expert output",
                "raw_response": response[:500],  # Include first 500 chars for debugging
            }
        except Exception as e:
            logger.error(f"Error parsing expert output: {e}")
            return {
                "output_type": "error",
                "error": str(e),
            }

    def log_skill_invocation(
        self,
        session_id: str,
        expert_id: str,
        consultation_id: str,
        skill_log: SkillCallLog,
    ) -> None:
        """
        Log a skill invocation to expert_logs directory.

        Args:
            session_id: Session identifier
            expert_id: Expert identifier
            consultation_id: Consultation identifier
            skill_log: Skill call log data
        """
        try:
            log_dir = Path("sessions") / session_id / "working_files" / "expert_logs" / expert_id / consultation_id
            log_dir.mkdir(parents=True, exist_ok=True)

            # Find next skill call number
            existing_logs = list(log_dir.glob("skill_call_*.json"))
            next_num = len(existing_logs) + 1

            log_path = log_dir / f"skill_call_{next_num:04d}.json"

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(skill_log.model_dump(), f, ensure_ascii=False, indent=2)

            logger.info(f"Logged skill invocation: {log_path}")

        except Exception as e:
            logger.error(f"Failed to log skill invocation: {e}")
            # Don't raise - logging failure shouldn't break consultation

    def validate_file_path(self, file_path: str, allowed_root: str) -> Path:
        """
        Validate that a file path is within the allowed sandbox.

        Args:
            file_path: File path to validate
            allowed_root: Allowed root directory

        Returns:
            Resolved Path object

        Raises:
            ExpertRunnerError: If path is outside sandbox
        """
        allowed_root_path = Path(allowed_root).resolve()
        target_path = (allowed_root_path / file_path).resolve()

        if not str(target_path).startswith(str(allowed_root_path)):
            raise ExpertRunnerError(f"Path {file_path} is outside allowed sandbox {allowed_root}")

        return target_path
