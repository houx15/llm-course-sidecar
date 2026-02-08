"""Consultation engine for RMA to manage expert consultations."""

import json
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio

from ..models.schemas import (
    ConsultingEnvelope,
    ConsultationMeta,
    ConsultationGuide,
    ConsultationScenario,
    ConsultationConfig,
    ConsultationContext,
    BindingRules,
    ExpertJudgmentBindingRule,
    YellowPage,
    ExpertMetadata,
    SessionState,
    TurnOutcome,
    MemoDigest,
    SessionScope,
    ExpectedOutputTemplate,
)
from .expert_runner import ExpertRunner
from .yellow_page_generator import load_yellow_page

logger = logging.getLogger(__name__)


class ConsultationEngineError(Exception):
    """Custom exception for consultation engine errors."""
    pass


class ConsultationEngine:
    """Engine for managing expert consultations triggered by RMA."""

    def __init__(
        self,
        experts_root: Path = Path("experts"),
        curriculum_root: Path = Path("curriculum"),
    ):
        """
        Initialize consultation engine.

        Args:
            experts_root: Root directory containing expert definitions
            curriculum_root: Root directory containing curriculum
        """
        self.experts_root = experts_root
        self.curriculum_root = curriculum_root
        self.sessions_root = Path(os.getenv("SESSIONS_DIR", "sessions"))
        self.expert_runner = ExpertRunner(experts_root)

        # Load yellow page with runtime-aware path resolution.
        yellow_page_path = self._resolve_yellow_page_path()
        self.yellow_page = load_yellow_page(yellow_page_path) if yellow_page_path else None
        if not self.yellow_page:
            raise ConsultationEngineError("Failed to load yellow page")

        # Consultation ID counter (session-global)
        self._consultation_counters: Dict[str, int] = {}

    def _get_next_consultation_id(self, session_id: str) -> str:
        """Get next consultation ID for a session."""
        if session_id not in self._consultation_counters:
            # Check existing consultations in session
            session_dir = self.sessions_root / session_id / "expert_workspace" / "consultations"
            if session_dir.exists():
                existing = list(session_dir.glob("consult_*"))
                self._consultation_counters[session_id] = len(existing)
            else:
                self._consultation_counters[session_id] = 0

        self._consultation_counters[session_id] += 1
        return f"consult_{self._consultation_counters[session_id]:04d}"

    def _resolve_yellow_page_path(self) -> Optional[Path]:
        """Resolve yellow page path with env/bundle-friendly fallbacks."""
        env_path = os.getenv("EXPERT_YELLOW_PAGE_PATH", "").strip()
        candidates = []
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend(
            [
                self.experts_root / "yellow_page.generated.json",
                self.experts_root / ".metadata" / "experts" / "yellow_page.generated.json",
                self.experts_root.parent / "yellow_page.generated.json",
                self.experts_root.parent / ".metadata" / "experts" / "yellow_page.generated.json",
                Path(".metadata/experts/yellow_page.generated.json"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_consultation_guide(self, chapter_id: str) -> Optional[ConsultationGuide]:
        """
        Load consultation guide for a chapter (v3.0 legacy format).

        Args:
            chapter_id: Chapter identifier (format: course_id/chapter_name)

        Returns:
            ConsultationGuide if found, None otherwise
        """
        try:
            # Support both new format (course_id/chapter_name) and legacy format
            if "/" in chapter_id:
                course_id, chapter_name = chapter_id.split("/", 1)
                guide_path = (
                    self.curriculum_root / "courses" / course_id / "chapters" / chapter_name / "consultation_guide.json"
                )
            else:
                guide_path = self.curriculum_root / "chapters" / chapter_id / "consultation_guide.json"

            if not guide_path.exists():
                logger.warning(f"Consultation guide not found for chapter: {chapter_id}")
                return None

            with open(guide_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return ConsultationGuide(**data)

        except Exception as e:
            logger.error(f"Failed to load consultation guide for {chapter_id}: {e}")
            return None

    def load_chapter_consultation_context(self, chapter_id: str) -> Optional[Union[ConsultationContext, ConsultationGuide]]:
        """
        Load consultation context for a chapter (v3.1 hybrid or v3.0 legacy).

        Args:
            chapter_id: Chapter identifier (format: course_id/chapter_name)

        Returns:
            ConsultationContext (v3.1) or ConsultationGuide (v3.0) if found, None otherwise
        """
        # Determine chapter directory
        if "/" in chapter_id:
            course_id, chapter_name = chapter_id.split("/", 1)
            chapter_dir = self.curriculum_root / "courses" / course_id / "chapters" / chapter_name
        else:
            chapter_dir = self.curriculum_root / "chapters" / chapter_id

        # Try v3.1 hybrid format first (YAML + Markdown)
        yaml_path = chapter_dir / "consultation_config.yaml"
        md_path = chapter_dir / "consultation_guide.md"

        if yaml_path.exists() and md_path.exists():
            try:
                # Load YAML config
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)

                # Parse binding rules (v3.2: new structure)
                binding_rules_data = yaml_data.get("binding_rules", {})

                # Parse expert_judgment_binding rules
                expert_judgment_binding = {}
                if "expert_judgment_binding" in binding_rules_data:
                    from ..models.schemas import ExpertJudgmentBindingRule
                    for rule_name, rule_data in binding_rules_data["expert_judgment_binding"].items():
                        expert_judgment_binding[rule_name] = ExpertJudgmentBindingRule(**rule_data)
                    binding_rules_data["expert_judgment_binding"] = expert_judgment_binding

                from ..models.schemas import BindingRules
                binding_rules = BindingRules(**binding_rules_data)
                yaml_data["binding_rules"] = binding_rules

                config = ConsultationConfig(**yaml_data)

                # Load Markdown guide
                with open(md_path, "r", encoding="utf-8") as f:
                    guide_text = f.read()

                context = ConsultationContext(
                    config=config,
                    guide_text=guide_text,
                    binding_rules=binding_rules,
                )

                logger.info(f"Loaded v3.1 hybrid consultation context for {chapter_id}")
                return context

            except Exception as e:
                logger.error(f"Failed to load v3.1 hybrid format for {chapter_id}: {e}")
                # Fall through to try legacy format

        # Fallback to v3.0 legacy JSON format
        guide = self._load_consultation_guide(chapter_id)
        if guide:
            logger.info(f"Loaded v3.0 legacy consultation guide for {chapter_id}")
            return guide

        logger.warning(f"No consultation configuration found for chapter: {chapter_id}")
        return None

    def check_triggers(
        self,
        session_state: SessionState,
        turn_outcome: TurnOutcome,
        memo_digest: MemoDigest,
    ) -> List[ConsultationScenario]:
        """
        Check if any consultation scenarios are triggered.

        **DEPRECATED (v3.0)**: This method is part of the legacy v3.0 automatic
        trigger system. In v3.1, RMA reads the consultation guide and proactively
        decides when to consult experts using `consult_expert()`.

        This method is kept for backward compatibility and will be removed in v4.0.

        Args:
            session_state: Current session state
            turn_outcome: Outcome of current turn
            memo_digest: Memo digest

        Returns:
            List of triggered scenarios
        """
        logger.warning("check_triggers() is deprecated. Use v3.1 RMA-initiated consultation instead.")
        # Load consultation guide
        guide = self._load_consultation_guide(session_state.chapter_id)
        if not guide:
            return []

        triggered_scenarios = []

        for scenario in guide.consultation_scenarios:
            if self._evaluate_trigger_conditions(
                scenario.trigger_conditions,
                session_state,
                turn_outcome,
                memo_digest,
            ):
                triggered_scenarios.append(scenario)
                logger.info(f"Triggered consultation scenario: {scenario.scenario_id}")

        return triggered_scenarios

    def _evaluate_trigger_conditions(
        self,
        conditions: Dict,
        session_state: SessionState,
        turn_outcome: TurnOutcome,
        memo_digest: MemoDigest,
    ) -> bool:
        """
        Evaluate if trigger conditions are met.

        Args:
            conditions: Trigger conditions from scenario
            session_state: Current session state
            turn_outcome: Turn outcome
            memo_digest: Memo digest

        Returns:
            True if all conditions are met
        """
        # Check user_uploaded_file
        if conditions.get("user_uploaded_file"):
            # This would need to be tracked in session state or turn outcome
            # For now, we'll assume it's tracked elsewhere
            pass

        # Check attempts_since_last_progress
        if "attempts_since_last_progress_gte" in conditions:
            threshold = conditions["attempts_since_last_progress_gte"]
            if session_state.attempts_since_last_progress < threshold:
                return False

        # Check student_sentiment
        if "student_sentiment" in conditions:
            allowed_sentiments = conditions["student_sentiment"]
            if isinstance(allowed_sentiments, list):
                if turn_outcome.student_sentiment not in allowed_sentiments:
                    return False
            elif turn_outcome.student_sentiment != allowed_sentiments:
                return False

        # Check blocker_type
        if "blocker_type" in conditions:
            allowed_blockers = conditions["blocker_type"]
            if isinstance(allowed_blockers, list):
                if turn_outcome.blocker_type not in allowed_blockers:
                    return False
            elif turn_outcome.blocker_type != allowed_blockers:
                return False

        # Check checkpoint_reached
        if "checkpoint_reached" in conditions:
            if turn_outcome.checkpoint_reached != conditions["checkpoint_reached"]:
                return False

        # All conditions met
        return True

    async def consult_expert(
        self,
        expert_id: str,
        question: str,
        context: Dict[str, Any],
        expected_output_type: str,
        session_id: str,
        session_state: SessionState,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Consult an expert with a custom question (v3.1 RMA-initiated consultation).

        Args:
            expert_id: ID of expert to consult
            question: The question to ask the expert
            context: Context data for the consultation
            expected_output_type: Expected output schema type
            session_id: Session identifier
            session_state: Current session state
            scenario_id: Optional scenario identifier for logging

        Returns:
            Dict containing expert output, binding rules triggered, and instruction updates
        """
        consultation_id = self._get_next_consultation_id(session_id)
        start_timestamp = datetime.now().isoformat()

        logger.info(f"RMA-initiated consultation {consultation_id} with expert {expert_id}")

        # Get expert metadata
        expert = self._get_expert_by_id(expert_id)
        if not expert:
            raise ConsultationEngineError(f"Expert {expert_id} not found in yellow page")

        # Create consultation workspace
        consult_dir = Path("sessions") / session_id / "expert_workspace" / "consultations" / consultation_id
        consult_dir.mkdir(parents=True, exist_ok=True)

        # Create envelope
        envelope = ConsultingEnvelope(
            consulting_letter_title=f"RMA Consultation: {scenario_id or 'custom'}",
            consultation_id=consultation_id,
            user_turn_index=session_state.turn_index,
            round_index=1,
            rounds_remaining_after_this=0,
            parallel_group_id=f"pg_{consultation_id}_r1",
            scenario_id=scenario_id or "rma_custom",
            expert_id=expert_id,
            question=question,
            session_scope=SessionScope(allowed_root=f"sessions/{session_id}/working_files/"),
            expected_output_template=ExpectedOutputTemplate(
                output_type=expected_output_type,
                required_fields=["output"],  # Minimal required field
                binding=False,
            ),
            context=context,
        )

        # Run expert consultation
        result = await self.expert_runner.run_consultation_round(
            expert_id=expert_id,
            envelope=envelope,
            session_id=session_id,
            consultation_history=[],
        )

        expert_output = result.get("expert_output", {})

        # Check for binding rules (v3.1)
        consultation_context = self.load_chapter_consultation_context(session_state.chapter_id)
        triggered_rules = []
        instruction_updates = {}

        if isinstance(consultation_context, ConsultationContext):
            # v3.2: Check expert judgment binding rules
            for rule_name, rule in consultation_context.binding_rules.expert_judgment_binding.items():
                if self._check_expert_judgment_binding_trigger(rule_name, rule, expert_output, expected_output_type):
                    triggered_rules.append({
                        "rule_name": rule_name,
                        "condition": rule.condition,
                        "enforcement": rule.enforcement,
                    })
                    instruction_updates.update(
                        self._parse_enforcement_action(rule.enforcement, session_state)
                    )
                    logger.info(f"Expert judgment binding rule triggered: {rule_name}")
                    logger.info(f"  Condition: {rule.condition}")
                    logger.info(f"  Enforcement: {rule.enforcement}")

        # Save consultation metadata
        end_timestamp = datetime.now().isoformat()
        meta = ConsultationMeta(
            consultation_id=consultation_id,
            user_turn_index=session_state.turn_index,
            scenario_id=scenario_id or "rma_custom",
            experts_involved=[expert_id],
            total_rounds=1,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            termination_reason="single_round_consultation",
            binding=len(triggered_rules) > 0,
        )

        meta_path = consult_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)

        # Save expert output
        output_path = consult_dir / f"expert_output_{expert_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(expert_output, f, ensure_ascii=False, indent=2)

        logger.info(f"Consultation {consultation_id} completed")

        return {
            "consultation_id": consultation_id,
            "expert_id": expert_id,
            "expert_output": expert_output,
            "binding_rules_triggered": triggered_rules,
            "instruction_updates": instruction_updates,
            "meta": meta.model_dump(),
        }

    def _check_expert_judgment_binding_trigger(
        self,
        rule_name: str,
        rule,  # ExpertJudgmentBindingRule
        expert_output: Dict,
        expected_output_type: str,
    ) -> bool:
        """
        Check if an expert judgment binding rule should be triggered.

        This method evaluates the condition string against the expert output.
        The condition is a natural language description that should be parsed
        to determine if it matches the expert's output.

        Args:
            rule_name: Name of the rule
            rule: ExpertJudgmentBindingRule configuration
            expert_output: Expert's output
            expected_output_type: Expected output type

        Returns:
            True if rule should be triggered
        """
        condition = rule.condition
        import re

        # Pattern 1: "返回field=value" or "expert_name返回field=value"
        # e.g., "data_inspector返回is_suitable=false" or "返回is_suitable=false"
        match = re.search(r'返回(\w+)=(true|false|[\w_]+)', condition, re.IGNORECASE)
        if match:
            field_name = match.group(1)
            expected_value_str = match.group(2).lower()

            # Convert string to appropriate type
            if expected_value_str == "true":
                expected_value = True
            elif expected_value_str == "false":
                expected_value = False
            else:
                expected_value = expected_value_str

            # Check if field exists and matches
            actual_value = expert_output.get(field_name)
            return actual_value == expected_value

        # Pattern 2: "field非空" or "返回field非空" or "expert_name返回field非空"
        # e.g., "warnings非空" or "analyzer返回warnings非空"
        match = re.search(r'返回(\w+)非空', condition)
        if match:
            field_name = match.group(1)
        else:
            # Try without "返回"
            match = re.search(r'(\w+)非空', condition)
            if match:
                field_name = match.group(1)
            else:
                field_name = None

        if match and field_name:
            field_value = expert_output.get(field_name)

            # Check if field is non-empty
            if field_value is not None:
                if isinstance(field_value, (list, dict, str)):
                    return len(field_value) > 0
                return True
            return False

        # Pattern 3: "field>value" or "field<value" or "field>=value" or "field<=value"
        # e.g., "score<60" or "quality_checker返回score<60"
        match = re.search(r'返回(\w+)\s*([><=]+)\s*(\d+(?:\.\d+)?)', condition)
        if match:
            field_name = match.group(1)
            operator = match.group(2)
            threshold = float(match.group(3))
        else:
            # Try without "返回"
            match = re.search(r'(\w+)\s*([><=]+)\s*(\d+(?:\.\d+)?)', condition)
            if match:
                field_name = match.group(1)
                operator = match.group(2)
                threshold = float(match.group(3))
            else:
                field_name = None

        if match and field_name:
            actual_value = expert_output.get(field_name)
            if actual_value is not None:
                try:
                    actual_value = float(actual_value)
                    if operator == '>':
                        return actual_value > threshold
                    elif operator == '<':
                        return actual_value < threshold
                    elif operator == '>=':
                        return actual_value >= threshold
                    elif operator == '<=':
                        return actual_value <= threshold
                except (ValueError, TypeError):
                    pass

        # If no pattern matched, log warning and return False
        logger.warning(f"Could not evaluate condition: {rule.condition}")
        return False

    def _parse_enforcement_action(
        self,
        enforcement: str,
        session_state: SessionState,
    ) -> Dict[str, Any]:
        """
        Parse enforcement action string and return instruction packet updates.

        The enforcement string contains instructions like:
        "必须阻止任务推进，设置lock_until='user_uploads_suitable_dataset_or_uses_example'，
         设置allow_setup_helper_code=false，设置current_focus='等待学生解决数据集问题'"

        Or simpler forms like:
        "可以继续任务，但必须在guidance_for_ca中要求CA提醒学生注意数据质量问题"

        Args:
            enforcement: Enforcement action string
            session_state: Current session state

        Returns:
            Dictionary of instruction packet updates
        """
        updates = {}

        # Parse lock_until
        import re
        lock_until_match = re.search(r"lock_until='([^']+)'", enforcement)
        if lock_until_match:
            updates["lock_until"] = lock_until_match.group(1)

        # Parse allow_setup_helper_code
        if "allow_setup_helper_code=false" in enforcement:
            updates["allow_setup_helper_code"] = False
        elif "allow_setup_helper_code=true" in enforcement:
            updates["allow_setup_helper_code"] = True

        # Parse current_focus
        current_focus_match = re.search(r"current_focus='([^']+)'", enforcement)
        if current_focus_match:
            updates["current_focus"] = current_focus_match.group(1)

        # Parse guidance_for_ca (with quotes)
        guidance_match = re.search(r"guidance_for_ca='([^']+)'", enforcement)
        if guidance_match:
            updates["guidance_for_ca"] = guidance_match.group(1)
        # Parse guidance_for_ca (without quotes - natural language description)
        elif "guidance_for_ca中" in enforcement or "guidance_for_ca" in enforcement:
            # Extract the guidance from natural language
            # e.g., "必须在guidance_for_ca中要求CA提醒学生注意数据质量问题"
            if "要求CA" in enforcement:
                guidance_start = enforcement.find("要求CA")
                guidance_text = enforcement[guidance_start:]
                updates["guidance_for_ca"] = guidance_text

        # Parse must_check (array)
        must_check_match = re.search(r"must_check=\['([^']+)'\]", enforcement)
        if must_check_match:
            updates["must_check"] = [must_check_match.group(1)]

        return updates

    def select_experts(
        self,
        scenario: ConsultationScenario,
        guide: ConsultationGuide,
    ) -> List[ExpertMetadata]:
        """
        Select experts for a scenario based on relevance scoring.

        **DEPRECATED (v3.0)**: This method is part of the legacy v3.0 automatic
        trigger system. In v3.1, RMA decides which expert to consult based on
        the consultation guide.

        This method is kept for backward compatibility and will be removed in v4.0.

        Args:
            scenario: Consultation scenario
            guide: Consultation guide

        Returns:
            List of selected experts
        """
        logger.warning("select_experts() is deprecated. Use v3.1 RMA-initiated consultation instead.")
        # Get available experts from guide
        available_expert_ids = guide.available_experts_list

        # Score each expert
        expert_scores = {}
        for expert_id in available_expert_ids:
            # Get expert metadata from yellow page
            expert = self._get_expert_by_id(expert_id)
            if not expert:
                logger.warning(f"Expert {expert_id} not found in yellow page")
                continue

            # Get scoring hint from scenario
            if expert_id in scenario.expert_scoring_hints:
                score = scenario.expert_scoring_hints[expert_id].get("base_score", 0)
            else:
                # Default scoring based on tags/output_modes
                score = self._calculate_default_score(expert, scenario)

            expert_scores[expert_id] = score

        # Apply selection rule
        selection_rule = guide.multi_consultation_policy.get("selection_rule", {})

        # Find full matches (score=2) and partial matches (score=1)
        full_matches = [eid for eid, score in expert_scores.items() if score == 2]
        partial_matches = [eid for eid, score in expert_scores.items() if score == 1]

        selected_expert_ids = []

        if full_matches and selection_rule.get("if_any_full_match_consult_all_full_match"):
            selected_expert_ids = full_matches
        elif partial_matches and selection_rule.get("if_only_partial_consult_all_partial"):
            selected_expert_ids = partial_matches
        elif not full_matches and not partial_matches and selection_rule.get("if_all_irrelevant_skip_consultation"):
            selected_expert_ids = []

        # Get expert metadata
        selected_experts = []
        for expert_id in selected_expert_ids:
            expert = self._get_expert_by_id(expert_id)
            if expert:
                selected_experts.append(expert)

        logger.info(f"Selected {len(selected_experts)} experts for scenario {scenario.scenario_id}: {selected_expert_ids}")
        return selected_experts

    def _get_expert_by_id(self, expert_id: str) -> Optional[ExpertMetadata]:
        """Get expert metadata by ID from yellow page."""
        for expert in self.yellow_page.experts:
            if expert.expert_id == expert_id:
                return expert
        return None

    def _calculate_default_score(self, expert: ExpertMetadata, scenario: ConsultationScenario) -> int:
        """Calculate default relevance score based on expert metadata and scenario."""
        # Check if expert's output_modes match expected output type
        expected_output_type = scenario.expected_output_template.get("output_type")
        if expected_output_type in expert.output_modes:
            return 2  # Full match

        # Check if any tags are relevant (this is a heuristic)
        # For now, return 0 (irrelevant) if no output mode match
        return 0

    async def run_consultation(
        self,
        session_id: str,
        session_state: SessionState,
        scenario: ConsultationScenario,
        guide: ConsultationGuide,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a complete consultation (up to 4 rounds, parallel experts, wait-for-all).

        **DEPRECATED (v3.0)**: This method is part of the legacy v3.0 automatic
        trigger system. In v3.1, use `consult_expert()` for RMA-initiated consultations.

        This method is kept for backward compatibility and will be removed in v4.0.

        Args:
            session_id: Session identifier
            session_state: Current session state
            scenario: Consultation scenario
            guide: Consultation guide
            context: Context data for consultation

        Returns:
            Consultation results including expert outputs and metadata
        """
        logger.warning("run_consultation() is deprecated. Use consult_expert() instead.")
        consultation_id = self._get_next_consultation_id(session_id)
        start_timestamp = datetime.now().isoformat()

        logger.info(f"Starting consultation {consultation_id} for scenario {scenario.scenario_id}")

        # Select experts
        selected_experts = self.select_experts(scenario, guide)
        if not selected_experts:
            logger.info(f"No experts selected for scenario {scenario.scenario_id}, skipping consultation")
            return {
                "consultation_id": consultation_id,
                "scenario_id": scenario.scenario_id,
                "experts_involved": [],
                "total_rounds": 0,
                "skipped": True,
                "reason": "no_relevant_experts",
            }

        # Create consultation workspace
        consult_dir = Path("sessions") / session_id / "expert_workspace" / "consultations" / consultation_id
        consult_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transcript
        transcript_path = consult_dir / "transcript.jsonl"
        transcript_path.touch()

        # Run consultation rounds (max 4)
        max_rounds = guide.multi_consultation_policy.get("max_rounds_per_user_turn", 4)
        consultation_history = []
        all_expert_outputs = {}

        for round_index in range(1, max_rounds + 1):
            logger.info(f"Consultation {consultation_id} - Round {round_index}")

            # Create envelopes for all experts in parallel
            envelopes = []
            for expert in selected_experts:
                envelope = self._create_envelope(
                    consultation_id=consultation_id,
                    user_turn_index=session_state.turn_index,
                    round_index=round_index,
                    rounds_remaining=max_rounds - round_index,
                    scenario=scenario,
                    context=context,
                    session_id=session_id,
                    expert_id=expert.expert_id,
                )
                envelopes.append((expert.expert_id, envelope))

            # Run all experts in parallel (wait-for-all)
            tasks = []
            for expert_id, envelope in envelopes:
                task = self.expert_runner.run_consultation_round(
                    expert_id=expert_id,
                    envelope=envelope,
                    session_id=session_id,
                    consultation_history=consultation_history,
                )
                tasks.append(task)

            # Wait for all experts to complete
            round_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            round_outputs = {}
            termination_signals = []

            for i, result in enumerate(round_results):
                expert_id = envelopes[i][0]

                if isinstance(result, Exception):
                    logger.error(f"Expert {expert_id} failed in round {round_index}: {result}")
                    continue

                round_outputs[expert_id] = result
                all_expert_outputs[expert_id] = result["expert_output"]

                # Check termination signal
                if result.get("termination_signal"):
                    termination_signals.append(result.get("termination_reason"))

                # Log to transcript
                self._append_to_transcript(
                    transcript_path,
                    round_index,
                    expert_id,
                    envelopes[i][1],
                    result,
                )

            # Add to consultation history
            consultation_history.append({
                "round_index": round_index,
                "outputs": round_outputs,
            })

            # Check termination conditions
            if self._should_terminate(termination_signals, scenario, round_index, max_rounds):
                logger.info(f"Consultation {consultation_id} terminated after round {round_index}")
                break

        # Save consultation metadata
        end_timestamp = datetime.now().isoformat()
        meta = ConsultationMeta(
            consultation_id=consultation_id,
            user_turn_index=session_state.turn_index,
            scenario_id=scenario.scenario_id,
            experts_involved=[e.expert_id for e in selected_experts],
            total_rounds=len(consultation_history),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            termination_reason=termination_signals[0] if termination_signals else "max_rounds_reached",
            binding=scenario.expected_output_template.get("binding", False),
        )

        meta_path = consult_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)

        # Save expert outputs
        for expert_id, output in all_expert_outputs.items():
            output_path = consult_dir / f"expert_output_{expert_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Consultation {consultation_id} completed with {len(consultation_history)} rounds")

        return {
            "consultation_id": consultation_id,
            "scenario_id": scenario.scenario_id,
            "experts_involved": [e.expert_id for e in selected_experts],
            "total_rounds": len(consultation_history),
            "expert_outputs": all_expert_outputs,
            "meta": meta.model_dump(),
            "binding": meta.binding,
        }

    def _create_envelope(
        self,
        consultation_id: str,
        user_turn_index: int,
        round_index: int,
        rounds_remaining: int,
        scenario: ConsultationScenario,
        context: Dict[str, Any],
        session_id: str,
        expert_id: str,
    ) -> ConsultingEnvelope:
        """Create consulting envelope for an expert."""
        # Fill in template placeholders
        template = scenario.consulting_envelope_template
        title = template.consulting_letter_title.format(**context)
        question = template.question.format(**context)

        envelope = ConsultingEnvelope(
            consulting_letter_title=title,
            consultation_id=consultation_id,
            user_turn_index=user_turn_index,
            round_index=round_index,
            rounds_remaining_after_this=rounds_remaining,
            parallel_group_id=f"pg_{consultation_id}_r{round_index}",
            scenario_id=scenario.scenario_id,
            question=question,
            session_scope=SessionScope(allowed_root=f"sessions/{session_id}/working_files/"),
            expected_output_template=scenario.expected_output_template,
            context=context,
        )

        return envelope

    def _append_to_transcript(
        self,
        transcript_path: Path,
        round_index: int,
        expert_id: str,
        envelope: ConsultingEnvelope,
        result: Dict,
    ) -> None:
        """Append a round to the consultation transcript."""
        try:
            transcript_entry = {
                "round_index": round_index,
                "expert_id": expert_id,
                "timestamp": result.get("timestamp"),
                "request": {
                    "question": envelope.question,
                    "context": envelope.context,
                },
                "response": result.get("expert_output"),
            }

            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(transcript_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"Failed to append to transcript: {e}")

    def _should_terminate(
        self,
        termination_signals: List[str],
        scenario: ConsultationScenario,
        current_round: int,
        max_rounds: int,
    ) -> bool:
        """Check if consultation should terminate."""
        # Max rounds reached
        if current_round >= max_rounds:
            return True

        # Check termination rules
        termination_rules = scenario.termination_rules

        for signal in termination_signals:
            if signal in termination_rules:
                return True

        return False
