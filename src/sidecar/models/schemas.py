"""Pydantic models for data validation and type safety."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re
from datetime import datetime


class SubtaskStatus(BaseModel):
    """Status of a single subtask."""
    status: Literal["not_started", "in_progress", "completed"]
    evidence: List[str] = Field(default_factory=list)


class SessionConstraints(BaseModel):
    """Session configuration constraints."""
    max_input_length: int = 10000
    batch_error_log_every_n_turns: int = 5
    max_attempts_before_unlock: int = 3


class SessionState(BaseModel):
    """Complete session state tracking progress and configuration."""
    session_id: str
    chapter_id: str
    turn_index: int = 0
    subtask_status: Dict[str, SubtaskStatus]
    end_suggested: bool = False
    end_confirmed: bool = False
    constraints: SessionConstraints
    # v2.0 fields for instruction locking
    current_instruction_version: int = 1
    attempts_since_last_progress: int = 0
    last_progress_turn: int = 0

    def update(self, state_update: Dict) -> None:
        """Update session state with new values."""
        for key, value in state_update.items():
            if key == "subtask_status":
                # Update subtask status
                for subtask_id, status_data in value.items():
                    if subtask_id in self.subtask_status:
                        self.subtask_status[subtask_id].status = status_data.get(
                            "status", self.subtask_status[subtask_id].status
                        )
                        if "evidence" in status_data:
                            self.subtask_status[subtask_id].evidence.extend(
                                status_data["evidence"]
                            )
                    else:
                        self.subtask_status[subtask_id] = SubtaskStatus(**status_data)
            elif hasattr(self, key):
                setattr(self, key, value)


class InstructionPacket(BaseModel):
    """Instructions from Roadmap Manager to Companion Agent."""
    current_focus: str
    guidance_for_ca: str
    must_check: List[str] = Field(max_length=2, description="Max 2 critical checks")
    nice_check: List[str] = Field(default_factory=list, max_length=1, description="Max 1 optional check")
    instruction_version: int
    lock_until: Literal[
        "checkpoint_reached",
        "attempts_exceeded",
        "new_error_type",
        "user_uploads_suitable_dataset_or_uses_example",
    ]
    allow_setup_helper_code: bool
    setup_helper_scope: Literal["none", "file_creation", "env_setup", "path_check", "data_generation"] = "none"
    task_type: Literal["core", "scaffolding"] = "core"

    @field_validator("must_check", mode="before")
    @classmethod
    def clamp_must_check(cls, v: List[str]) -> List[str]:
        return v[:2] if isinstance(v, list) else v

    @field_validator("nice_check", mode="before")
    @classmethod
    def clamp_nice_check(cls, v: List[str]) -> List[str]:
        return v[:1] if isinstance(v, list) else v


class SubtaskEvidence(BaseModel):
    """Evidence for a specific subtask."""
    subtask_id: str
    evidence: str


class TurnOutcome(BaseModel):
    """Structured outcome of a single turn from Companion Agent."""
    what_user_attempted: str
    what_user_observed: str
    ca_teaching_mode: Literal["socratic", "direct"]
    ca_next_suggestion: str
    checkpoint_reached: bool
    blocker_type: Literal["none", "scaffolding", "core_concept", "core_implementation", "external_resource_needed"]
    student_sentiment: Literal["engaged", "confused", "frustrated", "fatigued"]
    evidence_for_subtasks: List[SubtaskEvidence] = Field(default_factory=list)
    # v3.2.0: Expert consultation signal from CA
    expert_consultation_needed: bool = False
    expert_consultation_reason: str = ""  # e.g., "user_requested_data_analysis", "concept_clarification_needed", "error_diagnosis_needed"


class MemoDigest(BaseModel):
    """Condensed summary from Memo Agent for Roadmap Manager consumption."""
    key_observations: List[str]
    student_struggles: List[str]
    student_strengths: List[str]
    student_sentiment: Literal["engaged", "confused", "frustrated", "fatigued"]
    blocker_type: Literal["none", "scaffolding", "core_concept", "core_implementation", "external_resource_needed"]
    progress_delta: Literal["none", "evidence_added", "checkpoint_reached", "regressed"]
    diagnostic_log: List[str] = Field(default_factory=list)


class MemoryChunk(BaseModel):
    """Mid-term memory chunk summary."""
    turn_range: str
    summary: str


class MemoryState(BaseModel):
    """Layered memory state for a session."""
    long_term_summary: str = ""
    mid_term_summaries: List[MemoryChunk] = Field(default_factory=list)
    last_mid_term_turn: int = -1
    last_updated_turn: int = -1
    version: int = 1


class StudentErrorEntry(BaseModel):
    """A single student error entry for the error summary."""
    turn_index: int
    error_type: Literal["conceptual", "coding", "scaffolding"]
    description: str
    context: str


class MemoResult(BaseModel):
    """Result from Memo Agent including updated report and digest."""
    updated_report: str
    digest: MemoDigest
    error_entries: List[StudentErrorEntry] = Field(default_factory=list)


class ConsultationRequest(BaseModel):
    """Request from RMA to consult an expert (v3.1)."""
    expert_id: str = Field(..., min_length=1, description="Expert to consult")
    question: str = Field(..., min_length=1, description="Question to ask the expert")
    context: Dict = Field(default_factory=dict, description="Context for consultation")
    expected_output_type: str = Field(..., min_length=1, description="Expected output type")
    scenario_id: str = Field(..., min_length=1, description="Scenario identifier")
    reasoning: str = Field(..., min_length=1, description="RMA's reasoning for this consultation")
    consulting_letter_title: str = Field(..., min_length=1, max_length=120, description="One-line summary for UI display")


class RoadmapManagerResult(BaseModel):
    """Result from Roadmap Manager including instruction packet and state update."""
    instruction_packet: InstructionPacket
    state_update: Dict
    consultation_request: Optional[ConsultationRequest] = Field(
        None,
        description="Optional request to consult an expert (v3.1)"
    )


class RoadmapManagerFinalResult(BaseModel):
    """
    Final result from RMA after expert consultation (Phase 2).

    This is RMA's interpretation and integration of expert findings,
    formatted for CA to understand and act upon.
    """
    instruction_packet: InstructionPacket
    state_update: Dict
    expert_consultation_summary: str = Field(
        default="",
        description="RMA's summary and interpretation of expert findings for CA"
    )
    guidance_for_ca: str = Field(
        default="",
        description="RMA's guidance on how CA should use the expert findings in teaching"
    )


# ============================================================================
# v3.0 Expert System Models
# ============================================================================

# Helper validators for ID formats
def validate_expert_id(v: str) -> str:
    """Validate expert_id format: lowercase letter start, alphanumeric + underscore/hyphen."""
    if not re.match(r'^[a-z][a-z0-9_\-]{1,63}$', v):
        raise ValueError(f"Invalid expert_id format: {v}. Must start with lowercase letter, contain only lowercase alphanumeric, underscore, or hyphen, and be 2-64 chars long.")
    return v

def validate_consultation_id(v: str) -> str:
    """Validate consultation_id format: consult_NNNN+."""
    if not re.match(r'^consult_[0-9]{4,}$', v):
        raise ValueError(f"Invalid consultation_id format: {v}. Must be 'consult_' followed by at least 4 digits.")
    return v

def validate_scenario_id(v: str) -> str:
    """Validate scenario_id format: lowercase letter start, alphanumeric + underscore/hyphen."""
    if not re.match(r'^[a-z][a-z0-9_\-]{1,63}$', v):
        raise ValueError(f"Invalid scenario_id format: {v}. Must start with lowercase letter, contain only lowercase alphanumeric, underscore, or hyphen, and be 2-64 chars long.")
    return v

def validate_parallel_group_id(v: str) -> str:
    """Validate parallel_group_id format: pg_XXX."""
    if not re.match(r'^pg_[a-zA-Z0-9_\-]{3,64}$', v):
        raise ValueError(f"Invalid parallel_group_id format: {v}. Must be 'pg_' followed by 3-64 alphanumeric/underscore/hyphen chars.")
    return v

def validate_report_id(v: str) -> str:
    """Validate report_id format: report_NNNN_NN."""
    if not re.match(r'^report_[0-9]{4,}_[0-9]{2,}$', v):
        raise ValueError(f"Invalid report_id format: {v}. Must be 'report_NNNN_NN' format.")
    return v


class ExpertMetadata(BaseModel):
    """Expert metadata extracted from principles.md."""
    expert_id: str = Field(..., description="Unique expert identifier")
    description: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    skill_handles: List[str] = Field(default_factory=list)
    output_modes: List[str] = Field(default_factory=list)

    @field_validator('expert_id')
    @classmethod
    def validate_expert_id_format(cls, v: str) -> str:
        return validate_expert_id(v)


class YellowPage(BaseModel):
    """Yellow page registry of all available experts."""
    experts: List[ExpertMetadata]


class SessionScope(BaseModel):
    """Session scope defining allowed file access boundaries."""
    allowed_root: str = Field(..., min_length=1, description="Root directory path for file access")


class ExpectedOutputTemplate(BaseModel):
    """Template defining expected output structure from expert."""
    output_type: str = Field(..., pattern=r'^[a-z][a-z0-9_\-]{1,63}$')
    required_fields: List[str] = Field(..., min_length=1, description="Fields that must be present in output")
    binding: bool = Field(default=False, description="Whether this output triggers binding RMA decisions")


class ConsultingEnvelope(BaseModel):
    """Envelope sent from RMA to expert for consultation."""
    consulting_letter_title: str = Field(..., min_length=1, max_length=120)
    consultation_id: str = Field(..., description="Unique consultation identifier")
    user_turn_index: int = Field(..., ge=0)
    round_index: int = Field(..., ge=1, le=10, description="Current round number (1-10)")
    rounds_remaining_after_this: int = Field(..., ge=0, le=10)
    parallel_group_id: str = Field(..., description="Identifier for parallel consultation group")
    scenario_id: str = Field(..., description="Scenario identifier from consultation_guide")
    expert_id: Optional[str] = Field(None, description="Target expert identifier (optional if implied by routing)")
    question: str = Field(..., min_length=1)
    session_scope: SessionScope
    expected_output_template: ExpectedOutputTemplate
    context: Dict = Field(default_factory=dict, description="Additional context for the consultation")
    constraints: Dict = Field(default_factory=dict, description="Optional constraints (e.g., token budget hints)")

    @field_validator('consultation_id')
    @classmethod
    def validate_consultation_id_format(cls, v: str) -> str:
        return validate_consultation_id(v)

    @field_validator('parallel_group_id')
    @classmethod
    def validate_parallel_group_id_format(cls, v: str) -> str:
        return validate_parallel_group_id(v)

    @field_validator('scenario_id')
    @classmethod
    def validate_scenario_id_format(cls, v: str) -> str:
        return validate_scenario_id(v)

    @field_validator('expert_id')
    @classmethod
    def validate_expert_id_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_expert_id(v)
        return v


class ConsultationMeta(BaseModel):
    """Metadata for a consultation stored in expert_workspace."""
    consultation_id: str
    user_turn_index: int = Field(..., ge=0)
    scenario_id: str
    experts_involved: List[str]
    total_rounds: int = Field(..., ge=1)
    start_timestamp: str
    end_timestamp: Optional[str] = None
    termination_reason: Optional[str] = None
    binding: bool = False

    @field_validator('consultation_id')
    @classmethod
    def validate_consultation_id_format(cls, v: str) -> str:
        return validate_consultation_id(v)

    @field_validator('scenario_id')
    @classmethod
    def validate_scenario_id_format(cls, v: str) -> str:
        return validate_scenario_id(v)


class KnowledgeSource(BaseModel):
    """Structured knowledge source reference."""
    doc: str = Field(..., min_length=1, description="Document or file path")
    excerpt_used: bool = Field(..., description="Whether an excerpt from this source was used")


class ExpertOutput(BaseModel):
    """Base model for expert structured output."""
    output_type: str = Field(..., pattern=r'^[a-z][a-z0-9_\-]{1,63}$')
    info_gain: bool = Field(..., description="Whether this response provides meaningful new information relative to prior rounds")
    confidence: Literal["low", "medium", "high"] = Field(default="medium", description="Expert's confidence level in this output")
    knowledge_sources: List[KnowledgeSource] = Field(default_factory=list, description="Sources consulted for this output")
    answer_summary: Optional[str] = Field(None, description="Brief summary of the answer")
    recommended_next_step: Optional[str] = Field(None, description="Suggested next action for RMA or student")


class SuitabilityJudgment(ExpertOutput):
    """Expert output for dataset suitability judgment."""
    output_type: Literal["suitability_judgment"] = "suitability_judgment"
    is_suitable: bool = Field(..., description="Whether the dataset/resource is suitable for the task")
    blocking_issues: List[str] = Field(default_factory=list, description="Critical issues that prevent task completion")
    warning_issues: List[str] = Field(default_factory=list, description="Non-critical issues that may affect quality")
    evidence: Dict = Field(default_factory=dict, description="Structured evidence supporting the judgment")


class ConceptExplanation(ExpertOutput):
    """Expert output for concept explanation."""
    output_type: Literal["concept_explanation"] = "concept_explanation"
    concept_name: str = Field(..., min_length=1)
    definition: str = Field(..., min_length=1)
    use_cases: List[str] = Field(default_factory=list)
    syntax: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    simple_example: Optional[str] = None
    common_pitfalls: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)


class DatasetOverviewReport(ExpertOutput):
    """Expert output for dataset overview."""
    output_type: Literal["dataset_overview_report"] = "dataset_overview_report"
    summary: str = Field(..., min_length=1)
    columns: List[Dict] = Field(default_factory=list)
    row_count: int = Field(..., ge=0)
    duplicate_rows: int = Field(..., ge=0)
    overall_missing_rate: float = Field(..., ge=0.0, le=1.0)


class SkillInputs(BaseModel):
    """Structured inputs for skill invocation."""
    files_read: List[str] = Field(default_factory=list, description="List of files read during skill execution")
    parameters: Dict = Field(default_factory=dict, description="Parameters passed to the skill")


class SkillOutputs(BaseModel):
    """Structured outputs from skill invocation."""
    files_written: List[str] = Field(default_factory=list, description="List of files written during skill execution")


class SkillCallLog(BaseModel):
    """Log entry for a single skill invocation."""
    timestamp: str = Field(..., description="ISO 8601 timestamp of skill invocation")
    expert_id: str
    consultation_id: str
    user_turn_index: int = Field(..., ge=0)
    round_index: int = Field(..., ge=1, le=10)
    skill_name: str = Field(..., pattern=r'^[a-z][a-z0-9_\-]{1,63}$')
    cwd: str = Field(..., min_length=1, description="Working directory during skill execution")
    inputs: SkillInputs
    outputs: SkillOutputs
    stdout_stderr_paths: Dict[str, Optional[str]] = Field(default_factory=dict, description="Paths to stdout/stderr logs")
    execution_time_ms: Optional[int] = Field(None, ge=0, description="Execution time in milliseconds")
    success: bool = Field(default=True, description="Whether skill execution succeeded")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")

    @field_validator('expert_id')
    @classmethod
    def validate_expert_id_format(cls, v: str) -> str:
        return validate_expert_id(v)

    @field_validator('consultation_id')
    @classmethod
    def validate_consultation_id_format(cls, v: str) -> str:
        return validate_consultation_id(v)


class ExpertReport(BaseModel):
    """Student-visible expert report authored by RMA."""
    report_id: str = Field(..., description="Unique report identifier")
    consultation_id: str
    user_turn_index: int = Field(..., ge=0)
    scenario_id: str
    expert_id: Optional[str] = Field(None, description="Primary expert consulted for this report")
    title: str = Field(..., min_length=1, max_length=140, description="Report title")
    created_at: str = Field(..., description="ISO 8601 timestamp when report was created")
    binding_effect: Literal["none", "abort_task_path"] = Field(default="none", description="Whether this report triggers binding RMA decisions")
    summary: str = Field(..., min_length=1)
    key_findings: List[str] = Field(..., min_length=1)
    evidence: Dict = Field(default_factory=dict)
    implication_for_next_step: str = Field(..., min_length=1)
    expert_sources: List[str] = Field(default_factory=list, description="List of experts consulted")
    related_consultation_ids: List[str] = Field(default_factory=list, description="Related consultation IDs for traceability")

    @field_validator('report_id')
    @classmethod
    def validate_report_id_format(cls, v: str) -> str:
        return validate_report_id(v)

    @field_validator('consultation_id')
    @classmethod
    def validate_consultation_id_format(cls, v: str) -> str:
        return validate_consultation_id(v)

    @field_validator('scenario_id')
    @classmethod
    def validate_scenario_id_format(cls, v: str) -> str:
        return validate_scenario_id(v)

    @field_validator('expert_id')
    @classmethod
    def validate_expert_id_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_expert_id(v)
        return v


class ExpertReportIndex(BaseModel):
    """Index of all expert reports in a session."""
    reports: List[Dict]  # Simplified metadata for each report


class TriggerConditions(BaseModel):
    """Trigger conditions for a consultation scenario."""
    user_uploaded_file: Optional[bool] = None
    file_extensions: Optional[List[str]] = None
    attempts_since_last_progress_gte: Optional[int] = Field(None, ge=0)
    task_type: Optional[Literal["core", "scaffolding"]] = None
    detected_error_types_any: Optional[List[str]] = None
    student_sentiment: Optional[List[Literal["engaged", "confused", "frustrated", "fatigued"]]] = None
    blocker_type: Optional[str | List[str]] = None  # Can be string or list
    ca_teaching_mode: Optional[Literal["socratic", "direct"]] = None
    checkpoint_reached: Optional[bool] = None
    error_message_available: Optional[bool] = None
    evidence_quality: Optional[str] = None


class ConsultingEnvelopeTemplate(BaseModel):
    """Template for consulting envelope in a scenario."""
    consulting_letter_title: str = Field(..., min_length=1, max_length=120)
    question: str = Field(..., min_length=1)
    expected_output_type: str = Field(..., pattern=r'^[a-z][a-z0-9_\-]{1,63}$')
    context_fields: List[str] = Field(default_factory=list, description="Fields to be filled from context")
    preferred_expert_tags: Optional[List[str]] = None
    required_skill_handle: Optional[str] = Field(None, pattern=r'^[a-z][a-z0-9_\-]{1,63}$')


class ConsultationScenario(BaseModel):
    """A consultation scenario defined in consultation_guide.json."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    description: str = Field(..., min_length=1)
    trigger_conditions: TriggerConditions
    expert_selection_rule: Literal["rank_and_score_then_select", "fixed_expert_set"]
    fixed_expert_set: Optional[List[str]] = Field(None, description="Fixed set of experts (only if expert_selection_rule == fixed_expert_set)")
    expert_scoring_hints: Dict[str, Dict] = Field(default_factory=dict, description="Hints for scoring experts")
    consulting_envelope_template: ConsultingEnvelopeTemplate
    expected_output_template: ExpectedOutputTemplate
    termination_rules: List[str] = Field(..., min_length=1)
    binding_decision_rules: Dict = Field(default_factory=dict)

    @field_validator('scenario_id')
    @classmethod
    def validate_scenario_id_format(cls, v: str) -> str:
        return validate_scenario_id(v)

    @field_validator('fixed_expert_set')
    @classmethod
    def validate_fixed_expert_set(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        if v is not None:
            for expert_id in v:
                validate_expert_id(expert_id)
        return v


class RelevanceScale(BaseModel):
    """Relevance scale definitions."""
    model_config = {"populate_by_name": True}

    minus_one: str = Field(..., alias="-1")
    zero: str = Field(..., alias="0")
    one: str = Field(..., alias="1")
    two: str = Field(..., alias="2")


class MultiConsultationSelectionRule(BaseModel):
    """Selection rules for multi-consultation scenarios."""
    if_any_full_match_consult_all_full_match: bool = True
    ignore_partial_if_full_exists: bool = True
    if_only_partial_consult_all_partial: bool = True
    if_all_irrelevant_skip_consultation: Optional[bool] = None


class MultiConsultationPolicy(BaseModel):
    """Policy for handling multiple consultations."""
    parallel: bool = Field(default=True, description="Whether to run consultations in parallel")
    wait_for_all: bool = Field(default=True, description="Whether to wait for all consultations to complete")
    max_rounds_per_user_turn: int = Field(..., ge=1, le=10)
    selection_rule: MultiConsultationSelectionRule


class RMAExpertReportTemplate(BaseModel):
    """Template for RMA expert reports."""
    required_fields: List[str] = Field(..., min_length=1)
    format: Optional[str] = "markdown"
    student_visible: Optional[bool] = True
    ca_consumable: Optional[bool] = True


class ConsultationLogging(BaseModel):
    """Logging configuration for consultations."""
    log_all_envelopes: bool = True
    log_all_transcripts: bool = True
    log_all_expert_outputs: bool = True
    log_skill_invocations: bool = True
    audit_trail_required: bool = True


class ConsultationGuide(BaseModel):
    """
    Chapter-level consultation guide configuration (v3.0 legacy format).

    **DEPRECATED**: This model is part of the legacy v3.0 automatic trigger system.
    Use v3.1 ConsultationConfig + ConsultationContext instead.

    Will be removed in v4.0.
    """
    chapter_id: str = Field(..., min_length=1)
    version: str = Field(..., pattern=r'^\d+\.\d+(\.\d+)?$', description="Semantic version (e.g., 3.0 or 3.0.1)")
    available_experts_list: List[str] = Field(..., min_length=1)
    relevance_scale: Dict[str, str] = Field(..., description="Relevance scale definitions")
    multi_consultation_policy: MultiConsultationPolicy
    consultation_scenarios: List[ConsultationScenario] = Field(..., min_length=1)
    rma_expert_report_template: RMAExpertReportTemplate
    consultation_logging: ConsultationLogging
    chapter_specific_notes: Dict = Field(default_factory=dict)

    @field_validator('available_experts_list')
    @classmethod
    def validate_available_experts(cls, v: List[str]) -> List[str]:
        for expert_id in v:
            validate_expert_id(expert_id)
        return v


# ============================================================================
# v3.1 Hybrid Consultation Models (YAML + Markdown)
# ============================================================================

class ExpertJudgmentBindingRule(BaseModel):
    """Expert judgment binding rule with condition and enforcement."""
    condition: str = Field(..., description="Condition that triggers this rule (e.g., 'data_inspector返回is_suitable=false')")
    enforcement: str = Field(..., description="Enforcement action RMA must take (e.g., '必须阻止任务推进，设置lock_until=...')")


class BindingRules(BaseModel):
    """Complete binding rules configuration (principle-based)."""
    must_consult: List[str] = Field(default_factory=list, description="Principles for when RMA MUST consult experts")
    must_not_consult: List[str] = Field(default_factory=list, description="Principles for when RMA MUST NOT consult experts")
    expert_judgment_binding: Dict[str, ExpertJudgmentBindingRule] = Field(default_factory=dict, description="Rules for enforcing expert judgments")


class ConsultationConfig(BaseModel):
    """v3.1 YAML configuration for consultation (hard constraints)."""
    chapter_id: str = Field(..., min_length=1)
    version: str = Field(..., pattern=r'^\d+\.\d+(\.\d+)?$', description="Semantic version (e.g., 3.1)")
    available_experts: List[str] = Field(..., min_length=1)
    multi_consultation_policy: Dict = Field(default_factory=dict)
    binding_rules: BindingRules = Field(default_factory=BindingRules)
    logging: Dict = Field(default_factory=dict)
    schema_version: str = "3.0"
    expected_output_schemas: Dict[str, str] = Field(default_factory=dict)

    @field_validator('available_experts')
    @classmethod
    def validate_available_experts(cls, v: List[str]) -> List[str]:
        for expert_id in v:
            validate_expert_id(expert_id)
        return v


class ConsultationContext(BaseModel):
    """Combined consultation context (config + guide text) for v3.1."""
    config: ConsultationConfig
    guide_text: str = Field(..., min_length=1, description="Markdown guide content")
    binding_rules: BindingRules = Field(default_factory=BindingRules)
