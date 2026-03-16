"""Expert report generation service for RMA."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..models.schemas import (
    ExpertReport,
    ExpertReportIndex,
    ConsultationGuide,
)

logger = logging.getLogger(__name__)


class ExpertReportGeneratorError(Exception):
    """Custom exception for expert report generator errors."""
    pass


class ExpertReportGenerator:
    """Service for generating student-facing expert reports from consultation results."""

    def __init__(self):
        """Initialize expert report generator."""
        pass

    def generate_report(
        self,
        session_id: str,
        consultation_result: Dict[str, Any],
        guide: ConsultationGuide,
        rma_analysis: Optional[str] = None,
    ) -> ExpertReport:
        """
        Generate a student-facing expert report from consultation results.

        Args:
            session_id: Session identifier
            consultation_result: Results from consultation engine
            guide: Consultation guide for the chapter
            rma_analysis: Optional RMA analysis/summary text

        Returns:
            ExpertReport instance
        """
        try:
            consultation_id = consultation_result["consultation_id"]
            scenario_id = consultation_result["scenario_id"]
            expert_outputs = consultation_result["expert_outputs"]
            meta = consultation_result["meta"]

            # Find the scenario in guide
            scenario = None
            for s in guide.consultation_scenarios:
                if s.scenario_id == scenario_id:
                    scenario = s
                    break

            if not scenario:
                raise ExpertReportGeneratorError(f"Scenario {scenario_id} not found in guide")

            # Generate report ID
            report_id = self._generate_report_id(session_id, consultation_id)

            # Extract key findings from expert outputs
            key_findings = self._extract_key_findings(expert_outputs, scenario)

            # Extract evidence
            evidence = self._extract_evidence(expert_outputs)

            # Generate summary
            summary = self._generate_summary(expert_outputs, scenario, rma_analysis)

            # Generate implication for next step
            implication = self._generate_implication(expert_outputs, scenario)

            # Create report
            report = ExpertReport(
                report_id=report_id,
                consultation_id=consultation_id,
                user_turn_index=meta["user_turn_index"],
                scenario_id=scenario_id,
                title=self._generate_title(scenario, expert_outputs),
                created_at=datetime.now().isoformat(),
                binding_effect="abort_task_path" if meta["binding"] else "none",
                summary=summary,
                key_findings=key_findings,
                evidence=evidence,
                implication_for_next_step=implication,
                expert_sources=meta["experts_involved"],
                related_consultation_ids=[consultation_id],
            )

            # Save report
            self._save_report(session_id, report)

            # Update index
            self._update_report_index(session_id, report)

            logger.info(f"Generated expert report {report_id} for consultation {consultation_id}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate expert report: {e}")
            raise ExpertReportGeneratorError(f"Failed to generate expert report: {e}")

    def _generate_report_id(self, session_id: str, consultation_id: str) -> str:
        """Generate unique report ID."""
        # Count existing reports for this consultation
        reports_dir = Path("sessions") / session_id / "expert_reports"
        existing = list(reports_dir.glob(f"report_{consultation_id}_*.json"))
        seq = len(existing) + 1
        return f"report_{consultation_id}_{seq:02d}"

    def _extract_key_findings(
        self,
        expert_outputs: Dict[str, Dict],
        scenario: Any,
    ) -> List[str]:
        """Extract key findings from expert outputs."""
        findings = []

        for expert_id, output in expert_outputs.items():
            output_type = output.get("output_type")

            if output_type == "suitability_judgment":
                is_suitable = output.get("is_suitable", False)
                if is_suitable:
                    findings.append("数据集符合任务要求")
                else:
                    findings.append("数据集不符合任务要求")

                # Add blocking issues
                blocking_issues = output.get("blocking_issues", [])
                for issue in blocking_issues:
                    findings.append(f"阻塞性问题：{issue}")

                # Add warning issues
                warning_issues = output.get("warning_issues", [])
                for issue in warning_issues:
                    findings.append(f"警告：{issue}")

            elif output_type == "concept_explanation":
                concept_name = output.get("concept_name", "未知概念")
                findings.append(f"已解释概念：{concept_name}")

                # Add common pitfalls
                pitfalls = output.get("common_pitfalls", [])
                if pitfalls:
                    findings.append(f"常见陷阱：{', '.join(pitfalls[:2])}")

            elif output_type == "validation_report":
                is_valid = output.get("is_valid", False)
                if is_valid:
                    findings.append("任务完成验证通过")
                else:
                    findings.append("任务完成验证未通过")

                missing = output.get("missing_elements", [])
                if missing:
                    findings.append(f"缺失要素：{', '.join(missing)}")

        return findings

    def _extract_evidence(self, expert_outputs: Dict[str, Dict]) -> Dict:
        """Extract evidence from expert outputs."""
        evidence = {}

        for expert_id, output in expert_outputs.items():
            if "evidence" in output:
                evidence[expert_id] = output["evidence"]

        return evidence

    def _generate_summary(
        self,
        expert_outputs: Dict[str, Dict],
        scenario: Any,
        rma_analysis: Optional[str],
    ) -> str:
        """Generate summary text for the report."""
        if rma_analysis:
            return rma_analysis

        # Generate default summary based on output type
        summaries = []

        for expert_id, output in expert_outputs.items():
            output_type = output.get("output_type")

            if output_type == "suitability_judgment":
                is_suitable = output.get("is_suitable", False)
                if is_suitable:
                    summaries.append("专家检查确认数据集符合教学要求。")
                else:
                    summaries.append("专家检查发现数据集存在问题，不符合教学要求。")

            elif output_type == "concept_explanation":
                concept_name = output.get("concept_name", "相关概念")
                summaries.append(f"专家已对'{concept_name}'进行了详细解释。")

            elif output_type == "validation_report":
                is_valid = output.get("is_valid", False)
                if is_valid:
                    summaries.append("专家验证确认任务已正确完成。")
                else:
                    summaries.append("专家验证发现任务完成存在不足。")

        return " ".join(summaries) if summaries else "专家咨询已完成。"

    def _generate_implication(
        self,
        expert_outputs: Dict[str, Dict],
        scenario: Any,
    ) -> str:
        """Generate implication for next step."""
        implications = []

        for expert_id, output in expert_outputs.items():
            if "recommended_next_step" in output:
                implications.append(output["recommended_next_step"])
            elif "recommended_action" in output:
                implications.append(output["recommended_action"])

        if implications:
            return " ".join(implications)

        # Default implication
        return "请根据专家建议继续学习。"

    def _generate_title(self, scenario: Any, expert_outputs: Dict[str, Dict]) -> str:
        """Generate report title."""
        # Use scenario description as base
        title = scenario.description

        # Customize based on output type
        for expert_id, output in expert_outputs.items():
            output_type = output.get("output_type")

            if output_type == "suitability_judgment":
                is_suitable = output.get("is_suitable", False)
                if is_suitable:
                    title = "数据集适用性检查：通过"
                else:
                    title = "数据集适用性检查：不通过"
                break

            elif output_type == "concept_explanation":
                concept_name = output.get("concept_name", "")
                if concept_name:
                    title = f"概念解释：{concept_name}"
                break

        return title

    def _save_report(self, session_id: str, report: ExpertReport) -> None:
        """Save expert report to file."""
        reports_dir = Path("sessions") / session_id / "expert_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"{report.report_id}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved expert report to {report_path}")

    def _update_report_index(self, session_id: str, report: ExpertReport) -> None:
        """Update expert reports index."""
        index_path = Path("sessions") / session_id / "expert_reports" / "index.json"

        # Load existing index
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            index_data = {"reports": []}

        # Add new report metadata
        report_metadata = {
            "report_id": report.report_id,
            "consultation_id": report.consultation_id,
            "user_turn_index": report.user_turn_index,
            "scenario_id": report.scenario_id,
            "title": report.title,
            "created_at": report.created_at,
            "binding_effect": report.binding_effect,
            "summary": report.summary,
        }

        index_data["reports"].append(report_metadata)

        # Save updated index
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Updated expert reports index for session {session_id}")

    def enforce_binding_rules(
        self,
        consultation_result: Dict[str, Any],
        scenario: Any,
    ) -> Dict[str, Any]:
        """
        Enforce binding decision rules from consultation guide.

        Args:
            consultation_result: Results from consultation
            scenario: Consultation scenario with binding rules

        Returns:
            Dictionary with enforcement actions
        """
        if not consultation_result.get("binding"):
            return {"binding_enforced": False}

        expert_outputs = consultation_result["expert_outputs"]
        binding_rules = scenario.binding_decision_rules

        enforcement_actions = {
            "binding_enforced": True,
            "actions": [],
        }

        for expert_id, output in expert_outputs.items():
            output_type = output.get("output_type")

            # Check suitability judgment
            if output_type == "suitability_judgment":
                is_suitable = output.get("is_suitable", True)

                if not is_suitable and "if_is_suitable_false" in binding_rules:
                    rule = binding_rules["if_is_suitable_false"]
                    enforcement_actions["actions"].append({
                        "rule": "if_is_suitable_false",
                        "action": rule.get("action"),
                        "instruction_packet_update": rule.get("instruction_packet_update"),
                    })

                elif is_suitable and output.get("warning_issues") and "if_is_suitable_true_with_warnings" in binding_rules:
                    rule = binding_rules["if_is_suitable_true_with_warnings"]
                    enforcement_actions["actions"].append({
                        "rule": "if_is_suitable_true_with_warnings",
                        "action": rule.get("action"),
                        "instruction_packet_update": rule.get("instruction_packet_update"),
                    })

            # Check validation report
            elif output_type == "validation_report":
                is_valid = output.get("is_valid", True)

                if not is_valid and "if_is_valid_false" in binding_rules:
                    rule = binding_rules["if_is_valid_false"]
                    enforcement_actions["actions"].append({
                        "rule": "if_is_valid_false",
                        "action": rule.get("action"),
                        "instruction_packet_update": rule.get("instruction_packet_update"),
                    })

        return enforcement_actions
