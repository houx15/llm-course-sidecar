"""Data inspector skills - actual implementation for dataset analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Conditional pandas import - only needed when not using sandbox
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .decision_logger import DecisionLoggerMixin

logger = logging.getLogger(__name__)


class DataInspectorSkillsError(Exception):
    """Custom exception for data inspector skills errors."""
    pass


class DataInspectorSkills(DecisionLoggerMixin):
    """Implementation of data inspector expert skills."""

    def __init__(
        self,
        working_root: Path,
        use_sandbox: bool = None,
        session_id: Optional[str] = None,
        consultation_id: Optional[str] = None,
    ):
        """
        Initialize data inspector skills.

        Args:
            working_root: Root directory for file operations (sandbox)
            use_sandbox: Whether to use sandbox execution. If None, read from env.
            session_id: Session identifier (for decision logging)
            consultation_id: Consultation identifier (for decision logging)
        """
        self.working_root = Path(working_root)
        if not self.working_root.exists():
            raise DataInspectorSkillsError(f"Working root does not exist: {working_root}")

        # Determine sandbox mode
        if use_sandbox is None:
            use_sandbox = os.getenv("USE_EXPERT_SANDBOX", "false").lower() == "true"

        self.use_sandbox = use_sandbox

        # Initialize sandbox if needed
        if self.use_sandbox:
            from .expert_sandbox import ExpertSandbox
            self.sandbox = ExpertSandbox()
            if not self.sandbox.is_enabled():
                logger.warning("Sandbox requested but not available, falling back to direct execution")
                self.use_sandbox = False
        else:
            self.sandbox = None
            # Check pandas availability for direct execution
            if not PANDAS_AVAILABLE:
                raise DataInspectorSkillsError(
                    "pandas not available and sandbox not enabled. "
                    "Either install pandas or enable sandbox mode with USE_EXPERT_SANDBOX=true"
                )

        # Initialize decision logger if session/consultation IDs provided
        if session_id and consultation_id:
            self._init_decision_logger(
                session_id=session_id,
                expert_id="data_inspector",
                consultation_id=consultation_id,
            )
        else:
            self._decision_logger = None

    def _validate_path(self, file_path: str) -> Path:
        """
        Validate that file path is within sandbox.

        Args:
            file_path: Relative file path

        Returns:
            Resolved absolute path

        Raises:
            DataInspectorSkillsError: If path is outside sandbox
        """
        target = (self.working_root / file_path).resolve()
        if not str(target).startswith(str(self.working_root.resolve())):
            raise DataInspectorSkillsError(f"Path {file_path} is outside sandbox")
        return target

    def inspect_dataset(
        self,
        file_name: str,
        sample_rows: int = 50,
    ) -> Dict[str, Any]:
        """
        Inspect a dataset and generate basic profile.

        Args:
            file_name: Name of the dataset file in working_files
            sample_rows: Number of rows to sample for inspection

        Returns:
            Dictionary with inspection results
        """
        if self.use_sandbox:
            return self._inspect_dataset_sandbox(file_name, sample_rows)
        else:
            return self._inspect_dataset_direct(file_name, sample_rows)

    def _inspect_dataset_direct(
        self,
        file_name: str,
        sample_rows: int = 50,
    ) -> Dict[str, Any]:
        """Direct execution of inspect_dataset (no sandbox)."""
        try:
            file_path = self._validate_path(file_name)

            if not file_path.exists():
                raise DataInspectorSkillsError(f"File not found: {file_name}")

            # Read dataset based on extension
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=sample_rows)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=sample_rows)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
                if len(df) > sample_rows:
                    df = df.head(sample_rows)
            else:
                raise DataInspectorSkillsError(f"Unsupported file format: {file_path.suffix}")

            # Collect basic information
            columns_info = []
            for col in df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_rate": float(df[col].isna().sum() / len(df)),
                    "unique_count": int(df[col].nunique()),
                }

                # Add sample values
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    col_info["sample_values"] = non_null_values.head(3).tolist()
                else:
                    col_info["sample_values"] = []

                columns_info.append(col_info)

            # Check for duplicates
            duplicate_count = df.duplicated().sum()

            # Overall missing rate
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isna().sum().sum()
            overall_missing_rate = float(missing_cells / total_cells) if total_cells > 0 else 0.0

            profile = {
                "file_name": file_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": columns_info,
                "duplicate_rows": int(duplicate_count),
                "overall_missing_rate": overall_missing_rate,
                "sample_inspected": sample_rows,
                "timestamp": datetime.now().isoformat(),
            }

            # Write profile to working_files
            profile_path = self.working_root / "dataset_profile.json"
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)

            logger.info(f"Dataset profile generated: {profile_path}")
            return profile

        except pd.errors.EmptyDataError:
            raise DataInspectorSkillsError(f"File {file_name} is empty")
        except pd.errors.ParserError as e:
            raise DataInspectorSkillsError(f"Failed to parse {file_name}: {e}")
        except Exception as e:
            logger.error(f"Error inspecting dataset: {e}")
            raise DataInspectorSkillsError(f"Failed to inspect dataset: {e}")

    def _inspect_dataset_sandbox(
        self,
        file_name: str,
        sample_rows: int = 50,
    ) -> Dict[str, Any]:
        """Sandbox execution of inspect_dataset."""
        code = f"""
import pandas as pd
import json
from datetime import datetime

file_name = {json.dumps(file_name)}
sample_rows = {sample_rows}

try:
    # Read dataset
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name, nrows=sample_rows)
    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_name, nrows=sample_rows)
    elif file_name.endswith('.json'):
        df = pd.read_json(file_name)
        if len(df) > sample_rows:
            df = df.head(sample_rows)
    else:
        raise ValueError(f"Unsupported file format")

    # Collect column information
    columns_info = []
    for col in df.columns:
        col_info = {{
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().sum() / len(df)),
            "unique_count": int(df[col].nunique()),
        }}

        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            col_info["sample_values"] = non_null_values.head(3).tolist()
        else:
            col_info["sample_values"] = []

        columns_info.append(col_info)

    # Check duplicates
    duplicate_count = int(df.duplicated().sum())

    # Overall missing rate
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    overall_missing_rate = float(missing_cells / total_cells) if total_cells > 0 else 0.0

    profile = {{
        "file_name": file_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
        "duplicate_rows": duplicate_count,
        "overall_missing_rate": overall_missing_rate,
        "sample_inspected": sample_rows,
        "timestamp": datetime.now().isoformat(),
    }}

    # Write profile
    with open("dataset_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(json.dumps({{"success": True, "result": profile}}))

except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""

        result = self.sandbox.execute(code, self.working_root, timeout=30)

        if not result["success"]:
            raise DataInspectorSkillsError(f"Sandbox execution failed: {result['stderr']}")

        try:
            output = json.loads(result["stdout"])
            if not output["success"]:
                raise DataInspectorSkillsError(f"Skill execution failed: {output['error']}")
            return output["result"]
        except json.JSONDecodeError:
            raise DataInspectorSkillsError(f"Failed to parse sandbox output: {result['stdout']}")


    def check_suitability(
        self,
        file_name: str,
        required_columns: list,
        minimum_rows: int = 100,
        maximum_missing_rate: float = 0.3,
        expected_dtypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Check if dataset is suitable for teaching task.

        Args:
            file_name: Name of the dataset file
            required_columns: List of required column names
            minimum_rows: Minimum number of rows required
            maximum_missing_rate: Maximum acceptable missing rate per column
            expected_dtypes: Optional dict of column -> expected dtype

        Returns:
            Suitability judgment dictionary
        """
        if self.use_sandbox:
            return self._check_suitability_sandbox(
                file_name, required_columns, minimum_rows, maximum_missing_rate, expected_dtypes
            )
        else:
            return self._check_suitability_direct(
                file_name, required_columns, minimum_rows, maximum_missing_rate, expected_dtypes
            )

    def _check_suitability_direct(
        self,
        file_name: str,
        required_columns: list,
        minimum_rows: int = 100,
        maximum_missing_rate: float = 0.3,
        expected_dtypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Direct execution of check_suitability (no sandbox)."""
        try:
            file_path = self._validate_path(file_name)

            if not file_path.exists():
                raise DataInspectorSkillsError(f"File not found: {file_name}")

            # Read full dataset to check row count
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            else:
                raise DataInspectorSkillsError(f"Unsupported file format: {file_path.suffix}")

            # Collect column information
            columns_info = []
            for col in df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_rate": float(df[col].isna().sum() / len(df)),
                    "unique_count": int(df[col].nunique()),
                }
                columns_info.append(col_info)

            # Check requirements
            blocking_issues = []
            warning_issues = []

            # Check row count
            if len(df) < minimum_rows:
                blocking_issues.append(
                    f"数据行数不足：需要至少{minimum_rows}行，实际只有{len(df)}行"
                )
                # Log decision
                self._log_decision(
                    skill_name="check_suitability",
                    decision_point="row_count_check",
                    input_data={"minimum": minimum_rows, "actual": len(df)},
                    output_data={"is_sufficient": False},
                    reasoning=f"Row count {len(df)} is below minimum {minimum_rows}",
                    confidence=1.0,
                )
            else:
                self._log_decision(
                    skill_name="check_suitability",
                    decision_point="row_count_check",
                    input_data={"minimum": minimum_rows, "actual": len(df)},
                    output_data={"is_sufficient": True},
                    reasoning=f"Row count {len(df)} meets minimum {minimum_rows}",
                    confidence=1.0,
                )

            # Check required columns
            actual_columns = [col["name"] for col in columns_info]
            missing_columns = [col for col in required_columns if col not in actual_columns]

            if missing_columns:
                blocking_issues.append(
                    f"缺少必需列：{', '.join(missing_columns)}"
                )
                # Log decision
                self._log_decision(
                    skill_name="check_suitability",
                    decision_point="column_check",
                    input_data={"required": required_columns, "actual": actual_columns},
                    output_data={"missing": missing_columns, "is_suitable": False},
                    reasoning=f"Missing required columns: {missing_columns}",
                    confidence=1.0,
                )
            else:
                self._log_decision(
                    skill_name="check_suitability",
                    decision_point="column_check",
                    input_data={"required": required_columns, "actual": actual_columns},
                    output_data={"missing": [], "is_suitable": True},
                    reasoning="All required columns found",
                    confidence=1.0,
                )

            # Check data types
            if expected_dtypes:
                for col_name, expected_dtype in expected_dtypes.items():
                    if col_name in actual_columns:
                        col_info = next((c for c in columns_info if c["name"] == col_name), None)
                        if col_info:
                            actual_dtype = col_info["dtype"]
                            # Simplified dtype matching (e.g., int64 matches int)
                            if not self._dtype_matches(actual_dtype, expected_dtype):
                                warning_issues.append(
                                    f"列'{col_name}'的数据类型不匹配：期望{expected_dtype}，实际为{actual_dtype}"
                                )

            # Check missing rates
            for col_info in columns_info:
                if col_info["name"] in required_columns:
                    if col_info["missing_rate"] > maximum_missing_rate:
                        warning_issues.append(
                            f"列'{col_info['name']}'缺失率过高：{col_info['missing_rate']:.1%}（阈值：{maximum_missing_rate:.1%}）"
                        )

            # Check duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                dup_rate = duplicate_count / len(df)
                if dup_rate > 0.1:  # More than 10% duplicates
                    warning_issues.append(
                        f"检测到{duplicate_count}个重复行（{dup_rate:.1%}）"
                    )

            # Determine suitability
            is_suitable = len(blocking_issues) == 0

            # Log final suitability decision
            self._log_decision(
                skill_name="check_suitability",
                decision_point="final_judgment",
                input_data={
                    "blocking_issues_count": len(blocking_issues),
                    "warning_issues_count": len(warning_issues),
                },
                output_data={"is_suitable": is_suitable},
                reasoning=f"Suitability determined: {is_suitable} (blocking: {len(blocking_issues)}, warnings: {len(warning_issues)})",
                confidence=1.0 if len(blocking_issues) == 0 and len(warning_issues) == 0 else 0.8,
            )

            # Generate recommendation
            if is_suitable and len(warning_issues) == 0:
                recommended_next_step = "数据集完全符合要求，可以继续任务。"
            elif is_suitable and len(warning_issues) > 0:
                recommended_next_step = "数据集基本符合要求，但存在一些质量问题。建议学生注意这些问题，或选择重新上传更高质量的数据集。"
            else:
                recommended_next_step = f"数据集不符合要求。请上传包含必需列{required_columns}的数据集，且至少有{minimum_rows}行数据。"

            judgment = {
                "output_type": "suitability_judgment",
                "is_suitable": is_suitable,
                "blocking_issues": blocking_issues,
                "warning_issues": warning_issues,
                "evidence": {
                    "columns_found": actual_columns,
                    "columns_required": required_columns,
                    "row_count": len(df),
                    "minimum_rows_required": minimum_rows,
                    "missing_rate_by_column": {
                        col["name"]: col["missing_rate"]
                        for col in columns_info
                        if col["name"] in required_columns
                    },
                },
                "recommended_next_step": recommended_next_step,
                "knowledge_sources": [],
            }

            # Write judgment to working_files
            judgment_path = self.working_root / "suitability_judgment.json"
            with open(judgment_path, "w", encoding="utf-8") as f:
                json.dump(judgment, f, ensure_ascii=False, indent=2)

            logger.info(f"Suitability judgment generated: {judgment_path}")
            return judgment

        except Exception as e:
            logger.error(f"Error checking suitability: {e}")
            raise DataInspectorSkillsError(f"Failed to check suitability: {e}")

    def _dtype_matches(self, actual: str, expected: str) -> bool:
        """
        Check if actual dtype matches expected dtype (simplified).

        Args:
            actual: Actual pandas dtype string
            expected: Expected dtype string

        Returns:
            True if types match
        """
        # Normalize types
        actual_lower = actual.lower()
        expected_lower = expected.lower()

        # Direct match
        if actual_lower == expected_lower:
            return True

        # Integer types
        if expected_lower in ["int", "integer"] and "int" in actual_lower:
            return True

        # Float types
        if expected_lower in ["float", "number", "numeric"] and "float" in actual_lower:
            return True

        # String types
        if expected_lower in ["str", "string", "text", "object"] and actual_lower in ["object", "string"]:
            return True

        return False

    def profile_dataset(
        self,
        file_name: str,
    ) -> Dict[str, Any]:
        """
        Generate detailed dataset profile (full read).

        Args:
            file_name: Name of the dataset file

        Returns:
            Detailed profile dictionary
        """
        try:
            file_path = self._validate_path(file_name)

            if not file_path.exists():
                raise DataInspectorSkillsError(f"File not found: {file_name}")

            # Read full dataset
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            else:
                raise DataInspectorSkillsError(f"Unsupported file format: {file_path.suffix}")

            # Detailed column analysis
            columns_detail = []
            for col in df.columns:
                col_detail = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_rate": float(df[col].isna().sum() / len(df)),
                    "unique_count": int(df[col].nunique()),
                }

                # Numeric columns: add statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_detail["statistics"] = {
                        "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                        "median": float(df[col].median()) if not df[col].isna().all() else None,
                        "std": float(df[col].std()) if not df[col].isna().all() else None,
                        "min": float(df[col].min()) if not df[col].isna().all() else None,
                        "max": float(df[col].max()) if not df[col].isna().all() else None,
                        "q25": float(df[col].quantile(0.25)) if not df[col].isna().all() else None,
                        "q75": float(df[col].quantile(0.75)) if not df[col].isna().all() else None,
                    }

                columns_detail.append(col_detail)

            profile = {
                "output_type": "dataset_overview_report",
                "summary": f"数据集包含{len(df)}行，{len(df.columns)}列",
                "columns": columns_detail,
                "row_count": len(df),
                "duplicate_rows": int(df.duplicated().sum()),
                "overall_missing_rate": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])),
                "knowledge_sources": [],
            }

            # Write detailed profile
            profile_path = self.working_root / "dataset_detailed_profile.json"
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)

            logger.info(f"Detailed dataset profile generated: {profile_path}")
            return profile

        except Exception as e:
            logger.error(f"Error profiling dataset: {e}")
            raise DataInspectorSkillsError(f"Failed to profile dataset: {e}")
