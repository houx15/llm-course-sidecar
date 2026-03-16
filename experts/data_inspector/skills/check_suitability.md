# Skill: check_suitability

## Purpose
基于教学任务要求，判断数据集是否适用，并提供binding judgment。

## Preconditions
1. 已经执行过`inspect_dataset`或`profile_dataset`，存在`dataset_profile.json`
2. RMA提供了明确的教学任务要求（required_columns, expected_dtypes, minimum_rows等）
3. 数据文件存在于`sessions/{session_id}/working_files/`

## Input Parameters
```json
{
  "profile_file": "dataset_profile.json",  // 之前生成的profile文件
  "task_requirements": {
    "required_columns": ["student_id", "name", "score"],
    "expected_dtypes": {
      "student_id": "int64",
      "score": "float64"
    },
    "minimum_rows": 100,
    "maximum_missing_rate": 0.2,
    "allow_duplicates": false
  }
}
```

## Execution Steps

### Step 1: Load existing profile
```python
import json
from pathlib import Path

working_files_root = Path(f"sessions/{session_id}/working_files")
profile_path = working_files_root / profile_file

if not profile_path.exists():
    raise FileNotFoundError(f"Profile file not found: {profile_file}. Please run inspect_dataset first.")

with open(profile_path, 'r', encoding='utf-8') as f:
    profile = json.load(f)
```

### Step 2: Check required columns
```python
blocking_issues = []
warning_issues = []

# Extract actual columns
actual_columns = [col["name"] for col in profile["columns"]]
required_columns = task_requirements.get("required_columns", [])

# Check missing required columns
missing_columns = set(required_columns) - set(actual_columns)
if missing_columns:
    blocking_issues.append(f"缺少必需列: {', '.join(missing_columns)}")

# Check extra columns (warning only)
extra_columns = set(actual_columns) - set(required_columns)
if extra_columns:
    warning_issues.append(f"包含额外列: {', '.join(extra_columns)}（不影响任务）")
```

### Step 3: Check data types
```python
expected_dtypes = task_requirements.get("expected_dtypes", {})

for col_name, expected_dtype in expected_dtypes.items():
    if col_name not in actual_columns:
        continue  # Already flagged as missing

    # Find column info
    col_info = next((c for c in profile["columns"] if c["name"] == col_name), None)
    if col_info:
        actual_dtype = col_info["dtype"]

        # Type compatibility check (simplified)
        if not _is_dtype_compatible(actual_dtype, expected_dtype):
            blocking_issues.append(
                f"列'{col_name}'数据类型不匹配: 期望{expected_dtype}，实际{actual_dtype}"
            )

def _is_dtype_compatible(actual, expected):
    """Check if actual dtype is compatible with expected"""
    # Exact match
    if actual == expected:
        return True

    # Numeric compatibility
    numeric_types = ["int64", "int32", "float64", "float32"]
    if actual in numeric_types and expected in numeric_types:
        return True

    # String compatibility
    if actual == "object" and expected == "str":
        return True

    return False
```

### Step 4: Check row count
```python
minimum_rows = task_requirements.get("minimum_rows", 0)
actual_rows = profile.get("row_count_sampled", 0)

if actual_rows < minimum_rows:
    blocking_issues.append(
        f"数据行数不足: 需要至少{minimum_rows}行，实际{actual_rows}行"
    )
```

### Step 5: Check missing rate
```python
maximum_missing_rate = task_requirements.get("maximum_missing_rate", 1.0)
overall_missing_rate = profile.get("overall_missing_rate", 0.0)

if overall_missing_rate > maximum_missing_rate:
    blocking_issues.append(
        f"整体缺失率过高: {overall_missing_rate:.1%} > {maximum_missing_rate:.1%}"
    )

# Check individual columns
for col_info in profile["columns"]:
    if col_info["name"] in required_columns:
        col_missing_rate = col_info.get("missing_rate", 0.0)
        if col_missing_rate > maximum_missing_rate:
            warning_issues.append(
                f"列'{col_info['name']}'缺失率{col_missing_rate:.1%}超过阈值"
            )
```

### Step 6: Check duplicates
```python
allow_duplicates = task_requirements.get("allow_duplicates", True)
duplicate_rows = profile.get("duplicate_rows", 0)

if not allow_duplicates and duplicate_rows > 0:
    warning_issues.append(
        f"检测到{duplicate_rows}个重复行（任务要求无重复）"
    )
```

### Step 7: Generate binding judgment
```python
is_suitable = len(blocking_issues) == 0

# Determine recommended next step
if not is_suitable:
    if missing_columns:
        recommended_next_step = f"请上传包含必需列{required_columns}的数据集，或使用课程提供的示例数据"
    elif "数据类型不匹配" in str(blocking_issues):
        recommended_next_step = "请检查数据类型，必要时进行数据清洗和类型转换"
    elif "数据行数不足" in str(blocking_issues):
        recommended_next_step = f"请上传至少包含{minimum_rows}行的数据集"
    else:
        recommended_next_step = "请解决上述阻塞性问题后重新上传数据集"
else:
    if warning_issues:
        recommended_next_step = "数据集可用，但建议注意警告问题（如缺失值处理）"
    else:
        recommended_next_step = "数据集完全符合要求，可以开始任务"

judgment = {
    "output_type": "suitability_judgment",
    "is_suitable": is_suitable,
    "blocking_issues": blocking_issues,
    "warning_issues": warning_issues,
    "evidence": {
        "columns_found": actual_columns,
        "columns_required": required_columns,
        "row_count": actual_rows,
        "missing_rate_overall": overall_missing_rate,
        "duplicate_rows": duplicate_rows,
        "dtype_checks": [
            {
                "column": col_name,
                "expected": expected_dtype,
                "actual": next((c["dtype"] for c in profile["columns"] if c["name"] == col_name), "N/A")
            }
            for col_name, expected_dtype in expected_dtypes.items()
        ]
    },
    "recommended_next_step": recommended_next_step,
    "knowledge_sources": []
}
```

### Step 8: Write judgment to working_files
```python
judgment_path = working_files_root / "suitability_judgment.json"
with open(judgment_path, 'w', encoding='utf-8') as f:
    json.dump(judgment, f, ensure_ascii=False, indent=2)
```

### Step 9: Log skill invocation
```python
log_dir = working_files_root / "expert_logs" / "data_inspector" / consultation_id
log_dir.mkdir(parents=True, exist_ok=True)

skill_log = {
    "timestamp": datetime.now().isoformat(),
    "expert_id": "data_inspector",
    "consultation_id": consultation_id,
    "user_turn_index": user_turn_index,
    "round_index": round_index,
    "skill_name": "check_suitability",
    "cwd": str(working_files_root),
    "inputs": {
        "profile_file": profile_file,
        "task_requirements": task_requirements
    },
    "outputs": {
        "files_written": ["suitability_judgment.json"],
        "judgment_summary": {
            "is_suitable": is_suitable,
            "blocking_issues_count": len(blocking_issues),
            "warning_issues_count": len(warning_issues)
        }
    },
    "stdout_stderr_paths": {
        "stdout": None,
        "stderr": None
    }
}

skill_log_path = log_dir / f"skill_call_{skill_call_counter:04d}.json"
with open(skill_log_path, 'w', encoding='utf-8') as f:
    json.dump(skill_log, f, ensure_ascii=False, indent=2)
```

## Output Contract
```json
{
  "output_type": "suitability_judgment",
  "is_suitable": false,
  "blocking_issues": [
    "缺少必需列: student_id",
    "列'score'数据类型不匹配: 期望float64，实际object"
  ],
  "warning_issues": [
    "列'age'缺失率30%超过阈值"
  ],
  "evidence": {
    "columns_found": ["id", "name", "age", "score"],
    "columns_required": ["student_id", "name", "score"],
    "row_count": 1500,
    "missing_rate_overall": 0.08,
    "duplicate_rows": 15,
    "dtype_checks": [...]
  },
  "recommended_next_step": "请上传包含必需列['student_id', 'name', 'score']的数据集，或使用课程提供的示例数据",
  "knowledge_sources": []
}
```

## Binding Decision Rules

**CRITICAL**: This judgment is **binding** for RMA.

- If `is_suitable == false`, RMA **MUST**:
  1. Block the current task path
  2. Update instruction packet to prevent CA from advancing the task
  3. Generate an expert report explaining the blocking issues
  4. Provide alternative actions (request new file, use example dataset, etc.)

- If `is_suitable == true` but `warning_issues` exist:
  1. RMA may proceed with the task
  2. RMA should inform CA to mention potential data quality concerns
  3. Student can choose to proceed or address warnings

## Error Handling
- **FileNotFoundError** (profile missing): 返回错误，要求先运行inspect_dataset
- **KeyError** (malformed profile): 返回错误，要求重新生成profile
- **ValueError** (invalid requirements): 返回错误，说明requirements格式问题

## Execution Constraints
- **Sandbox root**: 只能访问`sessions/{session_id}/working_files/`
- **Read-only profile**: 不修改profile文件
- **Write judgment**: 生成suitability_judgment.json
- **No network**: 不访问外部资源

## Example Usage in Consultation

**RMA Request** (Round 1):
```json
{
  "consulting_letter_title": "判断数据集是否适用于ch2:task2.1",
  "question": "检查student_scores.csv是否满足任务要求",
  "expected_output_template": {
    "output_type": "suitability_judgment",
    "required_fields": ["is_suitable", "blocking_issues", "evidence", "recommended_next_step"],
    "binding": true
  },
  "context": {
    "task_requirements": {
      "required_columns": ["student_id", "name", "score"],
      "expected_dtypes": {"student_id": "int64", "score": "float64"},
      "minimum_rows": 100,
      "maximum_missing_rate": 0.2
    }
  }
}
```

**Expert Action**:
1. 加载`dataset_profile.json`（假设已存在）
2. 调用`check_suitability`技能
3. 生成binding judgment

**Expert Response**:
```json
{
  "output_type": "suitability_judgment",
  "is_suitable": false,
  "blocking_issues": ["缺少必需列: student_id"],
  "warning_issues": [],
  "evidence": {...},
  "recommended_next_step": "请上传包含必需列['student_id', 'name', 'score']的数据集",
  "knowledge_sources": []
}
```

**RMA Action** (after receiving binding judgment):
- Mark task path as blocked
- Update instruction packet: `"current_focus": "等待学生上传符合要求的数据集"`
- Generate expert report for student UI
- CA will inform student about the issue (using RMA's instruction packet)
