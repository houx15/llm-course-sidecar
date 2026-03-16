# Skill: inspect_dataset

## Purpose
快速检查数据集的基本信息，包括结构、缺失值、数据类型和前几行样本。

## Preconditions
1. 数据文件存在于`sessions/{session_id}/working_files/`目录
2. 文件格式为CSV、Excel (.xlsx, .xls)或JSON
3. 文件大小合理（< 100MB，避免内存问题）

## Input Parameters
```json
{
  "file_path": "uploaded_data.csv",  // 相对于working_files的路径
  "sample_rows": 50,                  // 读取前N行，默认50
  "encoding": "utf-8"                 // 文件编码，默认utf-8
}
```

## Execution Steps

### Step 1: Validate file existence and format
```python
import os
import pandas as pd
from pathlib import Path

# Construct full path (sandbox enforced)
working_files_root = Path(f"sessions/{session_id}/working_files")
file_full_path = working_files_root / file_path

# Check existence
if not file_full_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

# Check format
file_ext = file_full_path.suffix.lower()
if file_ext not in ['.csv', '.xlsx', '.xls', '.json']:
    raise ValueError(f"Unsupported format: {file_ext}")
```

### Step 2: Read dataset (sample)
```python
try:
    if file_ext == '.csv':
        df = pd.read_csv(file_full_path, nrows=sample_rows, encoding=encoding)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_full_path, nrows=sample_rows)
    elif file_ext == '.json':
        df = pd.read_json(file_full_path, lines=True, nrows=sample_rows)
except Exception as e:
    raise RuntimeError(f"Failed to read file: {str(e)}")
```

### Step 3: Analyze basic structure
```python
# Column information
columns_info = []
for col in df.columns:
    col_info = {
        "name": col,
        "dtype": str(df[col].dtype),
        "missing_count": int(df[col].isna().sum()),
        "missing_rate": float(df[col].isna().sum() / len(df)),
        "unique_count": int(df[col].nunique()),
        "sample_values": df[col].dropna().head(3).tolist()
    }
    columns_info.append(col_info)

# Overall statistics
row_count = len(df)
duplicate_rows = int(df.duplicated().sum())
total_cells = df.size
missing_cells = int(df.isna().sum().sum())
overall_missing_rate = float(missing_cells / total_cells) if total_cells > 0 else 0.0
```

### Step 4: Generate profile report
```python
profile = {
    "file_name": file_path,
    "file_size_bytes": file_full_path.stat().st_size,
    "row_count_sampled": row_count,
    "column_count": len(df.columns),
    "columns": columns_info,
    "duplicate_rows": duplicate_rows,
    "overall_missing_rate": overall_missing_rate,
    "sample_data": df.head(10).to_dict(orient='records')
}
```

### Step 5: Write output to working_files
```python
import json

output_path = working_files_root / "dataset_profile.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(profile, f, ensure_ascii=False, indent=2)
```

### Step 6: Log skill invocation
```python
log_dir = working_files_root / "expert_logs" / "data_inspector" / consultation_id
log_dir.mkdir(parents=True, exist_ok=True)

skill_log = {
    "timestamp": datetime.now().isoformat(),
    "expert_id": "data_inspector",
    "consultation_id": consultation_id,
    "user_turn_index": user_turn_index,
    "round_index": round_index,
    "skill_name": "inspect_dataset",
    "cwd": str(working_files_root),
    "inputs": {
        "file_path": file_path,
        "sample_rows": sample_rows,
        "encoding": encoding
    },
    "outputs": {
        "files_written": ["dataset_profile.json"],
        "profile_summary": {
            "row_count": row_count,
            "column_count": len(df.columns),
            "missing_rate": overall_missing_rate
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
  "success": true,
  "output_file": "dataset_profile.json",
  "summary": {
    "row_count": 1500,
    "column_count": 10,
    "missing_rate": 0.08,
    "duplicate_rows": 15
  }
}
```

## Error Handling
- **FileNotFoundError**: 返回错误信息，建议检查文件路径
- **ValueError** (unsupported format): 返回支持的格式列表
- **RuntimeError** (read failure): 返回详细错误信息，可能是编码问题或文件损坏

## Execution Constraints
- **Sandbox root**: 只能访问`sessions/{session_id}/working_files/`
- **Read-only source**: 不修改原始数据文件
- **Write permissions**: 可以在working_files下创建输出文件
- **No network**: 不访问外部资源
- **Memory limit**: 只读取sample_rows行，避免大文件内存溢出

## Example Usage in Consultation

**RMA Request**:
```json
{
  "consulting_letter_title": "检查数据集基本信息",
  "question": "请检查用户上传的student_scores.csv的基本结构和数据质量",
  "session_scope": {
    "allowed_root": "sessions/sess_20260123_001/working_files/"
  }
}
```

**Expert Action**:
1. 调用`inspect_dataset`技能，参数：`{"file_path": "student_scores.csv", "sample_rows": 50}`
2. 生成`dataset_profile.json`
3. 分析profile，识别问题（如高缺失率列、数据类型异常）
4. 返回structured output给RMA

**Expert Response**:
```json
{
  "output_type": "dataset_overview_report",
  "summary": "数据集包含1500行（采样50行），10列，整体缺失率8%，发现15个重复行",
  "columns": [...],
  "row_count": 1500,
  "duplicate_rows": 15,
  "overall_missing_rate": 0.08,
  "key_findings": [
    "'age'列缺失率30%，可能影响分析",
    "'score'列包含非数值字符串'N/A'，需要清洗"
  ],
  "knowledge_sources": []
}
```
