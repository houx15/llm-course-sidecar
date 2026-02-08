# Consultation Request Tool Schema

## 工具名称：consult_expert

### 参数格式

```json
{
  "expert_id": "string - 专家ID（从可用专家列表中选择）",
  "question": "string - 向专家提出的具体问题（中文）",
  "context": {
    "file_name": "string - 文件名（如适用）",
    "task_id": "string - 当前任务ID",
    "task_requirements": "string - 任务要求说明",
    "student_confusion": "string - 学生的困惑点（如适用）",
    "error_message": "string - 错误信息（如适用）",
    "student_code_snippet": "string - 学生代码片段（如适用）",
    "learning_objectives": "string - 学习目标（如适用）",
    "student_evidence": "string - 学生提供的证据（如适用）"
  },
  "expected_output_type": "string - 期望的输出类型",
  "scenario_id": "string - 场景标识符",
  "reasoning": "string - 你为什么需要咨询这个专家的详细理由"
}
```

### 返回值格式

```json
{
  "consultation_id": "string - 咨询ID（如：consult_0001）",
  "expert_id": "string - 被咨询的专家ID",
  "expert_output": {
    "// 专家的输出内容，根据expected_output_type而定": "..."
  },
  "binding_rules_triggered": [
    {
      "rule_name": "string - 触发的规则名称",
      "action": "string - 规则动作"
    }
  ],
  "instruction_updates": {
    "// 如果触发了binding rule，这里包含必须应用的指令更新": "..."
  }
}
```

### 输出类型说明

**suitability_judgment** - 数据集适用性判断
```json
{
  "is_suitable": "boolean - 是否适用",
  "blocking_issues": ["string - 阻塞性问题列表"],
  "warning_issues": ["string - 警告性问题列表"],
  "evidence": ["string - 证据列表"],
  "recommended_next_step": "string - 建议的下一步"
}
```

**concept_explanation** - 概念解释
```json
{
  "concept_name": "string - 概念名称",
  "definition": "string - 定义",
  "use_cases": ["string - 使用场景列表"],
  "simple_example": "string - 简单示例（非任务答案）",
  "common_pitfalls": ["string - 常见陷阱列表"]
}
```

**error_diagnosis** - 错误诊断
```json
{
  "error_root_cause": "string - 错误根本原因",
  "explanation": "string - 解释",
  "suggested_fix_direction": "string - 建议的修复方向（不是完整答案）",
  "related_concepts": ["string - 相关概念列表"]
}
```

**validation_report** - 验证报告
```json
{
  "is_valid": "boolean - 是否有效",
  "validation_details": "string - 验证详情",
  "missing_elements": ["string - 缺失要素列表"],
  "recommended_action": "string - 建议的行动"
}
```

### 场景ID说明

- `dataset_suitability_check` - 数据集适用性检查
- `concept_clarification` - 概念澄清
- `error_diagnosis` - 错误诊断
- `progress_validation` - 进度验证

### 使用示例

#### 示例1: 检查数据集适用性

```json
{
  "expert_id": "data_inspector",
  "question": "请检查文件'student_data.csv'是否适合用于任务task_2_1。任务要求：数据集应包含至少100行数据，包含'name'、'age'、'city'列，用于练习pandas基础操作。",
  "context": {
    "file_name": "student_data.csv",
    "task_id": "task_2_1",
    "task_requirements": "数据集应包含至少100行数据，包含'name'、'age'、'city'列"
  },
  "expected_output_type": "suitability_judgment",
  "scenario_id": "dataset_suitability_check",
  "reasoning": "学生刚上传了新的数据文件，根据consultation guide的'Dataset Suitability Check'场景，应该验证数据集是否符合任务要求。这是一个核心任务，数据集质量直接影响学习效果。"
}
```

#### 示例2: 解释概念

```json
{
  "expert_id": "concept_explainer",
  "question": "学生在任务task_2_3中对'pandas.merge()'概念理解有困难。学生的困惑点：不清楚on参数和how参数的区别，以及什么时候用left join vs inner join。请提供清晰的解释，但不要给出任务答案。",
  "context": {
    "task_id": "task_2_3",
    "student_confusion": "不清楚on参数和how参数的区别",
    "concept_name": "pandas.merge()"
  },
  "expected_output_type": "concept_explanation",
  "scenario_id": "concept_clarification",
  "reasoning": "学生已经尝试了3次使用pandas.merge()但都失败了，表现出明显的困惑。根据consultation guide，当学生在核心概念上卡住≥2次尝试时，应该咨询concept_explainer进行概念澄清。"
}
```

#### 示例3: 诊断错误

```json
{
  "expert_id": "data_inspector",
  "question": "学生在任务task_2_2中遇到KeyError: 'age'错误。学生尝试的代码：df['age'].mean()。请帮助诊断问题根源并提供解决方向（不要给出完整答案）。",
  "context": {
    "task_id": "task_2_2",
    "error_message": "KeyError: 'age'",
    "student_code_snippet": "df['age'].mean()",
    "error_type": "KeyError"
  },
  "expected_output_type": "error_diagnosis",
  "scenario_id": "error_diagnosis",
  "reasoning": "学生遇到KeyError错误且已经尝试了3次仍未解决，表现出frustrated情绪。这个错误可能与数据列名有关，data_inspector可以检查数据集的实际列名并诊断问题。"
}
```

#### 示例4: 验证任务完成

```json
{
  "expert_id": "data_inspector",
  "question": "学生声称完成了任务task_2_4，提供的证据：'我成功使用groupby()计算了每个城市的平均年龄'。请验证该证据是否真正表明学生达到了学习目标：理解groupby()的使用方法并能正确应用。",
  "context": {
    "task_id": "task_2_4",
    "student_evidence": "我成功使用groupby()计算了每个城市的平均年龄",
    "learning_objectives": "理解groupby()的使用方法并能正确应用"
  },
  "expected_output_type": "validation_report",
  "scenario_id": "progress_validation",
  "reasoning": "学生声称完成了核心任务task_2_4，但只提供了口头描述而没有展示具体的代码或输出结果。根据consultation guide的'Progress Validation'场景，需要验证证据质量是否足以证明任务完成。"
}
```

### 重要提示

1. **Binding rules自动执行**：如果专家判断触发了binding rule（如数据集不合格），系统会自动应用`instruction_updates`。你必须在instruction_packet中反映这些更新。

2. **不要过度咨询**：只在真正需要时咨询。简单问题应该由CA引导学生自己解决。

3. **构造清晰的问题**：向专家提出的问题要具体、明确，包含足够的上下文。

4. **使用返回的建议**：专家的输出应该影响你的instruction_packet，特别是`guidance_for_ca`字段。

5. **reasoning字段很重要**：清晰解释为什么需要咨询，这有助于调试和改进系统。
