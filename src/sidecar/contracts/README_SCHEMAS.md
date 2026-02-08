# JSON Schema Files

This directory contains JSON Schema files generated from Pydantic models.

## Generated Schemas

### v3.0 Expert System Schemas

1. **consultation_guide_schema_v3.0.json**
   - Chapter-level consultation guide configuration
   - Defines available experts, scenarios, and policies

2. **consulting_envelope_schema_v3.0.json**
   - Envelope sent from RMA to expert for consultation
   - Contains question, context, and expected output template

3. **expert_report_schema_v3.0.json**
   - Student-visible expert report authored by RMA
   - Summarizes consultation results and implications

4. **skill_call_log_schema_v3.0.json**
   - Audit log for expert skill invocations
   - Tracks inputs, outputs, and execution details

5. **suitability_judgment_schema_v3.0.json**
   - Expert output for dataset/resource suitability judgment
   - Includes blocking issues, warnings, and evidence

6. **concept_explanation_schema_v3.0.json**
   - Expert output for concept explanation
   - Includes definition, examples, and common pitfalls

7. **dataset_overview_report_schema_v3.0.json**
   - Expert output for dataset overview
   - Includes column info, row count, and missing data stats

## Usage

These schemas can be used for:
- Validating JSON configuration files
- Generating documentation
- IDE autocomplete and validation
- External tool integration

## Regeneration

To regenerate these schemas after model changes:

```bash
python generate_json_schemas.py
```

## Version History

- v3.0 (2026-01-23): Initial generation from improved Pydantic models
  - Added strict ID format validation
  - Added structured sub-models
  - Added field constraints and descriptions
