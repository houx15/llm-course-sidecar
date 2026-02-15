# Chapter Bundle Spec (v2)

This document defines the canonical chapter bundle format for sidecar runtime content.

## Goals

- Keep backward compatibility with legacy flat bundles (v1).
- Support richer chapter resources beyond prompt markdown files.
- Make bundle contents verifiable by manifest hashes.

## Bundle Layout

```text
chapter_bundle.tar.gz
├── bundle.manifest.json
├── prompts/
│   ├── chapter_context.md
│   ├── task_list.md
│   ├── task_completion_principles.md
│   ├── interaction_protocol.md
│   ├── socratic_vs_direct.md
│   ├── consultation_config.yaml
│   ├── consultation_guide.md
│   └── consultation_guide.json
├── scripts/
│   ├── setup.py
│   ├── starter_code.py
│   └── solution.py
├── datasets/
│   ├── sample_data.csv
│   └── README.md
└── assets/
    └── images/
```

## Required vs Optional Files

Required prompt files:
- `prompts/chapter_context.md`
- `prompts/task_list.md`
- `prompts/task_completion_principles.md`

Recommended prompt files (global fallback allowed):
- `prompts/interaction_protocol.md`
- `prompts/socratic_vs_direct.md`

Optional prompt files:
- `prompts/consultation_config.yaml`
- `prompts/consultation_guide.md`
- `prompts/consultation_guide.json`

Optional resource directories:
- `scripts/`
- `datasets/`
- `assets/`

## Manifest

`bundle.manifest.json` should follow:

```json
{
  "format_version": "bundle-v2",
  "bundle_type": "chapter",
  "scope_id": "course1_python_pandas_basics/ch0_pandas_basics",
  "version": "1.0.0",
  "created_at": "2026-02-15T00:00:00Z",
  "chapter": {
    "course_id": "course1_python_pandas_basics",
    "chapter_code": "ch0_pandas_basics",
    "title": "Pandas Basics",
    "has_scripts": true,
    "has_datasets": true,
    "required_experts": ["data_inspector"]
  },
  "files": [
    {
      "path": "prompts/chapter_context.md",
      "sha256": "hex",
      "size_bytes": 1234
    }
  ]
}
```

## Sidecar Resolution Rules

Sidecar checks these layouts in chapter bundles:

1. v2 structured: required prompt files under `prompts/`
2. v1 flat: required prompt files directly under chapter directory root

When resolved, sidecar overlays chapter content into session runtime and copies:

- prompt files into the chapter root used by orchestrator
- `scripts/`, `datasets/`, and `assets/` directories as-is when present

## Backward Compatibility

- Existing v1 bundles remain valid.
- v2 bundles are preferred when `prompts/` exists and has required files.
- Missing recommended files (`interaction_protocol.md`, `socratic_vs_direct.md`) fall back to global prompt files.

## Dataset Size Guidance

- Prefer bundling small teaching datasets.
- For large datasets (>10MB), include a `datasets/manifest.json` with remote URLs/checksums and fetch out-of-band.

## Content Restore Procedure (Monorepo `content/` Tree)

The sidecar repo does not contain the monorepo `content/curriculum` source of truth.  
To satisfy the restore requirement for 5-file chapter prompts, run:

```bash
python scripts/restore_content_chapter_files.py \
  --content-curriculum /path/to/monorepo/content/curriculum \
  --demo-curriculum ./demo/curriculum \
  --companion-prompts-dir /path/to/monorepo/content/agents/companion
```

What this script does:

- For each `content/curriculum/courses/<course_id>/chapters/<chapter_code>`:
- Copies `interaction_protocol.md` and `socratic_vs_direct.md` from matching demo chapter when present.
- Falls back to `content/agents/companion/` for the two prompt files when chapter-specific demo files are missing.
- Copies consultation assets from demo chapter when present:
  - `consultation_config.yaml`
  - `consultation_guide.md`
  - `consultation_guide.json`

Use `--dry-run` first to inspect planned changes.
