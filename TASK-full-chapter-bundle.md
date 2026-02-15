# Task: full-chapter-bundle

## Context
The Knoweia sidecar (FastAPI, Python) runs the multi-agent tutoring loop. It was extracted from the `demo/` monolith. During extraction, chapter content was reduced from 5 markdown files to 3 (moving `interaction_protocol.md` and `socratic_vs_direct.md` to global fallbacks). The sidecar code actually already handles all 5 files — it loads them if present and falls back to globals if missing.

However, the **content directory** (`/content/curriculum/`) only has 3 files per chapter, and the **bundle structure** doesn't account for scripts and datasets that chapters may need.

The demo has the full 5-file chapters plus consultation configs. We need to:
1. Restore the `content/` directory to include all 5 files per chapter (consistent with demo)
2. Define and implement a complete chapter bundle structure that includes prompts (5 md files), scripts, and datasets
3. Update the sidecar's bundle resolution to handle scripts/datasets

Tech stack: Python 3.10+, FastAPI, setuptools.

## Objective
Restore full 5-file chapter content, define the canonical chapter bundle structure (prompts + scripts + datasets), and ensure the sidecar correctly resolves and serves all bundle components.

## Dependencies
- Depends on: none
- Branch: feature/full-chapter-bundle
- Base: main

## Scope

### Files to Modify

**In the sidecar repo (`llm-course-sidecar/`):**
- `src/sidecar/main.py` — Update `_find_chapter_dir_in_bundle()` to also resolve `scripts/` and `datasets/` subdirectories; update `_resolve_curriculum_dir()` overlay copy to include these directories
- `src/sidecar/services/orchestrator.py` — No code changes needed (already loads 5 files with fallback), but verify and add `consultation_config.yaml` loading if missing
- `docs/api_contract_v1.md` — Document the full chapter bundle structure

**In the content directory (NOT in sidecar repo, but referenced):**
- The `/content/curriculum/` chapters need `interaction_protocol.md` and `socratic_vs_direct.md` restored from demo

### Files to Create

**In the sidecar repo:**
- `scripts/build_chapter_bundle.py` — CLI tool to package a chapter directory into a bundle tar.gz with manifest
- `docs/chapter_bundle_spec.md` — Canonical documentation of the chapter bundle format
- `tests/test_bundle_resolution.py` — Tests for bundle path resolution with scripts/datasets

### Files NOT to Touch
- `src/sidecar/services/agent_runner.py` — Agent prompt assembly is fine
- `src/sidecar/services/consultation_engine.py` — Consultation loading works
- `src/sidecar/contracts/` — JSON Schema contracts are frozen

## Implementation Spec

### Step 1: Define the canonical chapter bundle structure
Document in `docs/chapter_bundle_spec.md`:

```
chapter_bundle.tar.gz
├── bundle.manifest.json
├── prompts/
│   ├── chapter_context.md          (required)
│   ├── task_list.md                (required)
│   ├── task_completion_principles.md (required)
│   ├── interaction_protocol.md     (recommended, falls back to global)
│   ├── socratic_vs_direct.md       (recommended, falls back to global)
│   ├── consultation_config.yaml    (optional)
│   ├── consultation_guide.md       (optional)
│   └── consultation_guide.json     (optional)
├── scripts/                        (optional)
│   ├── setup.py                    (chapter setup script)
│   ├── starter_code.py             (starter code for student)
│   └── solution.py                 (reference solution, not shown to student)
├── datasets/                       (optional)
│   ├── sample_data.csv
│   └── README.md                   (dataset description)
└── assets/                         (optional, future)
    └── images/
```

**bundle.manifest.json:**
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
    { "path": "prompts/chapter_context.md", "sha256": "...", "size_bytes": 1234 },
    { "path": "scripts/starter_code.py", "sha256": "...", "size_bytes": 567 }
  ]
}
```

### Step 2: Update bundle resolution in sidecar main.py

In `_find_chapter_dir_in_bundle()`:
- Currently checks for 3 required md files directly in the chapter directory
- Add support for the new `prompts/` subdirectory layout:
  ```python
  # Check new layout: prompts/ subdirectory
  prompts_dir = candidate / "prompts"
  if all((prompts_dir / name).exists() for name in required):
      return prompts_dir  # or return candidate with prompts_dir noted
  ```
- Return a richer result that includes paths to scripts/ and datasets/ if present

Update overlay copy logic to also copy `scripts/` and `datasets/` directories:
```python
# In addition to copying individual md files:
for subdir in ("scripts", "datasets", "assets"):
    src_subdir = chapter_dir_in_bundle / subdir
    if src_subdir.is_dir():
        shutil.copytree(src_subdir, overlay_dir / subdir, dirs_exist_ok=True)
```

### Step 3: Expose scripts/datasets paths to desktop
Add to the session creation response or a new endpoint:
- `GET /api/session/{session_id}/workspace` — Returns paths to scripts and datasets for the chapter
- Or include in the existing session creation response:
  ```json
  {
    "session_id": "...",
    "workspace": {
      "scripts_dir": "/path/to/scripts",
      "datasets_dir": "/path/to/datasets",
      "starter_code": "scripts/starter_code.py"
    }
  }
  ```

### Step 4: Create bundle packaging script
`scripts/build_chapter_bundle.py`:
- CLI: `python scripts/build_chapter_bundle.py --chapter-dir ./curriculum/courses/course1/chapters/ch0 --output ./dist/`
- Accepts a chapter directory (with prompts at root or in `prompts/` subdir)
- Optionally accepts `--scripts-dir` and `--datasets-dir`
- Generates `bundle.manifest.json` with SHA256 hashes for all files
- Creates `chapter_bundle.tar.gz`
- Prints manifest summary to stdout

### Step 5: Restore 5 files in content/ directory
This step involves the `content/` directory at the monorepo root (not in the sidecar git repo). Document this as a manual step or create a sync script:
- For each chapter in `content/curriculum/`, copy `interaction_protocol.md` and `socratic_vs_direct.md` from the corresponding demo chapter (if chapter-specific versions exist) or from the global `/content/agents/companion/` directory
- Also copy any `consultation_config.yaml`, `consultation_guide.md`, `consultation_guide.json` from demo chapters that have them

### Step 6: Write tests
`tests/test_bundle_resolution.py`:
- Test `_find_chapter_dir_in_bundle()` with old flat layout (md files at root)
- Test with new `prompts/` subdirectory layout
- Test with scripts/ and datasets/ present
- Test overlay copy includes scripts and datasets
- Test manifest generation in build script

## Testing Requirements
- Bundle with flat layout (backward compat) still resolves correctly
- Bundle with `prompts/` subdirectory layout resolves correctly
- Scripts and datasets are copied to overlay directory
- `build_chapter_bundle.py` produces valid tar.gz with correct manifest
- Sidecar starts and loads chapter content with both old and new bundle formats

## Acceptance Criteria
- [ ] Chapter bundle spec documented with prompts/, scripts/, datasets/ structure
- [ ] Sidecar resolves both old (flat) and new (prompts/ subdir) bundle layouts
- [ ] Scripts and datasets are copied to overlay during bundle extraction
- [ ] `build_chapter_bundle.py` packages chapters into valid bundles
- [ ] Content directory chapters have all 5 md files restored
- [ ] Backward compatibility: existing 3-file chapters still work via global fallback
- [ ] Tests pass for all bundle resolution scenarios

## Notes
- The sidecar orchestrator already handles 5 files with fallback — the code change is minimal. The main work is in bundle resolution and the packaging script.
- The `content/` directory is at the monorepo root and is NOT part of the sidecar git repo. The "restore 5 files" step should be documented as instructions or a script that operates on the content directory.
- For datasets, consider size limits. Large datasets (>10MB) should be downloaded separately, not bundled. Include a `datasets/manifest.json` that can reference external URLs for large files.
- The `consultation_config.yaml` is already supported by the sidecar's consultation engine. No code changes needed for that — just ensure it's included in the bundle spec.
- Keep backward compatibility with the desktop's existing `bundle_format_v1.md` expectations. The new format is v2 but v1 bundles should still work.
