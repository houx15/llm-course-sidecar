# llm-course-sidecar

Local Python sidecar runtime for the desktop learning platform.

## Scope

This repo owns the production sidecar runtime extracted from `demo/`:

- Multi-agent collaboration loop (CA/RMA/MA + experts)
- Expert code execution runtime
- User code / notebook execution runtime
- Sidecar API + streaming contract
- Platform-specific `python_runtime` bundle build and release

## Non-Goals

- Desktop UI implementation
- Cloud backend business logic
- Course/chapter content authoring

## Source of Truth Docs

- `docs/sidecar_implementation_plan.md`

## Current Status

Phase 1 bootstrap is started:

- demo server core is vendored under `src/sidecar/`
- sidecar API parity with `demo/app/server/main.py`
- `create_session` now accepts optional `desktop_context`
- runtime path resolution supports desktop chapter bundle paths
- yellow-page lookup supports runtime env/bundle paths (not only demo local `.metadata`)

## Run (Dev)

1. Create env and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux) or `.venv\\Scripts\\activate` (Windows)
   - `pip install -e .`
2. Start sidecar:
   - `uvicorn sidecar.main:app --host 127.0.0.1 --port 8000 --reload`

You can copy `.env.example` to `.env` for local configuration.

Optional runtime path env vars:

- `SESSIONS_DIR`
- `CURRICULUM_DIR`
- `EXPERTS_DIR`
- `MAIN_AGENTS_DIR`

## Relationship to Other Repos

- `llm-course-desktop`: downloads and launches sidecar runtime bundles
- `llm-course-backend`: provides bundle registry and OSS credential endpoints
- `demo`: legacy source for logic extraction; not production deployment target
