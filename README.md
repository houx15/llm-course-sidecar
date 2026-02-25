# llm-course-sidecar

Local Python sidecar runtime for the desktop learning platform.

## Scope

This repo owns the production sidecar runtime extracted from `demo/`:

- Multi-agent collaboration loop (CA/RMA/MA + experts)
- Expert code execution runtime
- User code / notebook execution runtime
- Sidecar API + streaming contract
- Platform-agnostic `python_runtime` bundle build and release (Python runtime provided by Miniconda on the desktop side)

## Non-Goals

- Desktop UI implementation
- Cloud backend business logic
- Course/chapter content authoring

## Source of Truth Docs

- `docs/sidecar_implementation_plan.md`
- `docs/api_contract_v1.md`

## Current Status

Phase 1 bootstrap is started:

- demo server core is vendored under `src/sidecar/`
- sidecar API parity with `demo/app/server/main.py`
- `create_session` now accepts optional `desktop_context`
- runtime path resolution supports desktop chapter bundle paths
- yellow-page lookup supports runtime env/bundle paths (not only demo local `.metadata`)
- core prompt/expert path defaults are now sidecar-repo aligned (no `app/server/*` dependency)
- initial user-code execution API added: `POST /api/session/{session_id}/code/run`
- async user-code job APIs added:
  - `POST /api/session/{session_id}/code/jobs`
  - `GET /api/session/{session_id}/code/jobs/{job_id}`
  - `POST /api/session/{session_id}/code/jobs/{job_id}/cancel`
- initial notebook APIs added:
  - `POST /api/session/{session_id}/notebook/cell/run`
  - `POST /api/session/{session_id}/notebook/reset`
- contract freeze endpoint added: `GET /api/contract`

## Run (Dev)

1. Create env and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
   - `pip install -e .`
2. Start sidecar:
   - `uvicorn sidecar.main:app --host 127.0.0.1 --port 8000 --reload`

You can copy `.env.example` to `.env` for local configuration.

Optional runtime path env vars:

- `SESSIONS_DIR`
- `CURRICULUM_DIR`
- `EXPERTS_DIR`
- `MAIN_AGENTS_DIR`
- `EXPERT_YELLOW_PAGE_PATH`
- `SIDECAR_SERVICES_DIR`

## Bundle Build

To build a platform-agnostic sidecar code bundle (for uploading to the backend):

```bash
python scripts/build_sidecar_code_bundle.py --version 0.2.0 --output /tmp/
```

This produces `/tmp/sidecar_code_<version>.tar.gz` containing the full sidecar source + `requirements.txt`. Upload to the backend as `bundle_type=python_runtime, scope_id=core`.

## Release

Releases are automated via GitHub Actions (`.github/workflows/build-and-upload-bundle.yml`). Pushing to `main` or `dev` triggers the pipeline automatically when source files change.

### Trigger conventions

| Trigger | Environment | Version format | Backend target |
|---|---|---|---|
| Tag `v*` | **prod** | Tag value (e.g. `0.1.0`) | Prod backend |
| Push to `dev` | **dev** | `0.1.0-dev.N` | Dev backend |

Dev builds trigger automatically when files in `src/`, `pyproject.toml`, or `scripts/build_sidecar_code_bundle.py` change on the `dev` branch. Prod builds only trigger on tags.

The sidecar itself is environment-agnostic — it receives `backend_url` and `auth_token` from the desktop at session creation time. The CI trigger only determines which backend the bundle gets uploaded to.

### How to release

```bash
# Dev release (push to dev branch — triggers automatically on source changes)
git checkout dev
git push

# Prod release (tag on main)
git checkout main
git merge dev
git tag v0.1.0
git push && git push --tags
```

### Required GitHub Secrets

| Secret | Purpose |
|---|---|
| `SIDECAR_BACKEND_URL` | Prod backend URL |
| `SIDECAR_ADMIN_KEY` | Prod admin API key |
| `SIDECAR_BACKEND_URL_DEV` | Dev backend URL |
| `SIDECAR_ADMIN_KEY_DEV` | Dev admin API key |

## Tests

1. Install dev dependencies:
   - `pip install -e .[dev]`
2. Run unit tests:
   - `pytest -q tests/`
3. Run E2E sidecar interaction test (requires a running sidecar + LLM key):
   - `RUN_INTEGRATION=1 pytest tests/test_e2e_sidecar_interaction.py -v -s`
   - Requires `/tmp/ch1_intro_bundle.tar.gz` (chapter bundle) and sidecar running at `http://127.0.0.1:8000`

## Relationship to Other Repos

- `llm-course-desktop`: downloads and launches sidecar runtime bundles
- `llm-course-backend`: provides bundle registry and OSS credential endpoints

