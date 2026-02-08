# Sidecar Implementation Plan

## Goal

Build a production sidecar that can be bundled with a platform-specific Python runtime and used by desktop as a stable local execution engine.

## Requirements (Frozen)

1. Multi-agent collaboration:
- CA/RMA/MA orchestration
- Prompt loading from chapter + global agent resources
- Streaming events to desktop

2. Experts running code:
- Task-oriented code execution for experts
- Isolated execution with resource limits

3. User running code / notebooks:
- User code execution in per-session workspace
- Notebook kernel lifecycle support

## Product Decision

Use split artifacts:

- `python_runtime` bundle (platform-specific, heavy, infrequent update)
- content bundles (`app_agents`, `experts_shared`, `experts`, `chapter`) (frequent update)

`python_runtime` target matrix:
- `macos-x64`
- `macos-arm64`
- `windows-x64`

## API and Contract Baseline (v1)

Desktop calls local sidecar under `/api/session/*`.

Must support:
- `POST /api/session/new`
- `POST /api/session/{session_id}/message/stream` (SSE)
- `GET /api/session/{session_id}/dynamic_report`
- `POST /api/session/{session_id}/end`
- `GET /health`

SSE event baseline:
- `start`
- `companion_chunk`
- `companion_complete`
- `consultation_start`
- `consultation_complete`
- `consultation_error`
- `complete`
- `error`

Desktop normalization remains:
- `complete -> done`
- consultation events -> `expert_consultation`

## Demo API Parity Audit (2026-02-08)

Current `demo/app/server/main.py` exposes more than the minimal session APIs.

Endpoints present in demo:
- Session core:
  - `POST /api/session/new`
  - `POST /api/session/{session_id}/message/stream`
  - `GET /api/session/{session_id}/dynamic_report`
  - `GET /api/session/{session_id}/state`
  - `POST /api/session/{session_id}/end`
  - `GET /api/sessions`
  - `GET /api/session/{session_id}/history`
- Session files:
  - `POST /api/session/{session_id}/upload`
  - `GET /api/session/{session_id}/files`
  - `DELETE /api/session/{session_id}/files/{filename}`
- Curriculum browsing:
  - `GET /api/courses`
  - `GET /api/courses/{course_id}/chapters`
  - `GET /api/chapters`

Plan adjustment:
- Keep strict compatibility for all session endpoints and SSE event types.
- Keep file upload/list/delete endpoints in sidecar v1 (needed by expert/data workflows).
- Curriculum browsing endpoints can be optional compatibility endpoints because desktop course/chapter metadata comes from backend.

## Critical Compatibility Adjustments

1. Create session payload extension:
- Demo currently only defines `chapter_id`.
- Desktop already sends `desktop_context` (bundle paths + prompt candidates).
- Sidecar v1 must support:
  - `chapter_id` (required)
  - `desktop_context` (optional, backward-compatible)

2. Bundle path resolution order:
- To support desktop-downloaded chapter bundles, sidecar must resolve content in this order:
  1. `desktop_context.bundle_paths.chapter_bundle_path` and related bundle paths
  2. environment overrides (`CURRICULUM_DIR`, `EXPERTS_DIR`, `MAIN_AGENTS_DIR`)
  3. demo-style local repo paths (dev fallback only)

3. Prompt source alignment:
- Keep demo behavior: chapter-local files first.
- Keep current split support: fallback to global main-agent prompts for:
  - `interaction_protocol.md`
  - `socratic_vs_direct.md`

4. Stream error event consistency:
- Demo can emit a non-typed error line on early validation failures.
- Sidecar v1 should always emit structured error events:
  - `{"type":"error","message":"..."}`
- Maintain existing event names otherwise for desktop compatibility.

## Runtime Isolation Model

Two execution lanes:

- Expert lane:
  - stricter permissions
  - deterministic temp workspace
  - explicit tool whitelist

- User lane:
  - session workspace persistence
  - notebook kernel state
  - bounded CPU/memory/time

Shared guardrails:
- process timeout
- memory limit
- max output size
- cancellation support
- structured execution logs

## Suggested Repo Structure

```text
src/sidecar/
  api/
  orchestration/
  experts/
  execution/
    expert_runner/
    user_runner/
    notebook/
  contracts/
  config/
tests/
docs/
scripts/
```

## Bundle Format (`python_runtime`)

Each release should include:
- embedded Python runtime
- sidecar code + pinned deps
- startup entrypoint
- manifest with compatibility fields

Manifest minimum:
- `format_version`
- `platform`
- `python_version`
- `sidecar_version`
- `desktop_min_version`
- `entrypoint` (relative)
- `python_executable_relpath`
- `sha256`

## Phased Execution

## Phase 1 Progress (current)

- [x] Sidecar repo scaffold (`pyproject.toml`, `src/sidecar` package, docs baseline)
- [x] Demo server core copied into `src/sidecar` for strict behavior alignment
- [x] Session API parity endpoints available (including file APIs and `dynamic_report`)
- [x] `create_session` supports optional `desktop_context`
- [x] Initial bundle-path runtime resolver implemented (chapter bundle -> curriculum overlay)
- [x] Added API parity test scaffold (`tests/test_api_parity.py`) for session endpoints and streaming event shape
- [x] Removed key demo-local assumptions in core services (yellow-page path resolution, sidecar prompt dir defaults, session path resolution)
- [ ] Replace remaining demo-local path assumptions (`.metadata`, legacy fallbacks) with bundle-aware resolvers
- [~] Expand tests to cover real bundle fixture loading (chapter files + global prompt fallback)
  - done: chapter bundle overlay fixture test
  - pending: global prompt fallback fixture coverage
- [x] Contract freeze endpoint implemented: `GET /api/contract` (routes + SSE event types)

## Phase 1 - Contract Freeze and Extraction

Deliverables:
- copy and refactor core logic from `demo/app/server/*` into this repo
- frozen API + event contract doc
- request/response schema definitions
- adapter for `desktop_context` and bundle-path based content loading

Exit criteria:
- desktop can run one end-to-end message turn using new sidecar repo in dev mode
- desktop chapter bundle content is actually used (not only demo local paths)
- sidecar serves a machine-readable contract (`/api/contract`) consumed by desktop integration tests

## Phase 2 - Execution Runtime Hardening

Deliverables:
- expert/user execution lanes implemented
- timeout/resource limits + cancellation
- stable error taxonomy
- initial user code execution endpoint scaffolded (`/api/session/{session_id}/code/run`)
- async code-job APIs scaffolded:
  - `POST /api/session/{session_id}/code/jobs`
  - `GET /api/session/{session_id}/code/jobs/{job_id}`
  - `POST /api/session/{session_id}/code/jobs/{job_id}/cancel`
- user-code memory guardrail added (`memory_limit_mb`)

Exit criteria:
- deterministic failure handling for code tasks
- no crash on bad user code or expert tool failure

## Phase 3 - Notebook Support

Deliverables:
- notebook execution service (kernel lifecycle)
- per-session workspace binding
- restart/reset kernel APIs
- initial in-process stateful notebook scaffolded with:
  - `POST /api/session/{session_id}/notebook/cell/run`
  - `POST /api/session/{session_id}/notebook/reset`

Exit criteria:
- run notebook cells in sequence with state continuity

## Phase 4 - Platform Bundle Build Pipeline

Deliverables:
- CI build jobs: `macos-x64`, `macos-arm64`, `windows-x64`
- signed artifacts + checksums + manifests
- OSS/CDN publish script

Exit criteria:
- backend registry can serve platform-matched `python_runtime` bundle metadata

## Phase 5 - Stability and Release Gates

Deliverables:
- integration tests:
  - chat turn
  - expert code execution
  - user code execution
  - notebook cell execution
- failure tests:
  - sidecar process crash + restart
  - bundle corruption detection
  - kernel crash recovery

Exit criteria:
- all P0 tests pass on all three target platforms

## Immediate Next Actions

1. Add fixture-driven tests for global prompt fallback (`interaction_protocol.md`, `socratic_vs_direct.md` from main agents).
2. Introduce execution error taxonomy constants shared by sync and async code APIs.
3. Add bundle manifest schema + validation command for `python_runtime` artifacts.
4. Implement desktop-side health and contract preflight checks before opening a session.
5. Build first CI draft for `macos-x64`, `macos-arm64`, `windows-x64` sidecar runtime bundles.
