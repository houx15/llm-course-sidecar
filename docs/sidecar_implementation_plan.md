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
- `GET /api/session/{session_id}/dynamic-report`
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

## Phase 1 - Contract Freeze and Extraction

Deliverables:
- copy and refactor core logic from `demo/app/server/*` into this repo
- frozen API + event contract doc
- request/response schema definitions

Exit criteria:
- desktop can run one end-to-end message turn using new sidecar repo in dev mode

## Phase 2 - Execution Runtime Hardening

Deliverables:
- expert/user execution lanes implemented
- timeout/resource limits + cancellation
- stable error taxonomy

Exit criteria:
- deterministic failure handling for code tasks
- no crash on bad user code or expert tool failure

## Phase 3 - Notebook Support

Deliverables:
- notebook execution service (kernel lifecycle)
- per-session workspace binding
- restart/reset kernel APIs

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

1. Scaffold project (`pyproject.toml`, package layout, test harness).
2. Port minimal session API from `demo` for a first runnable build.
3. Freeze JSON schemas for session creation and stream events.
4. Implement execution lane abstraction before feature growth.
5. Add initial runtime manifest generator for `python_runtime` bundle.
