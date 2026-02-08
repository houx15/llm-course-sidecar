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

## Relationship to Other Repos

- `llm-course-desktop`: downloads and launches sidecar runtime bundles
- `llm-course-backend`: provides bundle registry and OSS credential endpoints
- `demo`: legacy source for logic extraction; not production deployment target
