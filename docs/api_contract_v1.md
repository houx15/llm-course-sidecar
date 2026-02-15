# Sidecar API Contract v1

This document freezes the desktop-facing sidecar API and streaming event contract for integration stability.

Machine-readable source of truth:
- `GET /api/contract`

## Route Groups

Demo parity routes:
- `POST /api/session/new`
- `POST /api/session/{session_id}/message/stream`
- `GET /api/session/{session_id}/dynamic_report`
- `GET /api/session/{session_id}/state`
- `POST /api/session/{session_id}/end`
- `GET /api/sessions`
- `GET /api/session/{session_id}/history`
- `POST /api/session/{session_id}/upload`
- `GET /api/session/{session_id}/files`
- `DELETE /api/session/{session_id}/files/{filename}`
- `GET /api/courses`
- `GET /api/courses/{course_id}/chapters`
- `GET /api/chapters`
- `GET /health`

Sidecar extension routes:
- `GET /api/session/{session_id}/workspace`
- `POST /api/session/{session_id}/code/run`
- `POST /api/session/{session_id}/code/jobs`
- `GET /api/session/{session_id}/code/jobs/{job_id}`
- `POST /api/session/{session_id}/code/jobs/{job_id}/cancel`
- `POST /api/session/{session_id}/notebook/cell/run`
- `POST /api/session/{session_id}/notebook/reset`
- `GET /api/contract`

## SSE Event Types

`/api/session/{session_id}/message/stream` emits:
- `start`
- `companion_start`
- `companion_chunk`
- `companion_complete`
- `consultation_start`
- `consultation_complete`
- `consultation_error`
- `complete`
- `error`

Desktop-side normalization remains:
- `complete -> done`
- `consultation_* -> expert_consultation`

## Chapter Bundle Structure

Sidecar supports both chapter bundle layouts below:

1. v1 flat prompts (legacy): chapter markdown files at chapter root.
2. v2 structured prompts: markdown files under `prompts/` and optional `scripts/`, `datasets/`, `assets/`.

During `create_session`, sidecar copies resolved chapter files into a session overlay path and preserves optional resource directories:
- `scripts/`
- `datasets/`
- `assets/`

Desktop can query resolved runtime paths via:
- `GET /api/session/{session_id}/workspace`
