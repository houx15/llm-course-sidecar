# Task: code-editor-support

## Context
The Knoweia sidecar already has robust code execution APIs:
- `POST /api/session/{sid}/code/run` — Sync Python execution
- `POST /api/session/{sid}/code/jobs` — Async job-based execution
- `POST /api/session/{sid}/notebook/cell/run` — Notebook-style persistent namespace
- File upload/list/delete APIs

The desktop is building an integrated code editor (`desktop: feature/code-editor`). The desktop should call the sidecar's APIs rather than spawning Python directly, because the sidecar already has sandboxing, timeout management, and workspace integration with the AI agents.

However, the sidecar is missing three things the code editor needs:
1. **Bundle scripts/datasets exposure** — When a chapter bundle includes `scripts/` (starter code) and `datasets/` (sample data), these aren't copied to the session workspace or exposed via API.
2. **General workspace file read/write** — The current file APIs only handle uploaded files in `working_files/`. The code execution workspace (`user_workspace/`) has no read/write API. The editor needs to read/write `.py` files in the workspace.
3. **Code execution output forwarding to agents** — When a student runs code, the CA should know what they ran and what output they got, so it can provide relevant Socratic guidance.

Tech stack: Python 3.10+, FastAPI.

## Objective
Extend the sidecar to fully support the desktop code editor by: exposing bundle scripts/datasets, adding workspace file management APIs, and forwarding code execution context to the AI agents.

## Dependencies
- Depends on: `sidecar: feature/full-chapter-bundle` (must define bundle structure with scripts/datasets first)
- Branch: feature/code-editor-support
- Base: main

## Scope

### Files to Modify
- `src/sidecar/main.py` — Add workspace file management endpoints; copy bundle scripts/datasets to session on creation
- `src/sidecar/services/storage.py` — Add methods for user_workspace file read/write/list
- `src/sidecar/services/orchestrator.py` — Include recent code execution results in agent context (pass to CA)

### Files to Create
- `tests/test_workspace_api.py` — Tests for new workspace endpoints

### Files NOT to Touch
- `src/sidecar/services/user_code_runner.py` — Execution logic is fine
- `src/sidecar/services/code_execution_manager.py` — Job management is fine
- `src/sidecar/services/notebook_manager.py` — Notebook logic is fine
- `src/sidecar/contracts/` — Frozen schemas

## Implementation Spec

### Step 1: Copy bundle scripts/datasets to session workspace on creation
In `main.py`, within the `POST /api/session/new` handler (or `_resolve_curriculum_dir`):
- After resolving the chapter bundle, check for `scripts/` and `datasets/` directories
- Copy `scripts/` contents to `{session_dir}/user_workspace/` (so starter code is available)
- Copy `datasets/` contents to `{session_dir}/working_files/` (so data is accessible to both code execution and data_inspector expert)
- Record what was copied in session metadata (so the desktop can show the file list)

```python
# After chapter overlay is created:
scripts_src = chapter_dir_in_bundle / "scripts"
if scripts_src.is_dir():
    for f in scripts_src.iterdir():
        if f.is_file() and f.name != "solution.py":  # Don't copy solutions
            shutil.copy2(f, workspace_dir / f.name)

datasets_src = chapter_dir_in_bundle / "datasets"
if datasets_src.is_dir():
    for f in datasets_src.iterdir():
        if f.is_file():
            shutil.copy2(f, working_files_dir / f.name)
```

### Step 2: Add workspace file management endpoints
In `main.py`, add these endpoints:

**`GET /api/session/{session_id}/workspace/files`** — List all files in user_workspace
- Returns: `[{ "name": "starter_code.py", "size_bytes": 1234, "modified_at": "...", "source": "bundle"|"user" }]`
- Lists `.py`, `.ipynb`, `.txt`, `.md` files in `user_workspace/`

**`GET /api/session/{session_id}/workspace/files/{filename}`** — Read a workspace file
- Returns: `{ "name": "starter_code.py", "content": "...", "size_bytes": 1234 }`
- Path validation: filename must be safe, within user_workspace/

**`PUT /api/session/{session_id}/workspace/files/{filename}`** — Write/update a workspace file
- Body: `{ "content": "import pandas as pd\n..." }`
- Creates or overwrites the file in user_workspace/
- Path validation: only allow safe filenames, no directory traversal
- Max file size: 1MB

**`DELETE /api/session/{session_id}/workspace/files/{filename}`** — Delete a workspace file
- Returns: 204 No Content

### Step 3: Add storage methods for user_workspace
In `services/storage.py`, add:

```python
def list_workspace_files(self, session_id: str) -> list[dict]:
    """List files in user_workspace/ directory."""

def read_workspace_file(self, session_id: str, filename: str) -> str:
    """Read a file from user_workspace/. Raises FileNotFoundError."""

def write_workspace_file(self, session_id: str, filename: str, content: str) -> dict:
    """Write content to a file in user_workspace/. Returns file metadata."""

def delete_workspace_file(self, session_id: str, filename: str) -> None:
    """Delete a file from user_workspace/."""
```

All methods must validate that the resolved path is within the session's user_workspace directory (prevent directory traversal).

### Step 4: Forward code execution context to CA
In `services/orchestrator.py`, within `process_turn()`:
- After loading session state, also load recent code execution history
- Add to CA context: last N code executions (code snippet + output summary + exit code)
- This lets the CA know what the student has been running and guide them accordingly

In `services/storage.py`, add:
```python
def get_recent_code_executions(self, session_id: str, limit: int = 3) -> list[dict]:
    """Return the last N code execution records (code, stdout, stderr, exit_code, timestamp)."""
```

The code execution records should be saved by `user_code_runner.py` after each run. Add a small persistence step:
```python
# After execution completes, save to {session_dir}/code_history/run_{timestamp}.json
```

In `services/agent_runner.py`, add a new template variable:
```python
"RECENT_CODE_EXECUTIONS": formatted_code_history  # Markdown formatted
```

Format as:
```markdown
## 学生最近的代码执行

### 执行 1 (2 分钟前)
```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
```
**输出:** (成功)
```
   col1  col2
0   ...   ...
```

### 执行 2 (5 分钟前)
...
```

### Step 5: Include workspace info in session creation response
Extend the `POST /api/session/new` response to include workspace information:
```json
{
  "session_id": "...",
  "workspace": {
    "has_starter_code": true,
    "has_datasets": true,
    "files": [
      { "name": "starter_code.py", "source": "bundle" },
      { "name": "sample_data.csv", "source": "bundle" }
    ]
  }
}
```

This tells the desktop code editor what's available so it can auto-open the starter code.

## Testing Requirements
- Session creation with a bundle containing scripts/ → starter code appears in workspace
- Session creation with a bundle containing datasets/ → data files appear in working_files
- `solution.py` is NOT copied to workspace
- Workspace file CRUD APIs work (list, read, write, delete)
- Path traversal attacks are blocked (e.g., `../../etc/passwd`)
- Code execution history is saved and retrievable
- CA receives recent code execution context in its prompt
- Large file writes (>1MB) are rejected

## Acceptance Criteria
- [ ] Bundle scripts/ are auto-copied to user_workspace/ on session creation
- [ ] Bundle datasets/ are auto-copied to working_files/ on session creation
- [ ] Workspace file list/read/write/delete APIs work correctly
- [ ] Path validation prevents directory traversal
- [ ] Recent code executions are persisted and passed to CA as context
- [ ] Session creation response includes workspace metadata
- [ ] solution.py files are excluded from student workspace
- [ ] Tests pass

## Notes
- The desktop code-editor should call these sidecar APIs instead of spawning Python directly via Electron. This keeps execution in the sidecar's managed environment with proper sandboxing.
- The `user_workspace/` is separate from `working_files/` (uploaded data) and `expert_workspace/` (expert execution). This is intentional — user code runs in user_workspace with access to read files from working_files.
- Code execution history should be lightweight — store only the last 5-10 runs, with truncated output (max 2000 chars per run). Old runs can be pruned.
- The existing `/api/session/{sid}/code/run` endpoint already uses user_workspace as cwd. The new workspace file APIs complement this by letting the desktop manage files there.
- This feature depends on `feature/full-chapter-bundle` defining the scripts/ and datasets/ structure. If that's not merged yet, the bundle-copy logic can be added but will be a no-op until bundles include those directories.
