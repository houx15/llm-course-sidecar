"""FastAPI application for multi-agent tutor system."""

import logging
import asyncio
import os
import shutil
import hashlib
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Dict, Any

from .services.orchestrator import Orchestrator, OrchestratorError
from .services.storage import Storage, StorageError
from .services.user_code_runner import UserCodeRunner
from .services.code_execution_manager import CodeExecutionManager
from .services.notebook_manager import NotebookManager
from .config import settings

# Configure logging
_log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_fmt)

# Optionally mirror all logs to a file (set LOG_FILE env var from the desktop).
_log_file = os.getenv("LOG_FILE", "").strip()
if _log_file:
    try:
        Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(_log_file, encoding="utf-8")
        _fh.setFormatter(logging.Formatter(_log_fmt))
        _fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(_fh)
    except Exception as _log_err:
        print(f"[sidecar] WARNING: could not open log file {_log_file!r}: {_log_err}", flush=True)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Tutor System",
    description="A multi-agent teaching assistant system with Socratic questioning",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve project root to avoid cwd-dependent path issues
project_root = Path(__file__).resolve().parents[2]
sessions_root = Path(
    os.getenv("SESSIONS_DIR", str(project_root / "sessions"))
).resolve()
default_curriculum_dir = Path(
    os.getenv("CURRICULUM_DIR", str(project_root / "curriculum"))
).resolve()
default_experts_dir = Path(
    os.getenv("EXPERTS_DIR", str(project_root / "experts"))
).resolve()
default_main_agents_dir = Path(
    os.getenv("MAIN_AGENTS_DIR", str(project_root / "content" / "agents"))
).resolve()

overlay_root = sessions_root / "_chapter_overlays"
overlay_root.mkdir(parents=True, exist_ok=True)


CHAPTER_PROMPT_FILES = (
    "chapter_context.md",
    "task_list.md",
    "task_completion_principles.md",
    "interaction_protocol.md",
    "socratic_vs_direct.md",
    "consultation_config.yaml",
    "consultation_guide.md",
    "consultation_guide.json",
)


@dataclass(frozen=True)
class ChapterBundleLayout:
    chapter_root: Path
    prompts_dir: Path
    scripts_dir: Optional[Path]
    datasets_dir: Optional[Path]
    assets_dir: Optional[Path]


def _sanitize_segment(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(value or "")
    )


def _validate_chapter_segment(segment: str, field_name: str) -> str:
    value = str(segment or "").strip()
    if not value:
        raise ValueError(f"invalid chapter_id: {field_name} is empty")
    if value in {".", ".."}:
        raise ValueError(f"invalid chapter_id: {field_name} cannot be '.' or '..'")
    if "/" in value or "\\" in value:
        raise ValueError(
            f"invalid chapter_id: {field_name} cannot contain path separators"
        )
    if _sanitize_segment(value) != value:
        raise ValueError(
            f"invalid chapter_id: {field_name} contains unsupported characters"
        )
    return value


def _split_chapter_id(chapter_id: str) -> tuple[str, str]:
    chapter = str(chapter_id or "").strip()
    if "/" in chapter:
        raw_course_id, raw_chapter_name = chapter.split("/", 1)
        course_id = _validate_chapter_segment(raw_course_id, "course_id")
        chapter_name = _validate_chapter_segment(raw_chapter_name, "chapter_name")
        return course_id, chapter_name
    chapter_name = _validate_chapter_segment(chapter, "chapter_name")
    return "", chapter_name


def _exists(path: Optional[Path]) -> bool:
    return bool(path and path.exists())


def _build_orchestrator(
    curriculum_dir: Path, experts_dir: Path, main_agents_dir: Optional[Path]
) -> Orchestrator:
    return Orchestrator(
        storage=Storage(base_dir=str(sessions_root)),
        curriculum_dir=str(curriculum_dir),
        experts_dir=str(experts_dir),
        main_agents_dir=str(main_agents_dir) if main_agents_dir else None,
    )


def _resolve_main_agents_dir(desktop_context: Optional[Dict[str, Any]]) -> Path:
    bundle_paths = (desktop_context or {}).get("bundle_paths") or {}
    app_agents_path = str(bundle_paths.get("app_agents_path") or "").strip()
    candidates = []
    if app_agents_path:
        root = Path(app_agents_path)
        candidates.extend([root / "content" / "agents", root / "agents", root])
    if _exists(default_main_agents_dir):
        candidates.append(default_main_agents_dir)
    for candidate in candidates:
        if _exists(candidate):
            return candidate
    return default_main_agents_dir


def _resolve_experts_dir(desktop_context: Optional[Dict[str, Any]]) -> Path:
    bundle_paths = (desktop_context or {}).get("bundle_paths") or {}
    expert_bundle_paths = bundle_paths.get("expert_bundle_paths") or {}
    candidates = []
    shared_path = str(bundle_paths.get("experts_shared_path") or "").strip()
    if shared_path:
        shared_root = Path(shared_path)
        candidates.extend([shared_root / "experts", shared_root])
    for value in expert_bundle_paths.values():
        root = Path(str(value))
        candidates.extend([root / "experts", root])
    if _exists(default_experts_dir):
        candidates.append(default_experts_dir)
    for candidate in candidates:
        if _exists(candidate):
            return candidate
    return default_experts_dir


def _resolve_chapter_bundle_layout(
    bundle_root: Path,
    course_id: str,
    chapter_name: str,
) -> Optional[ChapterBundleLayout]:
    candidates = []
    if course_id:
        candidates.extend(
            [
                bundle_root / "courses" / course_id / "chapters" / chapter_name,
                bundle_root
                / "content"
                / "curriculum"
                / "courses"
                / course_id
                / "chapters"
                / chapter_name,
                bundle_root / "content" / "curriculum" / course_id / chapter_name,
            ]
        )
    candidates.extend(
        [
            bundle_root / chapter_name,
            bundle_root / "content" / chapter_name,
            bundle_root / "content" / "curriculum" / chapter_name,
            bundle_root,
        ]
    )
    required = ("chapter_context.md", "task_list.md", "task_completion_principles.md")
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        prompts_dir = candidate / "prompts"
        if prompts_dir.is_dir() and all(
            (prompts_dir / name).exists() for name in required
        ):
            return ChapterBundleLayout(
                chapter_root=candidate,
                prompts_dir=prompts_dir,
                scripts_dir=(
                    (candidate / "scripts")
                    if (candidate / "scripts").is_dir()
                    else None
                ),
                datasets_dir=(
                    (candidate / "datasets")
                    if (candidate / "datasets").is_dir()
                    else None
                ),
                assets_dir=(
                    (candidate / "assets") if (candidate / "assets").is_dir() else None
                ),
            )
        if all((candidate / name).exists() for name in required):
            return ChapterBundleLayout(
                chapter_root=candidate,
                prompts_dir=candidate,
                scripts_dir=(
                    (candidate / "scripts")
                    if (candidate / "scripts").is_dir()
                    else None
                ),
                datasets_dir=(
                    (candidate / "datasets")
                    if (candidate / "datasets").is_dir()
                    else None
                ),
                assets_dir=(
                    (candidate / "assets") if (candidate / "assets").is_dir() else None
                ),
            )
    return None


def _find_chapter_dir_in_bundle(
    bundle_root: Path, course_id: str, chapter_name: str
) -> Optional[Path]:
    """
    Resolve prompt directory inside a chapter bundle.

    This keeps backward-compatible behavior for callers expecting a prompt-file directory path.
    """
    layout = _resolve_chapter_bundle_layout(bundle_root, course_id, chapter_name)
    return layout.prompts_dir if layout else None


def _ensure_overlay_templates(curriculum_root: Path) -> None:
    fallback_templates = []
    template_root_env = os.getenv("CURRICULUM_TEMPLATES_DIR", "").strip()
    if template_root_env and _exists(Path(template_root_env)):
        fallback_templates.append(Path(template_root_env))
    if _exists(default_curriculum_dir / "_templates"):
        fallback_templates.append(default_curriculum_dir / "_templates")
    if _exists(default_curriculum_dir / "templates"):
        fallback_templates.append(default_curriculum_dir / "templates")
    # Built-in fallback: templates shipped with the sidecar package itself
    _builtin = Path(__file__).parent / "default_templates"
    if _exists(_builtin):
        fallback_templates.append(_builtin)

    for source in fallback_templates:
        target = curriculum_root / source.name
        if target.exists():
            return
        shutil.copytree(source, target, dirs_exist_ok=True)
        return


def _compute_overlay_fingerprint(
    bundle_root: Path, bundle_layout: ChapterBundleLayout
) -> str:
    """
    Build a stable fingerprint for overlay isolation.

    Uses file metadata (relative path + size + mtime_ns) for prompt/resource files so
    overlay paths rotate when bundle content changes, without reading large file contents.
    """
    digest = hashlib.sha256()
    digest.update(str(bundle_root.resolve()).encode("utf-8"))

    manifest = bundle_root / "bundle.manifest.json"
    if manifest.is_file():
        stat = manifest.stat()
        digest.update(b"manifest")
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    for name in CHAPTER_PROMPT_FILES:
        path = bundle_layout.prompts_dir / name
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(f"prompt:{name}".encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    for subdir_name, source in (
        ("scripts", bundle_layout.scripts_dir),
        ("datasets", bundle_layout.datasets_dir),
        ("assets", bundle_layout.assets_dir),
    ):
        if not source or not source.is_dir():
            continue
        for file_path in sorted(p for p in source.rglob("*") if p.is_file()):
            rel = file_path.relative_to(source).as_posix()
            stat = file_path.stat()
            digest.update(f"{subdir_name}:{rel}".encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()[:12]


def _resolve_curriculum_dir(
    chapter_id: str, desktop_context: Optional[Dict[str, Any]]
) -> Path:
    bundle_paths = (desktop_context or {}).get("bundle_paths") or {}
    chapter_bundle_path = str(bundle_paths.get("chapter_bundle_path") or "").strip()

    if not chapter_bundle_path:
        return default_curriculum_dir

    bundle_root = Path(chapter_bundle_path)
    course_id, chapter_name = _split_chapter_id(chapter_id)
    bundle_layout = _resolve_chapter_bundle_layout(bundle_root, course_id, chapter_name)
    if not bundle_layout:
        logger.warning(
            f"desktop_context chapter bundle path is set but no chapter files found: {chapter_bundle_path}"
        )
        return default_curriculum_dir

    fingerprint = _compute_overlay_fingerprint(bundle_root, bundle_layout)
    overlay_id = "__".join(
        filter(
            None,
            [
                _sanitize_segment(course_id) or "legacy",
                _sanitize_segment(chapter_name),
                fingerprint,
            ],
        )
    )
    overlay_curriculum_root = (overlay_root / overlay_id).resolve()
    if course_id:
        chapter_target = (
            overlay_curriculum_root / "courses" / course_id / "chapters" / chapter_name
        ).resolve()
    else:
        chapter_target = (overlay_curriculum_root / "chapters" / chapter_name).resolve()
    if not chapter_target.is_relative_to(overlay_curriculum_root):
        raise ValueError(
            "invalid chapter_id: resolved chapter path escapes overlay root"
        )
    if chapter_target.exists():
        shutil.rmtree(chapter_target)
    chapter_target.mkdir(parents=True, exist_ok=True)

    for name in CHAPTER_PROMPT_FILES:
        src = bundle_layout.prompts_dir / name
        if src.exists():
            shutil.copy2(src, chapter_target / name)

    for src_subdir, subdir_name in (
        (bundle_layout.scripts_dir, "scripts"),
        (bundle_layout.datasets_dir, "datasets"),
        (bundle_layout.assets_dir, "assets"),
    ):
        if src_subdir and src_subdir.is_dir():
            shutil.copytree(src_subdir, chapter_target / subdir_name)

    _ensure_overlay_templates(overlay_curriculum_root)
    return overlay_curriculum_root


def _resolve_chapter_workspace(
    curriculum_dir: Path, chapter_id: str
) -> Dict[str, Optional[str]]:
    course_id, chapter_name = _split_chapter_id(chapter_id)
    if course_id:
        chapter_dir = curriculum_dir / "courses" / course_id / "chapters" / chapter_name
    else:
        chapter_dir = curriculum_dir / "chapters" / chapter_name

    scripts_dir = chapter_dir / "scripts"
    datasets_dir = chapter_dir / "datasets"
    assets_dir = chapter_dir / "assets"
    starter_code = scripts_dir / "starter_code.py"

    return {
        "chapter_dir": str(chapter_dir),
        "scripts_dir": str(scripts_dir) if scripts_dir.is_dir() else None,
        "datasets_dir": str(datasets_dir) if datasets_dir.is_dir() else None,
        "assets_dir": str(assets_dir) if assets_dir.is_dir() else None,
        "starter_code": str(starter_code) if starter_code.is_file() else None,
    }


def _resolve_runtime_paths(
    chapter_id: str, desktop_context: Optional[Dict[str, Any]]
) -> tuple[Path, Path, Path]:
    curriculum_dir = _resolve_curriculum_dir(chapter_id, desktop_context)
    experts_dir = _resolve_experts_dir(desktop_context)
    main_agents_dir = _resolve_main_agents_dir(desktop_context)
    return curriculum_dir, experts_dir, main_agents_dir


def _copy_chapter_bundle_assets(
    *,
    session_id: str,
    chapter_id: str,
    desktop_context: Optional[Dict[str, Any]],
    storage: Any,
) -> Dict[str, Any]:
    """Copy chapter bundle scripts/datasets into session directories."""
    workspace_info: Dict[str, Any] = {
        "has_starter_code": False,
        "has_datasets": False,
        "files": [],
    }

    if not desktop_context:
        return workspace_info

    bundle_paths = desktop_context.get("bundle_paths") or {}
    chapter_bundle_path = str(bundle_paths.get("chapter_bundle_path") or "").strip()
    if not chapter_bundle_path:
        return workspace_info
    if not hasattr(storage, "get_workspace_path") or not hasattr(
        storage, "get_working_files_path"
    ):
        return workspace_info

    bundle_root = Path(chapter_bundle_path)
    course_id, chapter_name = _split_chapter_id(chapter_id)
    chapter_dir = _find_chapter_dir_in_bundle(bundle_root, course_id, chapter_name)
    if not chapter_dir:
        return workspace_info

    workspace_dir = storage.get_workspace_path(session_id)
    working_files_dir = storage.get_working_files_path(session_id)
    files_added: set[str] = set()

    scripts_src = chapter_dir / "scripts"
    if scripts_src.is_dir():
        for item in sorted(scripts_src.iterdir()):
            if not item.is_file() or item.name == "solution.py":
                continue
            target = workspace_dir / item.name
            shutil.copy2(item, target)
            if hasattr(storage, "set_workspace_file_source"):
                storage.set_workspace_file_source(session_id, item.name, "bundle")
            workspace_info["has_starter_code"] = True
            if item.name not in files_added:
                workspace_info["files"].append({"name": item.name, "source": "bundle"})
                files_added.add(item.name)

    datasets_src = chapter_dir / "datasets"
    if datasets_src.is_dir():
        for item in sorted(datasets_src.iterdir()):
            if not item.is_file():
                continue
            target = working_files_dir / item.name
            shutil.copy2(item, target)
            workspace_info["has_datasets"] = True
            if item.name not in files_added:
                workspace_info["files"].append({"name": item.name, "source": "bundle"})
                files_added.add(item.name)

    return workspace_info


default_orchestrator = _build_orchestrator(
    curriculum_dir=default_curriculum_dir,
    experts_dir=default_experts_dir,
    main_agents_dir=default_main_agents_dir,
)
session_orchestrators: dict[str, Orchestrator] = {}
user_code_runner = UserCodeRunner(sessions_root=sessions_root)
code_execution_manager = CodeExecutionManager(runner=user_code_runner)
notebook_manager = NotebookManager()
MAX_WORKSPACE_FILE_SIZE_BYTES = 1 * 1024 * 1024


def _get_orchestrator(session_id: str) -> Orchestrator:
    return session_orchestrators.get(session_id, default_orchestrator)


def _build_code_job_summary(job: Any) -> "CodeJobSummary":
    return CodeJobSummary(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
    )


def _build_code_job_detail(job: Any) -> "CodeJobDetailResponse":
    result = RunCodeResponse(**job.result) if job.result else None
    return CodeJobDetailResponse(
        **_build_code_job_summary(job).model_dump(),
        result=result,
    )


# Request/Response models
class DesktopBundlePaths(BaseModel):
    chapter_bundle_path: Optional[str] = None
    app_agents_path: Optional[str] = None
    experts_shared_path: Optional[str] = None
    expert_bundle_paths: Dict[str, str] = Field(default_factory=dict)


class DesktopContext(BaseModel):
    bundle_paths: DesktopBundlePaths = Field(default_factory=DesktopBundlePaths)
    prompt_sources: Dict[str, Any] = Field(default_factory=dict)
    chapter_scope: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    chapter_id: str = Field(
        default="ch0_pandas_basics", description="Chapter identifier"
    )
    desktop_context: Optional[DesktopContext] = Field(
        default=None,
        description="Desktop-provided bundle and prompt resolution context",
    )


class WorkspaceBundleFile(BaseModel):
    """Workspace metadata file entry."""

    name: str
    source: str = "bundle"


class WorkspaceSummary(BaseModel):
    """Workspace summary in session creation response."""

    has_starter_code: bool = False
    has_datasets: bool = False
    files: List[WorkspaceBundleFile] = Field(default_factory=list)


class CreateSessionResponse(BaseModel):
    """Response with new session ID."""

    session_id: str
    initial_message: str
    workspace: WorkspaceSummary = Field(default_factory=WorkspaceSummary)


class SendMessageRequest(BaseModel):
    """Request to send a message."""

    message: str = Field(..., max_length=10000, description="User message")


class DynamicReportResponse(BaseModel):
    """Response with dynamic report."""

    report: str


class SessionStateResponse(BaseModel):
    """Response with session state."""

    state: dict


class EndSessionResponse(BaseModel):
    """Response with final report."""

    final_report: str


class SessionListItem(BaseModel):
    """Session list item."""

    session_id: str
    chapter_id: str
    turn_index: int
    created_at: str
    last_updated: str


class SessionListResponse(BaseModel):
    """Response with list of sessions."""

    sessions: list[SessionListItem]


class ChapterInfo(BaseModel):
    """Chapter information."""

    chapter_id: str
    title: str
    description: str
    context: str


class ChaptersListResponse(BaseModel):
    """Response with list of chapters."""

    chapters: list[ChapterInfo]


class CourseInfo(BaseModel):
    """Course information."""

    course_id: str
    title: str
    description: str
    chapter_count: int
    full_info: str


class CoursesListResponse(BaseModel):
    """Response with list of courses."""

    courses: list[CourseInfo]


class CourseChaptersResponse(BaseModel):
    """Response with chapters for a specific course."""

    course_id: str
    course_title: str
    chapters: list[ChapterInfo]


class UploadedFileInfo(BaseModel):
    """Uploaded file information."""

    filename: str
    size: int
    upload_time: float
    file_type: str


class UploadFilesResponse(BaseModel):
    """Response with uploaded files information."""

    files: list[UploadedFileInfo]
    message: str


class FilesListResponse(BaseModel):
    """Response with list of uploaded files."""

    files: list[UploadedFileInfo]


class DeleteFileResponse(BaseModel):
    """Response for file deletion."""

    success: bool
    message: str


class SessionWorkspaceResponse(BaseModel):
    """Resolved chapter workspace paths for a session."""

    chapter_id: str
    chapter_dir: str
    scripts_dir: Optional[str] = None
    datasets_dir: Optional[str] = None
    assets_dir: Optional[str] = None
    starter_code: Optional[str] = None


class WorkspaceFileInfo(BaseModel):
    """User workspace file metadata."""

    name: str
    size_bytes: int
    modified_at: str
    source: str


class WorkspaceFilesListResponse(BaseModel):
    """Response with user workspace files."""

    files: List[WorkspaceFileInfo]


class WorkspaceFileReadResponse(BaseModel):
    """Workspace file content response."""

    name: str
    content: str
    size_bytes: int


class WorkspaceFileWriteRequest(BaseModel):
    """Request body for workspace file write."""

    content: str


class RunCodeRequest(BaseModel):
    """Request to run user Python code in the session workspace."""

    code: str = Field(..., min_length=1, max_length=20000)
    timeout_seconds: int = Field(default=20, ge=1, le=120)
    memory_limit_mb: Optional[int] = Field(default=None, ge=64, le=8192)


class RunCodeResponse(BaseModel):
    """Response for user code execution."""

    success: bool
    stdout: str
    stderr: str
    returncode: int
    execution_time_ms: int
    cancelled: bool = False


class CreateCodeJobRequest(BaseModel):
    """Request to create an async user code execution job."""

    code: str = Field(..., min_length=1, max_length=20000)
    timeout_seconds: int = Field(default=20, ge=1, le=120)
    memory_limit_mb: Optional[int] = Field(default=None, ge=64, le=8192)


class CodeJobSummary(BaseModel):
    """Summary for a background code execution job."""

    job_id: str
    session_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class CodeJobDetailResponse(CodeJobSummary):
    """Detailed response for a background code execution job."""

    result: Optional[RunCodeResponse] = None


class CancelCodeJobResponse(BaseModel):
    """Response for async code job cancellation."""

    success: bool
    job_id: str
    status: str


class ContractRoute(BaseModel):
    """Stable route contract item."""

    method: str
    path: str
    stability: str
    source: str


class SidecarContractResponse(BaseModel):
    """Public sidecar API contract."""

    contract_version: str
    api_version: str
    routes: list[ContractRoute]
    sse_event_types: list[str]


class RunNotebookCellRequest(BaseModel):
    """Request to execute a notebook cell."""

    code: str = Field(..., min_length=1, max_length=20000)
    cell_id: Optional[str] = None


class RunNotebookCellResponse(BaseModel):
    """Response for notebook cell execution."""

    success: bool
    stdout: str
    stderr: str
    result_repr: str
    execution_time_ms: int


class ResetNotebookResponse(BaseModel):
    """Response for notebook kernel/session reset."""

    success: bool


# API Endpoints


@app.post("/api/session/new", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Create a new learning session.

    This initializes a new session with the specified chapter and returns
    the session ID along with the initial greeting message.
    """
    try:
        logger.info(f"Creating new session for chapter: {request.chapter_id}")
        _split_chapter_id(request.chapter_id)  # Validate chapter_id format early.
        desktop_context = (
            request.desktop_context.model_dump() if request.desktop_context else None
        )
        curriculum_dir, experts_dir, main_agents_dir = _resolve_runtime_paths(
            request.chapter_id, desktop_context
        )
        logger.info(
            "Resolved runtime paths: curriculum=%s experts=%s main_agents=%s",
            curriculum_dir,
            experts_dir,
            main_agents_dir,
        )

        orchestrator = _build_orchestrator(
            curriculum_dir=curriculum_dir,
            experts_dir=experts_dir,
            main_agents_dir=main_agents_dir,
        )
        session_id = await orchestrator.create_session(request.chapter_id)
        session_orchestrators[session_id] = orchestrator
        workspace_info = _copy_chapter_bundle_assets(
            session_id=session_id,
            chapter_id=request.chapter_id,
            desktop_context=desktop_context,
            storage=orchestrator.storage,
        )

        # Load the initial companion message (turn 0)
        turn_history = orchestrator.storage.load_turn_history(session_id)
        if turn_history and len(turn_history) > 0:
            initial_message = turn_history[0]["companion_response"]
        else:
            initial_message = "你好！欢迎来到学习平台。"

        logger.info(f"Session created: {session_id}")
        return CreateSessionResponse(
            session_id=session_id,
            initial_message=initial_message,
            workspace=WorkspaceSummary.model_validate(workspace_info),
        )

    except OrchestratorError as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="创建会话失败")


@app.post("/api/session/{session_id}/message/stream")
async def send_message_stream(session_id: str, request: SendMessageRequest):
    """
    Send a message with streaming response.

    Returns Server-Sent Events stream with:
    - Companion response (character by character)
    - Progress updates from other agents
    """

    async def event_generator():
        orchestrator = _get_orchestrator(session_id)
        try:
            # Validate session
            if not orchestrator.storage.session_exists(session_id):
                import json

                yield f"data: {json.dumps({'type': 'error', 'message': '会话不存在'}, ensure_ascii=False)}\n\n"
                return

            # Validate message length
            if len(request.message) > settings.max_input_length:
                import json

                yield f"data: {json.dumps({'type': 'error', 'message': '消息长度超过限制'}, ensure_ascii=False)}\n\n"
                return

            # Send start event
            import json

            yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"

            # Process turn with streaming
            async for event in orchestrator.process_turn_stream(
                session_id, request.message
            ):
                import json

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming

            # Send complete event
            state = orchestrator.storage.load_state(session_id)
            import json

            yield f"data: {json.dumps({'type': 'complete', 'turn_index': state.turn_index}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            import json

            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/api/session/{session_id}/dynamic_report", response_model=DynamicReportResponse
)
async def get_dynamic_report(session_id: str):
    """
    Get the current dynamic report for a session.

    Returns the latest learning progress report.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        report = orchestrator.storage.load_dynamic_report(session_id)

        return DynamicReportResponse(report=report)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dynamic report: {e}")
        raise HTTPException(status_code=500, detail="获取报告失败")


@app.get("/api/session/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(session_id: str):
    """
    Get the current session state (for debugging).

    Returns the complete session state including task status and configuration.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        state = orchestrator.storage.load_state(session_id)

        return SessionStateResponse(state=state.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session state: {e}")
        raise HTTPException(status_code=500, detail="获取状态失败")


@app.post("/api/session/{session_id}/end", response_model=EndSessionResponse)
async def end_session(session_id: str):
    """
    End a session and generate final report.

    Marks the session as ended and generates a comprehensive final learning report.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        logger.info(f"Ending session: {session_id}")

        final_report = await orchestrator.end_session(session_id)
        session_orchestrators.pop(session_id, None)

        logger.info("Session ended successfully")
        return EndSessionResponse(final_report=final_report)

    except OrchestratorError as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="结束会话失败")


@app.get("/api/courses", response_model=CoursesListResponse)
async def list_courses():
    """
    Get list of all available courses.

    Returns a list of all courses with their basic information.
    """
    try:
        courses_path = default_curriculum_dir / "courses"
        if not courses_path.exists():
            return CoursesListResponse(courses=[])
        courses = []

        # Iterate through course directories
        for course_dir in sorted(courses_path.iterdir()):
            if not course_dir.is_dir():
                continue

            course_id = course_dir.name
            course_info_file = course_dir / "course_info.md"

            if not course_info_file.exists():
                continue

            # Read course info
            course_info_content = course_info_file.read_text(encoding="utf-8")

            # Extract title from first heading
            title = course_id
            description = ""
            for line in course_info_content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Extract description (first paragraph after title)
            lines = course_info_content.split("\n")
            in_description = False
            description_lines = []
            for line in lines:
                if line.startswith("## 课程简介"):
                    in_description = True
                    continue
                if in_description and line.strip():
                    if line.startswith("#"):
                        break
                    description_lines.append(line.strip())
                    if len(description_lines) >= 2:  # First 2 lines
                        break

            description = " ".join(description_lines)

            # Count chapters
            chapters_dir = course_dir / "chapters"
            chapter_count = 0
            if chapters_dir.exists():
                chapter_count = len([d for d in chapters_dir.iterdir() if d.is_dir()])

            courses.append(
                CourseInfo(
                    course_id=course_id,
                    title=title,
                    description=description,
                    chapter_count=chapter_count,
                    full_info=course_info_content,
                )
            )

        return CoursesListResponse(courses=courses)

    except Exception as e:
        logger.error(f"Failed to list courses: {e}")
        raise HTTPException(status_code=500, detail="获取课程列表失败")


@app.get("/api/courses/{course_id}/chapters", response_model=CourseChaptersResponse)
async def list_course_chapters(course_id: str):
    """
    Get list of chapters for a specific course.

    Returns all chapters belonging to the specified course.
    """
    try:
        course_path = default_curriculum_dir / "courses" / course_id
        if not course_path.exists():
            raise HTTPException(status_code=404, detail="课程不存在")

        # Get course title
        course_info_file = course_path / "course_info.md"
        course_title = course_id
        if course_info_file.exists():
            course_info_content = course_info_file.read_text(encoding="utf-8")
            for line in course_info_content.split("\n"):
                if line.startswith("# "):
                    course_title = line[2:].strip()
                    break

        chapters_path = course_path / "chapters"
        chapters = []

        if not chapters_path.exists():
            return CourseChaptersResponse(
                course_id=course_id, course_title=course_title, chapters=[]
            )

        # Iterate through chapter directories
        for chapter_dir in sorted(chapters_path.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = f"{course_id}/{chapter_dir.name}"
            context_file = chapter_dir / "chapter_context.md"

            if not context_file.exists():
                continue

            # Read chapter context
            context_content = context_file.read_text(encoding="utf-8")

            # Extract title from first heading
            title = chapter_dir.name
            description = ""
            for line in context_content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Extract description (first paragraph after title)
            lines = context_content.split("\n")
            in_description = False
            description_lines = []
            for line in lines:
                if line.startswith("# "):
                    in_description = True
                    continue
                if in_description and line.strip():
                    if line.startswith("#"):
                        break
                    description_lines.append(line.strip())
                    if len(description_lines) >= 3:  # First 3 lines
                        break

            description = " ".join(description_lines)

            chapters.append(
                ChapterInfo(
                    chapter_id=chapter_id,
                    title=title,
                    description=description,
                    context=context_content,
                )
            )

        return CourseChaptersResponse(
            course_id=course_id, course_title=course_title, chapters=chapters
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list chapters for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="获取课程章节失败")


@app.get("/api/chapters", response_model=ChaptersListResponse)
async def list_chapters():
    """
    Get list of all available chapters.

    Returns a list of all chapters with their context information.
    """
    try:
        curriculum_path = default_curriculum_dir / "chapters"
        if not curriculum_path.exists():
            return ChaptersListResponse(chapters=[])
        chapters = []

        # Iterate through chapter directories
        for chapter_dir in sorted(curriculum_path.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = chapter_dir.name
            context_file = chapter_dir / "chapter_context.md"

            if not context_file.exists():
                continue

            # Read chapter context
            context_content = context_file.read_text(encoding="utf-8")

            # Extract title from first heading
            title = chapter_id
            description = ""
            for line in context_content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Extract description (first paragraph after title)
            lines = context_content.split("\n")
            in_description = False
            description_lines = []
            for line in lines:
                if line.startswith("# "):
                    in_description = True
                    continue
                if in_description and line.strip():
                    if line.startswith("#"):
                        break
                    description_lines.append(line.strip())
                    if len(description_lines) >= 3:  # First 3 lines
                        break

            description = " ".join(description_lines)

            chapters.append(
                ChapterInfo(
                    chapter_id=chapter_id,
                    title=title,
                    description=description,
                    context=context_content,
                )
            )

        return ChaptersListResponse(chapters=chapters)

    except Exception as e:
        logger.error(f"Failed to list chapters: {e}")
        raise HTTPException(status_code=500, detail="获取课程列表失败")


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    """
    Get list of all sessions.

    Returns a list of all sessions with metadata, sorted by last updated.
    """
    try:
        from datetime import datetime

        sessions_data = default_orchestrator.storage.list_sessions()

        sessions = [
            SessionListItem(
                session_id=s["session_id"],
                chapter_id=s["chapter_id"],
                turn_index=s["turn_index"],
                created_at=datetime.fromtimestamp(s["created_at"]).isoformat(),
                last_updated=datetime.fromtimestamp(s["last_updated"]).isoformat(),
            )
            for s in sessions_data
        ]

        return SessionListResponse(sessions=sessions)

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="获取会话列表失败")


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get complete turn history for a session.

    Returns all turns with user messages, companion responses, and turn outcomes.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        turn_history = orchestrator.storage.load_turn_history(session_id)

        return {"turns": turn_history}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail="获取会话历史失败")


@app.get("/api/session/{session_id}/workspace", response_model=SessionWorkspaceResponse)
async def get_session_workspace(session_id: str):
    """
    Get resolved chapter workspace paths, including optional scripts/datasets/assets directories.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        state = orchestrator.storage.load_state(session_id)
        workspace = _resolve_chapter_workspace(
            orchestrator.curriculum_dir, state.chapter_id
        )
        return SessionWorkspaceResponse(chapter_id=state.chapter_id, **workspace)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session workspace: {e}")
        raise HTTPException(status_code=500, detail="获取章节工作区失败")


@app.post("/api/session/{session_id}/upload", response_model=UploadFilesResponse)
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    """
    Upload files to a session's working_files directory.

    Accepts multiple files with the following constraints:
    - Total size must be less than 5MB
    - Allowed file types: .csv, .xlsx, .xls, .json
    - Filenames are sanitized to prevent path traversal
    - Duplicate filenames are handled by adding timestamps
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        # Validate session exists
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        # Validate file count
        if len(files) > settings.max_uploads_per_session:
            raise HTTPException(
                status_code=400,
                detail=f"一次最多上传 {settings.max_uploads_per_session} 个文件",
            )

        # Read all files and validate
        uploaded_files_info = []
        total_size = 0

        for file in files:
            # Read file content
            content = await file.read()
            file_size = len(content)
            total_size += file_size

            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in settings.allowed_file_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件格式：{file_ext}。支持的格式：{', '.join(settings.allowed_file_extensions)}",
                )

            # Check total size
            max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
            if total_size > max_size_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件总大小超过 {settings.max_upload_size_mb}MB 限制",
                )

            # Save file
            saved_path = orchestrator.storage.save_uploaded_file(
                session_id, content, file.filename
            )

            # Get file metadata
            stat = saved_path.stat()
            uploaded_files_info.append(
                UploadedFileInfo(
                    filename=saved_path.name,
                    size=stat.st_size,
                    upload_time=stat.st_mtime,
                    file_type=saved_path.suffix,
                )
            )

        logger.info(
            f"Uploaded {len(uploaded_files_info)} files to session {session_id}"
        )

        return UploadFilesResponse(
            files=uploaded_files_info,
            message=f"成功上传 {len(uploaded_files_info)} 个文件",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        raise HTTPException(status_code=500, detail="文件上传失败")


@app.get("/api/session/{session_id}/files", response_model=FilesListResponse)
async def list_files(session_id: str):
    """
    Get list of uploaded files for a session.

    Returns metadata for all files in the session's working_files directory.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        # Validate session exists
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        # Get files metadata
        files_metadata = orchestrator.storage.get_uploaded_files_metadata(session_id)

        files_info = [
            UploadedFileInfo(
                filename=f["filename"],
                size=f["size"],
                upload_time=f["upload_time"],
                file_type=f["file_type"],
            )
            for f in files_metadata
        ]

        return FilesListResponse(files=files_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail="获取文件列表失败")


@app.delete(
    "/api/session/{session_id}/files/{filename}", response_model=DeleteFileResponse
)
async def delete_file(session_id: str, filename: str):
    """
    Delete a file from a session's working_files directory.

    Validates the filename to prevent path traversal attacks.
    """
    try:
        orchestrator = _get_orchestrator(session_id)
        # Validate session exists
        if not orchestrator.storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")

        # Delete file
        deleted = orchestrator.storage.delete_uploaded_file(session_id, filename)

        if deleted:
            logger.info(f"Deleted file {filename} from session {session_id}")
            return DeleteFileResponse(success=True, message=f"文件 {filename} 已删除")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        raise HTTPException(status_code=500, detail="删除文件失败")


@app.get(
    "/api/session/{session_id}/workspace/files",
    response_model=WorkspaceFilesListResponse,
)
async def list_workspace_files(session_id: str):
    """List editable files in user_workspace."""
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        files = orchestrator.storage.list_workspace_files(session_id)
        return WorkspaceFilesListResponse(
            files=[WorkspaceFileInfo(**item) for item in files]
        )
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to list workspace files: {exc}")
        raise HTTPException(status_code=500, detail="获取工作区文件失败") from exc


@app.get(
    "/api/session/{session_id}/workspace/files/{filename}",
    response_model=WorkspaceFileReadResponse,
)
async def read_workspace_file(session_id: str, filename: str):
    """Read a file from user_workspace."""
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        content = orchestrator.storage.read_workspace_file(session_id, filename)
        return WorkspaceFileReadResponse(
            name=filename,
            content=content,
            size_bytes=len(content.encode("utf-8")),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="文件不存在") from exc
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to read workspace file {filename}: {exc}")
        raise HTTPException(status_code=500, detail="读取工作区文件失败") from exc


@app.put(
    "/api/session/{session_id}/workspace/files/{filename}",
    response_model=WorkspaceFileInfo,
)
async def write_workspace_file(
    session_id: str, filename: str, request: WorkspaceFileWriteRequest
):
    """Write or overwrite a file in user_workspace."""
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    content_bytes = request.content.encode("utf-8")
    if len(content_bytes) > MAX_WORKSPACE_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="文件内容超过 1MB 限制")

    try:
        metadata = orchestrator.storage.write_workspace_file(
            session_id, filename, request.content
        )
        return WorkspaceFileInfo(**metadata)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to write workspace file {filename}: {exc}")
        raise HTTPException(status_code=500, detail="写入工作区文件失败") from exc


@app.delete("/api/session/{session_id}/workspace/files/{filename}", status_code=204)
async def delete_workspace_file(session_id: str, filename: str):
    """Delete a file from user_workspace."""
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        orchestrator.storage.delete_workspace_file(session_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="文件不存在") from exc
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to delete workspace file {filename}: {exc}")
        raise HTTPException(status_code=500, detail="删除工作区文件失败") from exc

    return Response(status_code=204)


@app.post("/api/session/{session_id}/code/run", response_model=RunCodeResponse)
async def run_user_code(session_id: str, request: RunCodeRequest):
    """
    Execute user Python code in a per-session workspace.

    This is the initial user-code runtime API and is intentionally simple:
    subprocess execution with timeout, output capture, and output size clipping.
    """
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    result = user_code_runner.run_python(
        session_id=session_id,
        code=request.code,
        timeout_seconds=request.timeout_seconds,
        memory_limit_mb=request.memory_limit_mb,
    )
    return RunCodeResponse(**result)


@app.post("/api/session/{session_id}/code/jobs", response_model=CodeJobSummary)
async def create_code_job(session_id: str, request: CreateCodeJobRequest):
    """
    Create an asynchronous user-code execution job.
    """
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    job = code_execution_manager.create_job(
        session_id=session_id,
        code=request.code,
        timeout_seconds=request.timeout_seconds,
        memory_limit_mb=request.memory_limit_mb,
    )
    return _build_code_job_summary(job)


@app.get(
    "/api/session/{session_id}/code/jobs/{job_id}", response_model=CodeJobDetailResponse
)
async def get_code_job(session_id: str, job_id: str):
    """
    Get asynchronous user-code execution job status.
    """
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    job = code_execution_manager.get_job(job_id)
    if not job or job.session_id != session_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _build_code_job_detail(job)


@app.post(
    "/api/session/{session_id}/code/jobs/{job_id}/cancel",
    response_model=CancelCodeJobResponse,
)
async def cancel_code_job(session_id: str, job_id: str):
    """
    Cancel asynchronous user-code execution job.
    """
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    job = code_execution_manager.get_job(job_id)
    if not job or job.session_id != session_id:
        raise HTTPException(status_code=404, detail="任务不存在")

    updated = code_execution_manager.cancel_job(job_id)
    if not updated:
        raise HTTPException(status_code=404, detail="任务不存在")

    return CancelCodeJobResponse(success=True, job_id=job_id, status=updated.status)


@app.post(
    "/api/session/{session_id}/notebook/cell/run",
    response_model=RunNotebookCellResponse,
)
async def run_notebook_cell(session_id: str, request: RunNotebookCellRequest):
    """
    Execute a notebook-style code cell with per-session state persistence.
    """
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    result = notebook_manager.execute_cell(session_id=session_id, code=request.code)
    return RunNotebookCellResponse(**result)


@app.post(
    "/api/session/{session_id}/notebook/reset", response_model=ResetNotebookResponse
)
async def reset_notebook_session(session_id: str):
    """
    Reset notebook session state (kernel-like reset).
    """
    orchestrator = _get_orchestrator(session_id)
    if not orchestrator.storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    notebook_manager.reset_session(session_id)
    return ResetNotebookResponse(success=True)


@app.get("/api/contract", response_model=SidecarContractResponse)
async def get_sidecar_contract():
    """
    Return frozen v1 sidecar API and streaming contract.
    """
    routes = [
        ContractRoute(
            method="POST",
            path="/api/session/new",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/message/stream",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/dynamic_report",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/state",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/end",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET", path="/api/sessions", stability="stable", source="demo_parity"
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/history",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/workspace",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/upload",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/files",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="DELETE",
            path="/api/session/{session_id}/files/{filename}",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/workspace/files",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/workspace/files/{filename}",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="PUT",
            path="/api/session/{session_id}/workspace/files/{filename}",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="DELETE",
            path="/api/session/{session_id}/workspace/files/{filename}",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="GET", path="/api/courses", stability="stable", source="demo_parity"
        ),
        ContractRoute(
            method="GET",
            path="/api/courses/{course_id}/chapters",
            stability="stable",
            source="demo_parity",
        ),
        ContractRoute(
            method="GET", path="/api/chapters", stability="stable", source="demo_parity"
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/code/run",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/code/jobs",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="GET",
            path="/api/session/{session_id}/code/jobs/{job_id}",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/code/jobs/{job_id}/cancel",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/notebook/cell/run",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="POST",
            path="/api/session/{session_id}/notebook/reset",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="GET",
            path="/api/contract",
            stability="stable",
            source="sidecar_extension",
        ),
        ContractRoute(
            method="GET", path="/health", stability="stable", source="demo_parity"
        ),
    ]
    sse_event_types = [
        "start",
        "companion_start",
        "companion_chunk",
        "companion_complete",
        "consultation_start",
        "consultation_complete",
        "consultation_error",
        "complete",
        "error",
    ]
    return SidecarContractResponse(
        contract_version="v1",
        api_version=app.version,
        routes=routes,
        sse_event_types=sse_event_types,
    )


# Serve static files (frontend)
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/")
    async def serve_index():
        """Serve the main web UI."""
        index_file = web_dir / "index.html"
        if index_file.exists():
            return FileResponse(
                index_file,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        else:
            return {"message": "Frontend not found. Please create app/web/index.html"}

else:

    @app.get("/")
    async def root():
        """Root endpoint when frontend is not available."""
        return {
            "message": "Multi-Agent Tutor System API",
            "version": "1.0.0",
            "docs": "/docs",
        }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
