"""File-based storage layer for session management."""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from filelock import FileLock
import tempfile
import shutil

from ..models.schemas import (
    SessionState,
    InstructionPacket,
    MemoDigest,
    MemoryState,
    StudentErrorEntry,
)

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Custom exception for storage errors."""
    pass


class Storage:
    """File-based storage manager for session data."""

    WORKSPACE_ALLOWED_EXTENSIONS = {".py", ".ipynb", ".txt", ".md"}

    def __init__(self, base_dir: str = "sessions"):
        """
        Initialize storage manager.

        Args:
            base_dir: Base directory for session storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get session directory path."""
        return self.base_dir / session_id

    def _get_turns_dir(self, session_id: str) -> Path:
        """Get turns directory path."""
        return self._get_session_dir(session_id) / "turns"

    def _get_workspace_dir(self, session_id: str) -> Path:
        """Get user workspace directory path."""
        return self._get_session_dir(session_id) / "user_workspace"

    def _get_code_history_dir(self, session_id: str) -> Path:
        """Get code history directory path."""
        return self._get_session_dir(session_id) / "code_history"

    def _get_workspace_sources_path(self, session_id: str) -> Path:
        """Get workspace file source metadata path."""
        return self._get_session_dir(session_id) / "workspace_sources.json"

    def _atomic_write_json(self, file_path: Path, data: Dict) -> None:
        """
        Write JSON file atomically using temp file + rename.

        Args:
            file_path: Target file path
            data: Data to write
        """
        # Create temp file in same directory
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=".tmp_",
            suffix=".json"
        )

        try:
            # Write to temp file
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomic rename
            shutil.move(temp_path, file_path)
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise StorageError(f"Failed to write {file_path}: {e}")

    def _read_json(self, file_path: Path) -> Dict:
        """
        Read JSON file with locking.

        Args:
            file_path: File path to read

        Returns:
            Parsed JSON data
        """
        lock_path = file_path.with_suffix(".lock")

        try:
            with FileLock(lock_path, timeout=10):
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except FileNotFoundError:
            raise StorageError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise StorageError(f"Invalid JSON in {file_path}: {e}")

    def _write_json(self, file_path: Path, data: Dict) -> None:
        """
        Write JSON file with locking and atomic write.

        Args:
            file_path: File path to write
            data: Data to write
        """
        lock_path = file_path.with_suffix(".lock")

        try:
            with FileLock(lock_path, timeout=10):
                self._atomic_write_json(file_path, data)
        except Exception as e:
            raise StorageError(f"Failed to write {file_path}: {e}")

    def create_session(self, session_id: str, chapter_id: str, initial_state: SessionState) -> None:
        """
        Create new session folder structure (v3.0 with expert system support).

        Args:
            session_id: Session identifier
            chapter_id: Chapter identifier
            initial_state: Initial session state
        """
        session_dir = self._get_session_dir(session_id)

        if session_dir.exists():
            raise StorageError(f"Session {session_id} already exists")

        # Create directory structure (v2.x + v3.0)
        session_dir.mkdir(parents=True)
        self._get_turns_dir(session_id).mkdir()

        # v3.0: Create expert system directories
        (session_dir / "working_files").mkdir(exist_ok=True)
        (session_dir / "working_files" / "expert_logs").mkdir(exist_ok=True)
        (session_dir / "user_workspace").mkdir(exist_ok=True)
        (session_dir / "code_history").mkdir(exist_ok=True)
        (session_dir / "expert_workspace").mkdir(exist_ok=True)
        (session_dir / "expert_workspace" / "consultations").mkdir(exist_ok=True)
        (session_dir / "expert_reports").mkdir(exist_ok=True)

        # v3.0: Initialize expert_reports/index.json
        expert_reports_index = {"reports": []}
        self._write_json(session_dir / "expert_reports" / "index.json", expert_reports_index)
        self._write_json(self._get_workspace_sources_path(session_id), {"sources": {}})

        # Save initial state
        self.save_state(session_id, initial_state)

        # Create empty reports
        self.save_dynamic_report(session_id, "")
        self.save_student_error_summary(session_id, "")

        logger.info(f"Created session {session_id} with v3.0 directory structure")
        self.save_final_learning_report(session_id, "")

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self._get_session_dir(session_id).exists()

    def save_state(self, session_id: str, state: SessionState) -> None:
        """Save session state."""
        file_path = self._get_session_dir(session_id) / "session_state.json"
        self._write_json(file_path, state.model_dump())

    def load_state(self, session_id: str) -> SessionState:
        """
        Load session state with soft migration for v1.2 sessions.

        Automatically adds missing v2.0 fields with default values.
        """
        file_path = self._get_session_dir(session_id) / "session_state.json"
        data = self._read_json(file_path)

        # Soft migration: add v2.0 fields if missing
        if "current_instruction_version" not in data:
            logger.info(f"Migrating session {session_id} from v1.2 to v2.0")
            data["current_instruction_version"] = 1
            data["attempts_since_last_progress"] = 0
            data["last_progress_turn"] = data.get("turn_index", 0)

            # Update constraints if needed
            if "constraints" in data:
                if "max_attempts_before_unlock" not in data["constraints"]:
                    data["constraints"]["max_attempts_before_unlock"] = 3

        return SessionState(**data)

    def save_instruction_packet(self, session_id: str, packet: InstructionPacket) -> None:
        """Save instruction packet."""
        file_path = self._get_session_dir(session_id) / "instruction_packet.json"
        self._write_json(file_path, packet.model_dump())

    def load_instruction_packet(self, session_id: str) -> InstructionPacket:
        """
        Load instruction packet with soft migration for v1.2 sessions.

        Converts old 'what_to_check' string to new 'must_check' array format.
        """
        file_path = self._get_session_dir(session_id) / "instruction_packet.json"
        data = self._read_json(file_path)

        # Soft migration: convert v1.2 format to v2.0
        if "what_to_check" in data and "must_check" not in data:
            logger.info(f"Migrating instruction packet for session {session_id} from v1.2 to v2.0")

            # Convert old string to array (split by newline or comma)
            old_check = data.pop("what_to_check")
            checks = [c.strip() for c in old_check.replace("\n", ",").split(",") if c.strip()]

            # Take first 2 as must_check
            data["must_check"] = checks[:2] if checks else ["验证学生理解当前任务"]
            data["nice_check"] = checks[2:3] if len(checks) > 2 else []

            # Add new required fields with defaults
            data["instruction_version"] = 1
            data["lock_until"] = "checkpoint_reached"
            data["allow_setup_helper_code"] = False
            data["setup_helper_scope"] = "none"
            data["task_type"] = "core"

        return InstructionPacket(**data)

    def save_memo_digest(self, session_id: str, digest: MemoDigest) -> None:
        """Save memo digest."""
        file_path = self._get_session_dir(session_id) / "memo_digest.json"
        self._write_json(file_path, digest.model_dump())

    def load_memo_digest(self, session_id: str) -> Optional[MemoDigest]:
        """Load memo digest."""
        file_path = self._get_session_dir(session_id) / "memo_digest.json"
        if not file_path.exists():
            return None
        data = self._read_json(file_path)
        return MemoDigest(**data)

    def save_turn(
        self,
        session_id: str,
        turn_index: int,
        user_message: str,
        companion_response: str,
        turn_outcome: Dict,
    ) -> None:
        """
        Save turn history files.

        Args:
            session_id: Session identifier
            turn_index: Turn index (0-indexed)
            user_message: User's message
            companion_response: Companion's response
            turn_outcome: Turn outcome data
        """
        turns_dir = self._get_turns_dir(session_id)

        # Zero-padded turn index (e.g., 000, 001, 002)
        turn_str = f"{turn_index:03d}"

        # Save user message
        user_file = turns_dir / f"{turn_str}_user.txt"
        with open(user_file, "w", encoding="utf-8") as f:
            f.write(user_message)

        # Save companion response
        companion_file = turns_dir / f"{turn_str}_companion.txt"
        with open(companion_file, "w", encoding="utf-8") as f:
            f.write(companion_response)

        # Save turn outcome
        outcome_file = turns_dir / f"{turn_str}_turn_outcome.json"
        self._write_json(outcome_file, turn_outcome)

    def load_turn_history(self, session_id: str) -> List[Dict]:
        """
        Load complete turn history.

        Returns:
            List of turn data dictionaries
        """
        turns_dir = self._get_turns_dir(session_id)
        if not turns_dir.exists():
            return []

        # Find all turn outcome files
        outcome_files = sorted(turns_dir.glob("*_turn_outcome.json"))

        history = []
        for outcome_file in outcome_files:
            turn_str = outcome_file.stem.split("_")[0]

            user_file = turns_dir / f"{turn_str}_user.txt"
            companion_file = turns_dir / f"{turn_str}_companion.txt"

            # Read files
            with open(user_file, "r", encoding="utf-8") as f:
                user_message = f.read()

            with open(companion_file, "r", encoding="utf-8") as f:
                companion_response = f.read()

            turn_outcome = self._read_json(outcome_file)

            history.append({
                "turn_index": int(turn_str),
                "user_message": user_message,
                "companion_response": companion_response,
                "turn_outcome": turn_outcome,
            })

        return history

    def get_last_turn_timestamp(self, session_id: str) -> Optional[float]:
        """
        Get the most recent turn file modification time.

        Returns:
            Unix timestamp of the latest saved turn, or None if no turns exist.
        """
        turns_dir = self._get_turns_dir(session_id)
        if not turns_dir.exists():
            return None

        outcome_files = list(turns_dir.glob("*_turn_outcome.json"))
        if not outcome_files:
            return None

        return max(f.stat().st_mtime for f in outcome_files)

    def save_dynamic_report(self, session_id: str, report: str) -> None:
        """Save dynamic report."""
        file_path = self._get_session_dir(session_id) / "dynamic_report.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report)

    def load_dynamic_report(self, session_id: str) -> str:
        """Load dynamic report."""
        file_path = self._get_session_dir(session_id) / "dynamic_report.md"
        if not file_path.exists():
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_student_error_summary(self, session_id: str, summary: str) -> None:
        """Save student error summary."""
        file_path = self._get_session_dir(session_id) / "student_error_summary.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary)

    def append_student_error_summary(
        self,
        session_id: str,
        error_entries: List[StudentErrorEntry]
    ) -> None:
        """
        Append error entries to student error summary.

        Args:
            session_id: Session identifier
            error_entries: List of error entries to append
        """
        current_summary = self.load_student_error_summary(session_id)

        # Format new entries
        new_entries_text = "\n\n"
        for entry in error_entries:
            error_type_cn = "概念性错误" if entry.error_type == "conceptual" else "编码错误"
            new_entries_text += f"### 回合 {entry.turn_index} - {error_type_cn}\n\n"
            new_entries_text += f"**描述**: {entry.description}\n\n"
            new_entries_text += f"**上下文**: {entry.context}\n\n"

        # Append to summary
        updated_summary = current_summary + new_entries_text
        self.save_student_error_summary(session_id, updated_summary)

    def load_student_error_summary(self, session_id: str) -> str:
        """Load student error summary."""
        file_path = self._get_session_dir(session_id) / "student_error_summary.md"
        if not file_path.exists():
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_final_learning_report(self, session_id: str, report: str) -> None:
        """Save final learning report."""
        file_path = self._get_session_dir(session_id) / "final_learning_report.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report)

    def load_final_learning_report(self, session_id: str) -> str:
        """Load final learning report."""
        file_path = self._get_session_dir(session_id) / "final_learning_report.md"
        if not file_path.exists():
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_memory_state(self, session_id: str, memory_state: MemoryState) -> None:
        """Save session memory state."""
        file_path = self._get_session_dir(session_id) / "memory_state.json"
        self._write_json(file_path, memory_state.model_dump())

    def load_memory_state(self, session_id: str) -> MemoryState:
        """Load session memory state."""
        file_path = self._get_session_dir(session_id) / "memory_state.json"
        if not file_path.exists():
            return MemoryState()
        data = self._read_json(file_path)
        return MemoryState(**data)

    def load_recent_turns(self, session_id: str, limit: int) -> List[Dict]:
        """
        Load most recent N turns.

        Returns:
            List of turn data dictionaries
        """
        if limit <= 0:
            return []

        turns_dir = self._get_turns_dir(session_id)
        if not turns_dir.exists():
            return []

        outcome_files = sorted(turns_dir.glob("*_turn_outcome.json"))
        outcome_files = outcome_files[-limit:]

        history = []
        for outcome_file in outcome_files:
            turn_str = outcome_file.stem.split("_")[0]
            user_file = turns_dir / f"{turn_str}_user.txt"
            companion_file = turns_dir / f"{turn_str}_companion.txt"

            if not user_file.exists() or not companion_file.exists():
                continue

            with open(user_file, "r", encoding="utf-8") as f:
                user_message = f.read()

            with open(companion_file, "r", encoding="utf-8") as f:
                companion_response = f.read()

            turn_outcome = self._read_json(outcome_file)

            history.append({
                "turn_index": int(turn_str),
                "user_message": user_message,
                "companion_response": companion_response,
                "turn_outcome": turn_outcome,
            })

        return history

    def load_turn(self, session_id: str, turn_index: int) -> Optional[Dict]:
        """
        Load a single turn by index.

        Returns:
            Turn data dictionary, or None if files are missing
        """
        turns_dir = self._get_turns_dir(session_id)
        if not turns_dir.exists():
            return None

        turn_str = f"{turn_index:03d}"
        outcome_file = turns_dir / f"{turn_str}_turn_outcome.json"
        user_file = turns_dir / f"{turn_str}_user.txt"
        companion_file = turns_dir / f"{turn_str}_companion.txt"

        if not outcome_file.exists() or not user_file.exists() or not companion_file.exists():
            return None

        with open(user_file, "r", encoding="utf-8") as f:
            user_message = f.read()

        with open(companion_file, "r", encoding="utf-8") as f:
            companion_response = f.read()

        turn_outcome = self._read_json(outcome_file)

        return {
            "turn_index": turn_index,
            "user_message": user_message,
            "companion_response": companion_response,
            "turn_outcome": turn_outcome,
        }

    def load_turn_range(self, session_id: str, start_turn: int, end_turn: int) -> List[Dict]:
        """
        Load turns within a specific inclusive range.

        Returns:
            List of turn data dictionaries
        """
        if start_turn > end_turn:
            return []

        turns_dir = self._get_turns_dir(session_id)
        if not turns_dir.exists():
            return []

        history = []
        for turn_index in range(start_turn, end_turn + 1):
            turn_str = f"{turn_index:03d}"
            outcome_file = turns_dir / f"{turn_str}_turn_outcome.json"
            user_file = turns_dir / f"{turn_str}_user.txt"
            companion_file = turns_dir / f"{turn_str}_companion.txt"

            if not outcome_file.exists() or not user_file.exists() or not companion_file.exists():
                continue

            with open(user_file, "r", encoding="utf-8") as f:
                user_message = f.read()

            with open(companion_file, "r", encoding="utf-8") as f:
                companion_response = f.read()

            turn_outcome = self._read_json(outcome_file)

            history.append({
                "turn_index": turn_index,
                "user_message": user_message,
                "companion_response": companion_response,
                "turn_outcome": turn_outcome,
            })

        return history

    def list_sessions(self) -> List[Dict]:
        """
        List all sessions with metadata.

        Returns:
            List of session metadata dictionaries
        """
        sessions = []

        if not self.base_dir.exists():
            return sessions

        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            state_file = session_dir / "session_state.json"

            if not state_file.exists():
                continue

            try:
                # Load session state
                state = self.load_state(session_id)

                # Get file timestamps
                created_at = state_file.stat().st_ctime
                modified_at = state_file.stat().st_mtime

                sessions.append({
                    "session_id": session_id,
                    "chapter_id": state.chapter_id,
                    "turn_index": state.turn_index,
                    "created_at": created_at,
                    "last_updated": modified_at,
                    "end_confirmed": state.end_confirmed,
                })

            except Exception as e:
                # Skip invalid sessions
                continue

        # Sort by last updated (most recent first)
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)

        return sessions

    def append_system_event(
        self,
        session_id: str,
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a system event to the session log."""
        log_path = self._get_session_dir(session_id) / "system_events.jsonl"
        lock_path = log_path.with_suffix(".lock")
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "details": details or {},
        }

        try:
            with FileLock(lock_path, timeout=10):
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append system event: {e}")

    # ========================================================================
    # v3.0 Expert System Storage Methods
    # ========================================================================

    def _ensure_v3_directories(self, session_id: str) -> None:
        """Ensure v3.0 directories exist (for backward compatibility with v2.x sessions).

        Args:
            session_id: Session identifier
        """
        session_dir = self._get_session_dir(session_id)

        # Check if working_files exists
        if not (session_dir / "working_files").exists():
            logger.warning(f"Session {session_id} missing v3.0 directories, creating them now")

        (session_dir / "working_files").mkdir(exist_ok=True)
        (session_dir / "working_files" / "expert_logs").mkdir(exist_ok=True)
        (session_dir / "user_workspace").mkdir(exist_ok=True)
        (session_dir / "code_history").mkdir(exist_ok=True)
        (session_dir / "expert_workspace").mkdir(exist_ok=True)
        (session_dir / "expert_workspace" / "consultations").mkdir(exist_ok=True)
        (session_dir / "expert_reports").mkdir(exist_ok=True)

        # Initialize expert_reports/index.json if missing
        index_path = session_dir / "expert_reports" / "index.json"
        if not index_path.exists():
            self._write_json(index_path, {"reports": []})

        workspace_sources_path = self._get_workspace_sources_path(session_id)
        if not workspace_sources_path.exists():
            self._write_json(workspace_sources_path, {"sources": {}})

    def _validate_workspace_filename(self, filename: str) -> str:
        """Validate workspace filename and return normalized value."""
        normalized = str(filename or "").strip()
        if not normalized:
            raise StorageError("Filename is required")
        if normalized != Path(normalized).name:
            raise StorageError(f"Invalid filename: {filename}")
        if "/" in normalized or "\\" in normalized or ".." in normalized:
            raise StorageError(f"Invalid filename: {filename}")
        return normalized

    def _validate_workspace_extension(self, filename: str) -> None:
        """Validate workspace filename extension."""
        suffix = Path(filename).suffix.lower()
        if suffix not in self.WORKSPACE_ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(self.WORKSPACE_ALLOWED_EXTENSIONS))
            raise StorageError(f"Unsupported workspace file extension: {suffix}. Allowed: {allowed}")

    def _resolve_workspace_file_path(self, session_id: str, filename: str) -> Path:
        """Resolve and validate workspace file path."""
        self._ensure_v3_directories(session_id)
        safe_filename = self._validate_workspace_filename(filename)
        workspace_dir = self._get_workspace_dir(session_id).resolve()
        file_path = (workspace_dir / safe_filename).resolve()

        try:
            file_path.relative_to(workspace_dir)
        except ValueError as exc:
            raise StorageError(f"Path traversal detected: {filename}") from exc

        return file_path

    def _load_workspace_sources(self, session_id: str) -> Dict[str, str]:
        """Load workspace source metadata map."""
        sources_path = self._get_workspace_sources_path(session_id)
        if not sources_path.exists():
            return {}

        payload = self._read_json(sources_path)
        sources = payload.get("sources", {})
        if isinstance(sources, dict):
            return {str(k): str(v) for k, v in sources.items()}
        return {}

    def _save_workspace_sources(self, session_id: str, sources: Dict[str, str]) -> None:
        """Persist workspace source metadata map."""
        self._write_json(self._get_workspace_sources_path(session_id), {"sources": sources})

    def set_workspace_file_source(self, session_id: str, filename: str, source: str) -> None:
        """Set source metadata for a workspace file."""
        safe_filename = self._validate_workspace_filename(filename)
        source_value = source if source in {"bundle", "user"} else "user"
        sources = self._load_workspace_sources(session_id)
        sources[safe_filename] = source_value
        self._save_workspace_sources(session_id, sources)

    def list_workspace_files(self, session_id: str) -> List[Dict]:
        """List files in user_workspace directory."""
        self._ensure_v3_directories(session_id)
        workspace_dir = self._get_workspace_dir(session_id)
        source_map = self._load_workspace_sources(session_id)

        files = []
        for item in workspace_dir.iterdir():
            if not item.is_file():
                continue
            if item.suffix.lower() not in self.WORKSPACE_ALLOWED_EXTENSIONS:
                continue
            stat = item.stat()
            files.append({
                "name": item.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "source": source_map.get(item.name, "user"),
            })

        files.sort(key=lambda x: x["name"])
        return files

    def read_workspace_file(self, session_id: str, filename: str) -> str:
        """Read a file from user_workspace. Raises FileNotFoundError."""
        self._validate_workspace_extension(filename)
        file_path = self._resolve_workspace_file_path(session_id, filename)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(filename)
        return file_path.read_text(encoding="utf-8")

    def write_workspace_file(self, session_id: str, filename: str, content: str) -> Dict:
        """Write content to a file in user_workspace and return file metadata."""
        self._validate_workspace_extension(filename)
        file_path = self._resolve_workspace_file_path(session_id, filename)
        file_path.write_text(content, encoding="utf-8")
        self.set_workspace_file_source(session_id, file_path.name, "user")
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "source": "user",
        }

    def delete_workspace_file(self, session_id: str, filename: str) -> None:
        """Delete a file from user_workspace."""
        self._validate_workspace_extension(filename)
        file_path = self._resolve_workspace_file_path(session_id, filename)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(filename)
        file_path.unlink()

        sources = self._load_workspace_sources(session_id)
        sources.pop(file_path.name, None)
        self._save_workspace_sources(session_id, sources)

    def get_recent_code_executions(self, session_id: str, limit: int = 3) -> List[Dict]:
        """Return the last N code execution records (most recent first)."""
        self._ensure_v3_directories(session_id)
        if limit <= 0:
            return []

        history_dir = self._get_code_history_dir(session_id)
        results: List[Dict[str, Any]] = []
        for history_file in sorted(history_dir.glob("run_*.json"), reverse=True):
            try:
                payload = self._read_json(history_file)
                timestamp = float(payload.get("timestamp", history_file.stat().st_mtime))
                if not math.isfinite(timestamp):
                    raise ValueError("timestamp is not finite")
                exit_code = int(payload.get("exit_code", payload.get("returncode", 0)))
            except Exception as exc:
                logger.warning(f"Failed reading code history file {history_file}: {exc}")
                continue

            results.append(
                {
                    "code": payload.get("code", ""),
                    "stdout": payload.get("stdout", ""),
                    "stderr": payload.get("stderr", ""),
                    "exit_code": exit_code,
                    "timestamp": timestamp,
                }
            )
            if len(results) >= limit:
                break

        return results

    def save_consultation_meta(self, session_id: str, consultation_id: str, meta: Dict) -> None:
        """Save consultation metadata.

        Args:
            session_id: Session identifier
            consultation_id: Consultation identifier
            meta: Consultation metadata dict
        """
        self._ensure_v3_directories(session_id)

        consult_dir = self._get_session_dir(session_id) / "expert_workspace" / "consultations" / consultation_id
        consult_dir.mkdir(parents=True, exist_ok=True)

        meta_path = consult_dir / "meta.json"
        self._write_json(meta_path, meta)

    def save_consulting_envelope(self, session_id: str, consultation_id: str, envelope: Dict) -> None:
        """Save consulting envelope.

        Args:
            session_id: Session identifier
            consultation_id: Consultation identifier
            envelope: Consulting envelope dict
        """
        self._ensure_v3_directories(session_id)

        consult_dir = self._get_session_dir(session_id) / "expert_workspace" / "consultations" / consultation_id
        consult_dir.mkdir(parents=True, exist_ok=True)

        envelope_path = consult_dir / "consulting_envelope.json"
        self._write_json(envelope_path, envelope)

    def append_consultation_transcript(self, session_id: str, consultation_id: str, round_entry: Dict) -> None:
        """Append a round entry to consultation transcript (JSONL format).

        Args:
            session_id: Session identifier
            consultation_id: Consultation identifier
            round_entry: Round entry dict (one line of JSONL)
        """
        self._ensure_v3_directories(session_id)

        consult_dir = self._get_session_dir(session_id) / "expert_workspace" / "consultations" / consultation_id
        consult_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = consult_dir / "transcript.jsonl"

        # Append to JSONL file
        with open(transcript_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(round_entry, ensure_ascii=False) + '\n')

    def save_expert_output(self, session_id: str, consultation_id: str, expert_output: Dict) -> None:
        """Save expert structured output.

        Args:
            session_id: Session identifier
            consultation_id: Consultation identifier
            expert_output: Expert output dict
        """
        self._ensure_v3_directories(session_id)

        consult_dir = self._get_session_dir(session_id) / "expert_workspace" / "consultations" / consultation_id
        consult_dir.mkdir(parents=True, exist_ok=True)

        output_path = consult_dir / "expert_output.json"
        self._write_json(output_path, expert_output)

    def load_expert_output(self, session_id: str, consultation_id: str) -> Optional[Dict]:
        """Load expert output.

        Args:
            session_id: Session identifier
            consultation_id: Consultation identifier

        Returns:
            Expert output dict if exists, None otherwise
        """
        self._ensure_v3_directories(session_id)

        output_path = self._get_session_dir(session_id) / "expert_workspace" / "consultations" / consultation_id / "expert_output.json"

        if not output_path.exists():
            return None

        return self._read_json(output_path)

    def save_skill_call_log(self, session_id: str, expert_id: str, consultation_id: str,
                           skill_call_counter: int, log_data: Dict) -> None:
        """Save skill call log.

        Args:
            session_id: Session identifier
            expert_id: Expert identifier
            consultation_id: Consultation identifier
            skill_call_counter: Skill call counter (for filename)
            log_data: Skill call log dict
        """
        self._ensure_v3_directories(session_id)

        log_dir = self._get_session_dir(session_id) / "working_files" / "expert_logs" / expert_id / consultation_id
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"skill_call_{skill_call_counter:04d}.json"
        self._write_json(log_path, log_data)

    def save_skill_stdout_stderr(self, session_id: str, expert_id: str, consultation_id: str,
                                skill_call_counter: int, stdout: Optional[str], stderr: Optional[str]) -> None:
        """Save skill stdout/stderr output.

        Args:
            session_id: Session identifier
            expert_id: Expert identifier
            consultation_id: Consultation identifier
            skill_call_counter: Skill call counter
            stdout: Standard output (if any)
            stderr: Standard error (if any)
        """
        self._ensure_v3_directories(session_id)

        log_dir = self._get_session_dir(session_id) / "working_files" / "expert_logs" / expert_id / consultation_id
        log_dir.mkdir(parents=True, exist_ok=True)

        if stdout:
            stdout_path = log_dir / f"stdout_{skill_call_counter:04d}.txt"
            stdout_path.write_text(stdout, encoding='utf-8')

        if stderr:
            stderr_path = log_dir / f"stderr_{skill_call_counter:04d}.txt"
            stderr_path.write_text(stderr, encoding='utf-8')

    def save_expert_report(self, session_id: str, report: Dict) -> None:
        """Save expert report and update index.

        Args:
            session_id: Session identifier
            report: Expert report dict (must include report_id)
        """
        self._ensure_v3_directories(session_id)

        reports_dir = self._get_session_dir(session_id) / "expert_reports"
        report_id = report["report_id"]

        # Save report file
        report_path = reports_dir / f"{report_id}.json"
        self._write_json(report_path, report)

        # Update index
        index_path = reports_dir / "index.json"
        index_data = self._read_json(index_path)

        # Add report metadata to index
        report_metadata = {
            "report_id": report["report_id"],
            "consultation_id": report["consultation_id"],
            "user_turn_index": report["user_turn_index"],
            "scenario_id": report["scenario_id"],
            "title": report["title"],
            "timestamp": report["timestamp"],
            "binding": report["binding"],
            "summary": report["summary"]
        }

        index_data["reports"].append(report_metadata)
        self._write_json(index_path, index_data)

    def load_expert_reports_index(self, session_id: str) -> Dict:
        """Load expert reports index.

        Args:
            session_id: Session identifier

        Returns:
            Expert reports index dict
        """
        self._ensure_v3_directories(session_id)

        index_path = self._get_session_dir(session_id) / "expert_reports" / "index.json"

        if not index_path.exists():
            return {"reports": []}

        return self._read_json(index_path)

    def load_expert_report(self, session_id: str, report_id: str) -> Optional[Dict]:
        """Load a specific expert report.

        Args:
            session_id: Session identifier
            report_id: Report identifier

        Returns:
            Expert report dict if exists, None otherwise
        """
        self._ensure_v3_directories(session_id)

        report_path = self._get_session_dir(session_id) / "expert_reports" / f"{report_id}.json"

        if not report_path.exists():
            return None

        return self._read_json(report_path)

    def get_working_files_path(self, session_id: str) -> Path:
        """Get path to working_files directory.

        Args:
            session_id: Session identifier

        Returns:
            Path to working_files directory
        """
        self._ensure_v3_directories(session_id)
        return self._get_session_dir(session_id) / "working_files"

    def get_workspace_path(self, session_id: str) -> Path:
        """Get path to user_workspace directory."""
        self._ensure_v3_directories(session_id)
        return self._get_workspace_dir(session_id)

    def list_working_files(self, session_id: str) -> List[str]:
        """List all files in working_files directory (excluding expert_logs).

        Args:
            session_id: Session identifier

        Returns:
            List of file names
        """
        self._ensure_v3_directories(session_id)

        working_files_dir = self.get_working_files_path(session_id)
        files = []

        for item in working_files_dir.iterdir():
            if item.is_file():
                files.append(item.name)

        return files

    # ========================================================================
    # v3.2.0 File Upload Methods
    # ========================================================================

    def save_uploaded_file(self, session_id: str, file_content: bytes, filename: str) -> Path:
        """Save uploaded file to working_files directory.

        Args:
            session_id: Session identifier
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Path to saved file

        Raises:
            StorageError: If file save fails
        """
        self._ensure_v3_directories(session_id)

        # Sanitize filename (prevent path traversal)
        safe_filename = Path(filename).name

        working_files_dir = self.get_working_files_path(session_id)
        target_path = working_files_dir / safe_filename

        # Handle duplicate filenames by adding timestamp
        if target_path.exists():
            import time
            timestamp = int(time.time())
            name_parts = safe_filename.rsplit('.', 1)
            if len(name_parts) == 2:
                safe_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            else:
                safe_filename = f"{safe_filename}_{timestamp}"
            target_path = working_files_dir / safe_filename

        try:
            # Write file
            with open(target_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"Saved uploaded file: {target_path}")
            return target_path

        except Exception as e:
            raise StorageError(f"Failed to save uploaded file {filename}: {e}")

    def get_uploaded_files_metadata(self, session_id: str) -> List[Dict]:
        """Get metadata for all uploaded files in working_files directory.

        Args:
            session_id: Session identifier

        Returns:
            List of file metadata dictionaries with keys:
            - filename: File name
            - size: File size in bytes
            - upload_time: Upload timestamp
            - file_type: File extension
        """
        self._ensure_v3_directories(session_id)

        working_files_dir = self.get_working_files_path(session_id)
        files_metadata = []

        for item in working_files_dir.iterdir():
            # Skip directories (like expert_logs)
            if not item.is_file():
                continue

            try:
                stat = item.stat()
                files_metadata.append({
                    "filename": item.name,
                    "size": stat.st_size,
                    "upload_time": stat.st_mtime,
                    "file_type": item.suffix
                })
            except Exception as e:
                logger.warning(f"Failed to get metadata for {item}: {e}")
                continue

        # Sort by upload time (most recent first)
        files_metadata.sort(key=lambda x: x["upload_time"], reverse=True)

        return files_metadata

    def delete_uploaded_file(self, session_id: str, filename: str) -> bool:
        """Delete uploaded file from working_files directory.

        Args:
            session_id: Session identifier
            filename: File name to delete

        Returns:
            True if file was deleted, False if file not found

        Raises:
            StorageError: If deletion fails or path traversal detected
        """
        self._ensure_v3_directories(session_id)

        # Sanitize filename (prevent path traversal)
        safe_filename = Path(filename).name

        # Ensure filename matches (no path components)
        if safe_filename != filename:
            raise StorageError(f"Invalid filename: {filename}")

        working_files_dir = self.get_working_files_path(session_id)
        file_path = working_files_dir / safe_filename

        # Verify path is within working_files directory
        try:
            file_path.resolve().relative_to(working_files_dir.resolve())
        except ValueError:
            raise StorageError(f"Path traversal detected: {filename}")

        # Delete file if exists
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            except Exception as e:
                raise StorageError(f"Failed to delete file {filename}: {e}")

        return False
