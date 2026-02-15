from pathlib import Path

from fastapi.testclient import TestClient

import sidecar.main as sidecar_main
from sidecar.models.schemas import SessionConstraints, SessionState
from sidecar.services.storage import Storage
from sidecar.services.user_code_runner import UserCodeRunner


class MinimalOrchestrator:
    def __init__(self, storage: Storage, session_id: str = "sess-1"):
        self.storage = storage
        self._session_id = session_id

    async def create_session(self, chapter_id: str) -> str:
        if not self.storage.session_exists(self._session_id):
            initial_state = SessionState(
                session_id=self._session_id,
                chapter_id=chapter_id,
                subtask_status={},
                constraints=SessionConstraints(),
            )
            self.storage.create_session(self._session_id, chapter_id, initial_state)
        return self._session_id


def _create_session(storage: Storage, session_id: str = "sess-1", chapter_id: str = "course_1/ch_01") -> None:
    initial_state = SessionState(
        session_id=session_id,
        chapter_id=chapter_id,
        subtask_status={},
        constraints=SessionConstraints(),
    )
    storage.create_session(session_id, chapter_id, initial_state)


def test_session_new_copies_bundle_scripts_and_datasets(monkeypatch, tmp_path: Path):
    sidecar_main.session_orchestrators.clear()
    sessions_root = tmp_path / "sessions"
    storage = Storage(base_dir=str(sessions_root))
    orchestrator = MinimalOrchestrator(storage=storage, session_id="sess-bundle")

    bundle_root = tmp_path / "bundle"
    chapter_dir = bundle_root / "content" / "curriculum" / "courses" / "course_1" / "chapters" / "ch_01"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    (chapter_dir / "chapter_context.md").write_text("# chapter", encoding="utf-8")
    (chapter_dir / "task_list.md").write_text("- task", encoding="utf-8")
    (chapter_dir / "task_completion_principles.md").write_text("- principles", encoding="utf-8")

    scripts_dir = chapter_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "starter.py").write_text("print('starter')\n", encoding="utf-8")
    (scripts_dir / "config.json").write_text('{"mode": "demo"}\n', encoding="utf-8")
    (scripts_dir / "solution.py").write_text("print('answer')\n", encoding="utf-8")

    datasets_dir = chapter_dir / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "sample.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(
        sidecar_main,
        "_resolve_runtime_paths",
        lambda chapter_id, desktop_context: (tmp_path / "curriculum", tmp_path / "experts", tmp_path / "agents"),
    )
    monkeypatch.setattr(sidecar_main, "_build_orchestrator", lambda **kwargs: orchestrator)

    with TestClient(sidecar_main.app) as client:
        response = client.post(
            "/api/session/new",
            json={
                "chapter_id": "course_1/ch_01",
                "desktop_context": {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"]["has_starter_code"] is True
    assert payload["workspace"]["has_datasets"] is True
    assert {"name": "starter.py", "source": "bundle"} in payload["workspace"]["files"]
    assert {"name": "config.json", "source": "bundle"} in payload["workspace"]["files"]
    assert {"name": "sample.csv", "source": "bundle"} in payload["workspace"]["files"]
    assert all(item["name"] != "solution.py" for item in payload["workspace"]["files"])

    workspace_path = storage.get_workspace_path("sess-bundle")
    working_files_path = storage.get_working_files_path("sess-bundle")
    assert (workspace_path / "starter.py").exists()
    assert (workspace_path / "config.json").exists()
    assert not (workspace_path / "solution.py").exists()
    assert (working_files_path / "sample.csv").exists()


def test_workspace_file_crud_and_security(monkeypatch, tmp_path: Path):
    storage = Storage(base_dir=str(tmp_path / "sessions"))
    _create_session(storage, session_id="sess-work")
    orchestrator = MinimalOrchestrator(storage=storage, session_id="sess-work")
    monkeypatch.setattr(sidecar_main, "_get_orchestrator", lambda _sid: orchestrator)

    with TestClient(sidecar_main.app) as client:
        write_resp = client.put(
            "/api/session/sess-work/workspace/files/lesson.py",
            json={"content": "print('hi')\n"},
        )
        assert write_resp.status_code == 200
        assert write_resp.json()["source"] == "user"

        list_resp = client.get("/api/session/sess-work/workspace/files")
        assert list_resp.status_code == 200
        assert any(item["name"] == "lesson.py" for item in list_resp.json()["files"])

        read_resp = client.get("/api/session/sess-work/workspace/files/lesson.py")
        assert read_resp.status_code == 200
        assert read_resp.json()["content"] == "print('hi')\n"

        traversal_resp = client.put(
            "/api/session/sess-work/workspace/files/..%5Csecret.py",
            json={"content": "bad"},
        )
        assert traversal_resp.status_code == 400

        ext_resp = client.put(
            "/api/session/sess-work/workspace/files/data.csv",
            json={"content": "x,y\n1,2\n"},
        )
        assert ext_resp.status_code == 400

        delete_resp = client.delete("/api/session/sess-work/workspace/files/lesson.py")
        assert delete_resp.status_code == 204

        missing_resp = client.get("/api/session/sess-work/workspace/files/lesson.py")
        assert missing_resp.status_code == 404


def test_workspace_write_rejects_large_file(monkeypatch, tmp_path: Path):
    storage = Storage(base_dir=str(tmp_path / "sessions"))
    _create_session(storage, session_id="sess-large")
    orchestrator = MinimalOrchestrator(storage=storage, session_id="sess-large")
    monkeypatch.setattr(sidecar_main, "_get_orchestrator", lambda _sid: orchestrator)

    oversized_content = "a" * (1024 * 1024 + 1)
    with TestClient(sidecar_main.app) as client:
        response = client.put(
            "/api/session/sess-large/workspace/files/big.py",
            json={"content": oversized_content},
        )

    assert response.status_code == 400


def test_code_history_saved_and_retrievable(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    runner = UserCodeRunner(sessions_root=sessions_root)

    result = runner.run_python(session_id="sess-history", code="print('hello')\n")
    assert result["success"] is True

    storage = Storage(base_dir=str(sessions_root))
    records = storage.get_recent_code_executions("sess-history", limit=1)
    assert len(records) == 1
    assert "print('hello')" in records[0]["code"]
    assert "hello" in records[0]["stdout"]
    assert records[0]["exit_code"] == 0


def test_code_history_skips_nan_timestamp(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    storage = Storage(base_dir=str(sessions_root))
    _create_session(storage, session_id="sess-history-bad")

    history_dir = sessions_root / "sess-history-bad" / "code_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "run_1.json").write_text(
        '{"timestamp": "nan", "exit_code": 0, "code": "print(1)", "stdout": "1", "stderr": ""}',
        encoding="utf-8",
    )
    (history_dir / "run_2.json").write_text(
        '{"timestamp": 1700000000.0, "exit_code": 0, "code": "print(2)", "stdout": "2", "stderr": ""}',
        encoding="utf-8",
    )

    records = storage.get_recent_code_executions("sess-history-bad", limit=5)
    assert len(records) == 1
    assert "print(2)" in records[0]["code"]
