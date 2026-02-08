from types import SimpleNamespace
from pathlib import Path

from fastapi.testclient import TestClient

import sidecar.main as sidecar_main


class FakeStorage:
    def __init__(self, exists=True):
        self.exists = exists
        self._turn_history = []
        self._state = SimpleNamespace(turn_index=1)
        self._report = "dynamic report content"

    def session_exists(self, _session_id):
        return self.exists

    def load_turn_history(self, _session_id):
        return self._turn_history

    def load_state(self, _session_id):
        return self._state

    def load_dynamic_report(self, _session_id):
        return self._report


class FakeOrchestrator:
    def __init__(self, storage=None):
        self.storage = storage or FakeStorage()

    async def create_session(self, _chapter_id):
        return "sess-1"

    async def end_session(self, _session_id):
        return "final report"

    async def process_turn_stream(self, _session_id, _message):
        yield {"type": "companion_start"}
        yield {"type": "companion_chunk", "content": "h"}
        yield {"type": "companion_complete"}


def test_create_session_accepts_desktop_context(monkeypatch):
    sidecar_main.session_orchestrators.clear()
    fake_orchestrator = FakeOrchestrator(storage=FakeStorage(exists=True))

    monkeypatch.setattr(
        sidecar_main,
        "_resolve_runtime_paths",
        lambda chapter_id, desktop_context: (
            Path("/tmp/curriculum"),
            Path("/tmp/experts"),
            Path("/tmp/agents"),
        ),
    )
    monkeypatch.setattr(sidecar_main, "_build_orchestrator", lambda **kwargs: fake_orchestrator)

    with TestClient(sidecar_main.app) as client:
        response = client.post(
            "/api/session/new",
            json={
                "chapter_id": "course_1/ch_01",
                "desktop_context": {
                    "bundle_paths": {
                        "chapter_bundle_path": "/tmp/chapter_bundle",
                        "app_agents_path": "/tmp/app_agents",
                    }
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sess-1"
    assert payload["initial_message"] == "你好！欢迎来到学习平台。"
    assert "sess-1" in sidecar_main.session_orchestrators


def test_stream_nonexistent_session_emits_structured_error(monkeypatch):
    fake_orchestrator = FakeOrchestrator(storage=FakeStorage(exists=False))
    monkeypatch.setattr(sidecar_main, "_get_orchestrator", lambda _sid: fake_orchestrator)

    with TestClient(sidecar_main.app) as client:
        response = client.post("/api/session/missing/message/stream", json={"message": "hello"})

    assert response.status_code == 200
    text = response.text
    assert '"type": "error"' in text
    assert "会话不存在" in text


def test_dynamic_report_endpoint_keeps_demo_path_name(monkeypatch):
    fake_storage = FakeStorage(exists=True)
    fake_storage._report = "report-v1"
    fake_orchestrator = FakeOrchestrator(storage=fake_storage)
    monkeypatch.setattr(sidecar_main, "_get_orchestrator", lambda _sid: fake_orchestrator)

    with TestClient(sidecar_main.app) as client:
        response = client.get("/api/session/sess-1/dynamic_report")

    assert response.status_code == 200
    assert response.json() == {"report": "report-v1"}


def test_stream_success_includes_complete_event(monkeypatch):
    fake_storage = FakeStorage(exists=True)
    fake_storage._state = SimpleNamespace(turn_index=7)
    fake_orchestrator = FakeOrchestrator(storage=fake_storage)
    monkeypatch.setattr(sidecar_main, "_get_orchestrator", lambda _sid: fake_orchestrator)

    with TestClient(sidecar_main.app) as client:
        response = client.post("/api/session/sess-1/message/stream", json={"message": "hello"})

    assert response.status_code == 200
    text = response.text
    assert '"type": "companion_chunk"' in text
    assert '"type": "complete"' in text
    assert '"turn_index": 7' in text


def test_resolve_curriculum_dir_uses_chapter_bundle_overlay(monkeypatch, tmp_path):
    default_curriculum = tmp_path / "default_curriculum"
    (default_curriculum / "_templates").mkdir(parents=True, exist_ok=True)
    (default_curriculum / "_templates" / "dynamic_report_template.md").write_text("template", encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    chapter_dir = bundle_root / "content" / "curriculum" / "courses" / "course_1" / "chapters" / "ch_01"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    (chapter_dir / "chapter_context.md").write_text("# chapter", encoding="utf-8")
    (chapter_dir / "task_list.md").write_text("- task", encoding="utf-8")
    (chapter_dir / "task_completion_principles.md").write_text("- principle", encoding="utf-8")
    (chapter_dir / "interaction_protocol.md").write_text("- protocol", encoding="utf-8")
    (chapter_dir / "socratic_vs_direct.md").write_text("- mode", encoding="utf-8")

    monkeypatch.setattr(sidecar_main, "default_curriculum_dir", default_curriculum)
    monkeypatch.setattr(sidecar_main, "overlay_root", tmp_path / "sessions" / "_chapter_overlays")

    resolved = sidecar_main._resolve_curriculum_dir(
        "course_1/ch_01",
        {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
    )

    target_chapter = resolved / "courses" / "course_1" / "chapters" / "ch_01"
    assert resolved.exists()
    assert target_chapter.exists()
    assert (target_chapter / "chapter_context.md").exists()
    assert (target_chapter / "interaction_protocol.md").exists()
    assert (resolved / "_templates" / "dynamic_report_template.md").exists()


def test_run_user_code_endpoint(monkeypatch):
    fake_orchestrator = FakeOrchestrator(storage=FakeStorage(exists=True))
    monkeypatch.setattr(sidecar_main, "_get_orchestrator", lambda _sid: fake_orchestrator)
    monkeypatch.setattr(
        sidecar_main.user_code_runner,
        "run_python",
        lambda session_id, code, timeout_seconds: {
            "success": True,
            "stdout": "ok",
            "stderr": "",
            "returncode": 0,
            "execution_time_ms": 10,
        },
    )

    with TestClient(sidecar_main.app) as client:
        response = client.post(
            "/api/session/sess-1/code/run",
            json={"code": "print('ok')", "timeout_seconds": 5},
        )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["stdout"] == "ok"
