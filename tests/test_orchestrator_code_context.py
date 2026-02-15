import asyncio
import json

from sidecar.models.schemas import (
    InstructionPacket,
    MemoDigest,
    MemoResult,
    SessionConstraints,
    SessionState,
    TurnOutcome,
)
from sidecar.services.orchestrator import Orchestrator
from sidecar.services.storage import Storage


class FakeMemoryManager:
    def build_memory_sections(self, session_id: str, user_message_length: int):
        return {"long_term": "", "mid_term": "", "recent_turns": ""}

    async def update_after_turn(self, session_id, turn_index, user_message, companion_response, turn_outcome):
        return None


class CapturingAgentRunner:
    def __init__(self):
        self.calls = []

    async def run_companion(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "ok",
            TurnOutcome(
                what_user_attempted="attempted",
                what_user_observed="observed",
                ca_teaching_mode="socratic",
                ca_next_suggestion="next",
                checkpoint_reached=False,
                blocker_type="none",
                student_sentiment="engaged",
                evidence_for_subtasks=[],
            ),
        )

    async def run_memo(self, **kwargs):
        return MemoResult(
            updated_report=kwargs["current_report"],
            digest=MemoDigest(
                key_observations=[],
                student_struggles=[],
                student_strengths=[],
                student_sentiment="engaged",
                blocker_type="none",
                progress_delta="none",
                diagnostic_log=[],
            ),
            error_entries=[],
        )


def _bootstrap_session(storage: Storage, session_id: str = "sess-1"):
    state = SessionState(
        session_id=session_id,
        chapter_id="course_1/ch_01",
        turn_index=0,
        subtask_status={},
        constraints=SessionConstraints(),
    )
    storage.create_session(session_id, state.chapter_id, state)
    storage.save_instruction_packet(
        session_id,
        InstructionPacket(
            current_focus="focus",
            guidance_for_ca="guide",
            must_check=["check"],
            nice_check=[],
            instruction_version=1,
            lock_until="checkpoint_reached",
            allow_setup_helper_code=False,
            setup_helper_scope="none",
            task_type="core",
        ),
    )


def _build_orchestrator(tmp_path):
    storage = Storage(base_dir=str(tmp_path / "sessions"))
    _bootstrap_session(storage)
    runner = CapturingAgentRunner()
    memory = FakeMemoryManager()
    orchestrator = Orchestrator(
        storage=storage,
        agent_runner=runner,
        memory_manager=memory,
        curriculum_dir=str(tmp_path / "curriculum"),
        experts_dir=str(tmp_path / "experts"),
    )
    orchestrator._load_chapter_content = lambda chapter_id: {
        "chapter_context": "",
        "task_list": "",
        "task_completion_principles": "",
        "interaction_protocol": "",
        "socratic_vs_direct": "",
    }
    orchestrator._load_templates = lambda: {
        "dynamic_report_template": "",
        "student_error_summary_template": "",
        "final_learning_report_template": "",
    }
    orchestrator._load_available_experts_info = lambda chapter_id: ""
    return orchestrator, storage, runner


def test_process_turn_skips_malformed_history_records(tmp_path):
    orchestrator, storage, runner = _build_orchestrator(tmp_path)
    history_dir = storage.get_workspace_path("sess-1").parent / "code_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "run_1.json").write_text(
        json.dumps({"timestamp": "bad", "exit_code": "oops", "code": "print('x')"}),
        encoding="utf-8",
    )

    response = asyncio.run(orchestrator.process_turn("sess-1", "hello"))
    assert response == "ok"
    assert runner.calls
    assert runner.calls[0]["recent_code_executions"] == ""


def test_process_turn_injects_recent_code_context(tmp_path):
    orchestrator, storage, runner = _build_orchestrator(tmp_path)
    history_dir = storage.get_workspace_path("sess-1").parent / "code_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "run_2.json").write_text(
        json.dumps(
            {
                "timestamp": 1700000000.0,
                "exit_code": 0,
                "code": "print('hello world')",
                "stdout": "hello world",
                "stderr": "",
            }
        ),
        encoding="utf-8",
    )

    _ = asyncio.run(orchestrator.process_turn("sess-1", "hello"))
    prompt_block = runner.calls[0]["recent_code_executions"]
    assert "学生最近的代码执行" in prompt_block
    assert "print('hello world')" in prompt_block


def test_process_turn_stream_forwards_recent_code_context(monkeypatch, tmp_path):
    orchestrator, storage, _runner = _build_orchestrator(tmp_path)
    history_dir = storage.get_workspace_path("sess-1").parent / "code_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "run_3.json").write_text(
        json.dumps(
            {
                "timestamp": 1700000000.0,
                "exit_code": 0,
                "code": "print('stream')",
                "stdout": "stream",
                "stderr": "",
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        yield {"type": "complete"}

    import sidecar.services.streaming as streaming_module

    monkeypatch.setattr(streaming_module, "process_turn_stream", fake_stream)

    async def _collect():
        events = []
        async for event in orchestrator.process_turn_stream("sess-1", "hello"):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    assert events and events[0]["type"] == "complete"
    assert "学生最近的代码执行" in captured["recent_code_executions_text"]
