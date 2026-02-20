"""
E2E integration test: sidecar session create → send message → SSE stream → verify reply.

Run with:
    RUN_INTEGRATION=1 uv run pytest tests/test_e2e_sidecar_interaction.py -v -s

The test skips gracefully when:
- RUN_INTEGRATION env var is not set
- The chapter bundle tar.gz is missing from /tmp/ch1_intro_bundle.tar.gz
- The sidecar is not running at SIDECAR_URL (default http://127.0.0.1:8000)
- The LLM key is not configured (sidecar returns 500 on session create)
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUNDLE_PATH = Path("/tmp/ch1_intro_bundle.tar.gz")
COURSE_ID = "a2159fb9-5973-4cda-be1c-59a190a91d10"
CHAPTER_NAME = "ch1_intro"
CHAPTER_ID = f"{COURSE_ID}/{CHAPTER_NAME}"

SIDECAR_URL = os.environ.get("SIDECAR_URL", "http://127.0.0.1:8000")

# SSE "done-equivalent" event type used by the sidecar (NOT "done" — it is "complete")
STREAM_DONE_TYPE = "complete"

# All terminal event types that should stop stream collection
STREAM_TERMINAL_TYPES = {STREAM_DONE_TYPE, "error"}

# Timeout (seconds) for individual HTTP requests during the E2E test
HTTP_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_integration() -> bool:
    return os.environ.get("RUN_INTEGRATION", "").strip() not in ("", "0", "false", "False")


def _sidecar_is_running() -> bool:
    """Return True if the sidecar responds to /health."""
    try:
        resp = httpx.get(f"{SIDECAR_URL}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _parse_sse_events(raw_text: str) -> list[dict]:
    """
    Parse a raw SSE response body into a list of decoded JSON event dicts.

    Each SSE event looks like:
        data: {"type": "companion_chunk", "content": "H"}\n\n
    """
    events: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    print(f"[warn] Skipping malformed SSE line: {payload!r}")
    return events


# ---------------------------------------------------------------------------
# Pytest skip conditions (evaluated at collection time where possible,
# but sidecar availability is checked inside the test to avoid slow collection)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not _is_integration(),
    reason="Set RUN_INTEGRATION=1 to run E2E sidecar tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    """Extract the chapter bundle tar.gz to a temporary directory."""
    if not BUNDLE_PATH.exists():
        pytest.skip(
            f"Chapter bundle not found at {BUNDLE_PATH}. "
            "Build it first with the bundle script."
        )
    tmp = tmp_path_factory.mktemp("ch1_intro_bundle")
    with tarfile.open(BUNDLE_PATH, "r:gz") as tf:
        try:
            tf.extractall(tmp, filter='data')
        except TypeError:
            tf.extractall(tmp)  # Python < 3.12
    return tmp


@pytest.fixture(scope="module")
def sidecar_client():
    """Return a synchronous httpx client pointed at the running sidecar."""
    if not _sidecar_is_running():
        pytest.skip(
            f"Sidecar not running at {SIDECAR_URL}. "
            "Start it with: uvicorn sidecar.main:app --port 8000"
        )
    with httpx.Client(base_url=SIDECAR_URL, timeout=HTTP_TIMEOUT) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2ESidecarInteraction:
    """Full round-trip: create session → send message → read SSE stream."""

    def test_create_session_with_bundle(self, sidecar_client, bundle_dir):
        """
        POST /api/session/new with a desktop_context pointing at the extracted
        chapter bundle. The sidecar must return a session_id and a non-empty
        initial_message.
        """
        payload = {
            "chapter_id": CHAPTER_ID,
            "desktop_context": {
                "bundle_paths": {
                    "chapter_bundle_path": str(bundle_dir),
                },
                "prompt_sources": {},
                "chapter_scope": {},
            },
        }

        try:
            resp = sidecar_client.post("/api/session/new", json=payload)
        except httpx.RequestError as exc:
            pytest.skip(f"Sidecar connection error: {exc}")

        if resp.status_code == 500:
            body = resp.text
            if "LLM" in body or "API" in body or "key" in body.lower() or "token" in body.lower():
                pytest.skip("LLM key not configured in sidecar — skipping E2E test")
            pytest.fail(
                f"Session creation returned 500: {body}"
            )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

        data = resp.json()
        assert "session_id" in data, f"Response missing session_id: {data}"
        assert data["session_id"], "session_id is empty"
        assert "initial_message" in data, f"Response missing initial_message: {data}"
        assert data["initial_message"], "initial_message is empty"

        # Store on the class for subsequent tests in this class
        self.__class__._session_id = data["session_id"]
        self.__class__._initial_message = data["initial_message"]

    def test_initial_message_is_non_empty(self, sidecar_client, bundle_dir):
        """Verify the initial greeting returned during session creation is non-empty."""
        if not hasattr(self.__class__, "_initial_message"):
            pytest.skip("Depends on test_create_session_with_bundle")
        assert len(self._initial_message.strip()) > 0, (
            "initial_message must not be empty or whitespace"
        )

    def test_send_message_sse_stream(self, sidecar_client, bundle_dir):
        """
        POST /api/session/{session_id}/message/stream and collect SSE events until
        the 'complete' event (the sidecar's done-equivalent).

        Assertions:
        - The stream contains at least one 'companion_chunk' or 'companion_complete' event
        - The stream ends with a 'complete' event
        - The concatenated companion content is non-empty
        """
        if not hasattr(self.__class__, "_session_id"):
            pytest.skip("Depends on test_create_session_with_bundle")

        session_id = self._session_id
        student_message = "你好！我想开始学习今天的内容，我应该从哪里开始？"

        payload = {"message": student_message}

        try:
            resp = sidecar_client.post(
                f"/api/session/{session_id}/message/stream",
                json=payload,
                headers={"Accept": "text/event-stream"},
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            pytest.skip(f"Sidecar connection error during streaming: {exc}")

        assert resp.status_code == 200, (
            f"Expected 200 from stream endpoint, got {resp.status_code}: {resp.text}"
        )

        events = _parse_sse_events(resp.text)
        assert events, "SSE stream returned no events"

        event_types = [e.get("type") for e in events]

        # Must receive the done-equivalent 'complete' event
        assert STREAM_DONE_TYPE in event_types, (
            f"Stream did not contain a '{STREAM_DONE_TYPE}' event. "
            f"Event types received: {event_types}"
        )

        # Collect companion content
        companion_text = "".join(
            e.get("content", "")
            for e in events
            if e.get("type") == "companion_chunk"
        )

        # If companion_complete was emitted, the companion definitely ran
        has_companion_complete = "companion_complete" in event_types
        has_companion_chunks = "companion_chunk" in event_types

        assert has_companion_complete or has_companion_chunks, (
            f"No companion_complete or companion_chunk events in stream. "
            f"Event types: {event_types}"
        )

        assert companion_text.strip(), (
            f"Companion reply is empty after joining all companion_chunk events. "
            f"All events: {events[:20]}"  # show first 20 to avoid log spam
        )

        # Store for optional inspection
        self.__class__._companion_text = companion_text
        self.__class__._stream_events = events

    def test_stream_has_no_error_event(self, sidecar_client, bundle_dir):
        """Verify the SSE stream did not contain an 'error' event."""
        if not hasattr(self.__class__, "_stream_events"):
            pytest.skip("Depends on test_send_message_sse_stream")

        error_events = [e for e in self._stream_events if e.get("type") == "error"]
        assert not error_events, (
            f"SSE stream contained error event(s): {error_events}"
        )

    def test_complete_event_has_turn_index(self, sidecar_client, bundle_dir):
        """
        The 'complete' event must include a 'turn_index' field (per main.py implementation).
        """
        if not hasattr(self.__class__, "_stream_events"):
            pytest.skip("Depends on test_send_message_sse_stream")

        complete_events = [
            e for e in self._stream_events if e.get("type") == STREAM_DONE_TYPE
        ]
        assert complete_events, f"No '{STREAM_DONE_TYPE}' event found in stream"

        complete_event = complete_events[-1]
        assert "turn_index" in complete_event, (
            f"'complete' event is missing 'turn_index': {complete_event}"
        )
        assert isinstance(complete_event["turn_index"], int), (
            f"'turn_index' must be an integer, got: {type(complete_event['turn_index'])}"
        )


# ---------------------------------------------------------------------------
# Standalone smoke test (independent of class state)
# ---------------------------------------------------------------------------

def test_health_endpoint():
    """Sanity check: /health returns 200 when the sidecar is running."""
    if not _sidecar_is_running():
        pytest.skip(f"Sidecar not running at {SIDECAR_URL}")

    resp = httpx.get(f"{SIDECAR_URL}/health", timeout=5.0)
    assert resp.status_code == 200, f"/health returned {resp.status_code}"
