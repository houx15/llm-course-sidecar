"""Layered memory manager for session context."""

import json
import logging
from typing import Dict, List, Optional

from ..config import settings
from ..models.schemas import MemoryChunk, MemoryState, TurnOutcome
from .json_utils import extract_json_from_response
from .llm_client import LLMError, get_llm_client
from .storage import Storage

logger = logging.getLogger(__name__)


class MemoryManager:
    """Build and update layered memory for a session."""

    def __init__(self, storage: Storage, llm_client=None):
        self.storage = storage
        self.llm_client = llm_client or get_llm_client()

    def initialize_session(self, session_id: str) -> None:
        """Initialize memory state for a new session."""
        memory_state = MemoryState()
        self.storage.save_memory_state(session_id, memory_state)

    def build_memory_sections(
        self,
        session_id: str,
        user_message_length: int,
    ) -> Dict[str, str]:
        """
        Build memory sections for CA prompt with budget control.

        Returns:
            Dict with keys: long_term, mid_term, recent_turns
        """
        memory_state = self.storage.load_memory_state(session_id)

        total_budget = self._calculate_memory_budget(user_message_length)
        long_term_budget, short_term_budget, mid_term_budget = self._allocate_budgets(total_budget)

        long_term = self._truncate(memory_state.long_term_summary, long_term_budget)
        mid_term = self._format_mid_term(memory_state.mid_term_summaries, mid_term_budget)

        recent_turns = self.storage.load_recent_turns(
            session_id,
            settings.memory_recent_turns,
        )
        recent_text = self._format_recent_turns(
            recent_turns,
            settings.memory_recent_turn_chars,
            short_term_budget,
        )

        return {
            "long_term": long_term or "(none)",
            "mid_term": mid_term or "(none)",
            "recent_turns": recent_text or "(none)",
        }

    async def update_after_turn(
        self,
        session_id: str,
        turn_index: int,
        user_message: str,
        companion_response: str,
        turn_outcome: TurnOutcome,
    ) -> None:
        """Update memory after a turn is saved."""
        try:
            memory_state = self.storage.load_memory_state(session_id)
        except Exception as e:
            logger.warning(f"Failed to load memory state: {e}")
            return

        if memory_state.last_updated_turn >= turn_index:
            return

        updated_summary = await self._update_long_term_summary(
            memory_state.long_term_summary,
            user_message,
            companion_response,
            turn_outcome,
        )
        if updated_summary is not None:
            memory_state.long_term_summary = updated_summary

        memory_state.last_updated_turn = turn_index

        await self._maybe_summarize_mid_term(
            session_id=session_id,
            memory_state=memory_state,
            current_turn_index=turn_index,
        )

        try:
            self.storage.save_memory_state(session_id, memory_state)
        except Exception as e:
            logger.warning(f"Failed to save memory state: {e}")

    def _calculate_memory_budget(self, user_message_length: int) -> int:
        remaining = settings.max_context_chars - settings.memory_reserved_chars - user_message_length
        remaining = max(0, remaining)
        return min(settings.memory_total_budget_chars, remaining)

    def _allocate_budgets(self, total_budget: int) -> tuple[int, int, int]:
        long_term_budget = min(settings.memory_long_term_budget_chars, total_budget)
        remaining = max(0, total_budget - long_term_budget)
        short_term_budget = min(settings.memory_short_term_budget_chars, remaining)
        remaining = max(0, remaining - short_term_budget)
        mid_term_budget = min(settings.memory_mid_term_budget_chars, remaining)
        return long_term_budget, short_term_budget, mid_term_budget

    def _format_mid_term(self, summaries: List[MemoryChunk], budget: int) -> str:
        if budget <= 0 or not summaries:
            return ""

        lines: List[str] = []
        used = 0
        for chunk in reversed(summaries):
            line = f"[Turns {chunk.turn_range}] {chunk.summary}"
            if used + len(line) + 1 > budget:
                break
            lines.append(line)
            used += len(line) + 1

        return "\n".join(reversed(lines))

    def _format_recent_turns(
        self,
        turns: List[Dict],
        per_turn_chars: int,
        total_budget: int,
    ) -> str:
        if total_budget <= 0 or not turns:
            return ""

        parts: List[str] = []
        for turn in turns:
            user_text = self._truncate(turn.get("user_message", ""), per_turn_chars)
            assistant_text = self._truncate(turn.get("companion_response", ""), per_turn_chars)
            parts.append(
                f"[Turn {turn.get('turn_index')}]\n"
                f"User: {user_text}\n"
                f"Assistant: {assistant_text}"
            )

        combined = "\n\n".join(parts)
        return self._truncate(combined, total_budget)

    def _truncate(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3].rstrip() + "..."

    async def _update_long_term_summary(
        self,
        current_summary: str,
        user_message: str,
        companion_response: str,
        turn_outcome: TurnOutcome,
    ) -> Optional[str]:
        prompt = (
            "You are a memory compressor for a Chinese tutoring dialogue.\n"
            "Update the long-term memory summary based on the new turn.\n"
            "Rules:\n"
            "- Keep only stable facts: goals, constraints, environment, key decisions, progress.\n"
            "- Explicitly record user preferences, constraints, and key self-descriptions (e.g., name, background).\n"
            "- Remove transient details.\n"
            f"- Output must be Simplified Chinese and <= {settings.memory_long_term_max_chars} chars.\n"
            "Return JSON only:\n"
            '{ "long_term_summary": "..." }\n\n'
            "Current summary:\n"
            f"{current_summary}\n\n"
            "New turn:\n"
            f"User: {user_message}\n"
            f"Assistant: {companion_response}\n"
            "Turn outcome:\n"
            f"{turn_outcome.model_dump_json(indent=2)}\n"
        )

        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.2,
                max_tokens=512,
            )
        except LLMError as e:
            logger.warning(f"Memory update failed: {e}")
            return None

        json_str = extract_json_from_response(response)
        if not json_str:
            logger.warning("Memory update returned no JSON.")
            return None

        try:
            data = json.loads(json_str)
            summary = str(data.get("long_term_summary", "")).strip()
            return self._truncate(summary, settings.memory_long_term_max_chars)
        except Exception as e:
            logger.warning(f"Memory update parse error: {e}")
            return None

    async def _maybe_summarize_mid_term(
        self,
        session_id: str,
        memory_state: MemoryState,
        current_turn_index: int,
    ) -> None:
        max_summarize_end = current_turn_index - settings.memory_recent_turns
        next_start = memory_state.last_mid_term_turn + 1

        if max_summarize_end < next_start:
            return

        if max_summarize_end - next_start + 1 < settings.memory_mid_term_chunk_turns:
            return

        end_turn = next_start + settings.memory_mid_term_chunk_turns - 1
        turns = self.storage.load_turn_range(session_id, next_start, end_turn)
        if not turns:
            return

        summary = await self._summarize_turn_chunk(turns)
        if summary is None:
            return

        memory_state.mid_term_summaries.append(
            MemoryChunk(
                turn_range=f"{next_start}-{end_turn}",
                summary=summary,
            )
        )
        memory_state.last_mid_term_turn = end_turn

        if len(memory_state.mid_term_summaries) > settings.memory_mid_term_max_chunks:
            memory_state.mid_term_summaries = memory_state.mid_term_summaries[-settings.memory_mid_term_max_chunks :]

    async def _summarize_turn_chunk(self, turns: List[Dict]) -> Optional[str]:
        turns_text = self._format_recent_turns(
            turns,
            settings.memory_mid_term_turn_chars,
            settings.memory_mid_term_input_chars,
        )

        if not turns_text:
            return None

        prompt = (
            "You are a mid-term memory compressor for a tutoring dialogue.\n"
            "Summarize the dialogue chunk to preserve key actions, progress, blockers, and open items.\n"
            f"Output must be Simplified Chinese and <= {settings.memory_mid_term_chunk_max_chars} chars.\n"
            "Return JSON only:\n"
            '{ "summary": "..." }\n\n'
            "Dialogue chunk:\n"
            f"{turns_text}\n"
        )

        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.2,
                max_tokens=512,
            )
        except LLMError as e:
            logger.warning(f"Mid-term summary failed: {e}")
            return None

        json_str = extract_json_from_response(response)
        if not json_str:
            logger.warning("Mid-term summary returned no JSON.")
            return None

        try:
            data = json.loads(json_str)
            summary = str(data.get("summary", "")).strip()
            return self._truncate(summary, settings.memory_mid_term_chunk_max_chars)
        except Exception as e:
            logger.warning(f"Mid-term summary parse error: {e}")
            return None
