"""Short-term conversation memory for follow-up questions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .intents import is_followup


@dataclass
class ConversationMemory:
    max_turns: int = 20
    turns: deque[dict[str, Any]] = field(default_factory=deque)

    def add_turn(self, user: str, assistant: str, intent: str, sources: list[dict]) -> None:
        source_summaries = [
            {
                "subsection": source.get("subsection"),
                "title": source.get("title"),
                "chunk_id": source.get("chunk_id"),
            }
            for source in sources[:5]
        ]
        self.turns.append(
            {
                "user": user,
                "assistant": assistant,
                "intent": intent,
                "sources": source_summaries,
            }
        )
        while len(self.turns) > self.max_turns:
            self.turns.popleft()

    def last_intent(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.get("intent") and turn["intent"] != "other":
                return turn["intent"]
        return None

    def last_topic(self) -> str | None:
        for turn in reversed(self.turns):
            sources = turn.get("sources") or []
            if sources:
                source = sources[0]
                subsection = source.get("subsection")
                title = source.get("title")
                if subsection and title:
                    return f"{subsection} {title}"
        return None

    def contextualize(self, query: str) -> str:
        if not self.turns or not is_followup(query):
            return query
        topic = self.last_topic()
        intent = self.last_intent()
        context_bits = [bit for bit in [topic, intent] if bit]
        if not context_bits:
            return query
        return f"{query} (follow-up about {'; '.join(context_bits)})"

    def to_state(self) -> list[dict[str, Any]]:
        return list(self.turns)

    @classmethod
    def from_state(cls, state: list[dict[str, Any]] | None, max_turns: int = 20) -> "ConversationMemory":
        memory = cls(max_turns=max_turns)
        for item in state or []:
            memory.turns.append(item)
        while len(memory.turns) > max_turns:
            memory.turns.popleft()
        return memory
