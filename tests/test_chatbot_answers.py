from __future__ import annotations

from rag.embeddings import build_index
from rag.generator import HRPolicyAssistant
from rag.memory import ConversationMemory


def assistant():
    build_index(rebuild=True)
    return HRPolicyAssistant()


def test_answer_includes_sources():
    bot = assistant()
    result = bot.answer("How does FMLA work?", name="Alex", memory=ConversationMemory())
    assert "Sources:" in result.answer
    assert "Section 5.6" in result.answer
    assert result.sources


def test_unsupported_question_uses_fallback():
    bot = assistant()
    result = bot.answer("What is today's cafeteria menu?", name="Alex", memory=ConversationMemory())
    assert "I couldn't locate that in the KPM HR Policy Manual" in result.answer
    assert "Sources: None located" in result.answer


def test_ambiguous_leave_query_triggers_clarification():
    bot = assistant()
    result = bot.answer("leave", name="Alex", memory=ConversationMemory())
    assert "I do not want to guess" in result.answer
    assert "vacation/PTO" in result.answer
    assert not result.sources


def test_followup_query_uses_memory():
    bot = assistant()
    memory = ConversationMemory()
    first = bot.answer("How does FMLA work?", name="Alex", memory=memory)
    second = bot.answer("What about maternity?", name="Alex", memory=memory)
    assert "Section 5.7" in second.answer
    assert "maternity" in second.effective_query.lower()
    assert first.sources
