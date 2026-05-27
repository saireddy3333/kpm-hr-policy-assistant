from __future__ import annotations

from rag.intents import classify_intent


def test_required_intents_are_classified():
    examples = {
        "How does FMLA work?": "leave",
        "What benefits are available?": "benefits",
        "How does overtime work?": "overtime",
        "What happens if I'm late?": "attendance",
        "What is the drug policy?": "conduct",
        "What PPE do I need?": "PPE",
        "Can I work from home?": "remote_work",
        "What if my paycheck is wrong?": "payroll",
        "How do performance reviews work?": "performance",
        "Can I be terminated immediately?": "termination",
        "Can I use AI tools at work?": "technology",
        "Who catered lunch today?": "other",
    }
    for question, expected in examples.items():
        assert classify_intent(question).intent == expected


def test_ambiguous_leave_query_asks_for_clarification():
    result = classify_intent("leave")
    assert result.intent == "leave"
    assert result.ambiguous
    assert result.clarification
    assert "FMLA" in result.clarification
