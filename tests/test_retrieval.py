from __future__ import annotations

import pytest

from rag.embeddings import build_index
from rag.intents import classify_intent
from rag.retriever import PolicyRetriever


@pytest.fixture(scope="module")
def retriever():
    build_index(rebuild=True)
    return PolicyRetriever()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How does FMLA work?", {"5.6"}),
        ("What is the dress code?", {"6.5", "7.7"}),
        ("What happens if I'm late?", {"3.2", "10.6"}),
        ("Can I work from home?", {"3.4"}),
        ("What PPE do I need?", {"6.5", "7.7"}),
        ("What is the drug policy?", {"6.3"}),
        ("How do performance reviews work?", {"9.3"}),
        ("How does overtime work?", {"3.3"}),
        ("What if my paycheck is wrong?", {"4.5"}),
        ("What happens for no-call/no-show?", {"3.2", "10.6"}),
    ],
)
def test_key_questions_return_expected_section_in_top_3(retriever, question, expected):
    results = retriever.retrieve(question, intent=classify_intent(question), top_k=5)
    top3 = {result.chunk["subsection"] for result in results[:3]}
    assert expected & top3
