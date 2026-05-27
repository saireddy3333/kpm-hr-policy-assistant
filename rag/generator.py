"""Grounded answer generation for the HR Policy Assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .intents import classify_intent
from .memory import ConversationMemory
from .retriever import PolicyRetriever, RetrievalResult, tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SUBHEADING_HINTS = (
    "eligibility",
    "requirements",
    "coordination",
    "benefits",
    "duration",
    "process",
    "approval",
    "reporting",
    "tardiness",
    "examples",
    "behavior",
    "testing",
    "medications",
    "employees",
    "responsibilities",
)


def looks_like_subheading(line: str) -> bool:
    lower = line.lower()
    words = re.findall(r"[A-Za-z0-9&()'-]+", line)
    if len(words) > 8 or line.endswith((".", "?", "!")):
        return False
    if not any(hint in lower for hint in SUBHEADING_HINTS):
        return False
    significant = [word for word in words if word.lower() not in {"and", "or", "for", "of", "the", "to"}]
    if not significant:
        return False
    return all(word[:1].isupper() or word.isupper() for word in significant)


@dataclass
class AnswerResult:
    question: str
    effective_query: str
    answer: str
    intent: str
    sources: list[dict]
    retrieved_sections: list[str]
    expected_hit: bool | None = None


def source_citation(results: list[RetrievalResult]) -> str:
    seen = set()
    citations = []
    for result in results:
        subsection = result.chunk.get("subsection")
        title = result.chunk.get("title")
        key = (subsection, title)
        if key in seen:
            continue
        seen.add(key)
        citations.append(f"Section {subsection} {title}")
    if not citations:
        return "Sources: None located in the KPM HR Policy Manual."
    return "Sources: " + "; ".join(citations)


def split_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and re.match(r"^\d+\.\d+\s+", lines[0]):
        lines = lines[1:]

    units: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if looks_like_subheading(line) and index + 1 < len(lines):
            details: list[str] = []
            cursor = index + 1
            while cursor < len(lines):
                next_line = lines[cursor]
                if looks_like_subheading(next_line) and details:
                    break
                details.append(next_line.rstrip(":"))
                if len(" ".join(details).split()) >= 42:
                    cursor += 1
                    break
                cursor += 1
            unit = f"{line.rstrip(':')}: {'; '.join(details)}"
            unit = unit.replace(":;", ":").replace(" with;", " with:")
            unit = re.sub(r"\b(must|wear|not|include|includes);", r"\1:", unit)
            units.append(unit)
            index = cursor
            continue
        units.extend(sentence.strip(" -\t") for sentence in SENTENCE_RE.split(line) if sentence.strip())
        index += 1

    return units


def select_grounded_sentences(query: str, results: list[RetrievalResult], limit: int = 4) -> list[str]:
    query_tokens = set(tokenize(query))
    selected: list[str] = []
    seen: set[str] = set()
    scored: list[tuple[float, str]] = []
    for rank, result in enumerate(results):
        for sentence in split_sentences(result.chunk.get("text", "")):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(query_tokens & sentence_tokens)
            score = overlap + max(0, 3 - rank) * 0.2
            if any(term in sentence.lower() for term in ["must", "required", "eligible", "should", "may"]):
                score += 0.2
            if sentence.lower().startswith("eligible"):
                score += 1.0
            scored.append((score, sentence))

    for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True):
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= limit:
            break

    if not selected and results:
        selected = split_sentences(results[0].chunk.get("text", ""))[:limit]
    return selected


def answer_relevant_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not results:
        return []
    top_confidence = results[0].confidence
    selected: list[RetrievalResult] = []
    for result in results:
        if not selected:
            selected.append(result)
            continue
        if result.confidence >= top_confidence * 0.62 or result.metadata_score >= 0.75:
            selected.append(result)
        if len(selected) >= config.ANSWER_TOP_K:
            break
    return selected


def normalize_name(name: str | None) -> str:
    cleaned = (name or "").strip()
    return cleaned if cleaned else "there"


class HRPolicyAssistant:
    def __init__(self, retriever: PolicyRetriever | None = None) -> None:
        self.retriever = retriever or PolicyRetriever()

    def answer(
        self,
        question: str,
        name: str = "Alex",
        memory: ConversationMemory | None = None,
    ) -> AnswerResult:
        memory = memory or ConversationMemory()
        question = question.strip()
        effective_query = memory.contextualize(question)
        intent = classify_intent(effective_query)
        display_name = normalize_name(name)

        if intent.ambiguous and intent.clarification:
            answer = (
                f"Hi {display_name}, I can help with leave policies, but I do not want to guess. "
                f"{intent.clarification}"
            )
            memory.add_turn(question, answer, intent.intent, [])
            return AnswerResult(question, effective_query, answer, intent.intent, [], [])

        results = self.retriever.retrieve(effective_query, intent=intent, top_k=config.ANSWER_TOP_K)
        top_confidence = results[0].confidence if results else 0.0
        has_clear_policy_signal = intent.intent != "other" or any(
            result.metadata_score > 0.25 for result in results[:3]
        )

        if not results or not has_clear_policy_signal or top_confidence < config.MIN_CONFIDENCE:
            answer = (
                f"Hi {display_name}, I couldn't locate that in the KPM HR Policy Manual. "
                "You might try asking about leave, attendance, payroll, safety, conduct, "
                "technology, performance reviews, or termination policies.\n\n"
                "Sources: None located in the KPM HR Policy Manual.\n\n"
                "For official interpretation or personal employment situations, contact HR.\n\n"
                "Did this answer your question?"
            )
            memory.add_turn(question, answer, "other", [])
            return AnswerResult(question, effective_query, answer, "other", [], [])

        answer_results = answer_relevant_results(results)
        sentences = select_grounded_sentences(effective_query, answer_results, limit=4)
        bullets = "\n".join(f"- {sentence}" for sentence in sentences)
        citations = source_citation(answer_results)
        answer = (
            f"Hi {display_name}, based on the KPM HR Policy Manual, here's what I found:\n\n"
            f"{bullets}\n\n"
            f"{citations}\n\n"
            "For official interpretation or personal employment situations, contact HR.\n\n"
            "Did this answer your question?"
        )
        sources = [result.to_source() for result in answer_results]
        memory.add_turn(question, answer, intent.intent, sources)
        return AnswerResult(
            question=question,
            effective_query=effective_query,
            answer=answer,
            intent=intent.intent,
            sources=sources,
            retrieved_sections=[source["subsection"] for source in sources],
        )
