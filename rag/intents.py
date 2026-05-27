"""Lightweight intent handling and disambiguation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    ambiguous: bool = False
    clarification: str | None = None


INTENT_PATTERNS: dict[str, list[str]] = {
    "leave": [
        r"\bleave\b",
        r"\bfmla\b",
        r"\bpto\b",
        r"\bsick\b",
        r"\bmaternity\b",
        r"\bpaternity\b",
        r"\bholiday\b",
        r"\bvacation\b",
        r"\babsence\b",
    ],
    "benefits": [r"\bbenefits?\b", r"\binsurance\b", r"\bhealth plan\b", r"\bretirement\b", r"\b401k\b"],
    "overtime": [r"\bovertime\b", r"\bover 40\b", r"\bextra hours?\b"],
    "attendance": [
        r"\blattendance\b",
        r"\blate\b",
        r"\btardy\b",
        r"\bpunctual",
        r"\bno[- ]?call\b",
        r"\bno[- ]?show\b",
        r"\babsen",
        r"\bschedule\b",
        r"\bshift swap\b",
    ],
    "conduct": [
        r"\bconduct\b",
        r"\bharass",
        r"\bbully",
        r"\bviolence\b",
        r"\bdrug\b",
        r"\balcohol\b",
        r"\bdress code\b",
        r"\bmisconduct\b",
    ],
    "safety": [r"\bsafety\b", r"\bosha\b", r"\bincident\b", r"\bemergency\b", r"\bequipment\b"],
    "PPE": [r"\bppe\b", r"\bpersonal protective\b", r"\bprotective equipment\b", r"\bglasses\b", r"\bgloves\b"],
    "remote_work": [r"\bremote\b", r"\bwork from home\b", r"\bwfh\b", r"\btelework\b"],
    "payroll": [r"\bpayroll\b", r"\bpaycheck\b", r"\bpay check\b", r"\bwage\b", r"\bsalary\b", r"\bdeduction\b"],
    "performance": [r"\bperformance\b", r"\breview\b", r"\bcoaching\b", r"\bpip\b", r"\btraining\b"],
    "termination": [r"\btermination\b", r"\bterminated\b", r"\bfired\b", r"\bresignation\b", r"\bdiscipline\b"],
    "technology": [
        r"\btechnology\b",
        r"\bdata\b",
        r"\bcyber",
        r"\bemail\b",
        r"\bsoftware\b",
        r"\binternet\b",
        r"\bdevice\b",
        r"\bgenai\b",
        r"\bai tools?\b",
    ],
}

AMBIGUOUS_LEAVE_RE = re.compile(
    r"^\s*(?:what\s+)?(?:are\s+)?(?:is\s+)?(?:the\s+)?(?:company\s+)?(?:leave|time\s+off|absence)\s*(?:policy|options|benefits)?\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def classify_intent(query: str) -> IntentResult:
    normalized = query.lower().strip()
    if AMBIGUOUS_LEAVE_RE.match(normalized):
        return IntentResult(
            intent="leave",
            confidence=0.55,
            ambiguous=True,
            clarification=(
                "Are you asking about vacation/PTO, sick leave, FMLA, maternity leave, "
                "paternity leave, holidays, or another leave type?"
            ),
        )

    scores: dict[str, float] = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, normalized):
                score += 1.0
        if score:
            scores[intent] = score

    if not scores:
        return IntentResult(intent="other", confidence=0.0)

    intent, score = max(scores.items(), key=lambda item: item[1])
    confidence = min(0.95, 0.45 + 0.15 * score)
    if intent == "conduct" and re.search(r"\bdress code\b", normalized):
        confidence = 0.9
    if intent == "safety" and re.search(r"\bppe\b|personal protective", normalized):
        intent = "PPE"
        confidence = 0.95
    if intent in config.INTENT_LABELS:
        return IntentResult(intent=intent, confidence=confidence)
    return IntentResult(intent="other", confidence=0.0)


def is_followup(query: str) -> bool:
    normalized = query.strip().lower()
    return normalized.startswith(
        (
            "what about",
            "how about",
            "and ",
            "what if that",
            "does that",
            "would that",
            "can that",
            "is that",
        )
    )
