"""Section-aware chunking for the KPM HR policy manual."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import config
from .document_loader import save_clean_text


MAJOR_RE = re.compile(r"^(?P<section>\d+)\.\s+(?P<title>.+)$")
SUBSECTION_RE = re.compile(r"^(?P<subsection>\d+\.\d+)\s+(?P<title>.+)$")
CROSSREF_RE = re.compile(r"\b[Ss]ection\s+(\d+(?:\.\d+)?)")
WORD_RE = re.compile(r"\b[\w'-]+\b")
MAJOR_TITLES = {
    "Introduction & Company Overview",
    "Employment Policies",
    "Work Hours, Attendance & Scheduling",
    "Compensation & Payroll",
    "Benefits & Leave Policies",
    "Workplace Conduct & Expectations",
    "Health, Safety & Compliance",
    "Technology & Data Policies",
    "Performance Management",
    "Disciplinary Action & Termination",
    "Acknowledgment & Receipt",
}


@dataclass
class PolicySection:
    section: str
    section_title: str
    subsection: str
    title: str
    paragraphs: list[str] = field(default_factory=list)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def short_title(title: str) -> str:
    title = re.sub(r"\([^)]*\)", "", title)
    words = re.findall(r"[A-Za-z0-9]+", title)
    if not words:
        return "Policy"
    return "".join(word[:16] for word in words[:6])


def detect_crossrefs(text: str) -> list[str]:
    refs = sorted(set(CROSSREF_RE.findall(text)), key=lambda value: [int(x) for x in value.split(".")])
    return refs


TAG_KEYWORDS = {
    "leave": ["leave", "fmla", "pto", "sick", "maternity", "paternity", "holiday", "absence"],
    "benefits": ["benefit", "insurance", "retirement", "401", "health"],
    "overtime": ["overtime"],
    "attendance": ["attendance", "punctuality", "late", "tardy", "absence", "no-call", "no-show", "schedule"],
    "conduct": ["conduct", "harassment", "bullying", "violence", "drug", "alcohol", "misconduct", "dress"],
    "safety": ["safety", "osha", "incident", "emergency", "equipment", "housekeeping", "audit"],
    "PPE": ["ppe", "personal protective", "protective equipment", "glasses", "gloves", "hearing"],
    "remote_work": ["remote", "work from home", "staff-only"],
    "payroll": ["pay", "payroll", "wage", "salary", "timekeeping", "deduction", "reimbursement"],
    "performance": ["performance", "review", "coaching", "pip", "training", "development"],
    "termination": ["discipline", "termination", "resignation", "investigation", "corrective"],
    "technology": ["technology", "data", "cybersecurity", "email", "software", "internet", "device", "genai"],
}


def infer_tags(title: str, text: str) -> list[str]:
    haystack = f"{title} {text}".lower()
    tags: list[str] = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            tags.append(tag)
    if "fmla" in haystack:
        tags.append("FMLA")
    if "kentucky" in haystack:
        tags.append("Kentucky")
    if not tags:
        tags.append("policy")
    return sorted(set(tags), key=str.lower)


def parse_sections(clean_text: str) -> list[PolicySection]:
    current_major = ""
    current_major_title = ""
    current: PolicySection | None = None
    sections: list[PolicySection] = []

    for raw_line in clean_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        sub_match = SUBSECTION_RE.match(line)
        major_match = MAJOR_RE.match(line)

        if sub_match:
            if current:
                sections.append(current)
            subsection = sub_match.group("subsection")
            if not current_major:
                current_major = subsection.split(".")[0]
            current = PolicySection(
                section=current_major,
                section_title=current_major_title,
                subsection=subsection,
                title=sub_match.group("title"),
            )
            continue

        if major_match and major_match.group("title") in MAJOR_TITLES:
            if current:
                sections.append(current)
                current = None
            current_major = major_match.group("section")
            current_major_title = major_match.group("title")
            current = PolicySection(
                section=current_major,
                section_title=current_major_title,
                subsection=current_major,
                title=current_major_title,
            )
            continue

        if current is not None:
            current.paragraphs.append(line)

    if current:
        sections.append(current)

    return [section for section in sections if section.paragraphs]


def split_oversized_paragraph(paragraph: str) -> list[str]:
    words = paragraph.split()
    if len(words) <= config.CHUNK_MAX_WORDS:
        return [paragraph]
    pieces = []
    start = 0
    while start < len(words):
        end = min(start + config.CHUNK_IDEAL_MAX_WORDS, len(words))
        pieces.append(" ".join(words[start:end]))
        start = end
    return pieces


def build_chunk_text(section: PolicySection, paragraphs: Iterable[str]) -> str:
    heading = f"{section.subsection} {section.title}"
    body = "\n".join(paragraphs).strip()
    return f"{heading}\n{body}".strip()


def split_section(section: PolicySection) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []

    for paragraph in section.paragraphs:
        for piece in split_oversized_paragraph(paragraph):
            piece_words = word_count(piece)
            current_words = word_count(" ".join(current))

            if current and current_words + piece_words > config.CHUNK_MAX_WORDS:
                chunks.append(current)
                current = [piece]
            else:
                current.append(piece)

            if word_count(" ".join(current)) >= config.CHUNK_IDEAL_MAX_WORDS:
                chunks.append(current)
                current = []

    if current:
        if chunks and word_count(" ".join(current)) < config.CHUNK_MIN_WORDS:
            merged = chunks[-1] + current
            if word_count(" ".join(merged)) <= config.CHUNK_MAX_WORDS:
                chunks[-1] = merged
            else:
                chunks.append(current)
        else:
            chunks.append(current)

    return [build_chunk_text(section, chunk) for chunk in chunks if chunk]


def chunk_id_for(section: PolicySection, part_index: int, total_parts: int) -> str:
    base = f"KPM-HR-{section.section}-{section.subsection}-{short_title(section.title)}"
    if total_parts > 1:
        return f"{base}-Part{part_index + 1}"
    return base


def create_chunk_records(clean_text: str) -> list[dict]:
    records: list[dict] = []
    for section in parse_sections(clean_text):
        chunk_texts = split_section(section)
        for index, text in enumerate(chunk_texts):
            records.append(
                {
                    "chunk_id": chunk_id_for(section, index, len(chunk_texts)),
                    "section": section.section,
                    "section_title": section.section_title,
                    "subsection": section.subsection,
                    "title": section.title,
                    "tags": infer_tags(section.title, text),
                    "source": config.SOURCE_NAME,
                    "industry": config.INDUSTRY,
                    "jurisdiction": config.JURISDICTION,
                    "version": config.VERSION,
                    "crossrefs": detect_crossrefs(text),
                    "text": text,
                    "word_count": word_count(text),
                    "naturally_short": word_count(text) < config.CHUNK_MIN_WORDS,
                }
            )
    return records


def save_chunks(
    clean_text_path: Path = config.CLEAN_POLICY_PATH,
    output_path: Path = config.PROCESSED_CHUNKS_PATH,
) -> Path:
    config.ensure_directories()
    if not clean_text_path.exists():
        save_clean_text()
    clean_text = clean_text_path.read_text(encoding="utf-8")
    chunks = create_chunk_records(clean_text)
    output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_chunks(path: Path = config.PROCESSED_CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        save_chunks()
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create section-aware HR policy chunks.")
    parser.add_argument("--input", type=Path, default=config.CLEAN_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=config.PROCESSED_CHUNKS_PATH)
    args = parser.parse_args()
    output = save_chunks(args.input, args.output)
    chunks = load_chunks(output)
    print(f"Saved {len(chunks)} chunks to {output}")


if __name__ == "__main__":
    main()
