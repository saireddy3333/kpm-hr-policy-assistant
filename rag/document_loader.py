"""DOCX ingestion for the employee-facing HR policy manual."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document

from . import config


SECTION_RE = re.compile(r"^(?:\d+\.\d+|\d+\.)\s+\S+")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries elsewhere."""

    return re.sub(r"\s+", " ", text).strip()


def extract_docx_text(path: Path = config.RAW_POLICY_PATH) -> str:
    """Extract clean text from the KPM HR policy DOCX.

    The HR policy document is the only document this loader accepts for the
    employee-facing knowledge base.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Missing HR policy document. Place it at "
            f"{config.RAW_POLICY_PATH} and rerun python -m rag.document_loader."
        )

    if path.name != "HR Policy for RAG v1.docx":
        raise ValueError("Only 'HR Policy for RAG v1.docx' may be loaded into the KB.")

    doc = Document(path)
    lines: list[str] = []
    for paragraph in doc.paragraphs:
        text = clean_text(paragraph.text)
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style else ""
        is_heading = style_name.startswith("Heading") or SECTION_RE.match(text)
        if is_heading and lines and lines[-1] != "":
            lines.append("")
        lines.append(text)

    return "\n".join(lines).strip() + "\n"


def save_clean_text(
    source_path: Path = config.RAW_POLICY_PATH,
    output_path: Path = config.CLEAN_POLICY_PATH,
) -> Path:
    config.ensure_directories()
    text = extract_docx_text(source_path)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the KPM HR policy DOCX.")
    parser.add_argument("--source", type=Path, default=config.RAW_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=config.CLEAN_POLICY_PATH)
    args = parser.parse_args()
    output = save_clean_text(args.source, args.output)
    print(f"Saved clean HR policy text to {output}")


if __name__ == "__main__":
    main()
