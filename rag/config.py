"""Configuration for the KPM HR Policy Assistant."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STORAGE_DIR = PROJECT_ROOT / "storage"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

RAW_POLICY_PATH = RAW_DIR / "HR Policy for RAG v1.docx"
CLEAN_POLICY_PATH = PROCESSED_DIR / "hr_policy_clean.txt"
PROCESSED_CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
FAISS_INDEX_PATH = STORAGE_DIR / "faiss.index"
STORAGE_CHUNKS_PATH = STORAGE_DIR / "chunks.json"
INDEX_META_PATH = STORAGE_DIR / "index_meta.json"
TFIDF_VECTORIZER_PATH = STORAGE_DIR / "tfidf_vectorizer.pkl"

EMBEDDING_MODEL = os.getenv(
    "KPM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
ALTERNATE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_BACKEND = os.getenv("KPM_EMBEDDING_BACKEND", "auto").lower()

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_CROSS_ENCODER = os.getenv("KPM_USE_CROSS_ENCODER", "0") == "1"

SOURCE_NAME = "KPM HR Manual 2026"
INDUSTRY = "Automotive Manufacturing"
JURISDICTION = "Kentucky, USA"
VERSION = "v1.0"

VECTOR_TOP_K = 15
FINAL_TOP_K = 5
ANSWER_TOP_K = 4
MIN_CONFIDENCE = 0.18

CHUNK_MIN_WORDS = 80
CHUNK_IDEAL_MIN_WORDS = 120
CHUNK_IDEAL_MAX_WORDS = 220
CHUNK_MAX_WORDS = 300

INTENT_LABELS = [
    "leave",
    "benefits",
    "overtime",
    "attendance",
    "conduct",
    "safety",
    "PPE",
    "remote_work",
    "payroll",
    "performance",
    "termination",
    "technology",
    "other",
]

INTENT_TO_SECTIONS = {
    "leave": {"5.3", "5.5", "5.6", "5.7", "5.8", "5.9", "5.10", "5.11"},
    "benefits": {"5.1", "5.2", "5.3", "5.4"},
    "overtime": {"3.3", "4.3"},
    "attendance": {"3.2", "3.5", "10.6"},
    "conduct": {"6.1", "6.2", "6.3", "6.4", "6.5", "6.10", "6.11"},
    "safety": {"7.1", "7.2", "7.3", "7.5", "7.7", "7.9", "10.5"},
    "PPE": {"6.5", "7.7"},
    "remote_work": {"3.4", "8.1", "8.2", "8.3", "8.6"},
    "payroll": {"4.1", "4.2", "4.3", "4.4", "4.5"},
    "performance": {"9.1", "9.2", "9.3", "9.4", "9.5"},
    "termination": {"10.1", "10.2", "10.8", "10.10", "10.11", "10.12"},
    "technology": {"8.1", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9"},
}

QUERY_SECTION_BOOSTS = {
    "fmla": {"5.6": 1.0},
    "family medical leave": {"5.6": 1.0},
    "dress code": {"6.5": 1.0, "7.7": 0.35},
    "ppe": {"7.7": 1.0, "6.5": 0.8},
    "personal protective": {"7.7": 1.0, "6.5": 0.8},
    "late": {"3.2": 1.0, "10.6": 0.8},
    "tardy": {"3.2": 1.0, "10.6": 0.8},
    "no-call": {"3.2": 1.0, "10.6": 1.0, "10.8": 0.35},
    "no call": {"3.2": 1.0, "10.6": 1.0, "10.8": 0.35},
    "no-show": {"3.2": 1.0, "10.6": 1.0, "10.8": 0.35},
    "no show": {"3.2": 1.0, "10.6": 1.0, "10.8": 0.35},
    "work from home": {"3.4": 1.0, "8.2": 0.45, "8.3": 0.35},
    "remote": {"3.4": 1.0, "8.2": 0.35, "8.3": 0.25},
    "drug": {"6.3": 1.0},
    "alcohol": {"6.3": 1.0},
    "performance review": {"9.3": 1.0},
    "reviews": {"9.3": 0.8},
    "overtime": {"3.3": 1.0},
    "paycheck": {"4.5": 1.0, "4.1": 0.35},
    "paycheck wrong": {"4.5": 1.0},
    "pay wrong": {"4.5": 1.0},
    "payroll error": {"4.5": 1.0},
    "maternity": {"5.7": 1.0, "5.6": 0.45},
    "paternity": {"5.8": 1.0, "5.6": 0.35},
    "sick leave": {"5.5": 1.0},
    "pto": {"5.3": 1.0},
    "vacation": {"5.3": 0.8},
}


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, STORAGE_DIR, EVALUATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
