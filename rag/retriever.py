"""Hybrid vector and keyword retrieval for KPM HR policy chunks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import config
from .embeddings import (
    chunk_text_for_embedding,
    ensure_index,
    load_embedding_bundle,
    load_vector_index,
    search_index,
)
from .intents import IntentResult, classify_intent


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")


@dataclass
class RetrievalResult:
    chunk: dict
    vector_score: float
    keyword_score: float
    metadata_score: float
    confidence: float

    def to_source(self) -> dict:
        text = self.chunk.get("text", "")
        snippet = text.replace("\n", " ")
        if len(snippet) > 360:
            snippet = snippet[:357].rstrip() + "..."
        return {
            "chunk_id": self.chunk.get("chunk_id"),
            "section": self.chunk.get("section"),
            "subsection": self.chunk.get("subsection"),
            "title": self.chunk.get("title"),
            "confidence": round(self.confidence, 3),
            "snippet": snippet,
        }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def simple_keyword_score(query_tokens: set[str], chunk: dict) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(tokenize(chunk_text_for_embedding(chunk)))
    overlap = query_tokens & chunk_tokens
    return len(overlap) / max(1, math.sqrt(len(query_tokens)))


class KeywordScorer:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized = [tokenize(chunk_text_for_embedding(chunk)) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.tokenized)
        except Exception:
            self.bm25 = None

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunks), dtype="float32")
        if self.bm25 is not None:
            raw = np.array(self.bm25.get_scores(tokens), dtype="float32")
        else:
            query_set = set(tokens)
            raw = np.array(
                [simple_keyword_score(query_set, chunk) for chunk in self.chunks],
                dtype="float32",
            )
        max_score = float(raw.max()) if raw.size else 0.0
        if max_score <= 0:
            return raw
        return raw / max_score


def query_boosts(query: str) -> dict[str, float]:
    normalized = query.lower()
    boosts: dict[str, float] = {}
    for phrase, section_boosts in config.QUERY_SECTION_BOOSTS.items():
        pattern = r"\b" + re.escape(phrase.lower()).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, normalized):
            for section, boost in section_boosts.items():
                boosts[section] = max(boosts.get(section, 0.0), boost)
    return boosts


def metadata_score(chunk: dict, intent: IntentResult, boosts: dict[str, float]) -> float:
    score = 0.0
    subsection = chunk.get("subsection", "")
    tags = set(chunk.get("tags", []))
    if subsection in boosts:
        score += boosts[subsection]
    if intent.intent in config.INTENT_TO_SECTIONS and subsection in config.INTENT_TO_SECTIONS[intent.intent]:
        score += 0.35
    if intent.intent in tags:
        score += 0.25
    if intent.intent == "PPE" and ("PPE" in tags or subsection in {"6.5", "7.7"}):
        score += 0.45
    return min(score, 1.5)


class PolicyRetriever:
    def __init__(self) -> None:
        ensure_index()
        self.chunks = json.loads(config.STORAGE_CHUNKS_PATH.read_text(encoding="utf-8"))
        self.embedding_bundle = load_embedding_bundle()
        self.index = load_vector_index()
        self.keyword_scorer = KeywordScorer(self.chunks)
        self.cross_encoder = None
        if config.USE_CROSS_ENCODER:
            try:
                from sentence_transformers import CrossEncoder

                self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
            except Exception as exc:
                print(f"Cross-encoder reranker unavailable; using score fusion: {exc}")

    def retrieve(
        self,
        query: str,
        intent: IntentResult | None = None,
        top_k: int = config.FINAL_TOP_K,
        vector_top_k: int = config.VECTOR_TOP_K,
    ) -> list[RetrievalResult]:
        intent = intent or classify_intent(query)
        query_vector = self.embedding_bundle.encode([query])
        vector_scores, indices = search_index(self.index, query_vector, vector_top_k)
        keyword_scores = self.keyword_scorer.scores(query)
        boosts = query_boosts(query)

        candidates: dict[int, RetrievalResult] = {}
        for raw_score, raw_index in zip(vector_scores, indices):
            if raw_index < 0:
                continue
            idx = int(raw_index)
            chunk = self.chunks[idx]
            v_score = float(max(0.0, raw_score))
            k_score = float(keyword_scores[idx])
            m_score = metadata_score(chunk, intent, boosts)
            confidence = 0.55 * v_score + 0.30 * k_score + 0.35 * m_score
            candidates[idx] = RetrievalResult(chunk, v_score, k_score, m_score, confidence)

        boosted_sections = set(boosts)
        if intent.intent in config.INTENT_TO_SECTIONS:
            boosted_sections |= config.INTENT_TO_SECTIONS[intent.intent]
        for idx, chunk in enumerate(self.chunks):
            if chunk.get("subsection") not in boosted_sections:
                continue
            k_score = float(keyword_scores[idx])
            m_score = metadata_score(chunk, intent, boosts)
            if k_score <= 0 and m_score <= 0:
                continue
            existing = candidates.get(idx)
            confidence = 0.45 * k_score + 0.45 * m_score
            result = RetrievalResult(chunk, existing.vector_score if existing else 0.0, k_score, m_score, confidence)
            if existing is None or result.confidence > existing.confidence:
                candidates[idx] = result

        ranked = sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)
        ranked = self._cross_encoder_rerank(query, ranked[: max(top_k, 12)])
        return ranked[:top_k]

    def _cross_encoder_rerank(
        self, query: str, candidates: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if self.cross_encoder is None or not candidates:
            return candidates
        pairs = [(query, candidate.chunk.get("text", "")) for candidate in candidates]
        scores = self.cross_encoder.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.confidence = 0.65 * float(score) + 0.35 * candidate.confidence
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)


def format_sources(results: Iterable[RetrievalResult]) -> str:
    lines = []
    for result in results:
        source = result.to_source()
        lines.append(
            f"- Section {source['subsection']} {source['title']} "
            f"({source['chunk_id']}, confidence {source['confidence']}): {source['snippet']}"
        )
    return "\n".join(lines)
