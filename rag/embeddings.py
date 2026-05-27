"""Embedding and vector index persistence."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunker import load_chunks, save_chunks


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32", copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass
class EmbeddingBundle:
    backend: str
    model_name: str
    dimension: int
    model: object | None = None
    vectorizer: object | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence_transformers" and self.model is not None:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vectors.astype("float32")
        if self.backend == "tfidf" and self.vectorizer is not None:
            vectors = self.vectorizer.transform(texts).astype("float32").toarray()
            return normalize(vectors)
        raise RuntimeError("Embedding bundle is not initialized.")


def chunk_text_for_embedding(chunk: dict) -> str:
    tags = " ".join(chunk.get("tags", []))
    return (
        f"Section {chunk.get('subsection')} {chunk.get('title')}. "
        f"Tags: {tags}. {chunk.get('text', '')}"
    )


def fit_embedding_bundle(texts: list[str]) -> tuple[EmbeddingBundle, np.ndarray]:
    backend = config.EMBEDDING_BACKEND
    if backend in {"auto", "sentence_transformers", "sentence-transformer"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.EMBEDDING_MODEL)
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")
            return (
                EmbeddingBundle(
                    backend="sentence_transformers",
                    model_name=config.EMBEDDING_MODEL,
                    dimension=vectors.shape[1],
                    model=model,
                ),
                vectors,
            )
        except Exception as exc:
            if backend not in {"auto"}:
                raise
            print(f"SentenceTransformers unavailable; falling back to TF-IDF embeddings: {exc}")

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
    )
    vectors = vectorizer.fit_transform(texts).astype("float32").toarray()
    vectors = normalize(vectors)
    return (
        EmbeddingBundle(
            backend="tfidf",
            model_name="sklearn-tfidf",
            dimension=vectors.shape[1],
            vectorizer=vectorizer,
        ),
        vectors,
    )


def load_embedding_bundle() -> EmbeddingBundle:
    if not config.INDEX_META_PATH.exists():
        raise FileNotFoundError("Index metadata is missing. Run python -m rag.embeddings first.")
    meta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
    backend = meta["backend"]
    if backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(meta["model_name"])
        return EmbeddingBundle(
            backend=backend,
            model_name=meta["model_name"],
            dimension=meta["dimension"],
            model=model,
        )

    if backend == "tfidf":
        with config.TFIDF_VECTORIZER_PATH.open("rb") as handle:
            vectorizer = pickle.load(handle)
        return EmbeddingBundle(
            backend=backend,
            model_name=meta["model_name"],
            dimension=meta["dimension"],
            vectorizer=vectorizer,
        )

    raise ValueError(f"Unknown embedding backend: {backend}")


def save_vector_index(vectors: np.ndarray, path: Path = config.FAISS_INDEX_PATH) -> str:
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype("float32"))
        faiss.write_index(index, str(path))
        return "faiss"
    except Exception as exc:
        print(f"FAISS unavailable; saving NumPy fallback index to {path}: {exc}")
        with path.open("wb") as handle:
            np.save(handle, vectors.astype("float32"))
        return "numpy"


def load_vector_index(path: Path = config.FAISS_INDEX_PATH):
    meta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
    index_backend = meta.get("index_backend", "faiss")
    if index_backend == "faiss":
        import faiss

        return faiss.read_index(str(path))
    with path.open("rb") as handle:
        return np.load(handle)


def search_index(index, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
    if meta.get("index_backend", "faiss") == "faiss":
        scores, indices = index.search(query_vector.astype("float32"), top_k)
        return scores[0], indices[0]
    vectors = index
    scores = vectors @ query_vector[0]
    order = np.argsort(scores)[::-1][:top_k]
    return scores[order], order


def build_index(rebuild: bool = False) -> dict:
    config.ensure_directories()
    if rebuild or not config.PROCESSED_CHUNKS_PATH.exists():
        save_chunks()
    chunks = load_chunks(config.PROCESSED_CHUNKS_PATH)
    texts = [chunk_text_for_embedding(chunk) for chunk in chunks]
    bundle, vectors = fit_embedding_bundle(texts)
    index_backend = save_vector_index(vectors)

    config.STORAGE_CHUNKS_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if bundle.backend == "tfidf":
        with config.TFIDF_VECTORIZER_PATH.open("wb") as handle:
            pickle.dump(bundle.vectorizer, handle)
    meta = {
        "backend": bundle.backend,
        "model_name": bundle.model_name,
        "dimension": bundle.dimension,
        "index_backend": index_backend,
        "chunk_count": len(chunks),
    }
    config.INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ensure_index() -> dict:
    if not (
        config.FAISS_INDEX_PATH.exists()
        and config.STORAGE_CHUNKS_PATH.exists()
        and config.INDEX_META_PATH.exists()
    ):
        return build_index(rebuild=True)
    return json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the KPM HR policy FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild processed text, chunks, and index.")
    args = parser.parse_args()
    meta = build_index(rebuild=args.rebuild)
    print(
        "Built vector index: "
        f"{meta['chunk_count']} chunks, embedding={meta['backend']}:{meta['model_name']}, "
        f"index={meta['index_backend']}"
    )


if __name__ == "__main__":
    main()
