# KPM HR Policy Assistant

Browser-based Retrieval-Augmented Generation (RAG) chatbot for the course project "HR Policy Assistant: A Retrieval-Augmented Conversational Chatbot."

The employee-facing knowledge base indexes only `data/raw/HR Policy for RAG v1.docx`. The proposal, chunking guide, and Microsoft CUX guide were used as implementation references and are not indexed into the chatbot.

## Features

- DOCX ingestion with clean text saved to `data/processed/hr_policy_clean.txt`
- Section-aware chunks with metadata saved to `data/processed/chunks.json`
- SentenceTransformers embeddings using `sentence-transformers/all-MiniLM-L6-v2` by default, with `sentence-transformers/all-mpnet-base-v2` configurable through `KPM_EMBEDDING_MODEL`
- FAISS vector index persisted to `storage/faiss.index`
- Hybrid retrieval with vector search, BM25/keyword scoring, intent/metadata boosting, and optional cross-encoder reranking
- Grounded extractive answer fallback that answers only from retrieved policy chunks
- Gradio browser UI with source citations, quick questions, clear chat, and rebuild index controls
- Evaluation script and pytest suite

## Setup

```bash
python -m pip install -r requirements.txt
```

The HR policy file must exist at:

```text
data/raw/HR Policy for RAG v1.docx
```

## Required Commands

```bash
python -m rag.document_loader
python -m rag.chunker
python -m rag.embeddings
python evaluation/evaluate.py
pytest -q
python app.py --smoke-test
python app.py
```

Launch the app at the URL printed by Gradio, usually:

```text
http://127.0.0.1:7860
```

## Notes

The answer generator defaults to a reliable grounded summarization path instead of open-ended generation. This keeps responses policy-based and avoids hallucinating when a local LLM is unavailable or slow.
