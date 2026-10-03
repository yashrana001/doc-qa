# Intelligent Document Understanding & Q&A System

A RAG (Retrieval-Augmented Generation) system: upload PDFs, ask questions, get answers
with page-level citations. If the documents don't contain the answer, it says so.

**Stack:** Python, PyTorch, Hugging Face, FastAPI, FAISS / pgvector, Docker

## Pipeline

```
PDF -> extract (PyMuPDF) -> chunk (page-aware) -> embed (bge) -> FAISS / pgvector
Question -> embed -> top-k retrieval (dense or hybrid dense+BM25)
         -> cross-encoder rerank -> confidence check -> LLM answer with [n] citations
         -> citation validation -> JSON {answer, sources[doc, page, score]}
```

**Reducing unsupported answers (3 guards):**
1. Top reranker score below `RERANK_THRESHOLD` -> refuse, don't call the LLM.
2. The LLM is told to answer only from numbered context, or say it can't.
3. Citations are validated against the chunks actually supplied.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt

# put PDFs in data/pdfs/, then:
python scripts_ingest_folder.py data/pdfs

# fast smoke test without downloading an LLM:
LLM_BACKEND=extractive uvicorn app.main:app --reload
# real answers (downloads Qwen2.5-1.5B-Instruct on first use):
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs to try the API.

```bash
curl -X POST localhost:8000/ingest -F "files=@data/pdfs/yourfile.pdf"
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What is the refund policy?"}'
```

## Docker (FastAPI + Postgres/pgvector)

```bash
docker compose up --build
```

## Evaluation

1. Add 5-10 PDFs to `data/pdfs/` and ingest them.
2. Draft questions: `python -m eval.generate_questions --n 220`, then **review by hand**,
   add ~25 unanswerable questions, and save as `eval/dataset.jsonl` (format in the sample file).
3. Retrieval + latency across configs: `python -m eval.eval_retrieval`
4. Repeat with `INDEX_TYPE=hnsw`, and with `VECTOR_BACKEND=pgvector` (re-ingest first).
5. Answer quality: `python -m eval.eval_answers` (also try `--no-rerank`).
6. **Tune** `RERANK_THRESHOLD` using the refusal accuracy / false refusal numbers.

Paste your numbers here:

| Store | Hybrid | Rerank | k | Recall@5 | MRR@10 | p50 ms | p95 ms |
|-------|--------|--------|---|----------|--------|--------|--------|
|       |        |        |   |          |        |        |        |

## Configuration (env vars)

`EMBED_MODEL`, `RERANK_MODEL`, `LLM_MODEL`, `LLM_BACKEND`, `VECTOR_BACKEND`, `INDEX_TYPE`,
`CHUNK_WORDS`, `CHUNK_OVERLAP`, `RETRIEVE_K`, `FINAL_K`, `RERANK_THRESHOLD`, `DENSE_THRESHOLD`
(see `app/config.py`).

## Limitations

- Scanned PDFs (images) need OCR first; PyMuPDF only reads real text.
- Tables are extracted as plain text, so numbers in tables can lose structure.
- A 1.5B local LLM is small; swap `LLM_MODEL` for a bigger one for better answers.
