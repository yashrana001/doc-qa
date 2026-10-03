# Intelligent Document Understanding & Q&A System

A RAG (Retrieval-Augmented Generation) system: upload PDFs, ask questions, and get answers with page-level citations. If the documents don't contain the answer, the system says so instead of making something up.

**Stack:** Python, PyTorch, Hugging Face, FastAPI, FAISS / pgvector, Docker

## Features

- End-to-end pipeline over multiple PDF documents: extraction, chunking, embedding, retrieval and answer generation
- Semantic retrieval with a choice of vector store (FAISS or pgvector) and index type (Flat or HNSW)
- Cross-encoder reranking of retrieved chunks
- Source-level citations (document and page) on every answer
- Refusal threshold to reduce unsupported answers
- REST API built with FastAPI, with interactive docs at `/docs`
- Evaluation suite: 220 curated queries, answer-quality metrics and latency benchmarks
- Docker and docker-compose support

## How it works

```
PDF -> extract text -> chunk (page-aware) -> embed (bge) -> FAISS / pgvector
Question -> embed -> retrieve top-k -> rerank -> threshold check -> answer + citations
```

If the best reranked chunk scores below `RERANK_THRESHOLD`, the system refuses to answer rather than guess.

## Project structure

```
doc-qa/
├── app/
│   ├── main.py        # FastAPI app
│   ├── pipeline.py    # ties the full RAG flow together
│   ├── ingest.py      # PDF extraction and chunking
│   ├── embed.py       # embedding model
│   ├── store.py       # FAISS and pgvector stores
│   ├── retrieve.py    # semantic retrieval
│   ├── rerank.py      # reranking
│   ├── generate.py    # answer generation with citations
│   └── config.py      # all settings (env vars)
├── eval/
│   ├── generate_questions.py
│   ├── eval_retrieval.py
│   ├── eval_answers.py
│   └── dataset.jsonl  # evaluation queries
├── data/pdfs/         # put your PDFs here
├── scripts_ingest_folder.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Getting started

### Run locally

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs to try the API. The first run downloads the Hugging Face models, so it can take a while.

Add your PDFs to `data/pdfs/` and ingest them (through the API, or with `scripts_ingest_folder.py`) before asking questions.

### Run with Docker

```bash
docker compose up --build
```

## Evaluation

The system was evaluated on a curated set of **220 queries** (195 answerable, 25 unanswerable) over multiple PDF documents. Latency was measured on a Google Colab T4 GPU, so numbers on CPU will be higher.

### Answer quality

| Metric | Value |
|---|---|
| Refusal accuracy (unanswerable queries) | 100.0% |
| False refusal rate (answerable queries) | 13.8% |
| Citation accuracy (answered queries) | 88.7% |
| Answer token-F1 vs reference | 0.442 |
| End-to-end latency p50 / p95 | 2979 ms / 8225 ms |

The refusal threshold (`RERANK_THRESHOLD`) trades off the two refusal numbers: raising it makes the system stricter, lowering it makes it answer more often. It was tuned using the refusal accuracy and false refusal rate above.

### Retrieval benchmark

Retrieval quality and latency were benchmarked across configurations (Flat vs HNSW index). Raw results are in `eval/results_retrieval_flat.csv` and `eval/results_retrieval_hnsw.csv`.

### How to reproduce

1. Add your PDFs to `data/pdfs/` and ingest them.
2. Generate draft questions: `python -m eval.generate_questions --n 220`, then review them by hand and add unanswerable questions. Save as `eval/dataset.jsonl`.
3. Retrieval and latency: `python -m eval.eval_retrieval`
4. Repeat with `INDEX_TYPE=hnsw`, and with `VECTOR_BACKEND=pgvector` (re-ingest first).
5. Answer quality: `python -m eval.eval_answers --data eval/dataset.jsonl` (also try `--no-rerank`).
6. Tune `RERANK_THRESHOLD` using the refusal accuracy and false refusal numbers.

## Configuration (env vars)

`EMBED_MODEL`, `RERANK_MODEL`, `LLM_MODEL`, `LLM_BACKEND`, `VECTOR_BACKEND`, `INDEX_TYPE`, `CHUNK_WORDS`, `CHUNK_OVERLAP`, `RETRIEVE_K`, `FINAL_K`, `RERANK_THRESHOLD`, `DENSE_THRESHOLD` (see `app/config.py`).

## Limitations

- Answer quality depends on the LLM backend. The extractive backend is fastest but gives shorter answers.
- Scanned PDFs without a text layer need OCR, which is not included.
- Latency figures were measured on a GPU. Expect slower responses on CPU-only machines.
- The false refusal rate (13.8%) means some answerable questions are declined. Lower `RERANK_THRESHOLD` to reduce this, at the cost of fewer correct refusals.


 
