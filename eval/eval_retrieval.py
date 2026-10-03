"""Retrieval quality + latency across configurations.

  python -m eval.eval_retrieval --data eval/dataset.jsonl --out eval/results_retrieval.csv

To compare vector stores / index types, re-ingest and re-run with different env vars:
  INDEX_TYPE=hnsw python -m eval.eval_retrieval ...
  VECTOR_BACKEND=pgvector python -m eval.eval_retrieval ...
"""
import argparse
import csv
import json
import statistics
import time

from app.config import settings
from app.pipeline import DocQA

KS = (1, 3, 5, 10)


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def evaluate(qa, rows, hybrid, rerank, retrieve_k):
    ranks, latencies = [], []
    for r in rows:
        t0 = time.perf_counter()
        ranked, _ = qa.retrieve(r["question"], retrieve_k, rerank, hybrid)
        latencies.append((time.perf_counter() - t0) * 1000)
        gold = set(r["gold_pages"])
        rank = next((i for i, c in enumerate(ranked[:max(KS)], 1)
                     if c["doc_id"] == r["gold_doc"] and c["page"] in gold), None)
        ranks.append(rank)
    n = len(rows)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in KS}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r) / n, 3)
    out["p50_ms"] = round(statistics.median(latencies), 1)
    out["p95_ms"] = round(percentile(latencies, 95), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/dataset.jsonl")
    ap.add_argument("--out", default="eval/results_retrieval.csv")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    rows = [r for r in rows if r.get("answerable", True) and r.get("gold_doc")]
    print(f"{len(rows)} answerable queries | backend={settings.vector_backend} index={settings.index_type}")

    qa = DocQA()
    qa.retrieve("warm up", 5, True, False)  # load models before timing

    configs = []
    for hybrid in (False, True):
        configs.append((hybrid, False, 20))
        for k in (10, 20, 50):
            configs.append((hybrid, True, k))

    results = []
    for hybrid, rerank, k in configs:
        m = evaluate(qa, rows, hybrid, rerank, k)
        row = {"store": f"{settings.vector_backend}/{settings.index_type}",
               "hybrid": hybrid, "rerank": rerank, "retrieve_k": k, **m}
        results.append(row)
        print(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
