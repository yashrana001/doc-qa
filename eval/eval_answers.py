"""End-to-end answer quality: refusal accuracy, citation accuracy, answer F1, latency.

  python -m eval.eval_answers --data eval/dataset.jsonl --out eval/answers.jsonl
Then READ a sample of answers.jsonl by hand to check faithfulness.
"""
import argparse
import json
import re
import statistics
from collections import Counter

from app.pipeline import DocQA


def f1(pred: str, ref: str) -> float:
    p, r = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/dataset.jsonl")
    ap.add_argument("--out", default="eval/answers.jsonl")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    qa = DocQA()
    qa.query("warm up", use_rerank=not args.no_rerank)

    cite_hits = answered_ok = false_refusals = n_ans = 0
    correct_refusals = n_unans = 0
    f1s, lat = [], []

    with open(args.out, "w") as out:
        for r in rows:
            res = qa.query(r["question"], use_rerank=not args.no_rerank, hybrid=args.hybrid)
            lat.append(res["timings"]["total_ms"])
            if r.get("answerable", True):
                n_ans += 1
                if not res["supported"]:
                    false_refusals += 1
                else:
                    answered_ok += 1
                    gold = set(r["gold_pages"])
                    if any(s["doc_id"] == r["gold_doc"] and s["page"] in gold for s in res["sources"]):
                        cite_hits += 1
                    if r.get("reference_answer"):
                        f1s.append(f1(res["answer"], r["reference_answer"]))
            else:
                n_unans += 1
                if not res["supported"]:
                    correct_refusals += 1
            out.write(json.dumps({"question": r["question"], "result": res}) + "\n")

    print("---- results ----")
    print(f"answerable: {n_ans} | unanswerable: {n_unans}")
    if n_unans:
        print(f"refusal accuracy (unanswerable): {correct_refusals / n_unans:.1%}")
    print(f"false refusal rate (answerable):  {false_refusals / max(n_ans, 1):.1%}")
    print(f"citation accuracy (answered):     {cite_hits / max(answered_ok, 1):.1%}")
    if f1s:
        print(f"answer token-F1 vs reference:     {statistics.mean(f1s):.3f}")
    print(f"end-to-end latency p50/p95 (ms):  {statistics.median(lat):.0f} / {sorted(lat)[int(0.95 * (len(lat) - 1))]:.0f}")


if __name__ == "__main__":
    main()
