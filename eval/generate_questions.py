"""Bootstrap an eval set: ask the LLM to write one question per random chunk.
ALWAYS review the output by hand before trusting it.

  python -m eval.generate_questions --n 220 --out eval/dataset_draft.jsonl
"""
import argparse
import json
import random

from app.pipeline import DocQA

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=220)
p.add_argument("--out", default="eval/dataset_draft.jsonl")
args = p.parse_args()

qa = DocQA()
chunks = [c for c in qa.store.all_chunks() if len(c["text"].split()) >= 60]
random.seed(0)
random.shuffle(chunks)

with open(args.out, "w") as f:
    for c in chunks[: args.n]:
        messages = [
            {"role": "system", "content": "You write exam questions. Output ONLY the question and the short "
             "answer in this format:\nQ: <question>\nA: <answer>\nThe question must be answerable from the "
             "passage alone and must not mention 'the passage' or 'the text'."},
            {"role": "user", "content": c["text"]},
        ]
        out = qa.generator.chat(messages, max_new_tokens=120)
        if "Q:" not in out or "A:" not in out:
            continue
        q = out.split("Q:", 1)[1].split("A:", 1)[0].strip()
        a = out.split("A:", 1)[1].strip()
        f.write(json.dumps({"question": q, "gold_doc": c["doc_id"], "gold_pages": [c["page"]],
                            "answerable": True, "reference_answer": a}) + "\n")
print("Wrote", args.out, "- now review it, and add ~25 unanswerable questions by hand.")
