"""
Evaluate retrieval quality against a small labeled test set: for each
question, check whether the expected source document appears in the
top-k retrieved chunks. This only tests retrieval, not generation, so
it runs with no API key and no network — the part of RAG that's most
useful to catch regressions on when you change chunking or add docs.
"""

import json
from pathlib import Path

from retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "eval_questions.json"


def run_eval():
    with open(TEST_SET) as f:
        cases = json.load(f)

    retriever = Retriever()
    correct = 0
    results = []

    for case in cases:
        retrieved = retriever.retrieve(case["question"], top_k=3)
        retrieved_sources = {r["source"] for r in retrieved}
        hit = case["expected_source"] in retrieved_sources
        correct += hit
        results.append({
            "question": case["question"],
            "expected": case["expected_source"],
            "retrieved": sorted(retrieved_sources),
            "hit": hit,
        })

    accuracy = correct / len(cases) if cases else 0
    print(f"Retrieval accuracy: {correct}/{len(cases)} ({accuracy:.0%})\n")
    for r in results:
        status = "PASS" if r["hit"] else "FAIL"
        print(f"[{status}] {r['question']}")
        print(f"       expected: {r['expected']} | retrieved: {r['retrieved']}")

    return accuracy


if __name__ == "__main__":
    run_eval()
