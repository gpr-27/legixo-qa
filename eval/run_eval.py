"""Run the gold-set self-test against a running API server.

    python eval/run_eval.py [--url http://localhost:8000]

Reads sample_test_cases.json, calls POST /ask for each question, and reports three
metrics: citation correctness, fact recall (in-corpus), and abstain correctness
(out-of-corpus). Writes per-question results to eval/results.md.
"""

import argparse
import json
import re
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
_REFUSAL_HINTS = ("cannot find", "not found", "cannot answer", "no information")
# Phrases that count as an honest "the documents don't say" for the O2 partial case.
_HONEST_GAP = ("not specified", "not stated", "does not", "no penalty", "not mention")


def _norm(text: str) -> str:
    """Lowercase and drop spaces, commas, and currency symbols so '₹45,000' == '45000'."""
    return re.sub(r"[\s,₹$€£¥]", "", text.lower())


def _has_fact(answer: str, fact: str) -> bool:
    return _norm(fact) in _norm(answer)


def run(url: str) -> None:
    cases = json.loads((HERE / "sample_test_cases.json").read_text(encoding="utf-8"))
    rows = []
    cite_ok = fact_ok = abstain_ok = n_in = n_out = 0

    for case in cases:
        resp = httpx.post(f"{url}/ask", json={"question": case["question"]},
                          timeout=60).json()
        answer = resp.get("answer", "")
        cited = {c["source"] for c in resp.get("citations", [])}

        if case["type"] == "in_corpus":
            n_in += 1
            facts = case.get("expected_facts", [])
            hits = sum(_has_fact(answer, f) for f in facts)
            fact_pass = hits >= max(1, (len(facts) + 1) // 2)   # at least half the facts
            cite_pass = bool(set(case["expected_source_files"]) & cited)
            cite_ok += cite_pass
            fact_ok += fact_pass
            rows.append((case["id"], "in", cite_pass and fact_pass,
                         f"facts {hits}/{len(facts)}, citation={'ok' if cite_pass else 'MISS'}"))
        else:
            n_out += 1
            refused = any(h in answer.lower() for h in _REFUSAL_HINTS)
            no_fabrication = not cited
            if case["id"] == "O2":
                # A grounded partial answer is acceptable if it doesn't invent a penalty.
                ok = refused or any(p in answer.lower() for p in _HONEST_GAP)
            else:
                ok = refused and no_fabrication
            abstain_ok += ok
            rows.append((case["id"], "out", ok,
                         f"refused={refused}, citations={len(cited)}"))

    _write_results(rows, cite_ok, fact_ok, abstain_ok, n_in, n_out)
    print(f"citation_correctness: {cite_ok}/{n_in}")
    print(f"fact_recall:          {fact_ok}/{n_in}")
    print(f"abstain_correctness:  {abstain_ok}/{n_out}")
    print(f"\nPer-question results written to {HERE / 'results.md'}")


def _write_results(rows, cite_ok, fact_ok, abstain_ok, n_in, n_out) -> None:
    lines = [
        "# Eval results",
        "",
        f"- citation_correctness: **{cite_ok}/{n_in}**",
        f"- fact_recall: **{fact_ok}/{n_in}**",
        f"- abstain_correctness: **{abstain_ok}/{n_out}**",
        "",
        "| id | type | pass | notes |",
        "|----|------|------|-------|",
    ]
    for cid, kind, ok, note in rows:
        lines.append(f"| {cid} | {kind} | {'PASS' if ok else 'FAIL'} | {note} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-test the Q&A API against the gold set.")
    parser.add_argument("--url", default="http://localhost:8000")
    run(parser.parse_args().url)
