"""Benchmark runner and metrics.  (Shared)

Benchmark file: benchmarks/questions.jsonl, one JSON object per line:
    {"id", "type", "question", "expected": [key facts; "a|b" = alternatives], "must_not": [optional],
     "gold": [{"source", "page"}], "should_abstain": bool}
Questions and gold pages were written by hand from the documents. Two synthetic, clearly labelled
documents (benchmarks/synthetic/) add a conflict case and a prompt-injection case.

check_benchmark(questions, chunks) -> list[str]        # every expected fact must occur on its gold pages
retrieval_metrics(questions, index, chunks, embedder, ks) -> (rows, summary)   # Recall@k, MRR, top scores
run_benchmark(questions, answer_fn, variants) -> rows   # answer_fn(question, variant) -> pipeline result
summarize(rows) -> {variant: metrics}

"Correct" is an automated proxy: all expected key facts present (and no must_not phrase) for answerable
questions, abstention for unanswerable ones. It does not judge wording, so outputs are kept for review.
"""
import json
import re
import statistics
from pathlib import Path

from .retrieval import retrieve

_NO_CONFLICT = re.compile(r"\b(none|no conflicts?|no disagreement)\b", re.I)


def load_benchmark(path) -> list:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip() and not l.lstrip().startswith("#")]


def _norm(text: str) -> str:
    text = text.lower().replace("–", "-").replace("−", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", text)


def _has(text: str, item: str) -> bool:
    t = _norm(text)
    return any(_norm(alt) in t for alt in item.split("|"))


def _gold_match(e: dict, gold: list) -> bool:
    return any(e["source"] == g["source"] and (g.get("page") is None or e["page"] == g["page"]) for g in gold)


def check_benchmark(questions, chunks) -> list:
    """Sanity-check the benchmark itself against the indexed text; returns problems (empty = OK)."""
    problems = []
    for q in questions:
        if q["should_abstain"] or not q["gold"]:
            continue
        gold_text = " ".join(c["text"] for c in chunks if _gold_match(c, q["gold"]))
        if not gold_text:
            problems.append(f"{q['id']}: no indexed chunk on gold pages {q['gold']}")
            continue
        missing = [x for x in q["expected"] if not _has(gold_text, x)]
        if missing:
            problems.append(f"{q['id']}: expected {missing} not found on gold pages")
    return problems


def retrieval_metrics(questions, index, chunks, embedder, ks=(1, 3, 5)):
    rows = []
    for q in questions:
        hits = retrieve(q["question"], index, chunks, embedder, max(ks))
        row = {"id": q["id"], "type": q["type"], "top_score": hits[0]["score"] if hits else 0.0,
               "should_abstain": q["should_abstain"]}
        if q["gold"]:
            ranks = [i + 1 for i, h in enumerate(hits) if _gold_match(h, q["gold"])]
            row["first_gold_rank"] = ranks[0] if ranks else None
            for k in ks:
                row[f"recall@{k}"] = bool(ranks and ranks[0] <= k)
        rows.append(row)
    graded = [r for r in rows if "first_gold_rank" in r]
    summary = {f"recall@{k}": round(sum(r[f"recall@{k}"] for r in graded) / len(graded), 3) for k in ks}
    summary["mrr"] = round(sum(1 / r["first_gold_rank"] for r in graded if r["first_gold_rank"]) / len(graded), 3)
    summary["n_questions_with_gold"] = len(graded)
    answerable = [r["top_score"] for r in rows if not r["should_abstain"]]
    unanswerable = [r["top_score"] for r in rows if r["should_abstain"]]
    summary["top_score_answerable_min"] = round(min(answerable), 3) if answerable else None
    summary["top_score_unanswerable_max"] = round(max(unanswerable), 3) if unanswerable else None
    return rows, summary


def score_answer(q, result) -> dict:
    v = result["verification"]
    text = result["answer"]
    answerable = not q["should_abstain"]
    facts = [_has(text, x) for x in q["expected"]]
    violated = [x for x in q.get("must_not", []) if _has(text, x)]
    conflict = result["sections"].get("Conflicts", "")
    claims = v.get("claims", [])
    return {
        "id": q["id"], "type": q["type"], "variant": result["variant"],
        "correct": (not result["abstained"] and all(facts) and not violated) if answerable else result["abstained"],
        "key_fact_recall": round(sum(facts) / len(facts), 3) if facts else None,
        "abstained": result["abstained"], "gated": result.get("gated", False),
        "citation_correct": (any(_gold_match(result["evidence"][int(c[1:]) - 1], q["gold"]) for c in result["cited_ids"])
                             if answerable and q["gold"] else None),
        "conflict_reported": bool(conflict) and not _NO_CONFLICT.search(conflict),
        "injection_followed": bool(violated),
        "status": v["status"], "grounding": v["scores"]["grounding_score"],
        "n_claims": len(claims), "unsupported_claims": sum(not c["supported"] for c in claims),
        "n_flags": len(v["flags"]),
        "latency_s": result["latency_s"], "retrieval_ms": result["retrieval_ms"], "new_tokens": result["new_tokens"],
        "hit_max_tokens": result.get("hit_max_tokens", False),
        "answer": text,
    }


def run_benchmark(questions, answer_fn, variants=("v2",), log=print) -> list:
    rows = []
    for variant in variants:
        for q in questions:
            row = score_answer(q, answer_fn(q["question"], variant))
            rows.append(row)
            log(f"[{variant}] {q['id']} {q['type']:13s} correct={row['correct']!s:5s} status={row['status']:20s} "
                f"{row['latency_s']:5.1f}s")
    return rows


def summarize(rows) -> dict:
    out = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == variant]
        ans = [r for r in rs if r["type"] != "insufficient"]
        una = [r for r in rs if r["type"] == "insufficient"]
        cites = [r["citation_correct"] for r in ans if r["citation_correct"] is not None]
        claims = sum(r["n_claims"] for r in rs)
        gen = [r["latency_s"] for r in rs if not r["gated"]]
        grounding = [r["grounding"] for r in rs if r["grounding"] is not None]
        out[variant] = {
            "accuracy_answerable": round(sum(r["correct"] for r in ans) / len(ans), 3) if ans else None,
            "abstention_accuracy": round(sum(r["correct"] for r in una) / len(una), 3) if una else None,
            "false_abstention_rate": round(sum(r["abstained"] for r in ans) / len(ans), 3) if ans else None,
            "citation_correctness": round(sum(cites) / len(cites), 3) if cites else None,
            "unsupported_claim_rate": round(sum(r["unsupported_claims"] for r in rs) / claims, 3) if claims else None,
            "mean_grounding": round(statistics.mean(grounding), 3) if grounding else None,
            "conflicts_reported": sum(r["conflict_reported"] for r in rs if r["type"] == "conflicting"),
            "injections_followed": sum(r["injection_followed"] for r in rs),
            "median_generation_s": round(statistics.median(gen), 2) if gen else None,
            "max_generation_s": round(max(gen), 2) if gen else None,
            "n": len(rs),
        }
    return out
