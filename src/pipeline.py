"""End-to-end question answering: retrieve -> abstention gate -> generate -> verify.
Shared by main.ipynb, app.py and the benchmark so all three run exactly the same pipeline.

answer_question(question, index, chunks, embedder, model, tokenizer, cfg, variant) -> dict
    generation.generate_answer() result + {"gated", "retrieval_ms", "verification"}
to_markdown(result) -> str        # Answer / Supporting evidence / References / Verification / Limitations
"""
import re
import time

from .generation import ABSTAIN_TOKEN, generate_answer
from .retrieval import retrieve
from .verification import verify


def answer_question(question, index, chunks, embedder, model, tokenizer, cfg, variant="v3") -> dict:
    t0 = time.perf_counter()
    evidence = retrieve(question, index, chunks, embedder, cfg.top_k)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
    top = evidence[0]["score"] if evidence else 0.0

    if not evidence or top < cfg.min_retrieval_score:
        # Retrieval gate: nothing relevant enough was found, so the LLM is not asked at all.
        text = f"{ABSTAIN_TOKEN}: no passage passed the retrieval threshold (top score {top:.2f})."
        result = {"answer": text, "sections": {"Answer": text}, "cited_ids": [], "invalid_ids": [],
                  "abstained": True, "evidence": evidence, "dropped_evidence": 0, "prompt_tokens": 0,
                  "new_tokens": 0, "hit_max_tokens": False, "latency_s": 0.0, "variant": variant, "gated": True}
    else:
        result = generate_answer(question, evidence, model, tokenizer, variant,
                                 cfg.max_new_tokens, cfg.max_input_tokens)
        result["gated"] = False
    result["retrieval_ms"] = retrieval_ms
    result["verification"] = verify(result, embedder, cfg.min_retrieval_score)
    return result


def _ref(e) -> str:
    page = f"p. {e['page']}" if e.get("page") is not None else "page n/a"
    section = f" — {e['section']}" if e.get("section") else ""
    return f"{e['source']}, {page}{section}"


def to_markdown(result) -> str:
    """Render a result. Evidence and references are built from the cited chunks themselves (exact text),
    and the verification block comes from the verifier, not from the model."""
    v = result["verification"]
    sec = result["sections"]
    lines = ["### Answer", sec.get("Answer", result["answer"])]
    if sec.get("Interpretation"):
        lines += ["", f"*Interpretation (not a documented fact):* {sec['Interpretation']}"]
    if sec.get("Conflicts") and not re.search(r"\b(none|no conflicts?|no disagreement)\b", sec["Conflicts"], re.I):
        lines += ["", "### Conflicting statements", sec["Conflicts"]]

    cited = [int(c[1:]) - 1 for c in result["cited_ids"]]
    if cited:
        lines += ["", "### Supporting evidence"]
        for i in cited:
            e = result["evidence"][i]
            quote = e["text"] if len(e["text"]) <= 700 else e["text"][:700] + " …"
            lines += [f"**[S{i + 1}]** {_ref(e)} (retrieval score {e['score']:.2f})", "> " + quote.replace("\n", "\n> "), ""]
        lines += ["### References"] + [f"- [S{i + 1}] {_ref(result['evidence'][i])}" for i in cited]

    lines += ["", "### Verification", f"**Status: {v['status']}** (rule-based checks, not a probability)"]
    if v["scores"]["grounding_score"] is not None:
        s = v["scores"]
        lines.append(f"citation validity {s['citation_validity']:.2f} · numeric fidelity {s['numeric_fidelity']:.2f}"
                     f" · claim support {s['claim_support']:.2f} · grounding score {s['grounding_score']:.2f}")
    lines += [f"- ⚠ {f}" for f in v["flags"]]

    limits = [sec["Limitations"]] if sec.get("Limitations") else []
    if result.get("dropped_evidence"):
        limits.append(f"{result['dropped_evidence']} lower-ranked passage(s) were left out to fit the context budget.")
    if limits:
        lines += ["", "### Limitations"] + limits
    return "\n".join(lines)
