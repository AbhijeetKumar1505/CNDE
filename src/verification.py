"""Deterministic evidence verification and hallucination scoring.  (Owner: Member 2)

Contract
--------
verify(answer_result, embedder, min_retrieval_score, sem_threshold, lex_threshold) -> dict
    answer_result: output of generation.generate_answer (its "evidence" list defines S1..Sn)
    {"status": "Supported" | "Partially supported" | "Unsupported"
               | "Insufficient evidence" | "Conflicting evidence",
     "scores": {"citation_validity", "numeric_fidelity", "claim_support", "grounding_score"},
     "claims": [...], "numbers": [...], "flags": list[str], "max_retrieval_score": float}

All scores are rule-based measurements in [0, 1] computed from the retrieved text. They are NOT
calibrated probabilities, and the generator model never grades its own answer.
"""
import re

import numpy as np

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

_UNIT = (r"(mm|cm|m|μm|µm|um|nm|in\.?|inch(?:es)?|ft|mils?|MHz|kHz|GHz|Hz|°F|°C|ksi|psi|MPa|GPa|kV|mA|V|A|"
         r"dB|lbs?|kg|g|sec(?:onds?)?|s|min(?:utes?)?|h(?:ours?)?|%|years?|months?)")
# A measurement number is not glued to letters/hyphens on either side, so identifiers such as
# "NASA-STD-5009C", "S1" or "E1742M" are ignored; an optional unit may follow.
_NUMBER = re.compile(r"(?<![\w.\-])(-?\d+(?:\.\d+)?)(?:\s?" + _UNIT + r"(?![A-Za-z]))?(?![\w\-])")
_CITATION = re.compile(r"\[[^\[\]]{1,40}\]")
_STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this these those it
its which shall should must may can not no than then also into such per any all each other used use using when where
what how does do""".split())
_UNIT_ALIASES = {"µm": "μm", "um": "μm", "in.": "in", "inch": "in", "inches": "in", "mil": "mils", "sec": "s",
                 "second": "s", "seconds": "s", "minute": "min", "minutes": "min", "hour": "h", "hours": "h",
                 "lb": "lbs", "year": "years", "month": "months"}


def _norm_text(t: str) -> str:
    t = t.replace("−", "-")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)              # 1,000 -> 1000


def _norm_unit(u):
    """Case-insensitive unit identity with common aliases (µm/um -> μm, inches/in. -> in, ...)."""
    if not u:
        return None
    key = u.lower()
    return _UNIT_ALIASES.get(key, key)


def extract_numbers(text: str) -> list:
    """(value, unit) pairs; citation markers like [S1] are removed first so their digits don't count."""
    text = _norm_text(_CITATION.sub(" ", text))
    return [(float(m.group(1)), _norm_unit(m.group(2)), m.group(0).strip()) for m in _NUMBER.finditer(text)]


_NO_CONFLICT = re.compile(r"\b(none|no conflicts?|no disagreement|do not (conflict|disagree))\b", re.I)


def _split_sentences(text: str) -> list:
    # No split before "[": a citation written after the full stop ("... cleaning. [S1]") stays with its sentence.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-zμ°%≤≥±]+[\w-]*", text.lower()) if w not in _STOP and len(w) > 2}


def _number_check(numbers, evidence_text: str) -> list:
    ev = extract_numbers(evidence_text)
    out = []
    for value, unit, raw in numbers:
        same_value = [u for v, u, _ in ev if abs(v - value) < 1e-9]
        found = bool(same_value)
        # Unit mismatch only when the value appears in the evidence exclusively with a different explicit unit
        # (table cells often carry the unit in a header, so a bare evidence value is accepted).
        unit_ok = (not found) or unit is None or unit in same_value or None in same_value
        out.append({"text": raw, "found": found, "unit_ok": unit_ok})
    return out


def verify(answer_result, embedder, min_retrieval_score=0.0, sem_threshold=0.80, lex_threshold=0.60,
           strong_retrieval=0.75) -> dict:
    """Thresholds are heuristics with placeholder defaults; they are tuned on the benchmark."""
    evidence = answer_result["evidence"]
    text = answer_result["answer"]
    flags = []
    max_score = max((e.get("score", 0.0) for e in evidence), default=0.0)
    base = {"claims": [], "numbers": [], "max_retrieval_score": round(max_score, 4)}

    # -- abstention / retrieval gate
    if not evidence or max_score < min_retrieval_score:
        flags.append(f"Top retrieval score {max_score:.2f} is below the threshold {min_retrieval_score:.2f}.")
    if answer_result.get("abstained"):
        if max_score >= strong_retrieval:
            flags.append(f"Model abstained although retrieval was strong (top score {max_score:.2f}); check manually.")
        return {**base, "status": "Insufficient evidence", "flags": flags,
                "scores": {"citation_validity": None, "numeric_fidelity": None, "claim_support": None,
                           "grounding_score": None}}

    # -- 1. citation validity (pages come from chunk metadata, never from the model's text)
    valid, invalid = answer_result["cited_ids"], answer_result["invalid_ids"]
    n_cites = len(valid) + len(invalid)
    citation_validity = len(valid) / n_cites if n_cites else 0.0
    if not n_cites:
        flags.append("Answer contains no citations.")
    if invalid:
        flags.append(f"Citations to non-existent passages: {', '.join(invalid)}.")
    pages = {str(e["page"]) for e in evidence if e.get("page") is not None}
    for p in re.findall(r"\b(?:page|p\.)\s*(\d+)", _CITATION.sub(" ", text), flags=re.I):
        if p not in pages:
            flags.append(f"Answer mentions page {p}, which is not among the retrieved pages.")
    if answer_result.get("hit_max_tokens"):
        flags.append("Generation hit the max_new_tokens limit; the answer may be truncated.")
    for sid in valid:
        e = evidence[int(sid[1:]) - 1]
        if "[Table" in e["text"]:
            # Numeric fidelity proves each number occurs in the evidence, not that it was read from the
            # right row/column of a table, so table-derived answers are explicitly flagged for review.
            flags.append(f"[{sid}] is a table ({e['source']}, p. {e['page']}): row/column pairing of values is "
                         f"not machine-checked; confirm against the source page.")

    # -- 2. claims: answer (+ conflicts) sentences, checked against their own cited passages
    sections = answer_result.get("sections") or {"Answer": text}
    conflict = sections.get("Conflicts", "")
    if _NO_CONFLICT.search(conflict):
        conflict = ""                                   # "There is no conflict" is not a claim to check
    claim_text = " ".join([sections.get("Answer", ""), conflict])
    # A claim = sentence with >=2 content words or a number; short answers ("5 MHz [S1].") stay checkable.
    claims = [s for s in _split_sentences(claim_text)
              if len(_content_words(_CITATION.sub(" ", s))) >= 2 or extract_numbers(s)]
    if not claims and claim_text.strip():
        claims = [claim_text.strip()]
    ev_sents = [(i, s) for i, e in enumerate(evidence) for s in _split_sentences(e["text"])]
    if claims and ev_sents:
        vecs = embedder.encode([_CITATION.sub(" ", c) for c in claims] + [s for _, s in ev_sents],
                               normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        claim_vecs, ev_vecs = vecs[:len(claims)], vecs[len(claims):]
        ev_idx = np.array([i for i, _ in ev_sents])
    all_numbers, claim_rows = [], []
    for k, claim in enumerate(claims):
        cited = [int(n) - 1 for g in _CITATION.findall(claim) for n in re.findall(r"S\s*(\d+)", g)]
        cited = [i for i in cited if 0 <= i < len(evidence)]
        scope = cited or list(range(len(evidence)))         # uncited claims are checked against all evidence

        def support(idx):
            words = _content_words(_CITATION.sub(" ", claim))
            ev_words = set().union(*(_content_words(evidence[i]["text"]) for i in idx))
            lexical = len(words & ev_words) / len(words) if words else 0.0
            mask = np.isin(ev_idx, idx) if ev_sents else None
            semantic = float((ev_vecs[mask] @ claim_vecs[k]).max()) if ev_sents and mask.any() else 0.0
            return lexical, semantic

        def passes(idx):
            lex, sem = support(idx)
            nums = _number_check(extract_numbers(claim), " ".join(evidence[i]["text"] for i in idx))
            return (sem >= sem_threshold or lex >= lex_threshold) and all(n["found"] and n["unit_ok"] for n in nums), lex, sem, nums

        ok, lexical, semantic, nums = passes(scope)
        all_numbers += nums
        if not cited:
            flags.append(f"Uncited claim: \"{claim[:80]}\"")
        elif not ok:
            others = [i for i in range(len(evidence)) if i not in cited]
            if others and passes(others)[0]:     # fully supported (numbers included) only by uncited passages
                flags.append(f"Claim is supported by a passage other than the one cited: \"{claim[:80]}\"")
        claim_rows.append({"text": claim, "supported": ok, "semantic": round(semantic, 3),
                           "lexical": round(lexical, 3), "cited": [f"S{i + 1}" for i in cited]})

    for n in all_numbers:
        if not n["found"]:
            flags.append(f"Number not found in the evidence: {n['text']}")
        elif not n["unit_ok"]:
            flags.append(f"Unit differs from the evidence: {n['text']}")
    numeric_fidelity = (sum(n["found"] and n["unit_ok"] for n in all_numbers) / len(all_numbers)) if all_numbers else 1.0
    claim_support = (sum(c["supported"] for c in claim_rows) / len(claim_rows)) if claim_rows else 0.0
    if not claim_rows:
        flags.append("No checkable claims found in the answer.")

    # -- 3. status
    scores = {"citation_validity": round(citation_validity, 3), "numeric_fidelity": round(numeric_fidelity, 3),
              "claim_support": round(claim_support, 3)}
    scores["grounding_score"] = round(sum(scores.values()) / 3, 3)
    conflict_ids = set(re.findall(r"S\s*(\d+)", " ".join(_CITATION.findall(conflict))))
    if conflict and len(conflict_ids) >= 2:
        status = "Conflicting evidence"
    elif min(scores["citation_validity"], numeric_fidelity, claim_support) >= 0.999:
        status = "Supported"
    elif claim_support >= 0.5 and numeric_fidelity >= 0.5 and citation_validity > 0:
        status = "Partially supported"
    else:
        status = "Unsupported"
    return {**base, "status": status, "scores": scores, "claims": claim_rows, "numbers": all_numbers, "flags": flags}
