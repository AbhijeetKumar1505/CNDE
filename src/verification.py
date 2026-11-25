"""Deterministic evidence verification and hallucination scoring.  (Owner: Member 2)

Contract
--------
verify(answer_result, evidence, embedder) -> dict
    {"status": "Supported" | "Partially supported" | "Unsupported"
               | "Insufficient evidence" | "Conflicting evidence",
     "scores": {"citation_validity": float, "numeric_fidelity": float, "claim_support": float},
     "flags": list[str]}
Scores are rule-based measurements, NOT calibrated probabilities.
"""


def verify(answer_result, evidence, embedder):
    raise NotImplementedError("Milestone 4")
