"""Local LLM loading, prompt construction and answer generation.  (Owner: Member 2)

Contract
--------
load_llm(model_id, compute_dtype) -> (model, tokenizer)     # 4-bit NF4, double quant, FP16 compute
build_messages(question, evidence, variant) -> list[dict]   # chat messages; evidence labelled [S1]..[Sn]
generate_answer(question, evidence, model, tokenizer, variant, max_new_tokens) -> dict
    {"answer": str, "cited_ids": list[str], "abstained": bool,
     "prompt_tokens": int, "new_tokens": int, "latency_s": float}
"""


def load_llm(model_id, compute_dtype=None):
    raise NotImplementedError("Milestone 3")


def smoke_test(model, tokenizer):
    """Short generation that checks logits are finite (FP16 overflow) and reports VRAM + tokens/s."""
    raise NotImplementedError("Milestone 3")


def build_messages(question, evidence, variant="v2"):
    raise NotImplementedError("Milestone 3")


def generate_answer(question, evidence, model, tokenizer, variant="v2", max_new_tokens=512):
    raise NotImplementedError("Milestone 3")
