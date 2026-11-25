"""Benchmark runner and metrics (Recall@K, answer correctness, citation correctness,
unsupported-claim rate, abstention accuracy, latency).  (Shared)

Benchmark file: benchmarks/questions.jsonl, one JSON object per line:
    {"id": str, "type": str, "question": str, "expected": list[str],
     "gold": [{"source": str, "page": int}], "should_abstain": bool}
"""


def load_benchmark(path):
    raise NotImplementedError("Milestone 5")


def run_benchmark(questions, answer_fn):
    raise NotImplementedError("Milestone 5")
