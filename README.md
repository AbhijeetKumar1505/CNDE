# CNDE NDT Document QA — Grounded RAG with Automated Verification

Prototype for the CNDE (IIT Madras) AI/ML hackathon: ask questions about Non-Destructive Testing (NDT)
documents and get answers that are grounded in the documents, cited to document + page, and automatically
scored for hallucination risk.

> Document-grounded technical assistance only — not a certified inspection or engineering diagnostic system.

## Architecture

```
Upload (PDF/DOCX/TXT) -> PyMuPDF / python-docx extraction -> page-bounded chunking (BGE-token budget)
 -> BGE-small embeddings (normalized) -> FAISS inner-product index
Question -> top-k retrieval -> abstention gate -> prompt with [S1..Sn] evidence -> Qwen3-4B-Instruct (NF4, T4)
 -> answer with [S#] citations -> deterministic verifier (citations, numbers+units, claim support, conflicts)
 -> Gradio: answer, evidence, references, verification scores
```

| Component | Choice |
|---|---|
| LLM | `Qwen/Qwen3-4B-Instruct-2507`, 4-bit NF4, double quant, FP16 compute |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d, normalized) |
| Vector search | FAISS `IndexFlatIP` (exact cosine) |
| Extraction | PyMuPDF, python-docx |
| UI | Gradio |

## Repository layout

| Path | Purpose | Owner |
|---|---|---|
| `src/config.py` | Paths + tunables shared by notebook and app | Shared |
| `src/ingestion.py` | Extraction + chunking | Member 1 |
| `src/retrieval.py` | Embeddings + FAISS + retrieval | Member 1 |
| `src/generation.py` | Qwen loading, prompts, generation | Member 2 |
| `src/verification.py` | Deterministic answer verification / scoring | Member 2 |
| `src/evaluation.py` | Benchmark metrics | Shared |
| `app.py` | Gradio interface | Member 2 |
| `main.ipynb` | Colab entry point, sections 1–14 | Shared (one editor at a time) |

## Running (Google Colab, T4)

1. Open `main.ipynb` in Colab, `Runtime > Change runtime type > T4 GPU`.
2. Run sections in order. The notebook clones this repo, installs `requirements.txt`, and mounts Drive.
3. Persistent data lives in `MyDrive/CNDE_Hackathon/` (documents, indexes, results). Nothing private is committed.

## Branches

- `main` — stable, integrated code
- `feature/retrieval` — Member 1 (ingestion, retrieval)
- `feature/llm-verification` — Member 2 (generation, verification, app)

## Status / results

Filled in with measured numbers only. See Evaluation section once benchmarks are run.
