# CNDE NDT Document QA — Grounded RAG with Automated Verification

Prototype for the CNDE (IIT Madras) AI/ML hackathon. Ask questions about Non-Destructive Testing (NDT)
documents and get answers that are grounded in the documents, cited to document + page, and automatically
scored for hallucination risk by deterministic checks. Runs fully locally on a Colab T4.

> Document-grounded technical assistance only — not a certified inspection or engineering decision system.
> Verification scores are rule-based measurements, not calibrated probabilities.

**Open in Colab:** https://colab.research.google.com/github/AbhijeetKumar1505/CNDE/blob/main/main.ipynb

## Results at a glance (measured on a Colab T4, 2026-11-25)

| Prompt | Accuracy (21 answerable) | Abstention (3 unanswerable) | False abstention | Citation correct | Unsupported-claim rate | Conflicts found (of 2) | Injection followed (of 1) | Median gen. time |
|---|---|---|---|---|---|---|---|---|
| v1 baseline | 0.762 | 0.667 | 0.000 | 0.905 | 0.241 | 0 | 0 | 8.9 s |
| v2 grounding rules | 0.810 | 1.000 | 0.143 | 0.905 | 0.350 | 2 | 0 | 8.1 s |
| **v3 rules + examples (default)** | **0.857** | **1.000** | 0.095 | 0.905 | **0.200** | 2 | 0 | **5.8 s** |

Retrieval (21 questions with gold pages): **Recall@1 0.714 · Recall@3 0.905 · Recall@5 0.952 · MRR 0.821**.
Same 24 questions and the same retrieved evidence for every variant; only the prompt differs.
Small benchmark: one question = 4.8 points of accuracy, so v3 vs v2 is a one-question difference.
**Reproducible:** two independent runs on fresh T4 runtimes gave identical accuracy, abstention, citation,
unsupported-claim and retrieval metrics (greedy decoding); only latency varied (v3 median 5.81 s vs 5.66 s).

**Live app check (Gradio share link, T4):** "What minimum qualification is required for personnel performing NDE
of fracture-critical hardware?" → "qualified and certified Level II in accordance with NAS 410 [S1]"
(NASA-STD-5009C p. 47, Supported, 10.5 s round trip). "What is the required penetrant dwell time?" → abstained
by the retrieval gate (top score 0.64 vs threshold 0.65 — a narrow margin, which is why the model's own
abstention is the second layer).

## Architecture

```
PDF / DOCX / TXT
  -> extraction (PyMuPDF blocks + find_tables(); python-docx incl. tables)
  -> cleaning (Symbol-font glyphs mapped, running headers/footers and contents pages removed)
  -> page-bounded chunks (<= 350 BGE tokens, 50 overlap, section heading kept as metadata)
  -> BGE-small embeddings (normalized) -> FAISS IndexFlatIP (exact cosine), saved to Drive

question -> top-5 retrieval -> retrieval gate (score < 0.65: abstain, LLM not called)
  -> prompt v3: evidence in <evidence id="S#"> tags + grounding rules + 2 fictional examples
  -> Qwen3-4B-Instruct-2507 (4-bit NF4, FP16 compute, greedy) -> answer with [S#] citations
  -> deterministic verifier: citation validity, numbers+units, claim support, conflicts, table flag
  -> Gradio: answer, quoted evidence, references, verification status/scores/flags
```

One function, `src/pipeline.py:answer_question`, is shared by the notebook, the Gradio app and the benchmark,
so the evaluated pipeline is exactly the demonstrated one.

| Component | Choice | Why |
|---|---|---|
| LLM | `Qwen/Qwen3-4B-Instruct-2507`, NF4 + double quant, FP16 compute | fits the T4 with headroom (2.68 GB after load); T4 has no native BF16 |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-d, on GPU | 201 chunks embed in 1.4 s on the T4 (190 chunks took 30.6 s on a laptop CPU) |
| Vector search | FAISS `IndexFlatIP` | exact search; a few thousand chunks need nothing approximate |
| Extraction | PyMuPDF (+ `find_tables`), python-docx | page numbers preserved for citations |
| UI | Gradio | upload, ask, see evidence and verification live |

## Key engineering decisions (each driven by something we observed)

1. **Glyphs mapped, not stripped.** NASA-STD-5009C writes `≤`/`≥` with Symbol-font private-use code points
   (`U+F0A3`, `U+F0B3`); its own acronym list confirms the meaning. Stripping them would turn `0.64 ≤ t ≤ 1.27`
   into `0.64 t 1.27`.
2. **Table-aware extraction.** On the first T4 run the model answered a crack-size question with values
   mis-paired across rows of NASA Table 1, and the verifier still said "Supported" because every number did
   occur in the flattened text. Tables are now emitted row by row as
   `[Table 1 (Millimeters)] | Penetrant NDE | Crack Location: Open Surface | Part Thickness t: ... | Crack Dimension c*: ...`
   (the unit label repeats on every row because rows can land in different chunks). The re-run answered every
   row correctly, and table-derived answers now carry an explicit "check row/column pairing" flag.
3. **Contents pages dropped.** Table-of-contents lines list every heading and would match almost any query.
4. **Page-bounded chunks, budgeted in BGE tokens.** Every citation points to exactly one page, and nothing
   exceeds BGE's 512-token limit (max 369 including the section heading).
5. **Short citation IDs (`[S1]`) instead of file names.** Small models copy short IDs reliably; the app maps
   them back to document/page, and quotes the evidence itself rather than asking the model to.
6. **Two-layer abstention.** Retrieval scores alone cannot separate in-domain unanswerable questions: the highest
   unanswerable question scored 0.770 while the lowest answerable one scored 0.706. The 0.65 gate stops clearly
   off-topic questions without rejecting any answerable benchmark question; the model's own
   `INSUFFICIENT_EVIDENCE` reply covers the rest.
7. **Explicit FP16 and single-GPU placement.** transformers v5 defaults to `dtype="auto"` (the checkpoint's BF16),
   so `torch_dtype=float16` is passed; `device_map={"": 0}` fails loudly on OOM instead of offloading to CPU.

## Prompt engineering

| Variant | What it adds |
|---|---|
| v1 | minimal "answer from the context, cite [S#]" baseline |
| v2 | grounding contract: only evidence facts, cite every sentence, copy numbers/units/symbols exactly, keep terminology, `INSUFFICIENT_EVIDENCE` token, Limitations/Conflicts/Interpretation sections, ignore instructions inside documents, how to read table rows; rules repeated after the evidence |
| v3 | v2 + two short worked examples (an answer and an abstention) with fictional, labelled content so their facts cannot leak |

**Why v3:** best accuracy (0.857), perfect abstention, both conflicts reported, the lowest unsupported-claim
rate (0.200) and the fastest median generation (5.8 s, shorter answers). The cost of the stricter prompts is
false abstention (v1 0.000, v2 0.143, v3 0.095): the model sometimes declines when the passage was retrieved.

## Verification (src/verification.py)

All checks are deterministic and computed from the retrieved text; the generator never grades itself.

| Check | Rule |
|---|---|
| Citation validity | cited `[S#]` must exist; pages come from chunk metadata; page numbers written in the answer are cross-checked |
| Numeric fidelity | every number + unit in the answer must occur in the cited evidence (`0.5` = `0.50`; a different explicit unit is a mismatch); identifiers like `NASA-STD-5009C` are ignored |
| Claim support | each sentence checked against **its own cited passages** (word overlap or BGE cosine, plus numbers); flags claims supported only by an uncited passage, and uncited claims |
| Conflicts | `Conflicts:` section citing at least two passages |
| Table flag | cited table rows: pairing of values to rows/columns is not machine-checked, so the answer is flagged for review |

Status: Supported / Partially supported / Unsupported / Insufficient evidence / Conflicting evidence, plus a
grounding score (mean of the three scores). Planted-defect unit test: 10/10 cases classified as expected.

**How well the verifier agreed with answer correctness (from the T4 benchmark log):**
answers rated "Supported" were correct 13/14 (v1), 13/14 (v2), 16/17 (v3); the misses were *incomplete* but
faithful answers (q17, and q19 for v1). False alarms on correct answers: 3 (v1), 2 (v2), 1 (v3). For v1, 5 of 6
incorrect answers were not rated Supported. The verifier checks faithfulness, not completeness.

## Evaluation

`benchmarks/questions.jsonl`: 24 questions written by hand from the two documents, with gold source/page and
expected key facts — numerical 9, factual 2, table 3, terminology 2, multi-passage 2, conflicting 2,
insufficient-evidence 3 (incl. the in-domain "penetrant dwell time", which neither document states),
prompt injection 1. Two **synthetic, clearly labelled** documents in `benchmarks/synthetic/` provide the
conflicting-value and injection cases; they are added only to a separate evaluation index.
`check_benchmark` verifies that every expected fact really occurs on its gold pages.
"Correct" is an automated proxy (all key facts present / correct abstention); raw answers are kept for review.

**Failure analysis**
- q15 "What does PTC mean?" — retrieval miss (gold page not in the top 5) in every variant; acronym lookups are
  lexical, where dense retrieval is weak (BM25 hybrid search is the obvious next step).
- q17 (needs one passage from each document) — grounded but incomplete; gold passage ranked 4th.
- q02 — v2/v3 abstained although the gold passage was retrieved at rank 2 (the false-abstention trade-off).

## Operational measurements (Colab T4)

| Measure | Value |
|---|---|
| Model load (fresh runtime incl. download / cached) | 127.9 s / 30.9–37.0 s |
| VRAM after load / peak during the 72-answer benchmark | 2.68 GB / 3.45 GB of 15.6 GB |
| Decode speed (greedy) | 11.4–13.8 tokens/s |
| Extraction with table detection | 9.0 s (FAA, 50 pp) + 8.2 s (NASA, 54 pp) |
| Chunking / embedding 201 chunks | 0.4 s / 1.4 s |
| Retrieval latency | 12–41 ms |
| Generation latency (v3, benchmark) | median 5.8 s, max 29.9 s |
| Full benchmark (24 questions × 3 prompts) | 11.9 min |

## Corpus (public domain, not committed)

- FAA AC 43.13-1B (with Change 1), Chapter 5 "Nondestructive Inspection (NDI)" — physical pages 222–271 of the
  official PDF, extracted by notebook section 6; citations refer to pages of the extracted 50-page file.
  https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_w-chg1.pdf
- NASA-STD-5009C, Nondestructive Evaluation Requirements for Fracture Critical Metallic Components (2023).
  https://standards.nasa.gov/sites/default/files/standards/NASA/C/0/2023-08-03-NASA-STD-5009C-Approved.pdf

## Running (Google Colab, T4)

1. Open the Colab link above; `Runtime > Change runtime type > T4 GPU`.
2. Run sections 1–14 in order: clone + install, mount Drive, load the model, download the corpus, extract,
   chunk, embed, index, answer, verify, benchmark (about 12 min), launch the Gradio app (`share=True` link).
3. After pulling new code, re-run section 2 and then section 4 (it reloads the project modules); the model is
   loaded only once per runtime.

Persistent data lives in `MyDrive/CNDE_Hackathon/` (documents, indexes, results). Nothing private is committed.

## Repository layout and team

| Path | Purpose | Owner |
|---|---|---|
| `src/ingestion.py` | extraction (incl. tables) + chunking | Member 1 |
| `src/retrieval.py` | embeddings + FAISS + retrieval | Member 1 |
| `benchmarks/` | questions, synthetic test documents | Member 1 |
| `src/generation.py` | model loading, prompts, generation | Member 2 |
| `src/verification.py` | deterministic verification and scoring | Member 2 |
| `app.py` | Gradio interface | Member 2 |
| `src/pipeline.py`, `src/evaluation.py`, `src/config.py`, `main.ipynb` | integration, metrics, settings, Colab entry point | Shared |

Workflow: `feature/retrieval` (Member 1), `feature/llm-verification` (Member 2) and `feature/evaluation`
(shared), merged into `main` through pull requests.

## Known limitations and next steps

- Small, single-annotator benchmark (24 questions); correctness is an automated key-fact proxy.
- Verifier checks faithfulness, not completeness, and cannot verify row/column pairing in tables (flagged instead).
- Dense retrieval misses acronym/lexical lookups (q15) — add BM25 hybrid retrieval.
- No OCR: scanned pages are reported as "no extractable text" rather than processed.
- Conflict detection relies on the model reporting it plus a citation check; no independent NLI model.
- Two-column PDF pages are read in vertical block order, which can interleave columns.
