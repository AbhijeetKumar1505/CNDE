"""Gradio interface: upload documents, ask questions, see answer + evidence + verification.  (Owner: Member 2)

Launched from main.ipynb section 14:
    demo = build_app(cfg, model, tokenizer, embedder, index, chunks); demo.launch(share=True)

create_handlers() holds all logic (testable without a browser); build_app() only wires the UI.
"""
import shutil
import time
from pathlib import Path

from src.ingestion import SUPPORTED_EXTENSIONS, ingest_files
from src.pipeline import answer_question, to_markdown
from src.retrieval import add_to_index, save_index, token_counter

EXAMPLES = [
    "What frequency range do portable ultrasonic inspection instruments cover?",
    "For penetrant NDE of an open-surface crack in a part between 0.64 mm and 1.27 mm thick, what crack size must be reliably detectable?",
    "What minimum qualification is required for personnel performing NDE of fracture-critical hardware?",
    "What frequency range should an eddy current instrument cover?",
    "What is the required penetrant dwell time?",
]
DISCLAIMER = ("Document-grounded technical assistance only - not a certified inspection or engineering "
              "decision system. Verification scores are rule-based checks, not probabilities.")


def create_handlers(cfg, model, tokenizer, embedder, index, chunks):
    state = {"index": index, "chunks": chunks}      # uploads are added to this corpus in place
    count_tokens = token_counter(embedder)

    def corpus_table():
        docs = {}
        for c in state["chunks"]:
            d = docs.setdefault(c["source"], {"pages": set(), "chunks": 0})
            d["chunks"] += 1
            if c["page"] is not None:
                d["pages"].add(c["page"])
        return [[name, len(d["pages"]) or "n/a", d["chunks"]] for name, d in sorted(docs.items())]

    def upload(files):
        if not files:
            return "No files selected.", corpus_table()
        known = {c["source"] for c in state["chunks"]}
        notes, paths = [], []
        for f in files:
            p = Path(f if isinstance(f, (str, Path)) else f.name)    # Gradio 4+: path string; 3.x: tempfile
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                notes.append(f"✗ {p.name}: unsupported type (use PDF, DOCX or TXT)")
            elif p.name in known:
                notes.append(f"• {p.name}: already indexed - skipped")
            else:
                cfg.docs_dir.mkdir(parents=True, exist_ok=True)
                paths.append(Path(shutil.copy(p, cfg.docs_dir / p.name)))
        if paths:
            new_chunks, report = ingest_files(paths, count_tokens, cfg.chunk_tokens, cfg.chunk_overlap)
            add_to_index(state["index"], state["chunks"], new_chunks, embedder)
            save_index(state["index"], state["chunks"], cfg.index_dir,
                       meta={"embed_id": cfg.embed_id, "chunk_tokens": cfg.chunk_tokens, "chunk_overlap": cfg.chunk_overlap})
            for r in report:
                if r["error"]:
                    notes.append(f"✗ {r['source']}: {r['error']}")
                    continue
                notes.append(f"✓ {r['source']}: {r['pages']} page(s), {r['chunks']} chunks in {r['seconds']} s")
                if r["chunks"] == 0:
                    notes.append(f"  ⚠ no extractable text - scanned documents need OCR, which is not supported")
                elif r["empty_pages"]:
                    notes.append(f"  ⚠ pages without extractable text (scanned or contents-only): {r['empty_pages']}")
        return "\n".join(notes), corpus_table()

    def ask(question, variant="v3"):
        if not question or not question.strip():
            return "Please enter a question.", {}
        t0 = time.perf_counter()
        r = answer_question(question, state["index"], state["chunks"], embedder, model, tokenizer, cfg, variant)
        v = r["verification"]
        details = {
            "status": v["status"], "scores": v["scores"], "flags": v["flags"],
            "claims": v["claims"], "numbers_checked": v["numbers"],
            "retrieved": [{"id": f"S{i + 1}", "source": e["source"], "page": e["page"], "section": e["section"],
                           "score": e["score"]} for i, e in enumerate(r["evidence"])],
            "gated_by_retrieval_threshold": r["gated"], "prompt_variant": variant,
            "timing": {"retrieval_ms": r["retrieval_ms"], "generation_s": r["latency_s"],
                       "total_s": round(time.perf_counter() - t0, 2)},
            "tokens": {"prompt": r["prompt_tokens"], "generated": r["new_tokens"], "hit_max": r["hit_max_tokens"]},
            "raw_model_output": r["answer"],
        }
        return to_markdown(r), details

    return {"corpus_table": corpus_table, "upload": upload, "ask": ask}


def build_app(cfg, model, tokenizer, embedder, index, chunks):
    import gradio as gr

    h = create_handlers(cfg, model, tokenizer, embedder, index, chunks)
    with gr.Blocks(title="CNDE NDT Document QA") as demo:
        gr.Markdown(f"# NDT Document QA with automated verification\n{DISCLAIMER}")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Upload PDF / DOCX / TXT", file_count="multiple",
                                file_types=sorted(SUPPORTED_EXTENSIONS))
                upload_btn = gr.Button("Process documents")
                status = gr.Textbox(label="Processing status", lines=5, interactive=False)
                corpus = gr.Dataframe(value=h["corpus_table"](), headers=["Document", "Pages", "Chunks"],
                                      label="Indexed documents", interactive=False)
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", lines=2,
                                      placeholder="Ask about the indexed NDT documents...")
                with gr.Row():
                    variant = gr.Dropdown(choices=["v3", "v2", "v1"], value="v3", label="Prompt variant",
                                          info="v3 performed best on the benchmark")
                    ask_btn = gr.Button("Ask", variant="primary")
                gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question])
                answer = gr.Markdown(label="Answer")
                with gr.Accordion("Verification details, retrieval scores and timing", open=False):
                    details = gr.JSON()
        upload_btn.click(h["upload"], inputs=[files], outputs=[status, corpus])
        ask_btn.click(h["ask"], inputs=[question, variant], outputs=[answer, details])
        question.submit(h["ask"], inputs=[question, variant], outputs=[answer, details])
    return demo
