"""Embedding + FAISS semantic retrieval.  (Owner: Member 1)

Contract
--------
load_embedder(model_id, device) -> SentenceTransformer
token_counter(embedder) -> callable(str) -> int           # BGE token count, used by the chunker
build_index(chunks, embedder) -> faiss.Index              # normalized vectors + inner product == cosine
add_to_index(index, chunks, new_chunks, embedder)         # append uploaded documents in place
save_index(index, chunks, directory, meta) / load_index(directory, expected_embed_id) -> (index, chunks)
retrieve(question, index, chunks, embedder, top_k) -> list[Evidence]
    Evidence = {"text": str, "source": str, "page": int | None,
                "chunk_id": str, "section": str, "score": float}
"""
import json
from pathlib import Path

import faiss
import numpy as np

# BGE v1.5 model card: prefix short retrieval queries with this instruction (passages get no prefix).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_embedder(model_id, device="auto"):
    """device="auto" -> CUDA when available. bge-small is small enough to share the T4 with the LLM."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_id, device=device)


def token_counter(embedder):
    tokenizer = embedder.tokenizer
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _embedding_text(chunk: dict) -> str:
    """Section heading + verbatim text: the heading often carries context the body text lacks."""
    return f"{chunk['section']}\n{chunk['text']}" if chunk.get("section") else chunk["text"]


def embed_texts(embedder, texts, batch_size=32) -> np.ndarray:
    vectors = embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
    return np.ascontiguousarray(vectors, dtype="float32")


def build_index(chunks, embedder):
    if not chunks:
        raise ValueError("No chunks to index - did extraction produce any text?")
    vectors = embed_texts(embedder, [_embedding_text(c) for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; a few thousand chunks needs nothing fancier
    index.add(vectors)
    return index


def add_to_index(index, chunks, new_chunks, embedder) -> None:
    """Append new chunks to an existing index and its metadata list, keeping both in the same order."""
    if new_chunks:
        index.add(embed_texts(embedder, [_embedding_text(c) for c in new_chunks]))
        chunks.extend(new_chunks)


def save_index(index, chunks, directory, meta=None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "index.faiss"))
    payload = {"meta": meta or {}, "chunks": chunks}
    (directory / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(directory, expected_embed_id=None):
    directory = Path(directory)
    index = faiss.read_index(str(directory / "index.faiss"))
    payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    chunks, meta = payload["chunks"], payload.get("meta", {})
    if index.ntotal != len(chunks):
        raise ValueError(f"Index and metadata out of sync: {index.ntotal} vectors vs {len(chunks)} chunks")
    if expected_embed_id and meta.get("embed_id") not in (None, expected_embed_id):
        raise ValueError(f"Index built with {meta.get('embed_id')}, but {expected_embed_id} is loaded")
    return index, chunks


def retrieve(question, index, chunks, embedder, top_k=5) -> list:
    if index.ntotal == 0:
        return []
    query = embed_texts(embedder, [QUERY_INSTRUCTION + question.strip()])
    scores, ids = index.search(query, min(top_k, index.ntotal))
    return [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "page": chunks[i]["page"],
         "chunk_id": chunks[i]["chunk_id"], "section": chunks[i].get("section", ""),
         "score": round(float(s), 4)}
        for s, i in zip(scores[0], ids[0]) if i != -1
    ]
