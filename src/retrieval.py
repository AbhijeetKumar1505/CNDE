"""Embedding + FAISS semantic retrieval.  (Owner: Member 1)

Contract
--------
load_embedder(model_id, device) -> SentenceTransformer
build_index(chunks, embedder) -> faiss.Index          # normalized vectors + inner product == cosine
save_index(index, chunks, directory) / load_index(directory) -> (index, chunks)
retrieve(question, index, chunks, embedder, top_k) -> list[Evidence]
    Evidence = {"text": str, "source": str, "page": int | None,
                "chunk_id": str, "section": str, "score": float}
"""


def load_embedder(model_id, device="cpu"):
    raise NotImplementedError("Milestone 2")


def build_index(chunks, embedder):
    raise NotImplementedError("Milestone 2")


def save_index(index, chunks, directory):
    raise NotImplementedError("Milestone 2")


def load_index(directory):
    raise NotImplementedError("Milestone 2")


def retrieve(question, index, chunks, embedder, top_k=5):
    raise NotImplementedError("Milestone 2")
