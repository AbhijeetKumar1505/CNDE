"""Document ingestion: PDF / DOCX / TXT -> page records -> structure-aware chunks.  (Owner: Member 1)

Contract
--------
extract_document(path) -> list[PageRecord]
    PageRecord = {"source": str, "page": int | None, "blocks": list[str]}
    page is 1-based for PDFs and None for DOCX/TXT (no reliable pagination).

chunk_pages(pages, count_tokens, max_tokens, overlap) -> list[Chunk]
    Chunk = {"chunk_id": str, "source": str, "page": int | None,
             "section": str, "text": str, "n_tokens": int}
    A chunk never crosses a page boundary, so every chunk cites exactly one page.
"""


def extract_document(path):
    raise NotImplementedError("Milestone 1")


def chunk_pages(pages, count_tokens, max_tokens, overlap):
    raise NotImplementedError("Milestone 1")
