"""Document ingestion: PDF / DOCX / TXT -> page records -> structure-aware chunks.  (Owner: Member 1)

Contract
--------
extract_document(path) -> list[PageRecord]
    PageRecord = {"source": str, "page": int | None, "blocks": list[Block]}
    Block      = {"text": str, "heading": str | None}   # heading detected at the start of the block
    page is 1-based for PDFs (physical page in the file) and None for DOCX / single-page TXT.

chunk_pages(pages, count_tokens, max_tokens, overlap) -> list[Chunk]
    Chunk = {"chunk_id": str, "source": str, "page": int | None,
             "section": str, "text": str, "n_tokens": int}
    A chunk never crosses a page boundary, so every chunk cites exactly one page.

ingest_files(paths, count_tokens, max_tokens, overlap) -> (chunks, report)
    report: one dict per file with page count, empty_pages (no usable text after cleaning: scanned
    or table-of-contents only), chunk count, time, error.
"""
import re
import time
from collections import Counter
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Private-use code points emitted by PDFs that use the Symbol font. They are MAPPED, not stripped:
# dropping "≤" from "0.64 ≤ t ≤ 1.27" silently changes a requirement. Verified on NASA-STD-5009C
# (its acronym list reads "\uf0a3 less than or equal to") and FAA AC 43.13-1B ("5-8.\uf8e75-14.").
GLYPH_MAP = {
    "\uf0a3": "≤", "\uf0b3": "≥", "\uf0b1": "±", "\uf0b0": "°", "\uf06d": "μ",
    "\uf0b4": "×", "\uf0b9": "≠", "\uf0bb": "≈", "\uf0ae": "→", "\uf0b7": "•", "\uf8e7": "–",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00a0": " ", "\u00ad": "",  # non-breaking space, soft hyphen
}
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")

# Heading patterns seen in NDT standards / advisory circulars.
_HEADING_PATTERNS = [
    re.compile(r"^(CHAPTER|SECTION|APPENDIX|PART)\s+[\w.-]+.*$", re.I),
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][\w ,&()/'-]{2,80}$"),      # "4.2.1 General Requirements"
]
_INLINE_AC_HEADING = re.compile(r"^(\d+-\d+\.\s+[A-Z][A-Z0-9 ,&()/'-]{2,80}\.)")  # "5-1. GENERAL."


def clean_text(text: str) -> str:
    """Map Symbol-font glyphs, drop remaining private-use chars, re-join hyphenated line breaks."""
    for src, dst in GLYPH_MAP.items():
        text = text.replace(src, dst)
    text = _PRIVATE_USE.sub("", text)
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


def _detect_heading(raw_block: str):
    """Return the heading at the start of a block (whole short block, or an inline AC-style title)."""
    lines = [l.strip() for l in raw_block.splitlines() if l.strip()]
    if not lines:
        return None
    first = lines[0]
    if re.fullmatch(r"\d+(\.\d+)+\.?", first) and len(lines) > 1 and len(lines[1].split()) <= 10:
        return f"{first} {lines[1]}"                     # NASA style: number and title on separate lines
    if len(first.split()) <= 12 and any(p.match(first) for p in _HEADING_PATTERNS):
        return first
    m = _INLINE_AC_HEADING.match(" ".join(lines))
    return m.group(1) if m else None


_TOC_LEADER = re.compile(r"\.{6,}|(\.\s){6,}")  # "1.2 Applicability ........ 7"


def _make_block(raw: str) -> dict:
    raw = clean_text(raw)
    if _TOC_LEADER.search(raw):
        # Table-of-contents entries list every heading, so they would match almost any query and
        # crowd real content out of the top-k. Dropped at extraction time.
        return {"text": "", "heading": None}
    return {"text": re.sub(r"\s+", " ", raw).strip(), "heading": _detect_heading(raw)}


def _remove_boilerplate(pages: list) -> None:
    """Drop short blocks (running headers/footers) repeated on >=40% of a document's pages."""
    if len(pages) < 4:
        return
    key = lambda t: re.sub(r"\d+", "#", t.lower())
    counts = Counter()
    for p in pages:
        counts.update({key(b["text"]) for b in p["blocks"] if len(b["text"].split()) <= 15})
    threshold = max(3, int(0.4 * len(pages)))
    boiler = {k for k, n in counts.items() if n >= threshold}
    for p in pages:
        p["blocks"] = [b for b in p["blocks"] if key(b["text"]) not in boiler]


_TABLE_HINT = re.compile(r"\btable\b", re.I)   # find_tables() is slow (~0.3 s/page); only run it where useful


def _cell(text: str, sep: str = " / ") -> str:
    text = clean_text(text or "").replace("**", "").replace("<br>", sep)
    return re.sub(r"\s+", " ", text).strip(" ,")


def _table_blocks(page, text_blocks):
    """Tables as row-wise, self-describing lines ("group | Column: value | ..."), so each value stays
    attached to its row and column. Plain text extraction flattens tables and lets the LLM mis-pair
    values (seen on NASA-STD-5009C Table 1). Each row repeats a short label carrying the table's unit,
    because a row may end up in a different chunk from the caption.

    Returns (blocks with a "y" position, table rectangles used to drop the flattened duplicate text)."""
    import pymupdf

    out, rects = [], []
    for t in page.find_tables().tables:
        md = t.to_markdown()
        lines = [l for l in md.splitlines() if l.startswith("|")]
        if len(lines) < 3 or _TOC_LEADER.search(md):           # table-of-contents pages are detected as tables
            continue
        split = lambda l: l.strip().strip("|").split("|")
        rect = pymupdf.Rect(t.bbox)
        header, body = [_cell(h, " ") for h in split(lines[0])], lines[2:]
        caption = ""
        if re.match(r"\s*table\b", "".join(header), re.I) and body:   # caption detected as the header row
            caption, header, body = "".join(header), [_cell(h, " ") for h in split(body[0])], body[1:]
        if not caption:                                         # caption = text blocks between "Table ..." and the grid
            above = [b for b in text_blocks if b[3] <= rect.y0 + 2 and rect.y0 - b[3] < 80]
            starts = [b for b in above if re.match(r"\s*table\b", b[4], re.I)]
            if starts:
                top = max(starts, key=lambda b: b[1])[1]
                caption = " ".join(_cell(b[4], " ") for b in sorted(above, key=lambda b: b[1]) if b[1] >= top)
        m = re.match(r"\s*table\s+([\w-]+(?:\.\d+)?)", caption, re.I)
        unit = re.search(r"\((millimeters|inches|mm|in\.?)\)", caption, re.I)
        label = f"[Table{' ' + m.group(1) if m else ''}{' (' + unit.group(1) + ')' if unit else ''}]"
        rows, group = [], ""
        for line in body:
            cells = [_cell(c) for c in split(line)]
            values = [c for c in cells if c]
            if not values:
                continue
            if len(cells) > 1 and len(set(values)) == 1:      # spanning row, e.g. "Penetrant NDE"
                group = values[0]
                continue
            pairs = [f"{h}: {v}" if h else v for h, v in zip(header, cells) if v]
            rows.append(" | ".join([label] + ([group] if group else []) + pairs))
        if rows:
            out.append({"text": "\n".join(([caption] if caption else []) + rows), "heading": None, "y": rect.y0})
            rects.append(rect)
    return out, rects


def _extract_pdf(path: Path) -> list:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            raw = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]   # type 0 = text
            tables, rects = _table_blocks(page, raw) if _TABLE_HINT.search(page.get_text()) else ([], [])
            inside = lambda b: any(r.contains(pymupdf.Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for r in rects)
            blocks = [{**_make_block(b[4]), "y": b[1]} for b in raw if not inside(b)] + tables
            blocks.sort(key=lambda b: b["y"])                  # tables keep their reading-order position
            pages.append({"source": path.name, "page": i,
                          "blocks": [{"text": b["text"], "heading": b["heading"]} for b in blocks if b["text"]]})
    _remove_boilerplate(pages)
    return pages


def _extract_docx(path: Path) -> list:
    import docx
    from docx.table import Table

    blocks = []
    for item in docx.Document(path).iter_inner_content():
        if isinstance(item, Table):
            rows = [" | ".join(c.text.strip() for c in row.cells) for row in item.rows]
            text = clean_text("\n".join(rows)).strip()
            if text:
                blocks.append({"text": text, "heading": None})       # keep row breaks for splitting
        else:
            text = re.sub(r"\s+", " ", clean_text(item.text)).strip()
            if text:
                style = (item.style.name or "") if item.style is not None else ""
                is_heading = style.startswith(("Heading", "Title"))
                blocks.append({"text": text, "heading": text if is_heading else _detect_heading(text)})
    return [{"source": path.name, "page": None, "blocks": blocks}]


def _extract_txt(path: Path) -> list:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parts = raw.split("\f")                       # form feeds mark pages in many text exports
    pages = []
    for i, part in enumerate(parts, start=1):
        blocks = [_make_block(b) for b in re.split(r"\n\s*\n", part)]
        pages.append({"source": path.name, "page": i if len(parts) > 1 else None,
                      "blocks": [b for b in blocks if b["text"]]})
    return pages


def extract_document(path) -> list:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".txt":
        return _extract_txt(path)
    raise ValueError(f"Unsupported file type '{ext}' ({path.name}). Supported: {sorted(SUPPORTED_EXTENSIONS)}")


# ----------------------------------------------------------------------------------------- chunking
_ABBREVIATIONS = {"fig", "figs", "e.g", "i.e", "no", "vol", "ref", "approx", "etc", "par", "para", "sec", "eq", "vs"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list:
    """Sentence split that keeps decimals (0.5), figure refs (Fig. 5-2) and common abbreviations intact."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        before = text[start:m.start()].split()
        if before and before[-1].rstrip(".").lower() in _ABBREVIATIONS:
            continue
        out.append(text[start:m.start()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def _split_oversized(text: str, count_tokens, max_tokens: int) -> list:
    """Split a unit larger than max_tokens: by line (table rows), then sentence, then words."""
    if count_tokens(text) <= max_tokens:
        return [text]
    for splitter in (lambda t: t.split("\n"), split_sentences):
        parts = [p.strip() for p in splitter(text) if p.strip()]
        if len(parts) > 1:
            return [piece for p in parts for piece in _split_oversized(p, count_tokens, max_tokens)]
    pieces, current = [], []
    for w in text.split():                            # last resort: hard split on word boundaries
        if current and count_tokens(" ".join(current + [w])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(w)
    return pieces + ([" ".join(current)] if current else [])


def _safe_id(name: str) -> str:
    """Filename incl. extension, so report.pdf and report.txt never produce the same chunk ids."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def chunk_pages(pages, count_tokens, max_tokens=350, overlap=50) -> list:
    """Greedy, page-bounded packing of blocks into chunks of <= max_tokens, with whole-unit overlap.

    Section headings are tracked across pages and stored as chunk metadata (used in the embedding text
    and shown in citations); the chunk text itself stays verbatim.
    """
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    chunks, section, last_source = [], "", None
    for page in pages:
        if page["source"] != last_source:
            section, last_source = "", page["source"]
        units = []                                    # (text, n_tokens, section in force at this unit)
        for block in page["blocks"]:
            if block["heading"]:
                section = block["heading"]
            for piece in _split_oversized(block["text"], count_tokens, max_tokens):
                units.append((piece, count_tokens(piece), section))

        current, n = [], 0

        def flush():
            if current:
                chunks.append({
                    "chunk_id": f"{_safe_id(page['source'])}-p{page['page'] or 'na'}-c{len(chunks):04d}",
                    "source": page["source"], "page": page["page"], "section": current[0][2],
                    "text": "\n".join(u[0] for u in current), "n_tokens": sum(u[1] for u in current),
                })

        for unit in units:
            if current and n + unit[1] > max_tokens:
                flush()
                carry, carried = [], 0                # trailing whole units up to `overlap` tokens
                for u in reversed(current):
                    if carried + u[1] > overlap:
                        break
                    carry.insert(0, u)
                    carried += u[1]
                current, n = carry, carried
            current.append(unit)
            n += unit[1]
        flush()
    return chunks


def ingest_files(paths, count_tokens, max_tokens=350, overlap=50):
    """Extract + chunk several files. Failures are reported per file, never silently skipped."""
    all_chunks, report = [], []
    for path in map(Path, paths):
        t0 = time.perf_counter()
        entry = {"source": path.name, "pages": 0, "empty_pages": [], "chunks": 0, "seconds": 0.0, "error": None}
        try:
            pages = extract_document(path)
            chunks = chunk_pages(pages, count_tokens, max_tokens, overlap)
            entry.update(pages=len(pages), chunks=len(chunks),
                         empty_pages=[p["page"] for p in pages if not p["blocks"]])
            all_chunks.extend(chunks)
        except Exception as e:  # surfaced in the report / UI rather than swallowed
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        report.append(entry)
    return all_chunks, report
