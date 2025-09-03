# indexing/pdf_utils.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

def _id(c: dict) -> Optional[str]:
    return c.get("id") or c.get("chunk_id")

def _md(c: dict) -> dict:
    return c.get("metadata", {}) or {}

def shape_pdf_results(chunks: list[dict]) -> dict[str, list[dict]]:
    """
    Bucket raw retrieval results into semantic groups for downstream use.
    Buckets (stable order, de-duped by id): topics, questions, answers, tables, pages, other.

    - Accepts both metadata.chunk_type and legacy metadata.symbol_kind
    - Heuristics from metadata.name / first line of document if type is missing
    - 'toc' is treated as a topic (useful for overviews)
    """
    # fallbacks in case module-level helpers aren't present
    _id_fn = globals().get("_id", lambda c: c.get("id") or c.get("chunk_id"))
    _md_fn = globals().get("_md", lambda c: (c.get("metadata") or {}))

    TOPIC_TYPES = {
        "chapter", "section", "subsection", "paragraph", "heading", "title", "intro", "toc"
    }
    QUESTION_TYPES = {
        "questions", "question", "qa", "exercise", "exercises", "problems", "practice"
    }
    ANSWER_TYPES = {
        "answer", "answers", "solution", "solutions", "worked_solution"
    }
    TABLE_TYPES = {
        "table", "tables", "tabular", "grid"
    }
    PAGE_TYPES = {
        "pdf_page", "page"
    }

    import re

    def _infer_type(c: dict) -> str:
        md = _md_fn(c)
        t = (md.get("chunk_type") or md.get("symbol_kind") or "").strip().lower()

        if not t or t == "unknown":
            # try from name first
            name = (md.get("name") or "").strip()
            header = name or (c.get("document", "").splitlines()[:1] or [""])[0]
            h = header.strip().lower()

            if re.match(r"^(questions|exercises?|practice|problems)\b", h):
                t = "questions"
            elif re.match(r"^(answers?|solutions?)\b", h):
                t = "answer"
            elif re.match(r"^table\s*\d*", h) or "│" in header or "┼" in header:
                t = "table"
            elif re.match(r"^(chapter\s*\d+|[0-9]+\s+[A-Z][^\n]{1,80})$", header):
                t = "chapter"
            elif re.match(r"^\d+\.\d+(\.\d+)?\s+", header):
                t = "section"
            elif "page" in h:
                t = "pdf_page"

        # canonicalize to a known bucket key
        if t in TOPIC_TYPES:
            return "topic"
        if t in QUESTION_TYPES:
            return "question"
        if t in ANSWER_TYPES:
            return "answer"
        if t in TABLE_TYPES:
            return "table"
        if t in PAGE_TYPES:
            return "page"
        return t or "other"

    buckets = {
        "topics": [],
        "questions": [],
        "answers": [],
        "tables": [],
        "pages": [],
        "other": [],
    }
    seen_ids = set()

    for c in chunks or []:
        cid = _id_fn(c)
        if cid and cid in seen_ids:
            continue
        ctype = _infer_type(c)
        if ctype == "topic":
            buckets["topics"].append(c)
        elif ctype == "question":
            buckets["questions"].append(c)
        elif ctype == "answer":
            buckets["answers"].append(c)
        elif ctype == "table":
            buckets["tables"].append(c)
        elif ctype == "page":
            buckets["pages"].append(c)
        else:
            buckets["other"].append(c)
        if cid:
            seen_ids.add(cid)

    try:
        from loguru import logger
        logger.debug(
            "pdf_utils.shape: topics={} questions={} answers={} tables={} pages={} other={}",
            len(buckets["topics"]), len(buckets["questions"]), len(buckets["answers"]),
            len(buckets["tables"]), len(buckets["pages"]), len(buckets["other"])
        )
    except Exception:
        pass

    return buckets

def expand_pdf_neighbors(selected: List[dict], all_by_id: Dict[str, dict], cap_extra: int = 4) -> List[dict]:
    """
    Add high-value neighbors for each selected chunk:
    - parent_page (container)
    - prev_id / next_id (adjacent containers)
    """
    out = list(selected or [])
    seen = { _id(c) for c in out if _id(c) }
    added = 0

    def _add(cid: Optional[str]):
        nonlocal added
        if cid and cid in all_by_id and cid not in seen and added < cap_extra:
            out.append(all_by_id[cid])
            seen.add(cid)
            added += 1

    for c in list(selected or []):
        md = _md(c)
        _add(md.get("parent_page"))
        _add(md.get("prev_id"))
        _add(md.get("next_id"))

    logger.debug("pdf_utils.expand: +{} neighbors (cap={})", added, cap_extra)
    return out

def filter_by_scope(chunks: List[dict], scope: str, book: Optional[str], chapter: Optional[str]) -> List[dict]:
    """
    Scope results to 'chapter' or 'book' (or leave as-is for 'project').
    """
    # If we lack book/chapter hints, don't over-filter; pass-through.
    if not book and not chapter:
        return chunks or []
    
    res = []
    for c in chunks or []:
        md = _md(c)
        if scope == "book"    and md.get("book") == book:
            res.append(c)
        elif scope == "chapter" and md.get("book") == book and md.get("chapter") == chapter:
            res.append(c)
    return res

def select_book_chapter(shaped: Dict[str, List[dict]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic: peek into buckets to infer (book, chapter) from the first available hit.
    """
    for bucket in ("topics", "questions", "pages", "other"):
        arr = shaped.get(bucket) or []
        if arr:
            md = _md(arr[0])
            return md.get("book"), md.get("chapter")
    return None, None

def expand_pdf_neighbors_meta(
    seeds: List[dict],
    pool: List[dict],
    *,
    pages_before: int = 1,
    pages_after: int = 1,
    include_page_containers: bool = True,
    include_paragraphs: bool = True,
    include_sibling_paragraphs: bool = True,
    cap_extra: int = 6,
) -> List[dict]:
    """
    Add nearby context by metadata, not IDs:
      - previous/next pages
      - optional neighboring paragraph windows on the same page

    Works purely on the 'pool' you already retrieved. No extra DB calls.
    """

    def _key(c):
        md = _md(c)
        return md.get("file_path"), md.get("page_number")

    # Index pool by (file, page)
    by_fp_page: dict[tuple, list[dict]] = {}
    for c in pool or []:
        k = _key(c)
        if not all(k):  # need both file_path and page_number
            continue
        by_fp_page.setdefault(k, []).append(c)

    # Stable type filterers
    def _is_page(c): return (_md(c).get("chunk_type") == "pdf_page")
    def _is_para(c): return (_md(c).get("chunk_type") in {"paragraph", "section", "qa", "questions", "answer", "table"})

    # Try to order paragraph windows on a page by their window number if present
    def _sort_page_chunks(chs: list[dict]) -> list[dict]:
        import re
        def _order(c):
            name = (_md(c).get("name") or "")
            m = re.search(r"¶(?:window\s*)?(\d+)", name)
            return (0 if _is_page(c) else 1, int(m.group(1)) if m else 10**6)
        return sorted(chs, key=_order)

    # Build a quick lookup for position of a paragraph among siblings
    by_fp_sorted: dict[tuple, list[dict]] = {k: _sort_page_chunks(v) for k, v in by_fp_page.items()}
    pos_index: dict[str, tuple[tuple, int]] = {}
    for k, arr in by_fp_sorted.items():
        for idx, c in enumerate(arr):
            cid = c.get("id") or c.get("chunk_id")
            if cid:
                pos_index[cid] = (k, idx)

    out = list(seeds or [])
    seen = { (c.get("id") or c.get("chunk_id")) for c in out if (c.get("id") or c.get("chunk_id")) }
    added = 0

    def _try_add(c):
        nonlocal added
        cid = c.get("id") or c.get("chunk_id")
        if cid and cid not in seen and added < cap_extra:
            out.append(c); seen.add(cid); added += 1

    for s in seeds or []:
        md = _md(s)
        fp, p = md.get("file_path"), md.get("page_number")
        if not (fp and isinstance(p, int)):
            continue

        # ± page containers
        if include_page_containers:
            for d in range(-pages_before, pages_after + 1):
                if d == 0:  # current page handled below via siblings list too
                    continue
                for c in by_fp_page.get((fp, p + d), []):
                    if _is_page(c):
                        _try_add(c)

        # neighboring paragraph windows on the same page
        if include_paragraphs and include_sibling_paragraphs:
            cid = s.get("id") or s.get("chunk_id")
            if cid in pos_index:
                k, idx = pos_index[cid]
                arr = by_fp_sorted.get(k, [])
                # previous/next paragraphs (skip page container at index 0 if present)
                for j in (idx - 1, idx + 1):
                    if 0 <= j < len(arr):
                        c = arr[j]
                        if _is_para(c):
                            _try_add(c)

        # if seed itself is a page container, add a couple of paragraph windows from same page
        if include_paragraphs and _is_page(s):
            for c in by_fp_page.get((fp, p), [])[:4]:  # first few windows are usually the page header/body
                if _is_para(c):
                    _try_add(c)

    logger.debug("pdf_utils.expand_meta: +{} neighbors (cap={})", added, cap_extra)
    return out
