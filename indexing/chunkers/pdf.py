# indexing/chunkers/pdf.py
from __future__ import annotations

import io
import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

# --- PyMuPDF + optional OCR ---
import fitz  # PyMuPDF
try:
    import pytesseract
    from PIL import Image
    _OCR_OK = True
except Exception:  # pillow or tesseract not installed
    _OCR_OK = False


# ------------------------- small utils -------------------------

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def _get(cfg: dict, *path, default=None):
    cur = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _dehyphenate(text: str) -> str:
    # common: “exam-\nination” -> “examination”
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # normalize CRLF
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _normalize_ws(text: str) -> str:
    # collapse triple+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing spaces on lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text

def _first_non_empty(lines: List[str]) -> str:
    for ln in lines:
        s = ln.strip()
        if s:
            return s
    return ""

def _last_non_empty(lines: List[str]) -> str:
    for ln in reversed(lines):
        s = ln.strip()
        if s:
            return s
    return ""

def _classify_block(text: str) -> str:
    """
    Very light heuristics to help downstream grouping:
      - starts with Questions / Exercise / Practice => 'questions'
      - starts with Answers / Solutions              => 'answers'
      - looks like table (many delimiters)          => 'table'
      - single-line, Title/Heading-ish              => 'heading'
      - otherwise                                   => 'para'
    """
    header = (text.splitlines()[:1] or [""])[0].strip().lower()
    if re.match(r"^(questions?|exercises?|practice|problems?)\b", header):
        return "questions"
    if re.match(r"^(answers?|solutions?)\b", header):
        return "answers"
    # crude table signal: lots of separators, tabs or pipes on multiple lines
    tabish = sum(1 for ln in text.splitlines() if ln.count("|") >= 2 or "\t" in ln) >= 2
    if tabish:
        return "table"
    # heading if very short and title-like
    if len(text.splitlines()) == 1 and 2 <= len(text.strip()) <= 80:
        return "heading"
    return "para"


# ------------------------- page extraction -------------------------

def _extract_text_from_page(page: fitz.Page, use_ocr: bool) -> str:
    """
    Try text APIs first; fallback to OCR image if configured and page is mostly empty.
    """
    try:
        # prefer "text" (layout-preserving) over "blocks" / "words" for readability
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE) or ""
        if text.strip():
            return text
    except Exception:
        pass

    # fallback to simple text
    try:
        t2 = page.get_text() or ""
        if t2.strip():
            return t2
    except Exception:
        pass

    # OCR path
    if use_ocr and _OCR_OK:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x DPI for better OCR
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr = pytesseract.image_to_string(img)
            return ocr or ""
        except Exception as e:
            logger.warning("pdf.chunk: OCR failed on page %d: %s", page.number + 1, e)

    return ""


def _dedupe_headers_footers(page_texts: List[str], enable: bool) -> List[str]:
    """
    Identify repeated first/last lines across pages and drop them (headers/footers).
    """
    if not enable or not page_texts:
        return page_texts

    first_counts: Dict[str, int] = {}
    last_counts: Dict[str, int] = {}
    lines_per_page: List[List[str]] = []

    for t in page_texts:
        lines = [ln.rstrip() for ln in t.splitlines()]
        lines_per_page.append(lines)
        f = _first_non_empty(lines)
        l = _last_non_empty(lines)
        if f:
            first_counts[f] = first_counts.get(f, 0) + 1
        if l:
            last_counts[l] = last_counts.get(l, 0) + 1

    n = len(page_texts)
    hdr = {k for k, v in first_counts.items() if v >= max(3, int(0.6 * n))}
    ftr = {k for k, v in last_counts.items() if v >= max(3, int(0.6 * n))}

    cleaned: List[str] = []
    for lines in lines_per_page:
        # drop only one instance at head / tail if exactly matches
        if lines and lines[0].strip() in hdr:
            lines = lines[1:]
        if lines and lines[-1].strip() in ftr:
            lines = lines[:-1]
        cleaned.append("\n".join(lines))

    return cleaned


def _split_into_blocks(page_text: str) -> List[str]:
    """
    Split into roughly-paragraph blocks using blank lines as separators.
    """
    blocks: List[str] = []
    for raw in re.split(r"\n\s*\n", page_text.strip()):
        blk = raw.strip()
        if blk:
            blocks.append(blk)
    return blocks


def _semantic_windows(text: str, max_chars: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Sliding character windows over a paragraph-ish text.
    Returns (start_char, end_char) pairs (0-based, end exclusive).
    """
    out: List[Tuple[int, int]] = []
    n = len(text)
    if n <= max_chars:
        return [(0, n)]
    step = max(1, max_chars - overlap)
    i = 0
    while i < n:
        j = min(n, i + max_chars)
        out.append((i, j))
        if j == n:
            break
        i += step
    return out


# ------------------------- main entry -------------------------

def chunk(file_path: str, cfg: dict) -> List[Dict[str, Any]]:
    """
    Chunk a PDF into:
      - optional full page containers (chunk_type='pdf_page')
      - semantic blocks / windows (chunk_type in {'para','heading','questions','answers','table','section'})

    Standard metadata fields:
      - file_path, relpath, file_name, file_ext
      - profile: "document"
      - chunk_type: as above
      - page_number (1-based)
      - name: "<file> :: p<N> [<label>]"
      - (for windows) char_start/char_end within the block
    """
    rel = os.path.relpath(file_path)
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1].lower()

    # settings
    emit = set(_get(cfg, "pdf_emit", default=["pdf_page", "pdf_blocks"]) or [])
    ocr_fallback = bool(_get(cfg, "ocr_fallback", default=True))
    dehy = bool(_get(cfg, "dehyphenate", default=True))
    drop_headers = bool(_get(cfg, "dedupe_headers", default=True))
    max_page_chunk = int(_get(cfg, "max_pdf_chunk_chars", default=16000) or 16000)

    sem_enabled = bool(_get(cfg, "pdf_semantic", "enabled", default=True))
    sem_max = int(_get(cfg, "pdf_semantic", "max_chars", default=1100) or 1100)
    sem_ov = int(_get(cfg, "pdf_semantic", "overlap", default=120) or 120)
    preview_chars = int(_get(cfg, "metadata", "store_preview_chars", default=0) or 0)

    logger.debug("pdf.chunk: begin '{}' emit={} ocr={} sem={} (max={}, ov={})",
                 rel, sorted(emit), ocr_fallback, sem_enabled, sem_max, sem_ov)

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.warning("pdf.chunk: failed to open '%s': %s", rel, e)
        return []

    # pass 1: read each page text (with OCR fallback)
    page_texts: List[str] = []
    for page in doc:
        txt = _extract_text_from_page(page, ocr_fallback)
        if dehy:
            txt = _dehyphenate(txt)
        txt = _normalize_ws(txt)
        page_texts.append(txt)

    # optional header/footer cleanup
    if drop_headers:
        page_texts = _dedupe_headers_footers(page_texts, enable=True)

    chunks: List[Dict[str, Any]] = []

    # optional: page-level containers
    if "pdf_page" in emit:
        for i, text in enumerate(page_texts, start=1):
            if not text.strip():
                continue
            if len(text) > max_page_chunk:
                text = text[:max_page_chunk]
            meta = {
                "file_path": rel,
                "relpath": rel,
                "file_name": base,
                "file_ext": ext,
                "profile": "document",
                "chunk_type": "pdf_page",
                "name": f"{base} :: p{i}",
                "page_number": i,
            }
            if preview_chars > 0:
                meta["preview"] = text[:preview_chars]
            chunks.append({
                "id": _sha(f"{rel}:page:{i}"),
                "document": text,
                "metadata": meta,
            })

    # semantic block extraction
    if "pdf_blocks" in emit:
        for i, text in enumerate(page_texts, start=1):
            if not text.strip():
                continue
            blocks = _split_into_blocks(text)
            # If no blank-line blocks, treat whole page as one block
            if not blocks:
                blocks = [text]

            for bi, block in enumerate(blocks):
                kind = _classify_block(block)
                label = "section" if kind in {"para", "heading"} else kind

                if sem_enabled:
                    windows = _semantic_windows(block, max_chars=sem_max, overlap=sem_ov)
                else:
                    windows = [(0, min(len(block), sem_max))]

                for wi, (s, e) in enumerate(windows):
                    seg = block[s:e]
                    meta = {
                        "file_path": rel,
                        "relpath": rel,
                        "file_name": base,
                        "file_ext": ext,
                        "profile": "document",
                        "chunk_type": kind,         # e.g., 'para','heading','questions','answers','table'
                        "name": f"{base} :: p{i} :: {label} [{bi+1}.{wi+1}]",
                        "page_number": i,
                        "char_start": s,
                        "char_end": e,
                    }
                    if preview_chars > 0:
                        meta["preview"] = seg[:preview_chars]

                    chunks.append({
                        "id": _sha(f"{rel}:p{i}:b{bi}:w{wi}:{s}:{e}"),
                        "document": seg,
                        "metadata": meta,
                    })

    logger.info("pdf.chunk: done pages=%d chunks=%d", len(page_texts), len(chunks))
    return chunks
