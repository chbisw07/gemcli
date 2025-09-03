# indexing/pdf.py
from __future__ import annotations

import io
import os
import re
from typing import List, Dict, Any, Tuple, Optional

from loguru import logger

# --- PyMuPDF + optional OCR ---
import fitz  # PyMuPDF
try:
    import pytesseract
    from PIL import Image
    _OCR_OK = True
except Exception:
    _OCR_OK = False


# ---------- heuristics for school-book PDFs ----------
_HEADING_RX = {
    "chapter": re.compile(r"(?mi)^(?:Chapter\s+\d+|^\d+\s+[A-Z][^\n]{1,80})$"),
    "section": re.compile(r"(?mi)^\d+\.\d+\s+[^\n]{1,80}$"),
    "subsection": re.compile(r"(?mi)^\d+\.\d+\.\d+\s+[^\n]{1,80}$"),
}
_BLOCK_RX = {
    "questions": re.compile(r"(?mi)^(Questions|Exercises?)\b"),
    "qa":        re.compile(r"(?mi)^(Q\s*[:：]|Q\.)"),
    "answer":    re.compile(r"(?mi)^(A\s*[:：]|Ans\.)"),
    "figure":    re.compile(r"(?mi)^(Figure|Fig\.)\s*\d+"),
    "table":     re.compile(r"(?mi)^Table\s*\d+"),
    "activity":  re.compile(r"(?mi)^(Activity|Experiment)\b"),
    "biblio":    re.compile(r"(?mi)^(Bibliography|References)$"),
}


# ------------------------- small helpers -------------------------

def _clean_text(txt: str, dehyphenate: bool, dedupe_headers: bool) -> str:
    # normalize newlines
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    if dehyphenate:
        txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2\n", txt)  # join hyphenated words at line breaks
    if dedupe_headers:
        # remove simple "Page N" lines and repeated blank lines
        lines = [ln for ln in txt.splitlines() if not re.match(r"^\s*Page\s+\d+\s*$", ln)]
        txt = "\n".join(lines)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _infer_book_chapter(path: str) -> Tuple[Optional[str], Optional[str]]:
    base = os.path.basename(path)
    book = os.path.basename(os.path.dirname(path)) or None
    m = re.search(r"(?:chapter|ch)[_\s\-]?(\d+)", base, flags=re.I) or re.search(r"\b(\d{1,2})\b", base)
    return book, (m.group(1) if m else None)

def _page_texts(doc: fitz.Document, ocr_fallback: bool) -> List[str]:
    out: List[str] = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        t = page.get_text("text") or ""
        if ocr_fallback and len(t.strip()) < 8 and _OCR_OK:
            try:
                pix = page.get_pixmap(dpi=250)
                img = Image.open(io.BytesIO(pix.tobytes()))
                t = pytesseract.image_to_string(img) or ""
            except Exception as e:
                logger.debug("pdf.ocr: page {} OCR failed: {}", pno + 1, e)
        out.append(t)
    return out

def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def _window_paragraphs(paras: List[str], max_chars: int, overlap: int) -> List[str]:
    if max_chars <= 0:
        return paras
    out: List[str] = []
    buf: List[str] = []
    cur = 0
    for p in paras:
        # +2 for the "\n\n" joiner when present
        add_len = len(p) + (2 if buf else 0)
        if cur + add_len <= max_chars:
            buf.append(p); cur += add_len
        else:
            if buf:
                out.append("\n\n".join(buf))
            # simple trailing overlap (character-based)
            if overlap > 0 and out:
                carry = out[-1][-overlap:] if len(out[-1]) > overlap else out[-1]
                buf = [carry, p]
                cur = len(carry) + len(p) + 2
            else:
                buf = [p]; cur = len(p)
    if buf:
        out.append("\n\n".join(buf))
    return out

def _split_anchors(content: str):
    anchors = []
    for label, rx in _HEADING_RX.items():
        for m in rx.finditer(content):
            anchors.append((m.start(), label, m.group(0).strip()))
    for label, rx in _BLOCK_RX.items():
        for m in rx.finditer(content):
            anchors.append((m.start(), label, m.group(0).strip()))
    anchors = sorted(set(anchors), key=lambda x: x[0])
    return anchors


# ------------------------- main entry -------------------------

def chunk(file_path: str, cfg: dict) -> List[Dict[str, Any]]:
    """
    Emits chunks according to cfg:
      cfg["pdf_emit"] may include: "pdf_page", "pdf_structure", "pdf_blocks", "pdf_toc"
      cfg["ocr_fallback"]: bool
      cfg["dehyphenate"], cfg["dedupe_headers"], cfg["max_pdf_chunk_chars"]
      cfg["pdf_semantic"] = {"enabled": True, "max_chars": 1100, "overlap": 120}
    Each chunk has:
      - document: text
      - metadata: { file_path, chunk_type, name, page_number, book, chapter, ... }
    """
    rel = os.path.relpath(file_path)
    emit = cfg.get("pdf_emit") or ["pdf_page"]
    ocr_fallback = bool(cfg.get("ocr_fallback", True))
    dehyphenate = bool(cfg.get("dehyphenate", True))
    dedupe_headers = bool(cfg.get("dedupe_headers", True))
    cap = int(cfg.get("max_pdf_chunk_chars", 16000))

    sem = cfg.get("pdf_semantic") or {}
    sem_enabled = bool(sem.get("enabled", True))
    sem_max_chars = int(sem.get("max_chars", 1100))
    sem_overlap   = int(sem.get("overlap", 120))

    logger.info("pdf.chunk: begin '{}' emit={} sem={{enabled:{}, max_chars:{}, overlap:{}}}",
                rel, emit, sem_enabled, sem_max_chars, sem_overlap)

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error("pdf.chunk: open failed for '{}': {}", rel, e)
        return []

    book, chapter = _infer_book_chapter(rel)

    # ToC
    toc = []
    if "pdf_toc" in emit:
        try:
            toc = doc.get_toc(simple=True)  # [level, title, page]
        except Exception as e:
            logger.debug("pdf.chunk: get_toc failed: {}", e)

    pages_raw = _page_texts(doc, ocr_fallback=ocr_fallback)
    pages = [_clean_text(t, dehyphenate, dedupe_headers) for t in pages_raw]

    chunks: List[Dict[str, Any]] = []

    # ---- ToC chunks ----
    if "pdf_toc" in emit and toc:
        for idx, (level, title, page) in enumerate(toc, start=1):
            s = str(title).strip()
            if not s:
                continue
            chunks.append({
                "document": s,
                "metadata": {
                    "file_path": rel, "chunk_type": "toc", "level": int(level),
                    "page_number": int(page), "book": book, "chapter": chapter
                }
            })

    # ---- Per-page processing ----
    for pno, text in enumerate(pages, start=1):
        name = f"Page {pno}"

        # Page container
        if "pdf_page" in emit:
            doc_page = f"PDF {rel} {name}\n\n{text}"
            if cap and len(doc_page) > cap:
                doc_page = doc_page[:cap]
            chunks.append({
                "document": doc_page,
                "metadata": {
                    "file_path": rel,
                    "chunk_type": "pdf_page",
                    "name": name,
                    "page_number": pno,
                    "book": book,
                    "chapter": chapter,
                }
            })

        # Paragraph windows (semantic structure)
        if sem_enabled and "pdf_structure" in emit and text:
            paras = _paragraphs(text)
            windows = _window_paragraphs(paras, sem_max_chars, sem_overlap) if paras else []
            for j, win in enumerate(windows, start=1):
                s = f"Page {pno} — ¶window {j}\n\n{win}"
                if cap and len(s) > cap:
                    s = s[:cap]
                chunks.append({
                    "document": s,
                    "metadata": {
                        "file_path": rel,
                        "chunk_type": "paragraph",
                        "name": f"{name} ¶{j}",
                        "page_number": pno,
                        "book": book,
                        "chapter": chapter,
                    }
                })

        # Heuristic blocks (Questions/Answers/Tables/etc.)
        if "pdf_blocks" in emit and text:
            anchors = _split_anchors(text)
            if anchors:
                split_points = [a[0] for a in anchors] + [len(text)]
                for bi in range(len(anchors)):
                    start = anchors[bi][0]; end = split_points[bi + 1]
                    label = anchors[bi][1]; heading = anchors[bi][2]
                    sect = text[start:end].strip()
                    ctype = label if label in _BLOCK_RX else "section"
                    name2 = (heading[:80] or ctype.title())
                    s = f"Page {pno} — {name2}\n\n{sect}"
                    if cap and len(s) > cap:
                        s = s[:cap]
                    chunks.append({
                        "document": s,
                        "metadata": {
                            "file_path": rel,
                            "chunk_type": "table" if ctype == "table" else
                                          "qa"    if ctype == "qa"    else
                                          "answer"if ctype == "answer"else
                                          "questions" if ctype == "questions" else
                                          "section",
                            "name": name2,
                            "page_number": pno,
                            "book": book,
                            "chapter": chapter,
                        }
                    })

    logger.info("pdf.chunk: done pages={} chunks={}", len(pages), len(chunks))
    return chunks
