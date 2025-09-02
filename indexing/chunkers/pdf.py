# indexing/chunkers/pdf.py
import os, hashlib
from loguru import logger
from pypdf import PdfReader

def _sha(s: str): import hashlib; return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()

def chunk(file_path: str, cfg: dict):
    logger.debug("pdf.chunk: begin '{}'", file_path)
    reader = PdfReader(file_path)
    rel = os.path.relpath(file_path)
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.append({
            "id": _sha(rel + f":page{i}"),
            "document": f"PDF {rel} page {i+1}\n\n{text}",
            "metadata": {"file_path": rel, "symbol_kind": "pdf_page", "qualified_name": f"{rel}:page:{i+1}"}
        })
    logger.debug("pdf.chunk: done pages={} chunks={}", len(reader.pages), len(out))
    return out
