# indexing/chunkers/text.py
from __future__ import annotations

import os
import hashlib
from typing import List, Dict, Any
from loguru import logger


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _get(cfg: dict, *path, default=None):
    """Safe nested get: _get(cfg, "chunking", "text", "max_chars", default=1100)."""
    cur = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def chunk(file_path: str, cfg: dict) -> List[Dict[str, Any]]:
    """
    Chunk plain-text-ish files (.txt/.md/.rst/.log etc.) into fixed-size windows with overlap.

    Metadata fields (uniform across chunkers):
      - file_path: str (project-relative path)
      - relpath:   str (alias of file_path)
      - file_name: str (basename)
      - file_ext:  str (like ".txt")
      - profile:   "document"
      - chunk_type:"text_block"
      - name:      human-readable label
      - char_start, char_end: window bounds in the source
      - preview:   optional, first N chars (controlled by settings.metadata.store_preview_chars)
    """
    logger.debug("text.chunk: begin '{}'", file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
    except Exception as e:
        logger.warning("text.chunk: failed to read '{}': {}", file_path, e)
        return []

    # Settings with safe fallbacks
    maxc = int(_get(cfg, "chunking", "text", "max_chars", default=1100) or 1100)
    ov = int(_get(cfg, "chunking", "text", "overlap", default=120) or 120)
    preview_chars = int(_get(cfg, "metadata", "store_preview_chars", default=0) or 0)

    rel = os.path.relpath(file_path)
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1].lower()

    out: List[Dict[str, Any]] = []
    i = 0
    n = len(full_text)

    if n == 0:
        logger.debug("text.chunk: empty file '{}'", rel)
        return []

    step = max(1, maxc - ov)
    idx = 0

    while i < n:
        seg = full_text[i : i + maxc]
        meta = {
            "file_path": rel,
            "relpath": rel,
            "file_name": base,
            "file_ext": ext,
            "profile": "document",
            "chunk_type": "text_block",
            "name": f"{base} :: chunk {idx + 1}",
            "char_start": i,
            "char_end": i + len(seg),
        }
        if preview_chars > 0:
            meta["preview"] = seg[:preview_chars]

        out.append(
            {
                "id": _sha(f"{rel}:{i}"),
                "document": seg,
                "metadata": meta,
            }
        )

        idx += 1
        i += step

    logger.debug(
        "text.chunk: done file='{}' chars={} chunks={} maxc={} overlap={}",
        rel,
        n,
        len(out),
        maxc,
        ov,
    )
    return out
