# indexing/router.py
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from .settings import load as load_settings
from .retriever import retrieve as rag_retrieve
from config_home import project_rag_dir

CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c"}
DOC_EXTS  = {".pdf", ".md", ".txt", ".docx", ".rst"}
TAB_EXTS  = {".csv", ".xlsx", ".tsv"}


def _sim_from_distance(d: Optional[float]) -> float:
    """Map Chroma distance to [0,1] similarity."""
    if d is None:
        return 0.0
    try:
        d = float(d)
        if 0.0 <= d <= 1.0:
            s = 1.0 - d
        else:
            s = 1.0 / (1.0 + max(0.0, d))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0


def score_buckets(chunks: List[dict]) -> Dict[str, float]:
    """
    Sum similarities by rough profile/ext to produce bucket scores.
    Returns: {"code": float, "document": float, "tabular": float}
    """
    s = {"code": 0.0, "document": 0.0, "tabular": 0.0}
    for c in chunks or []:
        md = c.get("metadata") or {}
        sim = c.get("score") or _sim_from_distance(c.get("distance")) or 0.0
        fp = (md.get("file_path") or md.get("relpath") or "").lower()
        ext = "." + fp.split(".")[-1] if "." in fp else ""
        prof = (md.get("profile") or "").lower()

        if prof == "code" or ext in CODE_EXTS:
            s["code"] += sim
        elif prof == "document" or ext in DOC_EXTS or (md.get("chunk_type", "").startswith("pdf")):
            s["document"] += sim
        elif ext in TAB_EXTS:
            s["tabular"] += sim
        else:
            s["document"] += 0.5 * sim
    return s


def decide_route(
    scores: Dict[str, float],
    *,
    prefer_hybrid: bool = True,
    code_bias: float = 1.15,
    min_mix: float = 0.30,
) -> str:
    """
    Turn bucket scores into a route label.
    - If tabular dominates clearly → 'tabular'
    - If code >> docs → 'code'
    - If both code & docs present → 'hybrid' (when prefer_hybrid)
    - Else → 'document'
    """
    code = scores.get("code", 0.0)
    doc = scores.get("document", 0.0)
    tab = scores.get("tabular", 0.0)

    if tab > max(code, doc) * 1.20 and tab > 0.35:
        return "tabular"
    if code > doc * code_bias and code > 0.25:
        return "code"
    if prefer_hybrid and code > min_mix and doc > min_mix:
        return "hybrid"
    return "document"


def route_query(
    project_root: str,
    query: str,
    *,
    rag_path: Optional[str] = None,
    k: Optional[int] = None,
    min_score: Optional[float] = None,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One-call router:
      1) RAG retrieve (with optional filter/threshold),
      2) score buckets,
      3) pick route.

    Returns:
      {
        "route": "code" | "document" | "hybrid" | "tabular",
        "scores": {"code":..,"document":..,"tabular":..},
        "top_k": int,
        "chunks": [...],
        "used_where": {...},
        "filename_boosted": [...]
      }
    """
    cfg = load_settings(None)
    if k is None:
        k = int((cfg.get("retrieval") or {}).get("top_k", 8))

    if min_score is None:
        thr = (cfg.get("router") or {}).get("threshold")
        if isinstance(thr, (int, float)):
            min_score = float(thr)

    rag_dir = rag_path or cfg.get("chroma_dir") or project_rag_dir(project_root)

    logger.info("router.route_query: top_k={} min_score={} rag_dir='{}'", k, min_score, rag_dir)
    rag = rag_retrieve(
        project_root=project_root,
        rag_path=rag_dir,
        query=query,
        k=k,
        where=where,
        min_score=min_score,
    )

    chunks = rag.get("chunks") or []
    scores = score_buckets(chunks)
    route = decide_route(scores)

    out = {
        "route": route,
        "scores": scores,
        "top_k": rag.get("top_k"),
        "chunks": chunks,
        "used_where": rag.get("used_where"),
        "filename_boosted": rag.get("filename_boosted"),
    }
    logger.info("router.route_query: route={} scores={}", route, {k: round(v, 3) for k, v in scores.items()})
    return out


def grounding_summary(chunks: List[dict], max_items: int = 10) -> Dict[str, Any]:
    """
    Small utility for UIs: summarize which files/pages dominate the grounding.
    """
    by_file: Dict[str, float] = {}
    for c in chunks[: max_items or 10]:
        md = c.get("metadata") or {}
        key = (md.get("file_path") or md.get("relpath") or "?")
        by_file[key] = by_file.get(key, 0.0) + (c.get("score") or _sim_from_distance(c.get("distance")) or 0.0)

    # top-N
    top = sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)[: max_items or 10]
    return {
        "top_files": [{"file": k, "score": round(v, 3)} for k, v in top],
        "total_chunks_considered": len(chunks),
    }
