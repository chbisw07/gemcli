# =========================
# indexing/retriever.py
# =========================
from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional
from loguru import logger
import chromadb

from config_home import project_rag_dir
from .settings import load as load_settings
from .embedder import resolve_embedder
from .utils import collection_name_for as _collection_name_for
from .utils import resolve_models_json_path as _resolve_models_json_path

# e.g., "MathsStandard-SQP.pdf", "utils.py", "notes.md"
_FILENAME_RX = re.compile(
    r"\b([\w\-. ]+\.(?:pdf|docx|txt|md|csv|xlsx|pptx|py|js|jsx|ts|tsx))\b", re.I
)


def _to_score(distance: Optional[float]) -> float:
    """
    Convert Chroma 'distance' to a 0..1 'score' (higher is better).
    For cosine, distance is in [0, 2] typically; we clamp aggressively.
    """
    if distance is None:
        return 0.0
    try:
        d = float(distance)
        # If distance in [0,1], 1-d is a good similarity proxy.
        if 0.0 <= d <= 1.0:
            s = 1.0 - d
        else:
            # Otherwise, map to 1/(1+d) to keep it bounded.
            s = 1.0 / (1.0 + max(0.0, d))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0


def _merge_by_id(primary: List[dict], extra: List[dict], boost: float = 1.15) -> List[dict]:
    """
    Merge two result lists by id; keep highest score, optionally boosting the extra list.
    """
    by_id: Dict[str, dict] = {}
    for c in primary or []:
        by_id[c.get("id")] = dict(c)

    for c in extra or []:
        cid = c.get("id")
        if not cid:
            continue
        c2 = dict(c)
        if "score" in c2 and isinstance(c2["score"], (int, float)):
            c2["score"] = float(c2["score"]) * boost
        prev = by_id.get(cid)
        if prev is None or (c2.get("score", 0.0) > prev.get("score", 0.0)):
            by_id[cid] = c2

    out = list(by_id.values())
    out.sort(key=lambda x: (x.get("score") is not None, x.get("score") or 0.0), reverse=True)
    return out


def _query_collection(
    coll,
    qvec: List[float],
    n_results: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Run a single Chroma query and normalize the rows.
    """
    res = coll.query(query_embeddings=[qvec], n_results=int(n_results), where=where or {})
    ids: List[str] = res.get("ids", [[]])[0]
    docs: List[str] = res.get("documents", [[]])[0]
    metas: List[dict] = res.get("metadatas", [[]])[0]
    dists: List[float] = res.get("distances", [[None]])[0]

    out: List[dict] = []
    for i in range(len(ids)):
        dist = dists[i] if i < len(dists) else None
        out.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": dist,
                "score": _to_score(dist),
            }
        )
    return out


def retrieve(
    project_root: str,
    rag_path: str,
    query: str,
    k: int | None = None,
    where: Optional[Dict[str, Any]] = None,
    min_score: Optional[float] = None,
    enable_filename_boost: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Vector retrieve with optional metadata filter and filename-aware boosting.

    Returns:
      {
        "chunks": [{id, document, metadata, distance, score}],
        "top_k": int,
        "used_where": dict,
        "filename_boosted": [filenames...]
      }
    """
    q_preview = (query or "").strip().replace("\n", " ")[:160]
    logger.info("retriever.retrieve: q='{}…' root='{}' ragdir='{}'", q_preview, project_root, rag_path)

    # --- Load settings (global/default rag.json) ---
    # NOTE: settings.load expects a rag.json path; do not pass rag_dir here.
    cfg = load_settings(None)

    # knobs
    topk = int(k or (cfg.get("retrieval") or {}).get("top_k", 8))
    router_cfg = cfg.get("router") or {}
    if min_score is None:
        # Respect router threshold if present (can be 0)
        thr = router_cfg.get("threshold")
        min_score = float(thr) if isinstance(thr, (int, float)) else None
    if enable_filename_boost is None:
        enable_filename_boost = bool(router_cfg.get("enable_filename_boost", True))

    # --- Resolve rag directory and collection ---
    # Prefer explicit rag_path; else cfg['chroma_dir']; else project_rag_dir(project_root)
    rag_dir = str(rag_path or cfg.get("chroma_dir") or project_rag_dir(project_root))
    client = chromadb.PersistentClient(path=rag_dir)
    coll_name = _collection_name_for(cfg)
    coll = client.get_or_create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    logger.info("retriever: collection='{}' rag_dir='{}'", coll_name, rag_dir)

    # --- Build embedder ---
    models_path = _resolve_models_json_path(project_root)
    embed = resolve_embedder(models_path, cfg)
    qvecs = embed([query]) if query else []
    if not qvecs:
        return {"chunks": [], "top_k": topk, "used_where": where or {}, "filename_boosted": []}
    qvec = qvecs[0]

    # --- Base query (optionally filtered) ---
    base = _query_collection(coll, qvec, topk, where=where)

    # --- Filename-aware boosting pass ---
    filename_hits: List[str] = []
    merged = list(base)

    if enable_filename_boost and query:
        candidates = list(dict.fromkeys(_FILENAME_RX.findall(query)))  # dedupe, preserve order
        if candidates:
            # Make a slightly wider pull, then boost those with matching file_path
            wide = _query_collection(coll, qvec, n_results=max(2 * topk, 16), where=None)
            for name in candidates:
                # Filter 'wide' for rows referring to this filename
                name_low = name.lower()
                extra = [
                    r
                    for r in wide
                    if name_low in (r.get("metadata", {}).get("file_path", "") or "").lower()
                    or (r.get("metadata", {}).get("relpath", "") or "").lower().endswith(name_low)
                ]
                if extra:
                    filename_hits.append(name)
                    merged = _merge_by_id(merged, extra, boost=1.2)

    # --- Apply threshold & clip ---
    if isinstance(min_score, (int, float)):
        before = len(merged)
        merged = [c for c in merged if (c.get("score") or 0.0) >= float(min_score)]
        logger.info("retriever: min_score={} filtered {}→{}", min_score, before, len(merged))

    merged = merged[:topk]

    logger.info(
        "retriever: results={} first_id='{}' boosted={} where_keys={}",
        len(merged),
        merged[0]["id"] if merged else None,
        len(filename_hits),
        list((where or {}).keys()),
    )

    return {
        "chunks": merged,
        "top_k": topk,
        "used_where": where or {},
        "filename_boosted": filename_hits,
    }
