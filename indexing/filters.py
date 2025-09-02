# indexing/retriever.py
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Iterable
from loguru import logger
import chromadb

from config_home import project_rag_dir
from .settings import load as load_settings
from .embedder import resolve_embedder

__all__ = ["should_index", "iter_files"]

def should_index(path: str, cfg) -> bool:
    p = path.replace("\\", "/")
    _, ext = os.path.splitext(p)

    # Supported types
    supported = {e.lower() for e in (cfg.get("supported_file_types") or [])}
    if ext and supported and ext.lower() not in supported:
        logger.debug("filters: skip (unsupported ext='{}') '{}'", ext, p)
        return False

    name = os.path.basename(p)

    # Ignore lists
    ignore_files = set(cfg.get("ignore_files") or [])
    if name in ignore_files:
        logger.debug("filters: skip (ignore_files) '{}'", p)
        return False

    ignore_dir = set(cfg.get("ignore_dir") or [])
    if any(part in ignore_dir for part in p.split("/")):
        logger.debug("filters: skip (ignore_dir) '{}'", p)
        return False

    ignore_ext = set(cfg.get("ignore_ext") or [])
    if ext.lower() in ignore_ext:
        logger.debug("filters: skip (ignore_ext='{}') '{}'", ext, p)
        return False

    return True


def iter_files(root: str, cfg) -> Iterable[str]:
    """
    Walk `root` and yield files accepted by should_index().
    """
    logger.info("filters.iter_files: walking root='{}'", root)
    count = 0
    for dirpath, dirs, files in os.walk(root):
        # prune ignored dirs (in-place so os.walk doesn't descend)
        ignore_dir = set(cfg.get("ignore_dir") or [])
        dirs[:] = [d for d in dirs if d not in ignore_dir]

        for f in files:
            fp = os.path.join(dirpath, f)
            if should_index(fp, cfg):
                count += 1
                yield fp
    logger.info("filters.iter_files: yielded {} file(s)", count)
    
# --- must match indexer._collection_name_for -------------------------------
def _collection_name_for(cfg: dict) -> str:
    base = cfg.get("collection", "chunks")
    sel = ((cfg.get("embedder") or {}).get("selected_name")
           or (cfg.get("embedder") or {}).get("model_key")
           or "default")
    suffix = hashlib.sha1(str(sel).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{suffix}"


# --- mirror indexer’s models.json resolver (safe guesses) -------------------
def _resolve_models_json_path(project_root: str) -> str:
    """
    Locate data/models.json:
      1) <project_root>/data/models.json
      2) <repo_root>/data/models.json  (…/indexing/..)
    """
    cand1 = Path(project_root) / "data" / "models.json"
    if cand1.exists():
        return str(cand1)
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        repo_root = Path(project_root)
    cand2 = repo_root / "data" / "models.json"
    if cand2.exists():
        return str(cand2)
    # fall back to project path (will raise in resolve_embedder if missing)
    return str(cand1)


def retrieve(project_root: str, rag_path: str, query: str, k: int | None = None) -> Dict[str, Any]:
    logger.info("retriever.retrieve: begin project_root='{}' rag='{}' q='{}...'",
                project_root, rag_path, (query or "")[:120])

    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])
    coll_name = _collection_name_for(cfg)
    topk = k or int((cfg.get("retrieval") or {}).get("top_k", 8))
    logger.info("retriever: index_root='{}' collection='{}' top_k={} hybrid={}",
                root, coll_name, topk, bool((cfg.get("retrieval") or {}).get("hybrid", True)))

    # Open Chroma collection (no embedding_function; we embed client-side to avoid dim mismatches)
    rag_dir = project_rag_dir(root)
    client = chromadb.PersistentClient(path=str(rag_dir))
    coll = client.get_or_create_collection(name=coll_name)
    logger.debug("retriever: chroma dir='{}'", str(rag_dir))

    # Resolve the *same* embedder and embed the query here
    models_path = _resolve_models_json_path(project_root)
    logger.info("retriever: models.json='{}'", models_path)
    embed = resolve_embedder(models_path, cfg)

    logger.debug("retriever: embedding query text (len={})", len(query or ""))
    qvecs: List[List[float]] = embed([query])
    if not qvecs or not qvecs[0]:
        logger.error("retriever: embedder returned no vector for the query")
        return {"chunks": [], "top_k": topk}

    # Query by vector to guarantee compatibility with indexer’s stored embeddings
    res = coll.query(query_embeddings=[qvecs[0]], n_results=topk)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[None]])[0]

    results = []
    for i in range(len(ids)):
        results.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })

    logger.info("retriever: results={} first_id='{}'", len(results), results[0]["id"] if results else None)
    return {"chunks": results, "top_k": topk}
