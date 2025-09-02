# indexing/utils.py
from __future__ import annotations

import hashlib
from pathlib import Path


def collection_name_for(cfg: dict) -> str:
    """
    Match indexer/retriever partitioning exactly:
    base = cfg['collection'] (default handled by caller)
    suffix = sha1( selected_name or model_key or 'default' )[:8]
    """
    base = cfg.get("collection", "chunks")
    sel = ((cfg.get("embedder") or {}).get("selected_name")
           or (cfg.get("embedder") or {}).get("model_key")
           or "default")
    suffix = hashlib.sha1(str(sel).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{suffix}"


def resolve_models_json_path(project_root: str) -> str:
    """
    Locate data/models.json, trying:
      1) <project_root>/data/models.json
      2) <repo_root>/data/models.json   (…/indexing/..)
    Returns the best guess (even if nonexistent; callers handle errors).
    """
    pr = Path(project_root)
    cand1 = pr / "data" / "models.json"
    if cand1.exists():
        return str(cand1)

    repo_root = Path(__file__).resolve().parents[1]  # …/indexing/..
    cand2 = repo_root / "data" / "models.json"
    if cand2.exists():
        return str(cand2)

    # Fall back to project path; embedder resolver will raise if truly missing.
    return str(cand1)
