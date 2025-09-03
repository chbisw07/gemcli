# indexing/retriever.py
from __future__ import annotations

import os
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger
import chromadb

from config_home import project_rag_dir
from .settings import load as load_settings
from .embedder import resolve_embedder
from .utils import collection_name_for as _collection_name_for
from .utils import resolve_models_json_path as _resolve_models_json_path

def retrieve(project_root: str, rag_path: str, query: str, k: int | None = None) -> Dict[str, Any]:
    logger.info("retriever.retrieve: begin project_root='{}' rag='{}' q='{}...'", project_root, rag_path, (query or "")[:120])
    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])
    logger.info("retriever: index root='{}'", root)

    # Same collection naming as indexer (embedder-hash suffix)
    # Anchor the DB under the *project root* to match indexer & app expectations
    client = chromadb.PersistentClient(path=str(project_rag_dir(project_root)))
    coll_name = _collection_name_for(cfg)
    coll = client.get_or_create_collection(name=coll_name)  # no embedding_function

    # Client-side query embedding to ensure dim compatibility
    models_path = _resolve_models_json_path(project_root)
    embed = resolve_embedder(models_path, cfg)
    qvecs = embed([query]) if query else []
    if not qvecs:
        return {"chunks": [], "top_k": k or int(cfg["retrieval"]["top_k"])}

    topk = k or int(cfg["retrieval"]["top_k"])
    res = coll.query(query_embeddings=[qvecs[0]], n_results=topk)

    ids: List[str] = res.get("ids", [[]])[0]
    docs: List[str] = res.get("documents", [[]])[0]
    metas: List[dict] = res.get("metadatas", [[]])[0]
    dists: List[float] = res.get("distances", [[None]])[0]

    out = [{"id": ids[i], "document": docs[i], "metadata": metas[i], "distance": dists[i]} for i in range(len(ids))]
    logger.info("retriever: results={} first_id='{}'", len(out), out[0]["id"] if out else None)
    return {"chunks": out, "top_k": topk}
