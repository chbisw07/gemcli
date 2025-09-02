# indexing/retriever.py
import os
from typing import Dict, Any
from loguru import logger
import chromadb

from config_home import project_rag_dir
from .settings import load as load_settings
from .embedder import resolve_embedder

def retrieve(project_root: str, rag_path: str, query: str, k: int = None) -> Dict[str, Any]:
    logger.info("retriever.retrieve: begin project_root='{}' rag='{}' q='{}...'", project_root, rag_path, (query or "")[:120])
    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])
    logger.info("retriever: index root='{}'", root)

    # NOTE: Indexer partitions collections by embedder to avoid dim mismatches.
    # This retriever currently uses cfg['collection'] directly; if you switch
    # embedders between runs, you may query the wrong collection. Consider
    # reusing the same naming scheme as indexer._collection_name_for.
    ef = resolve_embedder(os.path.join(root, "data", "models.json"), cfg)
    client = chromadb.PersistentClient(path=str(project_rag_dir(root)))
    coll_name = cfg["collection"]
    logger.info("retriever: chroma dir='{}' collection='{}' top_k={} (hybrid={})",
                str(project_rag_dir(root)),
                coll_name,
                (k or int(cfg['retrieval']['top_k'])),
                bool(cfg.get("retrieval", {}).get("hybrid", True)))
    coll = client.get_or_create_collection(name=coll_name, embedding_function=ef)

    topk = k or int(cfg["retrieval"]["top_k"])
    res = coll.query(query_texts=[query], n_results=topk)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[None]])[0]

    out = []
    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })

    logger.info("retriever: results={} first_id='{}'", len(out), out[0]["id"] if out else None)
    return {"chunks": out, "top_k": topk}
