# indexing/retriever.py
import os
from typing import Dict, Any
import chromadb
from config_home import project_rag_dir
from .settings import load as load_settings
from .embedder import resolve_embedder

def retrieve(project_root: str, rag_path: str, query: str, k: int = None) -> Dict[str, Any]:
    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])

    # use the same per-project store as indexer.py
    ef = resolve_embedder(os.path.join(root, "data", "models.json"), cfg)
    client = chromadb.PersistentClient(path=str(project_rag_dir(root)))
    coll = client.get_or_create_collection(name=cfg["collection"], embedding_function=ef)

    topk = k or int(cfg["retrieval"]["top_k"])
    res = coll.query(query_texts=[query], n_results=topk)
    out = []
    for i in range(len(res.get("ids", [[]])[0])):
        out.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res.get("distances", [[None]])[0][i],
        })
    return {"chunks": out, "top_k": topk}
