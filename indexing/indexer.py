# indexing/indexer.py
from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Dict, List
from pathlib import Path

import chromadb

from config_home import project_rag_dir  # ~/.gencli/<project>/RAG
from .settings import load as load_settings
from .filters import iter_files
from .embedder import resolve_embedder
from .chunkers import py_ast, text as text_chunker, pdf as pdf_chunker

# Optional: load .env for CLI/indexing runs (Streamlit already loads it)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


# ------------------------------ env helper ------------------------------

def _load_env_for_project(project_root: str) -> None:
    """
    Ensure `.env` is loaded for CLI/indexing runs.
    Priority:
      - <project_root>/.env
      - CWD/.env
      - repo root (indexing/../..)/.env
    Never overrides existing env vars.
    """
    if load_dotenv is None:
        return
    candidates = [
        Path(project_root) / ".env",
        Path.cwd() / ".env",
    ]
    try:
        candidates.append(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass
    for p in candidates:
        try:
            if p.exists():
                load_dotenv(dotenv_path=p, override=False)
        except Exception:
            continue


# ------------------------------ small utilities ------------------------------

def _project_key(project_root: str) -> str:
    p = Path(project_root)
    name = p.name or "project"
    h = hashlib.sha1(os.path.abspath(project_root).encode("utf-8")).hexdigest()[:8]
    return f"{name}-{h}"

def _relpath(root: str, fp: str) -> str:
    try:
        return os.path.relpath(fp, root)
    except Exception:
        return fp

def _resolve_models_json_path(project_root: str) -> str:
    """
    Locate data/models.json:
      1) <project_root>/data/models.json
      2) <repo_root>/data/models.json  (…/indexing/..)
    """
    cand1 = Path(project_root) / "data" / "models.json"
    if cand1.exists():
        return str(cand1)
    repo_root = Path(__file__).resolve().parents[1]
    cand2 = repo_root / "data" / "models.json"
    if cand2.exists():
        return str(cand2)
    return str(cand1)  # let resolver fail later if truly missing

def _apply_max_chars(docs: List[str], metas: List[dict], cfg: dict) -> None:
    """
    Backstop guard to keep each embed payload light.
    (Your chunkers should already be token-aware.)
    """
    limit = int((cfg.get("chunking") or {}).get("max_embed_chars", 4000))
    if limit <= 0:
        return
    for i in range(len(docs)):
        if len(docs[i]) > limit:
            docs[i] = docs[i][:limit]
            try:
                metas[i] = dict(metas[i])
                metas[i]["truncated"] = True
                metas[i]["max_embed_chars"] = limit
            except Exception:
                pass

def _rekey(chunks: List[dict], project_root: str, root: str, fp: str) -> List[dict]:
    """
    Always assign canonical IDs:
      <project_key>::<relpath>::<mtime>::<seq>
    and ensure metadata carries relpath/mtime.
    """
    rel = _relpath(root, fp)
    try:
        mtime = int(os.path.getmtime(fp))
    except Exception:
        mtime = 0
    proj = _project_key(project_root)

    out: List[dict] = []
    seq = 0
    for ch in chunks:
        seq += 1
        c = dict(ch)
        c["id"] = f"{proj}::{rel}::{mtime}::{seq}"
        md = dict(c.get("metadata") or {})
        md.setdefault("relpath", rel)
        md.setdefault("mtime", mtime)
        c["metadata"] = md
        out.append(c)
    return out


# ----------------------------- chroma collection ------------------------------

def _collection_name_for(cfg: dict) -> str:
    """
    Partition by selected embedding model to avoid vector dimension mismatches
    when users switch embedders between runs.
    """
    base = cfg.get("collection", "chunks")
    sel = ((cfg.get("embedder") or {}).get("selected_name")
           or (cfg.get("embedder") or {}).get("model_key")
           or "default")
    suffix = hashlib.sha1(str(sel).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{suffix}"

def _chroma(root: str, cfg: dict):
    """
    Create/get a Chroma collection for this project.
    IMPORTANT: Do NOT pass an embedding_function here.
               We embed client-side and pass vectors explicitly.
    """
    rag_dir = project_rag_dir(root)
    rag_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(rag_dir))
    coll_name = _collection_name_for(cfg)
    return client.get_or_create_collection(name=coll_name)


# --------------------------------- chunking -----------------------------------

def _chunk_file(fp: str, cfg: dict) -> List[dict]:
    """
    Chunkers must return dicts like:
      { "id": str (optional), "document": str, "metadata": dict }
    (We will overwrite id with our canonical scheme.)
    """
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".py":
        return py_ast.chunk(fp, cfg)
    if ext in (".txt", ".md"):
        return text_chunker.chunk(fp, cfg)
    if ext == ".pdf":
        return pdf_chunker.chunk(fp, cfg)
    return []


# ------------------------ embed + upsert (client-side) ------------------------

def _embed_and_upsert(coll, embed_fn, ids, docs, metas, upsert_batch: int = 64):
    """
    Embed texts client-side (micro-batching is handled inside embed_fn)
    and upsert vectors to Chroma in straightforward batches.
    """
    # Embed once for this batch
    vectors = embed_fn(docs)
    if len(vectors) != len(docs):
        m = min(len(vectors), len(docs))
        ids, docs, metas, vectors = ids[:m], docs[:m], metas[:m], vectors[:m]

    # Ensure chunk_id in metadata
    for k in range(len(metas)):
        md = dict(metas[k] or {})
        md["chunk_id"] = ids[k]
        metas[k] = md

    # Upsert in simple fixed-size chunks
    n = len(ids)
    for i in range(0, n, upsert_batch):
        j = min(i + upsert_batch, n)
        coll.upsert(
            ids=ids[i:j],
            documents=docs[i:j],
            metadatas=metas[i:j],
            embeddings=vectors[i:j],
        )


# --------------------------------- public API --------------------------------

def full_reindex(project_root: str, rag_path: str | None) -> Dict:
    """
    Manual full indexing (clean slate):
      - Clear the collection completely
      - Chunk all supported files
      - Embed client-side + upsert in batches
    Returns: {"ok": True, "added": <int>, "files": {...}}
    """
    _load_env_for_project(project_root)  # make sure keys from .env are visible

    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])

    coll = _chroma(root, cfg)
    models_path = _resolve_models_json_path(project_root)
    embed_fn = resolve_embedder(models_path, cfg)

    # Wipe collection
    try:
        coll.delete(where={})
    except Exception:
        pass

    added = 0
    per_file_stats = {}
    embed_batch = int(cfg.get("embedding_batch_size") or 8)
    upsert_batch = int(cfg.get("upsert_batch_size") or 64)

    for fp in iter_files(root, cfg):
        chunks = _chunk_file(fp, cfg)
        if not chunks:
            continue

        # Always canonicalize IDs/metadata
        chunks = _rekey(chunks, project_root, root, fp)

        ids = [c["id"] for c in chunks]
        docs = [c["document"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        _apply_max_chars(docs, metas, cfg)

        # Let the embedder do micro-batching internally; we call once per file
        # If you prefer to double-batch here, you can split docs by embed_batch.
        _embed_and_upsert(coll, embed_fn, ids, docs, metas, upsert_batch=upsert_batch)

        added += len(chunks)
        per_file_stats[_relpath(root, fp)] = len(chunks)

    # Stamp for delta indexing
    stamp_path = project_rag_dir(root) / ".last_index.json"
    try:
        os.makedirs(os.path.dirname(stamp_path), exist_ok=True)
        with open(stamp_path, "w") as f:
            json.dump({"ts": time.time()}, f)
    except Exception:
        pass

    return {"ok": True, "added": added, "files": per_file_stats}


def delta_index(project_root: str, rag_path: str | None) -> Dict:
    """
    Delta indexing:
      - Find files with mtime > last stamp
      - For each changed file: delete its old chunks by relpath, re-chunk and upsert
    Returns: {"ok": True, "added": <int>, "changed_files": [relpath, ...]}
    """
    _load_env_for_project(project_root)  # make sure keys from .env are visible

    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])

    coll = _chroma(root, cfg)
    models_path = _resolve_models_json_path(project_root)
    embed_fn = resolve_embedder(models_path, cfg)

    # Read last stamp
    stamp_path = project_rag_dir(root) / ".last_index.json"
    last = 0.0
    if os.path.exists(stamp_path):
        try:
            with open(stamp_path, "r") as f:
                last = json.load(f).get("ts", 0.0)
        except Exception:
            last = 0.0

    changed, added = [], 0
    now = time.time()
    upsert_batch = int(cfg.get("upsert_batch_size") or 64)

    for fp in iter_files(root, cfg):
        try:
            mtime = os.path.getmtime(fp)
        except FileNotFoundError:
            continue
        if mtime <= last:
            continue

        rel = _relpath(root, fp)

        # Remove any prior chunks for this file (by relpath), then re-index it
        try:
            coll.delete(where={"relpath": rel})
        except Exception:
            pass

        chunks = _chunk_file(fp, cfg)
        if not chunks:
            continue

        chunks = _rekey(chunks, project_root, root, fp)

        ids = [c["id"] for c in chunks]
        docs = [c["document"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        _apply_max_chars(docs, metas, cfg)
        _embed_and_upsert(coll, embed_fn, ids, docs, metas, upsert_batch=upsert_batch)

        added += len(chunks)
        changed.append(rel)

    # Update stamp
    try:
        os.makedirs(os.path.dirname(stamp_path), exist_ok=True)
        with open(stamp_path, "w") as f:
            json.dump({"ts": now}, f)
    except Exception:
        pass

    return {"ok": True, "added": added, "changed_files": changed}
