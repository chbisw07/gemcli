# indexing/indexer.py
from __future__ import annotations

import os, json, time, hashlib, traceback
from typing import Dict, List
from pathlib import Path

from loguru import logger
import chromadb
import concurrent.futures as cf

from config_home import project_rag_dir  # ~/.gencli/<project>/RAG
from .settings import load as load_settings
from .filters import iter_files
from .embedder import resolve_embedder
from .chunkers import py_ast, text as text_chunker, pdf as pdf_chunker

# At the top with other imports:
from .utils import collection_name_for as _collection_name_for
from .utils import resolve_models_json_path as _resolve_models_json_path

# Optional: load .env for CLI/indexing runs (Streamlit already loads it)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


def _sanitize_metadata_value(v):
    """
    Chroma only accepts str/int/float/bool/None for metadata values.
    Convert anything else (lists, dicts, Paths, custom objects) to a safe string.
    """
    import json
    from pathlib import Path as _Path
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, _Path):
        return str(v)
    # Lists/tuples/sets → JSON string (after best-effort scalarization)
    if isinstance(v, (list, tuple, set)):
        try:
            return json.dumps([_sanitize_metadata_value(x) for x in v], ensure_ascii=False)
        except Exception:
            return str([str(x) for x in v])
    # Dicts → JSON string (after best-effort scalarization)
    if isinstance(v, dict):
        try:
            return json.dumps({str(k): _sanitize_metadata_value(x) for k, x in v.items()}, ensure_ascii=False)
        except Exception:
            return str({str(k): str(x) for k, x in v.items()})
    # Fallback for any custom/object types
    try:
        return str(v)
    except Exception:
        return repr(v)

def _sanitize_metadata(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        try:
            out[str(k)] = _sanitize_metadata_value(v)
        except Exception:
            # drop the bad key
            continue
    return out

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
        logger.debug("_load_env_for_project: python-dotenv not available; skipping")
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
                logger.debug("_load_env_for_project: loaded {}", str(p))
        except Exception as e:
            logger.debug("_load_env_for_project: failed to load {} → {}", str(p), e)


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

def _stop_flag_path(project_root: str) -> Path:
    # project_rag_dir returns str in current config_home; cast to Path
    return Path(project_rag_dir(project_root)) / ".index_stop"

def _status_path(project_root: str) -> Path:
    return Path(project_rag_dir(project_root)) / ".index_status.json"

def request_stop(project_root: str) -> Dict[str, bool]:
    """Create a stop flag; workers will finish in-flight and halt."""
    p = _stop_flag_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ts": time.time()}), encoding="utf-8")
    logger.warning("Indexing STOP requested (flag written to '{}')", str(p))
    return {"ok": True}

def index_status(project_root: str) -> Dict:
    """Read lightweight status (running/progress/dirty)."""
    try:
        p = _status_path(project_root)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}
    
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
            logger.debug("_apply_max_chars: truncating doc[{}] from {}→{} chars", i, len(docs[i]), limit)
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

    file_name = os.path.basename(rel)
    base_name, file_ext = os.path.splitext(file_name)
    out: List[dict] = []

    seq = 0
    for ch in chunks:
        seq += 1
        c = dict(ch)
        c["id"] = f"{proj}::{rel}::{mtime}::{seq}"
        md = dict(c.get("metadata") or {})
        md.setdefault("relpath", rel)
        md.setdefault("mtime", mtime)
        # enrich filename metadata so we can filter/boost by file name later
        md.setdefault("file_name", file_name)
        md.setdefault("base_name", base_name)
        md.setdefault("file_ext", file_ext.lower())
        c["metadata"] = md
        out.append(c)
    return out


# ----------------------------- chroma collection ------------------------------

def _chroma(project_root: str, cfg: dict):
    """
    Create/get a Chroma collection for this project.
    IMPORTANT: Do NOT pass an embedding_function here.
               We embed client-side and pass vectors explicitly.
    The RAG store path is anchored to the *project root* for consistency with the app.
    """
    # Ensure we always operate on a real Path object (mkdir used below)
    rag_dir = Path(project_rag_dir(project_root))
    rag_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(rag_dir))
    coll_name = _collection_name_for(cfg)
    logger.info("Chroma store: dir='{}' collection='{}'", str(rag_dir), coll_name)
    return client.get_or_create_collection(name=coll_name)


# --------------------------------- chunking -----------------------------------
# ---------------------------- parallel chunking -------------------------------

def _auto_workers(cfg: dict) -> int:
    w = int(cfg.get("indexing_workers", 0) or 0)
    if w <= 0:
        try:
            w = max(1, (os.cpu_count() or 2) - 1)
        except Exception:
            w = 1
    return w

def _chunk_file_worker(args):
    fp, cfg = args
    t0 = time.time()
    try:
        from indexing.indexer import _chunk_file as _router
        chunks = _router(fp, cfg)
        return fp, chunks, (time.time() - t0), None
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        return fp, [], (time.time() - t0), err

def _parallel_chunk_stream(files: List[str], cfg: dict, workers: int, project_root: str):
    """
    Submit at most `workers` tasks; yield results as they finish.
    If STOP flag appears, do not submit new tasks; drain in-flight only.
    """
    if workers <= 1:
        for fp in files:
            t0 = time.time()
            try:
                yield fp, _chunk_file(fp, cfg), (time.time() - t0), None
            except Exception as e:
                yield fp, [], (time.time() - t0), f"{type(e).__name__}: {e}"
        return

    stop_flag = _stop_flag_path(project_root)
    inflight: dict = {}
    it = iter(files)
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        # prefill
        while len(inflight) < workers:
            try:
                fp = next(it)
            except StopIteration:
                break
            f = ex.submit(_chunk_file_worker, (fp, cfg))
            inflight[f] = fp
        # stream
        while inflight:
            for fut in cf.as_completed(list(inflight.keys()), timeout=None):
                fp0 = inflight.pop(fut)
                try:
                    yield fut.result()
                except Exception as e:
                    yield fp0, [], 0.0, f"{type(e).__name__}: {e}"
                break  # re-check stop & backfill one by one
            # backfill only if no stop requested
            if not stop_flag.exists():
                try:
                    fp = next(it)
                except StopIteration:
                    continue
                f = ex.submit(_chunk_file_worker, (fp, cfg))
                inflight[f] = fp
        # done

def _chunk_file(fp: str, cfg: dict) -> List[dict]:
    """
    Chunkers must return dicts like:
      { "id": str (optional), "document": str, "metadata": dict }
    (We will overwrite id with our canonical scheme.)
    """
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".py":
        logger.debug("Chunker: py_ast for '{}'", fp)
        return py_ast.chunk(fp, cfg)
    if ext in (".txt", ".md"):
        logger.debug("Chunker: text for '{}'", fp)
        return text_chunker.chunk(fp, cfg)
    if ext == ".pdf":
        logger.debug("Chunker: pdf for '{}'", fp)
        return pdf_chunker.chunk(fp, cfg)
    logger.debug("Chunker: unsupported ext '{}' for '{}'", ext, fp)
    return []


# ------------------------ embed + upsert (client-side) ------------------------

def _embed_and_upsert(coll, embed_fn, ids, docs, metas, upsert_batch: int = 64):
    """
    Embed texts client-side (micro-batching is handled inside embed_fn)
    and upsert vectors to Chroma in straightforward batches.
    """
    # Embed once for this batch
    logger.debug("_embed_and_upsert: embedding {} text(s)", len(docs))
    vectors = embed_fn(docs)
    if len(vectors) != len(docs):
        logger.warning("_embed_and_upsert: vector/doc length mismatch: {} vs {}", len(vectors), len(docs))
        m = min(len(vectors), len(docs))
        ids, docs, metas, vectors = ids[:m], docs[:m], metas[:m], vectors[:m]

    # Ensure chunk_id in metadata and sanitize values to Chroma-safe types
    for k in range(len(metas)):
        md = dict(metas[k] or {})
        md["chunk_id"] = ids[k]
        metas[k] = _sanitize_metadata(md)
        # also guarantee document is a string
        try:
            docs[k] = "" if docs[k] is None else str(docs[k])
        except Exception:
            docs[k] = ""

    # Upsert in simple fixed-size chunks
    n = len(ids)
    for i in range(0, n, upsert_batch):
        j = min(i + upsert_batch, n)
        logger.debug("_embed_and_upsert: upserting [{}:{}] ({} items)", i, j, j - i)
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
    logger.info("full_reindex: begin project_root='{}' rag='{}'", project_root, rag_path)
    _load_env_for_project(project_root)  # make sure keys from .env are visible

    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])
    logger.info("Index root: '{}'", root)

    coll = _chroma(project_root, cfg)
    models_path = _resolve_models_json_path(project_root)
    logger.info("models.json resolved: '{}'", models_path)
    embed_fn = resolve_embedder(models_path, cfg)
    logger.info("Embedder resolved for selected='{}'", (cfg.get("embedder") or {}).get("selected_name") or (cfg.get("embedder") or {}).get("model_key"))

    # Clear any stale STOP and set status → running
    try:
        _stop_flag_path(project_root).unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass

    # Wipe collection
    try:
        coll.delete(where={})
        logger.info("Collection cleared")
    except Exception as e:
        logger.debug("Collection clear skipped/failed: {}", e)

    files = list(iter_files(root, cfg))
    workers = _auto_workers(cfg)
    upsert_batch = int(cfg.get("upsert_batch_size") or 64)
    max_pending = int(cfg.get("max_pending_chunks") or 2000)
    logger.info("Parallel chunking: files={} workers={} upsert_batch={} max_pending={}",
                len(files), workers, upsert_batch, max_pending)

    # status seed
    status = {"state": "running", "dirty": False, "processed_files": 0,
              "total_files": len(files), "started": time.time()}
    try:
        _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception:
        pass

    added, per_file_stats = 0, {}
    buf_ids: List[str] = []
    buf_docs: List[str] = []
    buf_meta: List[dict] = []

    def _flush():
        nonlocal added, buf_ids, buf_docs, buf_meta
        if not buf_docs:
            return
        _embed_and_upsert(coll, embed_fn, buf_ids, buf_docs, buf_meta, upsert_batch=upsert_batch)
        added += len(buf_docs)
        buf_ids.clear(); buf_docs.clear(); buf_meta.clear()

    stop_flag = _stop_flag_path(project_root)
    dirty = False

    for fp, chunks, elapsed, err in _parallel_chunk_stream(files, cfg, workers, project_root):
        rel = _relpath(root, fp)
        if err:
            logger.error("chunk FAIL '{}' in {:.1f}s: {}", rel, elapsed, err)
            # still update progress and continue
            status["processed_files"] += 1
            try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
            except Exception: pass
            continue

        logger.info("chunk OK   '{}' chunks={} time={:.1f}s", rel, len(chunks), elapsed)
        per_file_stats[rel] = len(chunks)
        status["processed_files"] += 1
        try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
        except Exception: pass

        # rekey + buffer
        chunks = _rekey(chunks, project_root, root, fp)
        for c in chunks:
            buf_ids.append(c["id"])
            buf_docs.append(c["document"])
            buf_meta.append(c["metadata"])
            if len(buf_docs) >= upsert_batch or len(buf_docs) >= max_pending:
                _apply_max_chars(buf_docs, buf_meta, cfg)
                _flush()

        # If a stop was requested while we were working, mark dirty (no new submits happen above)
        if stop_flag.exists():
            dirty = True

    # final flush
    _apply_max_chars(buf_docs, buf_meta, cfg)
    _flush()

    # Stamp for delta indexing
    # Stamp lives under the project root alongside the DB
    # Cast to Path to avoid 'str' has no attribute 'mkdir'
    stamp_path = Path(project_rag_dir(project_root)) / ".last_index.json"
    try:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), "dirty": bool(dirty)}
        with stamp_path.open("w") as f:
            json.dump(payload, f)
        logger.info("Index stamp updated at '{}' dirty={}", str(stamp_path), bool(dirty))
    except Exception as e:
        logger.warning("Failed to write index stamp '{}': {}", str(stamp_path), e)

    # update status file (final)
    status.update({"state": "stopped" if dirty else "complete", "dirty": bool(dirty), "ended": time.time()})
    try:
        _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception:
        pass

    logger.info("full_reindex: done added={} files={} dirty={}", added, len(per_file_stats), bool(dirty))
    return {"ok": True, "added": added, "files": per_file_stats, "dirty": bool(dirty),
            "processed_files": status["processed_files"], "total_files": status["total_files"]}


def delta_index(project_root: str, rag_path: str | None) -> Dict:
    """
    Delta indexing:
      - Find files with mtime > last stamp
      - For each changed file: delete its old chunks by relpath, re-chunk and upsert
    Returns: {"ok": True, "added": <int>, "changed_files": [relpath, ...]}
    """
    logger.info("delta_index: begin project_root='{}' rag='{}'", project_root, rag_path)
    _load_env_for_project(project_root)  # make sure keys from .env are visible

    cfg = load_settings(rag_path)
    root = os.path.join(project_root, cfg["index_root"])
    logger.info("Index root: '{}'", root)

    coll = _chroma(project_root, cfg)
    models_path = _resolve_models_json_path(project_root)
    logger.info("models.json resolved: '{}'", models_path)
    embed_fn = resolve_embedder(models_path, cfg)
    logger.info("Embedder resolved for selected='{}'", (cfg.get("embedder") or {}).get("selected_name") or (cfg.get("embedder") or {}).get("model_key"))

    # Read last stamp
    # Read the stamp co-located with the project-level DB
    # Cast to Path to ensure proper path ops
    stamp_path = Path(project_rag_dir(project_root)) / ".last_index.json"
    last = 0.0
    if os.path.exists(stamp_path):
        try:
            with open(stamp_path, "r") as f:
                last = json.load(f).get("ts", 0.0)
            logger.info("Last index stamp: ts={} ({})", last, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)))
        except Exception as e:
            last = 0.0
            logger.warning("Failed to read index stamp '{}': {}", str(stamp_path), e)
    else:
        logger.info("No previous index stamp found (full delta)")

    changed, added = [], 0
    now = time.time()
    upsert_batch = int(cfg.get("upsert_batch_size") or 64)
    workers = _auto_workers(cfg)
    max_pending = int(cfg.get("max_pending_chunks") or 2000)
    logger.info("Batch params: upsert_batch_size={}", upsert_batch)

    # Build list of candidates that actually changed since last
    candidates: List[str] = []
    for fp in iter_files(root, cfg):
        try:
            mtime = os.path.getmtime(fp)
        except FileNotFoundError:
            continue
        if mtime <= last:
            continue
        candidates.append(fp)

    status = {"state": "running", "dirty": False, "processed_files": 0,
              "total_files": len(candidates), "started": time.time()}
    try:
        _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception:
        pass

    buf_ids: List[str] = []; buf_docs: List[str] = []; buf_meta: List[dict] = []
    def _flush():
        nonlocal added, buf_ids, buf_docs, buf_meta
        if not buf_docs:
            return
        _embed_and_upsert(coll, embed_fn, buf_ids, buf_docs, buf_meta, upsert_batch=upsert_batch)
        added += len(buf_docs)
        buf_ids.clear(); buf_docs.clear(); buf_meta.clear()

    dirty = False
    stop_flag = _stop_flag_path(project_root)
    for fp, chunks, elapsed, err in _parallel_chunk_stream(candidates, cfg, workers, project_root):
        rel = _relpath(root, fp)
        if err:
            logger.error("delta chunk FAIL '{}' in {:.1f}s: {}", rel, elapsed, err)
            status["processed_files"] += 1
            try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
            except Exception: pass
            continue

        logger.info("delta chunk OK   '{}' chunks={} time={:.1f}s", rel, len(chunks), elapsed)
        # purge old chunks for this file before inserting new
        try:
            coll.delete(where={"relpath": rel})
        except Exception as e:
            logger.debug("delta delete old failed for '{}': {}", rel, e)
        changed.append(rel)

        chunks = _rekey(chunks, project_root, root, fp)
        for c in chunks:
            buf_ids.append(c["id"]); buf_docs.append(c["document"]); buf_meta.append(c["metadata"])
            if len(buf_docs) >= upsert_batch or len(buf_docs) >= max_pending:
                _apply_max_chars(buf_docs, buf_meta, cfg); _flush()

        status["processed_files"] += 1
        try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
        except Exception: pass
        if stop_flag.exists():
            dirty = True

    _apply_max_chars(buf_docs, buf_meta, cfg); _flush()

    # Update stamp
    try:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        with stamp_path.open("w") as f:
            json.dump({"ts": now, "dirty": bool(dirty)}, f)
        logger.info("Index stamp updated at '{}' dirty={}", str(stamp_path), bool(dirty))
    except Exception as e:
        logger.warning("Failed to write index stamp '{}': {}", str(stamp_path), e)
        
    status.update({"state": "stopped" if dirty else "complete", "dirty": bool(dirty), "ended": time.time()})
    try:
        _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception:
        pass

    logger.info("delta_index: done added={} changed_files={} dirty={}", added, len(changed), bool(dirty))
    return {"ok": True, "added": added, "changed_files": changed, "dirty": bool(dirty),
            "processed_files": status["processed_files"], "total_files": status["total_files"]}
