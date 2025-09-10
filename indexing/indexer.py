from __future__ import annotations

import re, os, json, time, hashlib, traceback
from typing import Dict, List
from pathlib import Path

from loguru import logger
import chromadb
import concurrent.futures as cf

from config_home import project_rag_dir, set_project_embedder  # ~/.tarkash/projects/<name>/RAG
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

def _resolve_rag_json_path(project_root: str, rag_path: str | None) -> str | None:
    """
    Prefer the project's rag.json. Rules:
      1) If rag_path is an explicit .json → use it.
      2) If rag_path is a directory and has rag.json → use it.
      3) <project_root>/rag.json if present.
      4) Parent of project RAG dir (…/<project>/rag.json) if present.
      5) Else None (caller may decide a fallback).
    """
    try:
        if rag_path and rag_path.lower().endswith(".json") and os.path.isfile(rag_path):
            return rag_path
        if rag_path and os.path.isdir(rag_path):
            cand = os.path.join(rag_path, "rag.json")
            if os.path.isfile(cand):
                return cand
        cand = os.path.join(project_root, "rag.json")
        if os.path.isfile(cand):
            return cand
        # ~/.tarkash/projects/<name>/rag.json (sibling of RAG dir)
        try:
            cand = str(Path(project_rag_dir(project_root)).parent / "rag.json")
            if os.path.isfile(cand):
                return cand
        except Exception:
            pass
    except Exception:
        pass
    return None


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
        # Guarantee a normalized 'document' field so downstream code/DB is consistent
        if "document" not in c:
            c["document"] = _as_doc_text(c)
        elif not isinstance(c["document"], str):
            try:
                c["document"] = str(c["document"] or "")
            except Exception:
                c["document"] = ""
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

 # ------------------------- manifest helpers for delta -------------------------
def _manifest_path(project_root: str) -> Path:
    return Path(project_rag_dir(project_root)) / ".manifest.json"

def _load_manifest_file(project_root: str) -> dict:
    p = _manifest_path(project_root)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

def _save_manifest_file(project_root: str, manifest: dict, stamp_ts: float | None = None, dirty: bool = False) -> None:
    # persist manifest and also refresh the legacy stamp used by UI
    p = _manifest_path(project_root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write manifest '{}': {}", str(p), e)
    # also write .last_index.json for backward compatibility
    try:
        ts = float(stamp_ts) if stamp_ts is not None else max([v.get("mtime", 0) for v in manifest.values()] + [time.time()])
        stamp = {"ts": ts, "dirty": bool(dirty)}
        (_status_path(project_root).parent / ".last_index.json").write_text(json.dumps(stamp), encoding="utf-8")
    except Exception as e:
        logger.debug("Save stamp failed: {}", e)

def _scan_current_manifest(root: str, cfg: dict) -> dict:
    cur: dict = {}
    for fp in iter_files(root, cfg):
        try:
            mt = int(os.path.getmtime(fp))
            sz = int(os.path.getsize(fp))
        except FileNotFoundError:
            continue
        rel = _relpath(root, fp)
        cur[rel] = {"mtime": mt, "size": sz}
    return cur


# ----------------------------- chunk sanitization ------------------------------
_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")
_WS_MULTI = re.compile(r"\s+")

def _min_chunk_chars() -> int:
    # allow override; default 8 chars to skip pure noise
    try:
        return int(os.environ.get("TARKASH_MIN_CHUNK_CHARS", "8"))
    except Exception:
        return 8

def _clean_for_embed(text: str) -> str:
    """Collapse whitespace and trim; keep content otherwise intact."""
    if not text:
        return ""
    return _WS_MULTI.sub(" ", text).strip()

def _is_meaningful(text: str) -> bool:
    """Heuristic: non-empty after trim, has at least one alpha/num, and length threshold."""
    if not text:
        return False
    s = text.strip()
    if len(s) < _min_chunk_chars():
        return False
    return bool(_HAS_ALNUM.search(s))

def _as_doc_text(chunk: dict) -> str:
    """
    Extract the chunk text consistently:
    prefer 'document', fall back to 'text', then clean & trim.
    """
    txt = chunk.get("document", None)
    if txt is None:
        txt = chunk.get("text", "")
    if not isinstance(txt, str):
        try:
            txt = str(txt or "")
        except Exception:
            txt = ""
    return _clean_for_embed(txt)


# ----------------------------- chroma collection ------------------------------

def _chroma(project_root: str, cfg: dict, embedder_name: str | None = None):
    """
    Create/get a Chroma collection for this project.
    IMPORTANT: Do NOT pass an embedding_function here.
               We embed client-side and pass vectors explicitly.
    The RAG store path is anchored to the *project root* for consistency with the app.
    Also stamps/refreshes collection metadata with the embedder name for mismatch checks.
    """
    # Ensure we always operate on a real Path object (mkdir used below)
    rag_dir = Path(project_rag_dir(project_root))
    rag_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(rag_dir))
    coll_name = _collection_name_for(cfg)
    logger.info("Chroma store: dir='{}' collection='{}'", str(rag_dir), coll_name)
    coll = client.get_or_create_collection(name=coll_name)
    # Best-effort metadata stamp (newer Chroma supports modify)
    try:
        meta = dict(getattr(coll, "metadata", {}) or {})
        if project_root:
            meta["project_root"] = project_root
        if embedder_name:
            meta["embedder_name"] = embedder_name
        if hasattr(coll, "modify"):
            coll.modify(metadata=meta)  # type: ignore[attr-defined]
    except Exception as _e:
        logger.debug("_chroma: could not set collection metadata: {}", _e)
    return coll


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
    # Normalize & filter before embedding (belt & suspenders)
    keep = []
    for i in range(len(docs)):
        try:
            s = "" if docs[i] is None else str(docs[i])
        except Exception:
            s = ""
        s = _clean_for_embed(s)
        docs[i] = s
        if _is_meaningful(s):
            keep.append(i)
    if not keep:
        logger.debug("_embed_and_upsert: nothing meaningful to embed")
        return
    ids   = [ids[i]   for i in keep]
    docs  = [docs[i]  for i in keep]
    metas = [metas[i] for i in keep]

    # Embed once for this filtered batch
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
        # docs already normalized above

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
    # Add contextual logging: component, project key, and root
    log = logger.bind(component="indexer",
                      project=_project_key(project_root),
                      project_root=project_root)
    log.info("full_reindex: begin project_root='{}' rag='{}'", project_root, rag_path)
    _load_env_for_project(project_root)  # make sure keys from .env are visible

    # Always prefer the project's rag.json (explicit .json overrides)
    cfg = load_settings(_resolve_rag_json_path(project_root, rag_path))
    root = os.path.join(project_root, cfg["index_root"])
    log.info("Index root: '{}'", root)

    models_path = _resolve_models_json_path(project_root)
    log.info("models.json resolved: '{}'", models_path)
    embed_fn = resolve_embedder(models_path, cfg)
    emb_name = getattr(embed_fn, "name", None) or (cfg.get("embedder") or {}).get("selected_name") or (cfg.get("embedder") or {}).get("model_key") or "unknown"
    # Enrich context with embedder
    log = log.bind(embedder=emb_name)
    log.info("Embedder resolved: '{}'", emb_name)
    # Persist the resolved embedder into the per-project rag.json (authoritative source of truth)
    try:
        set_project_embedder(project_root, emb_name)
    except Exception as e:
        log.debug("full_reindex: set_project_embedder failed: {}", e)
    coll = _chroma(project_root, cfg, embedder_name=emb_name)

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
    log.info("Parallel chunking: files={} workers={} upsert_batch={} max_pending={}",
                len(files), workers, upsert_batch, max_pending)

    # status seed
    status = {"state": "running", "dirty": False, "processed_files": 0,
              "total_files": len(files), "started": time.time()}
    try:
        _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception:
        pass

    added, per_file_stats = 0, {}
    # Track the newest mtime we actually process; stamp with this at the end.
    max_seen_mtime: float = 0.0
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
        # update max mtime for the stamp
        try:
            _mt = os.path.getmtime(fp)
            if _mt > max_seen_mtime:
                max_seen_mtime = _mt
        except Exception:
            pass
        if err:
            log.error("chunk FAIL '{}' in {:.1f}s: {}", rel, elapsed, err)
            # still update progress and continue
            status["processed_files"] += 1
            try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
            except Exception: pass
            continue

        # sanitize & filter chunks before enqueue (standardize to 'document')
        sanitized: List[dict] = []
        for c in chunks:
            doc = _as_doc_text(c)
            if not _is_meaningful(doc):
                continue
            sanitized.append({"document": doc, "metadata": c.get("metadata", {})})
        log.info("chunk OK   '{}' chunks={} kept={} time={:.1f}s", rel, len(chunks), len(sanitized), elapsed)
        per_file_stats[rel] = len(sanitized)
        status["processed_files"] += 1
        try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
        except Exception: pass

        # rekey + buffer
        for c in _rekey(sanitized, project_root, root, fp):
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

    # Persist manifest for delta baseline (and keep legacy stamp)
    try:
        cur_manifest = _scan_current_manifest(root, cfg)
        _save_manifest_file(project_root, cur_manifest,
                            stamp_ts=(max_seen_mtime or time.time()),
                            dirty=bool(dirty))
    except Exception as e:
        log.warning("Failed to save manifest/stamp: {}", e)

    # update status file (final)
    status.update({"state": "stopped" if dirty else "complete", "dirty": bool(dirty), "ended": time.time()})
    try:
        _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception:
        pass

    log.info("full_reindex: done added={} files={} dirty={}", added, len(per_file_stats), bool(dirty))
    # Normalize keys to align with delta_index summary
    return {
        "ok": True,
        "added": added,
        "updated": 0,
        "deleted": 0,
        "files_changed": len(per_file_stats),
        "files": per_file_stats,
        "dirty": bool(dirty),
        "processed_files": status["processed_files"],
        "total_files": status["total_files"],
        "embedder": emb_name,
    }


def delta_index(project_root: str, rag_path: str | None) -> Dict:
    """
    Delta indexing (manifest-based):
      - Build current manifest { rel: {mtime,size} }
      - Compare with previous manifest to find added/changed/deleted
      - deleted: purge by relpath
      - added/changed: purge old by relpath, re-chunk, embed, upsert
    """
    # Add contextual logging: component, project key, and root
    log = logger.bind(component="indexer",
                      project=_project_key(project_root),
                      project_root=project_root)
    log.info("delta_index(manifest): begin project_root='{}' rag='{}'", project_root, rag_path)
    _load_env_for_project(project_root)

    # Always prefer the project's rag.json (explicit .json overrides)
    cfg = load_settings(_resolve_rag_json_path(project_root, rag_path))
    root = os.path.join(project_root, cfg["index_root"])
    log.info("Index root: '{}'", root)

    models_path = _resolve_models_json_path(project_root)
    log.info("models.json resolved: '{}'", models_path)
    embed_fn = resolve_embedder(models_path, cfg)
    emb_name = getattr(embed_fn, "name", None) or (cfg.get("embedder") or {}).get("selected_name") or (cfg.get("embedder") or {}).get("model_key") or "unknown"
    # Enrich context with embedder
    log = log.bind(embedder=emb_name)
    log.info("Embedder resolved: '{}'", emb_name)
    # Keep per-project rag.json in sync with the embedder actually used for delta indexing
    try:
        set_project_embedder(project_root, emb_name)
    except Exception as e:
        log.debug("delta_index: set_project_embedder failed: {}", e)
    coll = _chroma(project_root, cfg, embedder_name=emb_name)

    prev = _load_manifest_file(project_root)
    cur  = _scan_current_manifest(root, cfg)

    prev_keys = set(prev.keys())
    cur_keys  = set(cur.keys())
    deleted = sorted(list(prev_keys - cur_keys))
    added   = sorted(list(cur_keys - prev_keys))
    changed = sorted([r for r in (cur_keys & prev_keys)
                      if (prev[r].get("mtime") != cur[r].get("mtime")
                          or prev[r].get("size")  != cur[r].get("size"))])

    logger.info("delta_index(manifest): diff added={} changed={} deleted={}", len(added), len(changed), len(deleted))

    candidates = [os.path.join(root, r) for r in (added + changed)]
    status = {"state": "running", "dirty": False, "processed_files": 0,
              "total_files": len(candidates), "started": time.time()}
    try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception: pass

    # purge deletions upfront
    for rel in deleted:
        try:
            coll.delete(where={"relpath": rel})
            logger.info("delta_index: purged deleted relpath='{}'", rel)
        except Exception as e:
            logger.warning("delta_index: purge failed for rel='{}': {}", rel, e)

    # chunk + upsert for added/changed
    workers = _auto_workers(cfg)
    upsert_batch = int(cfg.get("upsert_batch_size") or 64)
    max_pending = int(cfg.get("max_pending_chunks") or 2000)
    log.info("Parallel chunking delta: files={} workers={} upsert_batch={}", len(candidates), workers, upsert_batch)

    dirty = False
    per_file_stats = {}
    buf_ids: List[str] = []
    buf_docs: List[str] = []
    buf_meta: List[dict] = []

    def _flush():
        nonlocal buf_ids, buf_docs, buf_meta
        if not buf_docs:
            return
        _apply_max_chars(buf_docs, buf_meta, cfg)
        _embed_and_upsert(coll, embed_fn, buf_ids, buf_docs, buf_meta, upsert_batch=upsert_batch)
        buf_ids.clear(); buf_docs.clear(); buf_meta.clear()

    for fp, chunks, elapsed, err in _parallel_chunk_stream(candidates, cfg, workers, project_root):
        rel = _relpath(root, fp)
        if err:
            log.error("delta chunk FAIL '{}' in {:.1f}s: {}", rel, elapsed, err)
            status["processed_files"] += 1
            try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
            except Exception: pass
            continue

        # sanitize & filter chunks before enqueue (standardize to 'document')
        sanitized: List[dict] = []
        for c in chunks:
            doc = _as_doc_text(c)
            if not _is_meaningful(doc):
                continue
            sanitized.append({"document": doc, "metadata": c.get("metadata", {})})
        log.info("delta chunk OK   '{}' chunks={} kept={} time={:.1f}s", rel, len(chunks), len(sanitized), elapsed)
        per_file_stats[rel] = len(sanitized)
        status["processed_files"] += 1
        try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
        except Exception: pass

        # purge existing chunks for this relpath, then upsert
        try:
            coll.delete(where={"relpath": rel})
        except Exception as e:
            log.debug("delete before upsert skipped/failed for rel='{}': {}", rel, e)

        for c in _rekey(sanitized, project_root, root, fp):
            buf_ids.append(c["id"])
            buf_docs.append(c["document"])
            buf_meta.append(c.get("metadata") or {})
            if len(buf_docs) >= max_pending:
                _flush()

    _flush()

    # persist new manifest & legacy stamp
    stamp_ts = max([v.get("mtime", 0) for v in cur.values()] + [time.time()])
    _save_manifest_file(project_root, cur, stamp_ts=stamp_ts, dirty=dirty)

    status.update({"state": "stopped" if dirty else "complete", "dirty": bool(dirty), "ended": time.time()})
    try: _status_path(project_root).write_text(json.dumps(status), encoding="utf-8")
    except Exception: pass

    summary = {
        "ok": True,
        "files_changed": len(changed),
        "added": len(added),
        "updated": len(changed),
        "deleted": len(deleted),
        "processed_files": status["processed_files"],
        "total_files": status["total_files"],
        "changed_files": changed,
        "added_files": added,
        "deleted_files": deleted,
        "files": per_file_stats,
    }
    log.info("delta_index(manifest): done {}", summary)
    return summary
