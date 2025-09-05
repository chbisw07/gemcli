# indexing/embedder.py
# Simple, explicit embedder resolver for gemcli.
# Rules:
# - Choose embedder from rag_cfg['embedder']['selected_name'] (or model_key),
#   else models.json['default_embedding_model'], else first entry.
# - If api_key_reqd is true, read the key from env[api_key_env] (loaded via .env).
# - If endpoint is given, use OpenAI-compatible /v1/embeddings with micro-batching.
# - If no endpoint, fall back to local SentenceTransformers (if available).
# - Only treat the endpoint as "unavailable" when we cannot connect at all
#   (connection error / timeout). Otherwise let the POST raise (clear server error).

from __future__ import annotations

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
from loguru import logger
from urllib.parse import urlsplit, urlunsplit

# Optional: auto-load .env so API keys are available when running outside Streamlit
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


# --------------------------- small helpers ---------------------------

def _load_models_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug("embedder: loaded models.json from '{}'", path)
            return data
    except Exception as e:
        logger.error("embedder: failed to load models.json '{}': {}", path, e)
        return {}

def _select_embedder(models_cfg: Dict[str, Any], rag_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick by selected_name/model_key → default_embedding_model → first."""
    selected = ((rag_cfg.get("embedder") or {}).get("selected_name")
                or (rag_cfg.get("embedder") or {}).get("model_key"))
    embs = models_cfg.get("embedding_models") or []

    if selected:
        for e in embs:
            if e.get("name") == selected:
                logger.info("embedder: selected='{}' via rag_cfg", selected)
                return e

    default_name = models_cfg.get("default_embedding_model")
    if default_name:
        for e in embs:
            if e.get("name") == default_name:
                logger.info("embedder: using default_embedding_model='{}'", default_name)
                return e

    if embs:
        logger.info("embedder: using first entry='{}'", embs[0].get("name"))
    else:
        logger.error("embedder: no entries found in models.json")
    return embs[0] if embs else None

def _load_env(models_json_path: str) -> None:
    """Load .env from common spots without overriding existing env vars."""
    if load_dotenv is None:
        logger.debug("embedder: python-dotenv not available; skipping .env load")
        return
    try:
        mj = Path(models_json_path).resolve()
    except Exception:
        mj = None

    candidates = []
    if mj:
        # If models.json is <project_root>/data/models.json, try project_root/.env and data/.env
        candidates.append(mj.parent.parent / ".env")  # <project_root>/.env
        candidates.append(mj.parent / ".env")         # <project_root>/data/.env
    candidates.append(Path.cwd() / ".env")            # CWD/.env

    for p in candidates:
        try:
            if p and p.exists():
                load_dotenv(p, override=False)
                logger.debug("embedder: loaded env from '{}'", str(p))
        except Exception as e:
            logger.debug("embedder: failed to load env from '{}': {}", str(p), e)

def _norm_embeddings_url(endpoint: str) -> str:
    ep = endpoint.rstrip("/")
    if ep.endswith("/embeddings"):
        return ep
    if ep.endswith("/v1"):
        return ep + "/embeddings"
    return ep + "/v1/embeddings"

def _headers(entry: Dict[str, Any]) -> Dict[str, str]:
    """Build headers. If api_key_reqd, read from env[api_key_env]."""
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": "gemcli-embedder/1.0",
    }
    if entry.get("api_key_reqd"):
        key_env = entry.get("api_key_env") or "OPENAI_API_KEY"
        key = os.getenv(key_env)
        if not key:
            msg = f"Embedding '{entry.get('name')}' requires API key env '{key_env}' but it is not set."
            logger.error("embedder: {}", msg)
            raise RuntimeError(msg)
        headers["Authorization"] = f"Bearer {key}"
        logger.debug("embedder: auth via env '{}'", key_env)
    return headers

def _endpoint_reachable(url: str, timeout: float = 2.0) -> bool:
    """
    Connectivity probe: hit the **host root** with HEAD (not /v1/embeddings).
    We only care about socket reachability, not HTTP semantics of the embeddings route.
    """
    parts = urlsplit(url)
    base = urlunsplit((parts.scheme, parts.netloc, "/", "", ""))
    try:
        # Prefer HEAD to avoid noisy logs; fall back to GET if HEAD not allowed.
        try:
            requests.head(base, timeout=timeout, allow_redirects=True)
        except requests.RequestException:
            requests.get(base, timeout=timeout)
        logger.debug("embedder: endpoint reachable '{}'", base)
        return True
    except requests.RequestException as e:
        logger.debug("embedder: endpoint unreachable '{}': {}", base, e)
        return False


# ----------------------------- public API -----------------------------

def resolve_embedder(models_json_path: str, rag_cfg: Dict[str, Any]):
    """
    Returns a callable: embed(texts) -> List[List[float]]

    Behavior:
      - Loads .env so env[api_key_env] is available.
      - Selects the embedder per the simple rules agreed.
      - If entry has 'endpoint' → use OpenAI-compatible POST /v1/embeddings.
      - Else fall back to local sentence-transformers (if installed), else error.
    """
    logger.info("resolve_embedder: models='{}' selected='{}'",
                models_json_path,
                (rag_cfg.get('embedder') or {}).get('selected_name') or (rag_cfg.get('embedder') or {}).get('model_key'))

    _load_env(models_json_path)

    models_cfg = _load_models_json(models_json_path)
    entry = _select_embedder(models_cfg, rag_cfg)
    if not entry:
        raise RuntimeError("No embedding model entry found in models.json (embedding_models is empty).")

    # If there is an explicit endpoint, use OpenAI-compatible embeddings API.
    endpoint = entry.get("endpoint")
    if endpoint:
        url = _norm_embeddings_url(str(endpoint))
        headers = _headers(entry)
        model_name = entry.get("model") or entry.get("name") or "embedding-model"
        microbatch = int(rag_cfg.get("embedding_batch_size") or 8)
        logger.info("embedder: remote OpenAI-compatible url='{}' model='{}' microbatch={}", url, model_name, microbatch)

        # Only fail early if we can't connect to the host at all.
        if not _endpoint_reachable(url, timeout=2.0):
            raise RuntimeError(
                f"Embedding endpoint not reachable: {url}. "
                "If this is LM Studio, ensure the embedding model is loaded on that port."
            )

        def embed(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            out: List[List[float]] = []
            for i in range(0, len(texts), microbatch):
                batch = texts[i:i + microbatch]
                logger.debug("embedder: POST batch [{}:{}] (size={})", i, i + len(batch), len(batch))
                resp = requests.post(
                    url,
                    headers=headers,
                    json={"model": model_name, "input": batch},
                    timeout=120,
                )
                # Let the server speak clearly if something is wrong (401/404/500…)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("data") or data.get("embeddings") or []
                # Normalize common shapes
                if rows and isinstance(rows[0], dict) and "embedding" in rows[0]:
                    out.extend([r["embedding"] for r in rows])
                elif rows and isinstance(rows[0], list):
                    out.extend(rows)
                else:
                    # Defensive message if the server returns an unexpected schema
                    logger.error("embedder: unexpected response schema (keys={})", list(data.keys()))
                    raise RuntimeError("Embedding endpoint returned an unexpected response schema.")
            return out

        return embed

    # Fallback: local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("embedder: no endpoint and no sentence-transformers installed")
        raise RuntimeError(
            "No 'endpoint' specified for the selected embedder and 'sentence-transformers' "
            "is not installed for local embeddings."
        ) from e

    local_name = entry.get("model") or entry.get("name") or "all-MiniLM-L6-v2"
    logger.info("embedder: local SentenceTransformer='{}'", local_name)
    st_model = SentenceTransformer(local_name)

    def _local_embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # normalize_embeddings=True gives unit-length vectors (nice for cosine)
        return st_model.encode(texts, normalize_embeddings=True).tolist()

    return _local_embed
