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

# Optional: auto-load .env so API keys are available when running outside Streamlit
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


# --------------------------- small helpers ---------------------------

def _load_models_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _select_embedder(models_cfg: Dict[str, Any], rag_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick by selected_name/model_key → default_embedding_model → first."""
    selected = ((rag_cfg.get("embedder") or {}).get("selected_name")
                or (rag_cfg.get("embedder") or {}).get("model_key"))
    embs = models_cfg.get("embedding_models") or []

    if selected:
        for e in embs:
            if e.get("name") == selected:
                return e

    default_name = models_cfg.get("default_embedding_model")
    if default_name:
        for e in embs:
            if e.get("name") == default_name:
                return e

    return embs[0] if embs else None

def _load_env(models_json_path: str) -> None:
    """Load .env from common spots without overriding existing env vars."""
    if load_dotenv is None:
        return
    try:
        mj = Path(models_json_path).resolve()
    except Exception:
        mj = None

    candidates = []
    if mj:
        # If models.json is <project_root>/data/models.json, try project_root/.env and data/.env
        candidates.append(mj.parent.parent / ".env")
        candidates.append(mj.parent / ".env")
    candidates.append(Path.cwd() / ".env")  # current working dir

    for p in candidates:
        try:
            if p and p.exists():
                load_dotenv(p, override=False)
        except Exception:
            continue

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
        "Accept-Encoding": "identity",   # avoid large compressed responses
        "Connection": "close",
        "User-Agent": "gemcli-embedder/1.0",
    }
    if entry.get("api_key_reqd"):
        key_env = entry.get("api_key_env")
        if not key_env:
            # Keep it simple: default to OPENAI_API_KEY if not provided
            key_env = "OPENAI_API_KEY"
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(
                f"Embedding '{entry.get('name')}' requires API key env '{key_env}' but it is not set."
            )
        headers["Authorization"] = f"Bearer {key}"
    return headers

def _endpoint_reachable(url: str, timeout: float = 2.0) -> bool:
    """
    Only checks basic connectivity (host:port reachable). We do NOT interpret HTTP status;
    any response means it's up. Pure connection errors/timeouts → unreachable.
    """
    try:
        # Some servers may not support GET here; we still only care about connectivity.
        requests.get(url, timeout=timeout)
        return True
    except requests.RequestException:
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

        # Only fail early if we can't connect at all.
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
                    raise RuntimeError("Embedding endpoint returned an unexpected response schema.")
            return out

        return embed

    # Fallback: local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No 'endpoint' specified for the selected embedder and 'sentence-transformers' "
            "is not installed for local embeddings."
        ) from e

    local_name = entry.get("model") or entry.get("name") or "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(local_name)

    def _local_embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # normalize_embeddings=True gives unit-length vectors (nice for cosine)
        return st_model.encode(texts, normalize_embeddings=True).tolist()

    return _local_embed
