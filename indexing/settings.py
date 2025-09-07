# =========================
# indexing/settings.py (Tarkash)
# =========================
import json, os
from typing import Any, Dict
from pathlib import Path
from loguru import logger
from config_home import GLOBAL_RAG_PATH  # ~/.tarkash/rag.json (template/fallback)

DEFAULT = {
    # --- Core paths & collection ---
    "index_root": ".",
    # If empty, the indexer will use the value from rag.json["chroma_dir"] or compute a per-project dir.
    "chroma_dir": "",
    "collection": "code_chunks",

    # --- What files to index ---
    "supported_file_types": [
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c",
        ".txt", ".md", ".rst", ".json", ".yaml", ".yml",
        ".csv", ".xlsx", ".ipynb",
        ".pdf", ".docx", ".pptx"
    ],
    "ignore_dir": [
        "__pycache__", ".git", "tmp", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", "build", "dist", ".idea", ".vscode"
    ],
    "ignore_globs": [
        "**/.git/**", "**/node_modules/**", "**/.venv/**", "**/venv/**",
        "**/build/**", "**/dist/**", "**/.idea/**", "**/.vscode/**",
        "**/*.min.js", "**/*.min.css"
    ],

    # --- Chunking controls ---
    "chunking": {
        "max_embed_chars": 4000,
        "text": {"max_chars": 1100, "overlap": 120},
        "code": {"max_lines": 120, "overlap_lines": 20},
    },

    # --- Embedder selection knobs ---
    "embedder": {
        "selected_name": "",
        "model_key": "",
        "endpoint": "",
        "api_key_env": "OPENAI_API_KEY",
        "batch_size": 16,
        "timeout_sec": 30,
        "retry": {"retries": 2, "backoff_sec": 1.5},
    },

    # --- Retrieval defaults ---
    "retrieval": {"top_k": 8, "hybrid": True, "expand_callgraph_hops": 0},

    # --- Router defaults ---
    "router": {"threshold": 0.55, "enable_filename_boost": True},

    # --- Metadata controls ---
    "metadata": {
        "store_code_snippet": True,
        "store_preview_chars": 300,
        "store_page_text": False,
        "filename_into_metadata": True,
        "add_profile": True,
    },

    # --- PDF parsing & normalization ---
    "pdf_emit": ["pdf_page", "pdf_structure", "pdf_blocks", "pdf_toc"],
    "ocr_fallback": True,
    "dehyphenate": True,
    "dedupe_headers": True,
    "max_pdf_chunk_chars": 16000,

    # --- PDF semantic windowing ---
    "pdf_semantic": {"enabled": True, "max_chars": 1100, "overlap": 120},

    # --- Batching & parallelism (indexer) ---
    "embedding_batch_size": 16,
    "upsert_batch_size": 128,
    "indexing_workers": 0,

    # --- Auto-indexing / watcher ---
    "auto_indexing": {"enabled": True, "watch_interval_sec": 10},

    # --- UI / misc ---
    "ui": {"show_router_panel": True, "show_debug_json": False},
}


def _read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        logger.debug("settings._read_json: path '{}' does not exist (returning empty)", str(p))
        return {}
    logger.debug("settings._read_json: reading '{}'", str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def load(path: str | None) -> Dict[str, Any]:
    """
    Load RAG settings with precedence:
      1) explicit path (if provided and exists)
      2) global ~/.tarkash/rag.json (template/fallback)
      3) DEFAULT
    """
    merged = DEFAULT.copy()
    if path and Path(path).exists():
        src = Path(path)
    else:
        src = GLOBAL_RAG_PATH
    logger.info("settings.load: source='{}'", str(src))

    data = _read_json(src)
    for k, v in data.items():
        merged[k] = v

    logger.info(
        "settings.load: index_root='{}' collection='{}' embedder='{}' top_k={} pdf_emit={}",
        merged.get("index_root"),
        merged.get("collection"),
        (merged.get("embedder") or {}).get("selected_name")
            or (merged.get("embedder") or {}).get("model_key"),
        (merged.get("retrieval") or {}).get("top_k"),
        merged.get("pdf_emit"),
    )
    return merged


def save(path: str | None, cfg: Dict[str, Any]) -> None:
    """Save to explicit path if given, else to ~/.tarkash/rag.json."""
    target = Path(path) if path else GLOBAL_RAG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info("settings.save: wrote '{}'", str(target))
