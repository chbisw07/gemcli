# =========================
# indexing/settings.py
# =========================
import json, os
from typing import Any, Dict
from pathlib import Path
from loguru import logger
from config_home import GLOBAL_RAG_PATH

DEFAULT = {
    # --- Core paths & collection ---
    # Project-relative root to scan for indexing (joined with project_root at runtime)
    "index_root": ".",
    # If empty, we’ll use project_rag_dir(project_root). Can be overridden to point to a custom Chroma dir.
    "chroma_dir": "",
    # Base collection name; actual collection becomes f"{collection}-{hash(embedder)}"
    "collection": "code_chunks",

    # --- What files to index ---
    # File types we consider for indexing (chunkers decide how to parse each)
    "supported_file_types": [
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c",
        ".txt", ".md", ".rst", ".json", ".yaml", ".yml",
        ".csv", ".xlsx", ".ipynb",
        ".pdf", ".docx", ".pptx"
    ],
    # Directory names to ignore completely (by folder name)
    "ignore_dir": [
        "__pycache__", ".git", "tmp", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", "build", "dist", ".idea", ".vscode"
    ],
    # Path globs (relative to project root) to skip; checked in addition to ignore_dir
    "ignore_globs": [
        "**/.git/**", "**/node_modules/**", "**/.venv/**", "**/venv/**",
        "**/build/**", "**/dist/**", "**/.idea/**", "**/.vscode/**",
        "**/*.min.js", "**/*.min.css"
    ],

    # --- Chunking controls (applied by chunkers and indexer) ---
    "chunking": {
        # Hard cap before embedding to prevent model errors / cost spikes
        "max_embed_chars": 4000,

        # Text/document chunking defaults (used by .txt/.md and as a fallback)
        "text": {
            "max_chars": 1100,
            "overlap": 120
        },
        # Code chunking defaults (line-based)
        "code": {
            "max_lines": 120,
            "overlap_lines": 20
        }
    },

    # --- Embedder selection knobs (cooperate with data/models.json) ---
    # These are hints; resolve_embedder() primarily reads data/models.json
    "embedder": {
        # Preferred entry name from data/models.json (optional)
        "selected_name": "",
        # Or a model_key alias if you use those (optional)
        "model_key": "",
        # If you run an OpenAI-compatible endpoint; leave blank to use local ST if configured
        "endpoint": "",
        # Environment variable containing the API key (only used if endpoint requires it)
        "api_key_env": "OPENAI_API_KEY",
        # Micro-batching for remote embeddings; the local SentenceTransformer ignores this
        "batch_size": 16,
        # Network behavior
        "timeout_sec": 30,
        "retry": {"retries": 2, "backoff_sec": 1.5}
    },

    # --- Retrieval defaults ---
    "retrieval": {
        # Default top-k for queries (agent/tools may override per-call)
        "top_k": 8,
        # If your assistant wants to mix document/code signals during planning
        "hybrid": True,
        # Reserved for graph expansion in code retrieval (not used unless you wire it)
        "expand_callgraph_hops": 0
    },

    # --- Router (pre-RAG) defaults ---
    # Used by the router & retriever: post-filter by similarity and optional filename boost
    "router": {
        # Drop hits with similarity score below this (0..1). Set to 0.0 to disable.
        "threshold": 0.55,
        # If the user mentions a filename in the prompt, widen the pull and boost matching chunks
        "enable_filename_boost": True
    },

    # --- Metadata controls (what to store per chunk) ---
    # Chunkers always set core fields (file_path, relpath, file_ext, chunk_type, profile, etc.)
    # These toggles influence extra fields and sizes.
    "metadata": {
        "store_code_snippet": True,      # include a short code preview in metadata for code chunks
        "store_preview_chars": 300,      # preview length (for UI / debugging)
        "store_page_text": False,        # for PDFs, store full page text in metadata (heavy; off by default)
        "filename_into_metadata": True,  # duplicate basename as 'file_name' for easy filtering
        "add_profile": True              # add 'profile' = 'code' | 'document' | 'tabular'
    },

    # --- PDF parsing & normalization ---
    # Which PDF-level artifacts to emit as chunk_type (your pdf chunker will respect these)
    "pdf_emit": ["pdf_page", "pdf_structure", "pdf_blocks", "pdf_toc"],
    "ocr_fallback": True,     # try OCR if text extraction fails
    "dehyphenate": True,      # fix common hyphenation across line breaks
    "dedupe_headers": True,   # attempt to drop repeated headers/footers
    "max_pdf_chunk_chars": 16000,  # safety cap for any single PDF-derived chunk

    # --- PDF semantic windowing (preferred for prose) ---
    "pdf_semantic": {
        "enabled": True,
        "max_chars": 1100,     # good default for school-book prose and manuals
        "overlap": 120
    },

    # --- Batching & parallelism (indexer) ---
    "embedding_batch_size": 16,   # micro-batch size for embedding calls
    "upsert_batch_size": 128,     # group size for Chroma upserts
    # Number of worker processes for file chunking.
    # 0 => auto (cpu_count - 1, min 1). Use 1 to force serial (useful for debugging).
    "indexing_workers": 0,

    # --- Auto-indexing / watcher ---
    "auto_indexing": {
        "enabled": True,
        "watch_interval_sec": 10
    },

    # --- UI / misc (read by Streamlit app; safe to ignore in headless use) ---
    "ui": {
        "show_router_panel": True,
        "show_debug_json": False
    }
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
      2) global ~/.gencli/rag.json
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

    # Quick visibility of key knobs
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
    """Save to explicit path if given, else to ~/.gencli/rag.json."""
    target = Path(path) if path else GLOBAL_RAG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info("settings.save: wrote '{}'", str(target))
