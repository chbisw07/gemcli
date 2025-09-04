# indexing/settings.py
import json, os
from typing import Any, Dict
from pathlib import Path
from loguru import logger
from config_home import GLOBAL_RAG_PATH

DEFAULT = {
    # --- Core paths & collection ---
    "index_root": ".",
    "chroma_dir": "",               # if empty, project_rag_dir(project_root) is used
    "collection": "code_chunks",

    # --- What files to index ---
    "supported_file_types": [".py", ".txt", ".md", ".pdf"],
    "ignore_dir": ["__pycache__", ".git", "tmp", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"],
    "ignore_ext": [".pyc", ".pyo", ".log"],
    "ignore_files": ["__init__.py"],

    # --- Chunking defaults (code/text/pdf) ---
    "chunking": {
        "py":   {"max_body_chars": 8000},
        "text": {"max_chars": 2000, "overlap": 200},
        "pdf":  {"page_window": 1},          # container windows; semantic windows are below
        "max_embed_chars": 4000              # guardrail before embedding large docs
    },

    # --- Retrieval defaults ---
    "retrieval": {
        "top_k": 8,
        "hybrid": True,
        "expand_callgraph_hops": 0
    },

    # --- Metadata knobs ---
    "metadata": {
        "store_code_snippet": True,
        "store_qualified_name": True
    },

    # --- Embedder selection ---
    "embedder": {
        "provider": "auto",                  # "auto", "openai_compat", "sentence_transformers", etc.
        "model_key": "embedding_default"     # resolved via models.json
    },

    # --- PDF: what to emit + cleanup ---
    "pdf_emit": ["pdf_page", "pdf_structure", "pdf_blocks", "pdf_toc"],
    "ocr_fallback": True,
    "dehyphenate": True,
    "dedupe_headers": True,
    "max_pdf_chunk_chars": 16000,

    # --- PDF semantic windowing (different from code/text) ---
    "pdf_semantic": {
        "enabled": True,
        "max_chars": 1100,                   # good default for school-book prose
        "overlap": 120
    },

    "embedding_batch_size": 16,              # remote OpenAI-compat: 16–32; local ST: ~64
    "upsert_batch_size": 128,                # Chroma upsert batch size
    # --- Parallel chunking + backpressure ---
    "indexing_workers": 0,                   # 0=auto (cpu_count-1, min 1)
    "max_pending_chunks": 2000,              # memory guard: flush when buffered docs hit this

    # --- Auto-indexer ---
    "auto_indexing": {
        "enabled": True,
        "watch_interval_sec": 10
    },
    
    # Grounding / web policy
    "grounding": {
        "profile": "auto",           # auto | education | finance | general
        "min_chunks": 6,             # if fewer RAG chunks than this, consider fallback
        "min_pdf_ratio": 0.6,        # % of context that should be from PDFs
        "allow_web_fallback": True,
        "education_web_fallback": "ask"  # ask | yes | no
    },
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
