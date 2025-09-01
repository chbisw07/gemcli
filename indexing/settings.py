import json, os
from typing import Any, Dict
from pathlib import Path
from config_home import GLOBAL_RAG_PATH

DEFAULT = {
    "index_root": ".",
    "chroma_dir": "",
    "collection": "code_chunks",
    "supported_file_types": [".py", ".txt", ".md", ".pdf"],
    "ignore_dir": ["__pycache__", ".git", "tmp", "node_modules", ".venv", "venv"],
    "ignore_ext": [".pyc", ".pyo", ".log"],
    "ignore_files": ["__init__.py"],
    "chunking": {"py": {"max_body_chars": 8000}, "text": {"max_chars": 2000, "overlap": 200}, "pdf": {"page_window": 1}},
    "retrieval": {"top_k": 8, "hybrid": True, "expand_callgraph_hops": 0},
    "auto_indexing": {"enabled": True, "watch_interval_sec": 10},
    "metadata": {"store_code_snippet": True, "store_qualified_name": True},
    "embedder": {"provider": "auto", "model_key": "embedding_default"}
}

def _read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def load(path: str | None) -> Dict[str, Any]:
    """
    Load RAG settings with precedence:
      1) explicit path (if provided and exists)
      2) global ~/.gencli/rag.json
      3) DEFAULT
    """
    merged = DEFAULT.copy()
    # explicit
    if path and Path(path).exists():
        data = _read_json(Path(path))
    else:
        data = _read_json(GLOBAL_RAG_PATH)

    # shallow merge defaults
    for k, v in data.items():
        merged[k] = v
    return merged

def save(path: str | None, cfg: Dict[str, Any]) -> None:
    """Save to explicit path if given, else to ~/.gencli/rag.json."""
    target = Path(path) if path else GLOBAL_RAG_PATH
    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")