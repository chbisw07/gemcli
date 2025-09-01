# config_home.py
import hashlib
import os
from pathlib import Path

APP_DIR = Path(os.path.expanduser("~")) / ".gencli"
APP_DIR.mkdir(parents=True, exist_ok=True)

UI_SETTINGS_PATH = APP_DIR / "ui_settings.json"
GLOBAL_RAG_PATH = APP_DIR / "rag.json"

def project_key(project_root: str) -> str:
    # human-friendly name with short hash for collisions
    name = Path(project_root).name or "project"
    h = hashlib.sha1(os.path.abspath(project_root).encode("utf-8")).hexdigest()[:8]
    return f"{name}-{h}"

def project_rag_dir(project_root: str) -> Path:
    return APP_DIR / project_key(project_root) / "RAG"
