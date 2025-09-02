# config_home.py
import hashlib
import os
from pathlib import Path
from loguru import logger

APP_DIR = Path(os.path.expanduser("~")) / ".gencli"
try:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug("config_home.APP_DIR ensured at '{}'", str(APP_DIR))
except Exception as e:
    logger.error("Failed to create APP_DIR at '{}': {}", str(APP_DIR), e)

UI_SETTINGS_PATH = APP_DIR / "ui_settings.json"
GLOBAL_RAG_PATH = APP_DIR / "rag.json"
logger.debug("config_home paths → UI_SETTINGS_PATH='{}', GLOBAL_RAG_PATH='{}'", str(UI_SETTINGS_PATH), str(GLOBAL_RAG_PATH))


def project_key(project_root: str) -> str:
    # human-friendly name with short hash for collisions
    try:
        name = Path(project_root).name or "project"
        h = hashlib.sha1(os.path.abspath(project_root).encode("utf-8")).hexdigest()[:8]
        key = f"{name}-{h}"
        logger.debug("config_home.project_key('{}') → '{}'", project_root, key)
        return key
    except Exception as e:
        logger.error("project_key failed for '{}': {}", project_root, e)
        # fallback
        return f"project-{hash(project_root) & 0xffff:x}"


def project_rag_dir(project_root: str) -> Path:
    try:
        p = APP_DIR / project_key(project_root) / "RAG"
        logger.debug("config_home.project_rag_dir('{}') → '{}'", project_root, str(p))
        return p
    except Exception as e:
        logger.error("project_rag_dir failed for '{}': {}", project_root, e)
        # fallback
        return APP_DIR / "RAG"
