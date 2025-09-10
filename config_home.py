# config_home.py — Tarkash home & per-project paths (with project_rag_dir)
from __future__ import annotations

import os
import json
import shutil
import re
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

# ---------- App home ----------

def _resolve_home() -> Path:
    env = os.getenv("TARKASH_HOME", "").strip()
    base = Path(os.path.expanduser(env)) if env else (Path.home() / ".tarkash")
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error("Failed to create Tarkash home at '{}': {}", str(base), e)
        base = Path.cwd() / ".tarkash"
        base.mkdir(parents=True, exist_ok=True)
    return base

APP_DIR: Path = _resolve_home()
PROJECTS_DIR: Path = APP_DIR / "projects"
CHAT_HISTORY_DIR: Path = APP_DIR / "chat_history"
for _p in (PROJECTS_DIR, CHAT_HISTORY_DIR):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create '{}': {}", str(_p), e)

# Single-file, app-scoped artifacts
ENV_PATH: Path = APP_DIR / ".env"
MODELS_JSON_PATH: Path = APP_DIR / "models.json"
GLOBAL_RAG_PATH: Path = APP_DIR / "rag.json"                  # TEMPLATE ONLY (seed new projects; not read at runtime)
GLOBAL_UI_SETTINGS_PATH: Path = APP_DIR / "ui_settings.json"  # remembers last project, etc.

# ---------- helpers ----------

def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(p: Path, d: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save json '{}': {}", str(p), e)

def _ensure_global_templates() -> None:
    """
    Make sure the app-level template rag.json exists.
    This file is used only to seed new projects (copied into ~/.tarkash/projects/<name>/rag.json).
    Runtime always reads the per-project rag.json.
    """
    try:
        if not GLOBAL_RAG_PATH.exists():
            # Keep this minimal; per-project rag.json will inject a project-local chroma_dir.
            _write_json(GLOBAL_RAG_PATH, {
                "index_root": ".",
                "collection": "tarkash"
            })
            logger.info("Initialized template rag.json at {}", str(GLOBAL_RAG_PATH))
    except Exception as e:
        logger.warning("Could not initialize template rag.json: {}", e)

_name_safe_rx = re.compile(r"[^A-Za-z0-9_.-]+")

def _safe_name(name: str) -> str:
    name = (name or "").strip() or "default"
    name = _name_safe_rx.sub("_", name)
    return name[:80]

def current_project_name(project_root: Optional[str] = None) -> str:
    """Resolve the active project name from GLOBAL_UI_SETTINGS_PATH; fallback to folder name of project_root."""
    ui = _read_json(GLOBAL_UI_SETTINGS_PATH)
    name = ui.get("project_name")
    if name:
        return _safe_name(name)
    if project_root:
        try:
            return _safe_name(Path(project_root).resolve().name)
        except Exception:
            pass
    return "default"

# ---------- Per-project paths ----------

def project_dir(project_name: str) -> Path:
    p = PROJECTS_DIR / _safe_name(project_name)
    p.mkdir(parents=True, exist_ok=True)
    return p

def project_paths(project_name: str) -> Dict[str, Path]:
    base = project_dir(project_name)
    p = {
        "project_dir": base,
        "ui_settings": base / "ui_settings.json",
        "rag": base / "rag.json",
        "rag_index_dir": base / "RAG",
        "chat_dir": CHAT_HISTORY_DIR / _safe_name(project_name),
        "chat_db": CHAT_HISTORY_DIR / _safe_name(project_name) / "chat.db",
    }
    p["chat_dir"].mkdir(parents=True, exist_ok=True)
    p["rag_index_dir"].mkdir(parents=True, exist_ok=True)
    return p

def project_rag_json(project_root: Optional[str] = None, project_name: Optional[str] = None) -> Path:
    """
    Resolve the authoritative per-project rag.json path.
    Prefer explicit project_name; else derive from project_root or UI state.
    """
    name = project_name or current_project_name(project_root)
    return project_paths(name)["rag"]

def ensure_project_scaffold(
    project_name: str,
    *,
    template_rag: Optional[Path] = None,
    template_ui: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Ensure ~/.tarkash/projects/<name> exists with rag.json and ui_settings.json.
    Copies from provided templates if they exist; else copies from GLOBAL_* if present;
    else writes minimal placeholders.
    """
    paths = project_paths(project_name)

    # rag.json (copy from template once; thereafter per-project rag.json is the source of truth)
    if not paths["rag"].exists():
        src = template_rag if (template_rag and template_rag.exists()) else GLOBAL_RAG_PATH
        if src.exists():
            try:
                shutil.copyfile(src, paths["rag"])
                logger.info("Copied rag.json → {}", str(paths["rag"]))
            except Exception as e:
                logger.warning("Failed to copy rag.json from '{}': {}", str(src), e)
        if not paths["rag"].exists():
            try:
                minimal = {
                    "index_root": ".",
                    "chroma_dir": str(paths["rag_index_dir"]),
                    "collection": "code_chunks",
                }
                paths["rag"].write_text(json.dumps(minimal, indent=2), encoding="utf-8")
                logger.info("Wrote minimal rag.json at {}", str(paths["rag"]))
            except Exception as e:
                logger.error("Failed to write rag.json at '{}': {}", str(paths["rag"]), e)

    # ui_settings.json
    if not paths["ui_settings"].exists():
        src_ui = template_ui if (template_ui and template_ui.exists()) else GLOBAL_UI_SETTINGS_PATH
        if src_ui.exists():
            try:
                shutil.copyfile(src_ui, paths["ui_settings"])
                logger.info("Copied ui_settings.json → {}", str(paths["ui_settings"]))
            except Exception as e:
                logger.warning("Failed to copy ui_settings.json from '{}': {}", str(src_ui), e)
        if not paths["ui_settings"].exists():
            try:
                paths["ui_settings"].write_text(json.dumps({"project_name": project_name}, indent=2), encoding="utf-8")
            except Exception as e:
                logger.error("Failed to write ui_settings.json at '{}': {}", str(paths["ui_settings"]), e)

    return paths

def set_project_embedder(project_root: str, embedder_name: str) -> None:
    """
    Persist the embedder used for (re)indexing into the per-project rag.json.
    Called by indexer after it resolves the actual embedder.
    """
    try:
        p = project_rag_json(project_root)
        cfg = _read_json(p)
        cfg.setdefault("embedder", {})
        cfg["embedder"]["selected_name"] = embedder_name
        _write_json(p, cfg)
        logger.info("Updated project embedder in {}", str(p))
    except Exception as e:
        logger.warning("Failed to update project embedder: {}", e)

# ---------- Compatibility shim for older imports ----------

def project_rag_dir(project_root: str) -> str:
    """
    Backward-compatible resolver used by retriever/blueprint.
    We map the *active* project to ~/.tarkash/projects/<project_name>/RAG
    (project_name is taken from GLOBAL_UI_SETTINGS_PATH or derived from project_root).
    """
    name = current_project_name(project_root)
    paths = ensure_project_scaffold(name)
    return str(paths["rag_index_dir"])

_ensure_global_templates()

__all__ = [
    "APP_DIR",
    "ENV_PATH",
    "MODELS_JSON_PATH",
    "GLOBAL_RAG_PATH",
    "GLOBAL_UI_SETTINGS_PATH",
    "PROJECTS_DIR",
    "CHAT_HISTORY_DIR",
    "project_dir",
    "project_paths",
    "ensure_project_scaffold",
    "current_project_name",
    "project_rag_dir",
]
