#!/usr/bin/env python3
# streamlit_app.py — Tarkash pathing + per-project config (no functionality lost)

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO
import subprocess
import shutil
import tempfile
import html
from functools import lru_cache

import sqlite3
import base64
import pandas as pd
import requests
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from loguru import logger

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

# Playwright (optional)
try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:
    sync_playwright = None

# --- Logging (existing helper) ---
import logging_setup

logging_setup.configure_logging()

# --- Optional visuals ---
import graphviz  # kept for DOT graphs

# --- Config & indexing helpers ---
try:
    from config_home import (
        APP_DIR,
        ENV_PATH,
        MODELS_JSON_PATH,
        GLOBAL_UI_SETTINGS_PATH,
        GLOBAL_RAG_PATH,
        ensure_project_scaffold,
        project_paths,
    )
    from indexing.settings import load as load_rag, save as save_rag
    from indexing.indexer import full_reindex, delta_index, request_stop, index_status
except Exception as e:
    # Fallback home if imports fail (keeps app usable)
    HOME = Path(os.path.expanduser("~")) / ".tarkash"
    HOME.mkdir(parents=True, exist_ok=True)
    APP_DIR = HOME
    ENV_PATH = HOME / ".env"
    MODELS_JSON_PATH = HOME / "models.json"
    GLOBAL_UI_SETTINGS_PATH = HOME / "ui_settings.json"
    GLOBAL_RAG_PATH = HOME / "rag.json"
    logger.warning("Config/import fallback enabled: {}", e)

# --- Core app modules (unchanged) ---
from models import ModelRegistry
from tools.registry import ToolRegistry
from agent import Agent
import intents

# --- Enhanced planner / tools (optional) ---
try:
    from reasoning_agent import ReasoningAgent
    RAG_AGENT_AVAILABLE = True
    RAG_AGENT_ERR = ""
except Exception as e:
    RAG_AGENT_AVAILABLE = False
    RAG_AGENT_ERR = str(e)
    ReasoningAgent = None  # type: ignore
    logger.info("ReasoningAgent unavailable: {}", e)

try:
    from tools.enhanced_registry import EnhancedToolRegistry
    ENHANCED_TOOLS_AVAILABLE = True
except Exception as e:
    ENHANCED_TOOLS_AVAILABLE = False
    EnhancedToolRegistry = None  # type: ignore
    logger.info("EnhancedToolRegistry unavailable: {}", e)

# --- App constants ---
APP_TITLE = "Tarkash — Code Assistant"
EDIT_TOOLS = {
    "replace_in_file",
    "bulk_edit",
    "format_python_files",
    "rewrite_naive_open",
    "write_file",
}

# --- Load .env from Tarkash home ---
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment loaded from {}", str(ENV_PATH.resolve()))

# -------------------------- Markdown / LaTeX helpers --------------------------

_SVG_FENCE_RE = re.compile(r"```(?:svg|SVG)\s+([\s\S]*?)\s*```", re.MULTILINE)
_SVG_TAG_RE   = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)
_DATA_URI_RE  = re.compile(r"data:image/(png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/=]+)")
_IMG_DATA_URI_TAG_RE = re.compile(
    r"<img[^>]+src=[\"'](data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+)[\"'][^>]*>",
    re.IGNORECASE
)

def _render_images_and_svg_from_text(text: str) -> str:
    """
    Detect and render inline images (SVG blocks and base64 data URIs) and return
    the remaining markdown text (with those image blobs removed/replaced).
    This lets models like Gemini 2.5 Flash show visuals directly in Streamlit.
    """
    if not text or not isinstance(text, str):
        return text or ""

    clean = text

    # 1) Fenced ```svg ... ``` blocks
    for m in list(_SVG_FENCE_RE.finditer(clean)):
        svg = m.group(1).strip()
        if svg:
            # Render the SVG inline
            st.markdown(svg, unsafe_allow_html=True)
        # Remove block from text
        clean = clean.replace(m.group(0), "")

    # 2) Raw <svg>...</svg> tags
    for m in list(_SVG_TAG_RE.finditer(clean)):
        svg = m.group(0)
        st.markdown(svg, unsafe_allow_html=True)
        clean = clean.replace(svg, "")

    # 3) <img src="data:image/..."> tags
    for m in list(_IMG_DATA_URI_TAG_RE.finditer(clean)):
        uri = m.group(1)
        m2 = _DATA_URI_RE.search(uri)
        if m2:
            try:
                ext = m2.group(1).lower()
                b   = base64.b64decode(m2.group(2))
                st.image(BytesIO(b), use_column_width=True)
            except Exception:
                pass
        clean = clean.replace(m.group(0), "")

    # 4) Bare data:image/... URIs inside text
    for m in list(_DATA_URI_RE.finditer(clean)):
        try:
            b = base64.b64decode(m.group(2))
            st.image(BytesIO(b), use_column_width=True)
        except Exception:
            pass
        # Replace the data URI to avoid duplication in the markdown below
        clean = clean.replace(m.group(0), "")

    return clean.strip()

# -------------------------- Markdown / LaTeX helpers --------------------------

# Detect math delimiters to auto-enable KaTeX in HTML->PDF
_MATH_DELIMS_RE = re.compile(r"(\\\(|\\\)|\\\[|\\\]|\$\$[^$]+\$\$|\$[^$]+\$)")

def _md_has_math(s: str) -> bool:
    if not s:
        return False
    return bool(_MATH_DELIMS_RE.search(s))

# ========= Feature flags =========
# We no longer try headless browser PDF. We keep Markdown export always-on,
# and add a Pandoc (+xelatex or tectonic) PDF path that runs only if found.
PDF_EXPORT_ENABLED = True

# ---------- Lightweight environment checks (for status pills) ----------
def _have_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False

def _have_weasyprint() -> bool: return _have_import("weasyprint")
def _have_xhtml2pdf() -> bool:  return _have_import("xhtml2pdf")
def _have_pymupdf() -> bool:    return _have_import("fitz")
def _have_wkhtmltopdf() -> bool: return bool(shutil.which("wkhtmltopdf"))
def _log_env_status():
    try:
        have_pandoc = _pandoc_available()
        engine = _pandoc_engine() or "-"
        weasy = _have_weasyprint()
        x2pdf = _have_xhtml2pdf()
        pymu = _have_pymupdf()
        wk = _have_wkhtmltopdf()
        logger.info(
            "Env: pandoc=%s engine=%s weasyprint=%s xhtml2pdf=%s pymupdf=%s wkhtmltopdf=%s",
            have_pandoc, engine, weasy, x2pdf, pymu, wk
        )
    except Exception as e:
        logger.info("Env status check failed: %s", e)



# -------------------------------------------------------------------
# Streamlit rerun compatibility (st.rerun in newer versions)
# -------------------------------------------------------------------
def _rerun() -> None:
    """Call st.rerun() if available, else fall back to st.experimental_rerun()."""
    try:
        # New API (Streamlit >= 1.25)
        st.rerun()
    except Exception:
        # Older API
        try: st.experimental_rerun()
        except Exception: pass

# =============================================================================
# Utilities (formatting, LaTeX, charts, UI state)
# =============================================================================

def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def _sanitize_latex_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    SUP = {"⁰":"^{0}","¹":"^{1}","²":"^{2}","³":"^{3}","⁴":"^{4}",
           "⁵":"^{5}","⁶":"^{6}","⁷":"^{7}","⁸":"^{8}","⁹":"^{9}",
           "⁺":"^{+}","⁻":"^{-}","⁽":"^{(}","⁾":"^{)}","ⁿ":"^{n}"}
    SUB = {"₀":"_{0}","₁":"_{1}","₂":"_{2}","₃":"_{3}","₄":"_{4}",
           "₅":"_{5}","₆":"_{6}","₇":"_{7}","₈":"_{8}","₉":"_{9}",
           "₊":"_{+}","₋":"_{-}","₍":"_{(}","₎":"_{)}"}
    return "".join(SUP.get(ch, SUB.get(ch, ch)) for ch in text)

def _fix_common_latex_typos(text: str) -> str:
    if not text:
        return text
    s = text.replace("−", "-").replace("–", "-")
    s = s.replace(r"\left$", r"\left(").replace(r"\right$", r"\right)")
    s = re.sub(r"\\left\$(\\frac|\\sqrt)", r"\\left(\1", s)
    s = re.sub(r"\^\s+", "^", s)
    s = re.sub(r"_\s+", "_", s)
    s = re.sub(r"(\\frac\{[^}]+\}\{[^}]+\})\s*\^\s*(\d+)", r"\1^{\2}", s)
    return s

def _normalize_math_delimiters(text: str) -> str:
    if not text:
        return text
    s = text
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$",   s, flags=re.DOTALL)

    def _br(m):
        inner = m.group(1)
        return "$$" + inner.strip() + "$$" if re.match(r"^\s*\\[A-Za-z]", inner) else "[" + inner + "]"
    s = re.sub(r"\[(.*?)\]", _br, s, flags=re.DOTALL)

    def _pr(m):
        inner = m.group(1)
        return "$" + inner.strip() + "$" if len(inner) <= 180 and re.match(r"^\s*\\[A-Za-z]", inner) else "(" + inner + ")"
    s = re.sub(r"\((.*?)\)", _pr, s, flags=re.DOTALL)

    return s

def render_response_with_latex(text: str):
    if not isinstance(text, str) or not text.strip():
        return
    s = _sanitize_latex_text(text)
    s = _fix_common_latex_typos(s)
    s = _normalize_math_delimiters(s)
    st.markdown(s)
    for m in re.finditer(r"\$\$([\s\S]*?)\$\$", s, flags=re.DOTALL):
        expr = (m.group(1) or "").strip()
        if ("\n" in expr) or ("\\\\" in expr) or ("\\begin{" in expr):
            st.code(expr, language="latex")

def _human_dt(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        return str(ts)

# --- UI settings (per-project) ---
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

# =============================================================================
# Chat history — SQLite (per project under ~/.tarkash/<project>/chat_history/chat.db)
# =============================================================================
def _safe_name(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "default"))[:80]

def _chat_db_path(project_name: str) -> Path:
    """
    New layout per request:
      ~/.tarkash/projects/<project_name>/chat_history/chat.db
    """
    base = Path.home() / ".tarkash" / "projects" / _safe_name(project_name) / "chat_history"
    base.mkdir(parents=True, exist_ok=True)
    return base / "chat.db"

def _chat_db_init(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,                 -- unix epoch
                ts_iso TEXT,                      -- readable timestamp
                project TEXT NOT NULL,            -- project name
                project_path TEXT,                -- project root path
                mode TEXT NOT NULL,               -- Direct Chat / LLM Tools / Agent Plan & Run
                streaming INTEGER DEFAULT 0,      -- 0/1
                rag_on INTEGER DEFAULT 0,         -- 0/1
                model TEXT,                       -- LLM name
                embedder TEXT,                    -- Embedding model
                prompt TEXT NOT NULL,             -- user input
                answer TEXT NOT NULL              -- final response
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_project_ts ON chats(project, ts DESC)")
        # Add missing columns for older DBs (safe to re-run)
        cols = {r[1] for r in cur.execute("PRAGMA table_info(chats)").fetchall()}
        wanted = {
            "ts_iso":"TEXT","project_path":"TEXT","streaming":"INTEGER","rag_on":"INTEGER",
            "model":"TEXT","embedder":"TEXT"
        }
        for k, typ in wanted.items():
            if k not in cols:
                try:
                    cur.execute(f"ALTER TABLE chats ADD COLUMN {k} {typ}")
                except Exception:
                    pass
        con.commit()

def save_chat(
    *,
    project_name: str,
    project_path: str,
    mode: str,
    prompt: str,
    answer: str,
    streaming: bool,
    rag_on: bool,
    model: Optional[str],
    embedder: Optional[str],
) -> None:
    if not (project_name and isinstance(answer, str) and answer.strip()):
        return
    db_path = _chat_db_path(project_name)
    _chat_db_init(db_path)
    with sqlite3.connect(db_path) as con:
        now = time.time()
        iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        con.execute(
            """
            INSERT INTO chats(ts, ts_iso, project, project_path, mode, streaming, rag_on, model, embedder, prompt, answer)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now, iso, project_name, project_path, mode,
                1 if streaming else 0, 1 if rag_on else 0,
                model or "", embedder or "", prompt, answer
            ),
        )
        con.commit()

def list_chats(
    project_name: str,
    *,
    query: str = "",
    regex: bool = False,
    filter_mode: str = "top_n",   # all | top_n | last_n | past_7d | past_30d | date_range
    n: int = 20,
    date_from: Optional[str] = None,   # "YYYY-MM-DD"
    date_to: Optional[str] = None,     # "YYYY-MM-DD"
) -> List[Dict[str, Any]]:
    db_path = _chat_db_path(project_name)
    if not db_path.exists():
        return []
    import datetime as _dt
    now = _dt.datetime.now().timestamp()
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    # resolve filter preset
    fm = (filter_mode or "top_n").lower()
    if fm == "past_7d":
        start_ts = now - 7*86400
    elif fm == "past_30d":
        start_ts = now - 30*86400
    elif fm == "date_range":
        start_ts = _epoch_from_date_str(date_from)
        # end of day for 'to'
        et = _epoch_from_date_str(date_to)
        if et is not None:
            end_ts = et + 86399.0
    # else: for all/top_n/last_n we leave start/end None

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        # optional REGEXP for sqlite
        if regex:
            def _re(x, y):
                try:
                    return 1 if re.search(y, x or "", re.IGNORECASE) else 0
                except Exception:
                    return 0
            con.create_function("REGEXP", 2, _re)

        where = ["project=?"]
        params: List[Any] = [project_name]

        # time window
        if start_ts is not None:
            where.append("ts >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            where.append("ts <= ?")
            params.append(float(end_ts))

        # search
        if query:
            q = query.strip()
            if regex:
                where.append("(prompt REGEXP ? OR answer REGEXP ? OR mode REGEXP ? OR model REGEXP ? OR embedder REGEXP ?)")
                params.extend([q, q, q, q, q])
            else:
                like = f"%{q}%"
                where.append("(prompt LIKE ? OR answer LIKE ? OR mode LIKE ? OR model LIKE ? OR embedder LIKE ?)")
                params.extend([like, like, like, like, like])

        # sort + limit
        order_sql = "ORDER BY ts DESC"
        limit_sql = ""
        if fm == "last_n":
            order_sql = "ORDER BY ts ASC"
            limit_sql = "LIMIT ?"
            params.append(int(max(1, n)))
        elif fm == "top_n":
            order_sql = "ORDER BY ts DESC"
            limit_sql = "LIMIT ?"
            params.append(int(max(1, n)))
        elif fm in ("past_7d", "past_30d", "date_range", "all"):
            order_sql = "ORDER BY ts DESC"
            # no limit for these (user can still search to constrain)

        where_sql = " AND ".join(where)
        sql = f"""
            SELECT id, ts, ts_iso, mode, prompt, answer, model, embedder, rag_on, streaming, project_path
            FROM chats
            WHERE {where_sql}
            {order_sql}
            {limit_sql}
        """
        rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

def delete_chat(project_name: str, chat_id: int) -> None:
    db_path = _chat_db_path(project_name)
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as con:
        con.execute("DELETE FROM chats WHERE project=? AND id=?", (project_name, int(chat_id)))
        con.commit()

def _copy_button_html(text: str, key: str) -> str:
    """
    Small JS copy-to-clipboard button. Text is base64-encoded to avoid escaping issues.
    """
    b64 = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
    return f"""
    <button class="copy-btn" onclick="navigator.clipboard.writeText(atob('{b64}'))" title="Copy to clipboard" id="{key}">
      Copy
    </button>
    """

def _copy_button(text: str, key: str) -> None:
    """
    More reliable copy using a tiny components.html widget.
    Works well under Streamlit's CSP and sandboxing.
    """
    b64 = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <div style="width:100%;">
          <button style="width:100%;height:36px;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer"
            onclick="navigator.clipboard.writeText(atob('{b64}'))">
            Copy
          </button>
        </div>
        """,
        height=42,
    )

def _shorten_one_line(text: str, max_chars: int = 90) -> str:
    """Return a single-line, ellipsized preview."""
    if not text:
        return ""
    s = " ".join(str(text).split())  # collapse whitespace/newlines
    return s if len(s) <= max_chars else (s[: max_chars - 1] + "…")

# ---------- Time helpers ----------
def _epoch_from_date_str(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        import datetime as _dt
        d = _dt.datetime.strptime(s, "%Y-%m-%d")
        # interpret as start of day local time
        return d.timestamp()
    except Exception:
        return None

# ---------- Conversation context helpers ----------
def _build_context_messages(project_name: str, ui: Dict[str, Any], prompt: str) -> List[Dict[str, str]]:
    """
    Build LLM messages with optional conversational context.
    Priority:
      1) If per-project history is enabled, pull last N turns from SQLite.
      2) Else, use a transient in-memory buffer in st.session_state (this session only).
    """
    msgs: List[Dict[str, str]] = []
    ctx_on = bool(ui.get("ctx_on", False))      # default OFF unless user enables
    ctx_n = int(ui.get("ctx_turns", 4))
    ctx_n = max(0, min(ctx_n, 10))              # clamp 0..10
    if not ctx_on or ctx_n == 0:
        msgs.append({"role": "user", "content": prompt})
        return msgs

    hist_enabled = bool(ui.get("history_enabled", False))
    rows: List[Dict[str, Any]] = []
    if hist_enabled:
        # Use latest N project rows from DB
        try:
            rows = list_chats(project_name, filter_mode="top_n", n=ctx_n)
        except Exception:
            rows = []
    else:
        # Fall back to transient session buffer
        buf = st.session_state.get("transient_turns", [])
        # keep the last N
        rows = buf[-ctx_n:] if buf else []

    # Oldest -> newest sequence for correct dialogue order
    for r in list(reversed(rows)):
        up = r.get("prompt") or r.get("user") or ""
        an = r.get("answer") or r.get("assistant") or ""
        if up:
            msgs.append({"role": "user", "content": up})
        if an:
            msgs.append({"role": "assistant", "content": an})
    msgs.append({"role": "user", "content": prompt})
    return msgs


# ---------- History table helpers ----------
# ---------- History table helpers ----------
def _history_dataframe(
    project_name: str,
    *,
    flt: dict,
    show_cols: list[str] | None = None,   # any of: ["mode","model","embedder"]
) -> pd.DataFrame:
    rows = list_chats(project_name, **flt)
    show_cols = (show_cols or [])
    # compact snippets for the grid
    recs = []
    for r in rows:
        row = {
            "id": int(r["id"]),
            # slightly shorter previews so extra cols fit comfortably
            "prompt": _shorten_one_line(r.get("prompt",""), 90),
            "answer": _shorten_one_line(r.get("answer",""), 90),
        }
        if "mode" in show_cols:     row["mode"] = r.get("mode","")
        if "model" in show_cols:    row["model"] = r.get("model") or ""
        if "embedder" in show_cols: row["embedder"] = r.get("embedder") or ""
        recs.append(row)
    # order: id, mode, model, embedder, prompt, answer
    base = ["id"]
    for k in ["mode","model","embedder"]:
        if k in show_cols: base.append(k)
    base += ["prompt","answer"]
    df = pd.DataFrame(recs, columns=base)
    # attach a 'view' flag bound to session selection
    sel = set(st.session_state.get("hist_view_ids", set()))
    df["view"] = df["id"].apply(lambda i: (i in sel))
    return df

def _update_history_selection_from_editor(df_ret: pd.DataFrame):
    ids = set(df_ret.loc[df_ret["view"] == True, "id"].tolist())
    st.session_state["hist_view_ids"] = ids

def _apply_pending_view_edits_from_editor(df: pd.DataFrame, key: str = "history_table") -> pd.DataFrame:
    """
    If the user just clicked a checkbox, Streamlit stores that edit under
    st.session_state[key].edited_rows before our next st.data_editor runs.
    We merge those pending edits into the 'view' column so the UI never
    appears to "undo" the click on the immediate rerun.
    """
    try:
        state = st.session_state.get(key)
        # Try both dict-like and attribute access forms.
        edits = None
        if isinstance(state, dict):
            edits = state.get("edited_rows")
        else:
            edits = getattr(state, "edited_rows", None)
        if not edits:
            return df
        # edited_rows is typically a dict: { row_index: {"view": <bool>, ...}, ... }
        for idx_raw, changes in dict(edits).items():
            try:
                idx = int(idx_raw)
            except Exception:
                continue
            if isinstance(changes, dict) and "view" in changes and 0 <= idx < len(df):
                df.at[idx, "view"] = bool(changes["view"])
        return df
    except Exception:
        # Fail-quietly: if Streamlit changes the internal shape, we just skip.
        return df

# ---------- Small text cleanups for PDF ----------
def _fix_common_encoding_glitches(s: str) -> str:
    if not s: return s
    # Common UTF-8 seen as Latin-1 glitch for degree symbol
    return s.replace("Â°", "°")


# ======================= RICH PDF EXPORT (Markdown → HTML → PDF) =======================
def _clean_markdown_for_pdf(text: str) -> str:
    """Apply LaTeX/markdown sanitizers and encoding fixes before HTML conversion."""
    s = text or ""
    try:
        s = _normalize_math_delimiters(s)
    except Exception:
        pass
    try:
        s = _fix_common_latex_typos(s)
    except Exception:
        pass
    try:
        s = _sanitize_latex_text(s)
    except Exception:
        pass
    s = _fix_common_encoding_glitches(s)
    return s

# @lru_cache(maxsize=1)
# def _pygments_formatter():
#     try:
#         from pygments.formatters import HtmlFormatter
#         return HtmlFormatter(nowrap=False)
#     except Exception:
#         return None

# def _pygments_css() -> str:
#     fmt = _pygments_formatter()
#     if not fmt:
#         return ""
#     try:
#         return fmt.get_style_defs(".codehilite")
#     except Exception:
#         return ""

@lru_cache(maxsize=2)
def _md_renderer(use_mathml: bool):
    """
    Build a markdown-it renderer with useful plugins and a pygments highlighter.
    When use_mathml=True we convert math to MathML (for Weasy/xhtml2pdf).
    When use_mathml=False we LEAVE $…$ delimiters intact (for KaTeX+wkhtml).
    """
    from markdown_it import MarkdownIt
    from mdit_py_plugins.table import table_plugin
    from mdit_py_plugins.deflist import deflist_plugin
    from mdit_py_plugins.tasklists import tasklists_plugin
    from mdit_py_plugins.anchors import anchors_plugin
    from mdit_py_plugins.attrs import attrs_plugin
    if use_mathml:
        from mdit_py_plugins.texmath import texmath_plugin
    md = MarkdownIt("commonmark", {"typographer": True})
    # Plugins
    md.use(table_plugin)
    md.use(deflist_plugin)
    md.use(tasklists_plugin, enabled=True)
    md.use(anchors_plugin, max_level=3)
    md.use(attrs_plugin)
    # Math handling
    if use_mathml:
        md.use(texmath_plugin, renderer="mathml")  # Weasy/xhtml2pdf route
    # Pygments highlighting (works for fenced code)
    fmt = _pygments_formatter()
    if fmt:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, TextLexer
        def _hl(code: str, lang: str, attrs: str) -> str:
            try:
                lexer = get_lexer_by_name(lang or "", stripall=True)
            except Exception:
                lexer = TextLexer(stripall=True)
            return f'<pre class="codehilite">{highlight(code, lexer, fmt)}</pre>'
        md.options["highlight"] = _hl
    return md


def _md_to_html(md_text: str, *, use_mathml: bool) -> str:
    """Convert Markdown to HTML with plugins + Pygments.
       use_mathml=True: convert math to MathML (Weasy/xhtml2pdf)
       use_mathml=False: keep $…$ so KaTeX can render (wkhtml)"""
    src = _clean_markdown_for_pdf(md_text or "")
    try:
        md = _md_renderer(use_mathml)
        return md.render(src)
    except Exception:
        return f"<pre>{html.escape(src)}</pre>"
    
@lru_cache(maxsize=1)
def _pygments_formatter():
    try:
        from pygments.formatters import HtmlFormatter
        return HtmlFormatter(style="default", nowrap=False)
    except Exception:
        return None

def _pygments_css() -> str:
    """Inline Pygments CSS for codehilite blocks (no external files, no JS)."""
    try:
        from pygments.formatters import HtmlFormatter
        return HtmlFormatter(style="default").get_style_defs(".codehilite")
    except Exception:
        return ""

def _build_export_html(project_name: str, rows: List[Dict[str, Any]], *, include_katex: bool = False) -> str:
    """Compose a complete HTML document for selected chats (print-friendly).
       - Promotes fenced ```svg blocks to raw <svg> so they render as graphics
         (wkhtmltopdf would otherwise print them as code).
       - Adds print-grade CSS (tables, code, images/SVG).
       - Optionally injects KaTeX and signals readiness via window.status."""
    pyg_css = _pygments_css()
    # Print-grade CSS; WeasyPrint supports @page, wkhtml respects much of base CSS
    css = f"""
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
      @bottom-right {{
        content: "Page " counter(page);
        color: #777;
        font-size: 9pt;
      }}
    }}
    html, body {{ height: 100%; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue',
      Arial, 'Noto Sans', 'Liberation Sans', sans-serif; color: #111;
      -webkit-print-color-adjust: exact; print-color-adjust: exact;
    }}
    h1, h2, h3 {{ margin: 0 0 6px 0; }}
    h1 {{ font-size: 20px; }}
    h2 {{ font-size: 16px; margin-top: 20px; }}
    h3 {{ font-size: 14px; }}
    .meta {{ color:#666; font-size: 11px; margin: 2px 0 10px; }}
    .chat {{ page-break-before: always; }}
    .chat:first-child {{ page-break-before: auto; }}
    .label {{ font-weight: 700; margin: 10px 0 4px; }}
    p {{ line-height: 1.38; margin: 6px 0; white-space: normal; }}
    li {{ white-space: normal; }}
    ul, ol {{ margin: 6px 0 6px 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 6px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
    th {{ background: #f7f7f7; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; }}
    pre, pre code {{ white-space: pre-wrap; word-wrap: break-word; }}
    pre {{ border: 1px solid #eee; background: #fafafa; padding: 10px; border-radius: 6px; font-size: 12px; }}
    hr {{ border:0; border-top:1px solid #eee; margin: 14px 0; }}
    .title {{ font-size: 22px; font-weight: 800; margin: 0 0 6px 0; }}
    .subtitle {{ color:#666; font-size: 12px; margin-bottom: 12px; }}
    img, svg {{ max-width: 100%; height: auto; display:block; }}
    /* common inline markup */
    mark {{ background: #fffd8a; padding: 0 .15em; }}
    sub {{ vertical-align: sub; font-size: 75%; }}
    sup {{ vertical-align: super; font-size: 75%; }}
    /* Pygments for codehilite */
    {pyg_css}
    /* MathML tweaks (WeasyPrint) */
    math, mrow, mi, mo, mn, mfrac, msup, msub, msubsup, mtable, mtr, mtd {{
      font-family: STIXGeneral, 'DejaVu Serif', serif;
    }}
    """
    # Optional KaTeX (for wkhtmltopdf path with client-side math rendering).
    # We set window.status='katex-done' so wkhtml can wait until math is rendered.
    katex_head = ""
    if include_katex:
        # Uses CDN; if your environment blocks external fetches, prefer Pandoc/WeasyPrint.
        katex_head = """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
        <script>window.status='katex-loading';</script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
                onload="try{
                  renderMathInElement(document.body,{
                    delimiters:[
                      {left:'$$',right:'$$',display:true},
                      {left:'$', right:'$', display:false},
                      {left:'\\(', right:'\\)', display:false},
                      {left:'\\[', right:'\\]', display:true}
                    ]
                  });
                }catch(e){}finally{
                  window.status='katex-done';
                  setTimeout(function(){ if(window.status!=='katex-done'){window.status='katex-done';}},300);
                }">
        </script>
        """
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<style>{css}</style>",
        katex_head,
        "</head><body>",
        f"<div class='title'>Tarkash — Chat Export</div>",
        f"<div class='subtitle'>{html.escape(project_name)} · generated at {now_iso}</div>",
    ]
    # Helper to promote fenced ```svg ...``` into placeholders, restored post-render
    _SVG_FENCE = re.compile(r"```(?:svg|SVG)\s+([\s\S]*?)\s*```")
    def _promote_svg_fences(text: str):
        svgs = []
        def _repl(m):
            key = f"__SVG_{len(svgs)}__"
            svgs.append(m.group(1).strip())
            return key
        return _SVG_FENCE.sub(_repl, text), svgs

    for r in rows:
        cid = r.get("id")
        ts = r.get("ts"); ts_iso = r.get("ts_iso") or _human_dt(ts)
        mode = r.get("mode", "")
        model = r.get("model") or ""
        embedder = r.get("embedder") or ""
        rag_on = "on" if r.get("rag_on") else "off"
        streaming = "on" if r.get("streaming") else "off"
        prompt_md = (r.get("prompt") or "").rstrip()
        answer_md = (r.get("answer") or "").rstrip()
        # Promote fenced SVG to placeholders before Markdown conversion
        prompt_promoted, p_svgs = _promote_svg_fences(prompt_md)
        answer_promoted, a_svgs = _promote_svg_fences(answer_md)
        # Convert markdown → HTML
        # For wkhtml: include_katex=True ⇒ keep $…$ (no MathML). For server-side PDF: use MathML.
        _use_mathml = not include_katex
        prompt_html = _md_to_html(prompt_promoted, use_mathml=_use_mathml)
        answer_html = _md_to_html(answer_promoted, use_mathml=_use_mathml)
        # Restore SVG placeholders to raw <svg> so they render as graphics
        for i, s in enumerate(p_svgs):
            prompt_html = prompt_html.replace(f"__SVG_{i}__", s)
        for i, s in enumerate(a_svgs):
            answer_html = answer_html.replace(f"__SVG_{i}__", s)
        parts.append("<div class='chat'>")
        parts.append(f"<h2>Chat #{cid} — {html.escape(ts_iso)}</h2>")
        parts.append(f"<div class='meta'>{html.escape(mode)} · model={html.escape(model)} · emb={html.escape(embedder)} · RAG={rag_on} · stream={streaming}</div>")
        parts.append("<div class='label'>You:</div>")
        parts.append(prompt_html)
        parts.append("<div class='label'>Assistant:</div>")
        parts.append(answer_html)
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)

def _render_pdf_from_html(html_str: str) -> Optional[bytes]:
    """Render HTML → PDF with server-side engines only (WeasyPrint → xhtml2pdf → PyMuPDF).
       Respect feature flag to allow disabling PDF completely."""
    # Early exit when PDF export is disabled
    if not PDF_EXPORT_ENABLED:
        logger.info("PDF export disabled by flag; skipping render.")
        return None
    # 1) WeasyPrint (best fidelity)
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html_str).write_pdf()
        logger.info("PDF export via WeasyPrint (%d bytes)", len(pdf_bytes) if pdf_bytes else 0)
        return pdf_bytes
    except Exception as e:
        logger.info(f"WeasyPrint unavailable/failed: {e}")
    # 2) xhtml2pdf (pure Python)
    try:
        from xhtml2pdf import pisa
        buf = BytesIO()
        ok = pisa.CreatePDF(html_str, dest=buf, encoding="utf-8")
        if not ok.err:
            out = buf.getvalue()
            logger.info("PDF export via xhtml2pdf (%d bytes)", len(out))
            return out
    except Exception as e:
        logger.info(f"xhtml2pdf unavailable/failed: {e}")
    # 3) PyMuPDF (section-per-page HTML)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open()
        chats = re.findall(r"(<div class='chat'>.*?</div>)", html_str, flags=re.S)
        head = re.search(r"<head>(.*?)</head>", html_str, flags=re.S)
        head_html = head.group(1) if head else ""
        if not chats:
            chats = ["<div class='chat'><p>No content</p></div>"]
        for sec in chats:
            page = doc.new_page()
            w, h = page.rect.width, page.rect.height
            rect = fitz.Rect(45, 56, w - 45, h - 56)
            snippet = f"<!doctype html><html><head>{head_html}</head><body>{sec}</body></html>"
            try:
                page.insert_htmlbox(rect, snippet)  # PyMuPDF ≥ 1.22
            except Exception:
                page.insert_textbox(rect, "Rendering failed", fontsize=11)
        out = doc.tobytes()
        logger.info("PDF export via PyMuPDF (%d bytes)", len(out))
        return out
    except Exception as e:
        logger.warning(f"PyMuPDF HTML export failed: {e}")
        return None

def _export_chats_to_pdf_rich(project_name: str, rows: List[Dict[str, Any]]) -> Optional[bytes]:
    """End-to-end rich export using Markdown→HTML (+LaTeX cleanup) → PDF."""
    html_doc = _build_export_html(project_name, rows, include_katex=False)
    return _render_pdf_from_html(html_doc)

# --- Math preparation specifically for Pandoc→(xe)latex
def _prepare_math_for_pandoc(text: str) -> str:
    """
    Robust normalizer that prevents mixed/nested delimiters:
      1) Fix common typos/unicode.
      2) Strip any stray '$' **inside existing math** ($…$, $$…$$, \\[…\\], \\(…\\)).
      3) Stash code + existing math so we won't touch them further.
      4) Targeted inline wrap for safe short tokens (\\triangle, \\angle, degrees like 90^\\circ).
      5) Whole-line wrap only if the line looks like a standalone equation and has no delimiters.
    """
    if not text:
        return text
    s = text

    # (1) sanitize + common typos
    s = _sanitize_latex_text(s)
    s = _fix_common_latex_typos(s)
    s = s.replace(r"\left$", r"\left(").replace(r"\right$", r"\right)")

    # (2) strip stray dollars **inside** existing math spans
    def _strip_inside(pattern: str, opener: str, closer: str):
        nonlocal s
        def _repl(m: re.Match) -> str:
            inner = m.group(1).replace("$", "")
            return f"{opener}{inner}{closer}"
        s = re.sub(pattern, _repl, s, flags=re.S)
    _strip_inside(r"\$\$(.+?)\$\$", "$$", "$$")    # $$…$$
    _strip_inside(r"\$(.+?)\$", "$", "$")          # $…$
    _strip_inside(r"\\\[(.+?)\\\]", r"\[", r"\]")  # \[…]
    _strip_inside(r"\\\((.+?)\\\)", r"\(", r"\)")  # \(...)

    # (3) stash code + existing math (after cleanup)
    placeholders: dict[str, str] = {}
    def _stash(pattern: str):
        nonlocal s
        def _ph(m: re.Match) -> str:
            k = f"__PH_{len(placeholders)}__"
            placeholders[k] = m.group(0)
            return k
        s = re.sub(pattern, _ph, s, flags=re.S)
    _stash(r"```.*?```")          # fenced code
    _stash(r"`[^`]*`")            # inline code
    _stash(r"\$\$.*?\$\$")        # display math
    _stash(r"\$[^$]*\$")          # inline math
    _stash(r"\\\[[\s\S]*?\\\]")   # \[…]
    _stash(r"\\\([\s\S]*?\\\)")   # \(...)

    # (4) targeted inline wrap for *isolated* safe tokens
    def _wrap_token(pat: str):
        nonlocal s
        s = re.sub(pat, lambda m: f"${m.group(0)}$", s)
    # \triangle, \angle
    _wrap_token(r"(?<![\$\\])\\(?:triangle|angle)\b|(?<!\$)\\(?:triangle|angle)\b")
    # degrees like 90^\circ or 90^circ
    def _deg_repl(m: re.Match) -> str:
        num = m.group(1)
        return f"${num}^\\circ$"
    s = re.sub(r"(?<!\$)(\d+)\s*\^\s*(?:\\?circ)\b", _deg_repl, s)

    # (5) whole-line wrap only for strong equation-like lines with no delimiters/PH
    def _has_delim(t: str) -> bool:
        return bool(re.search(r"\$|\\\(|\\\)|\\\[|\\\]|__PH_", t))
    def _looks_equation(t: str) -> bool:
        if re.search(r"^\s*(\\begin\{|\\frac|\\sqrt|\\sum|\\int|\\prod|\\lim|\\displaystyle|\\[A-Za-z]+)\b", t):
            return True
        if re.search(r"=|\\frac|\\sqrt|\\binom", t) and not re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", t):
            return True
        # density heuristic
        mathish = len(re.findall(r"[\\^_{}+\-*/=<>]|[0-9]", t))
        return (mathish / max(len(t), 1)) >= 0.40
    lines = s.splitlines()
    for i, ln in enumerate(lines):
        t = ln.strip()
        if t and not _has_delim(t) and _looks_equation(t):
            lines[i] = f"${ln}$"
    s = "\n".join(lines)

    # restore stashed segments
    for k, v in placeholders.items():
        s = s.replace(k, v)
    return s

def _export_chats_to_markdown(project_name: str, rows: List[Dict[str, Any]], *, for_pdf: bool = False) -> bytes:
    """
    Build a single Markdown file containing the selected chats.
    Keeps prompts/answers as-is (so your external MD->PDF toolchain can render
    headings, code fences, and LaTeX).
    """
    lines: list[str] = []
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    title = f"# Tarkash — Chat Export"
    sub   = f"*{project_name} · generated at {now_iso}*"
    lines.extend([title, sub, ""])
    for r in rows:
        cid = r.get("id")
        ts  = r.get("ts")
        ts_iso = r.get("ts_iso") or _human_dt(ts)
        mode = r.get("mode", "")
        model = r.get("model") or ""
        embedder = r.get("embedder") or ""
        rag_on = "on" if r.get("rag_on") else "off"
        streaming = "on" if r.get("streaming") else "off"
        prompt = (r.get("prompt") or "").rstrip()
        answer = (r.get("answer") or "").rstrip()

        if for_pdf:
            prompt = _prepare_math_for_pandoc(prompt)
            answer = _prepare_math_for_pandoc(answer)

        lines.append(f"## Chat #{cid} — {ts_iso}")
        lines.append(f"`{mode}` · model=`{model}` · emb=`{embedder}` · RAG={rag_on} · stream={streaming}")
        lines.append("")
        lines.append("**You**")
        lines.append("")
        lines.append(prompt)
        lines.append("")
        lines.append("**Assistant**")
        lines.append("")
        lines.append(answer)
        lines.append("\n---\n")
    content = "\n".join(lines).strip() + "\n"
    return content.encode("utf-8")

#
# -------- Pandoc (+xelatex/tectonic) PDF export --------
#
def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None

def _pandoc_engine() -> Optional[str]:
    """Prefer xelatex; fall back to tectonic if installed."""
    if _which("xelatex"):
        return "xelatex"
    if _which("tectonic"):
        return "tectonic"
    return None

def _pandoc_available() -> bool:
    return bool(_which("pandoc") and _pandoc_engine())

def _export_chats_to_pdf_pandoc(project_name: str, rows: List[Dict[str, Any]]) -> Optional[bytes]:
    """
    Convert our Markdown export to PDF using pandoc + xelatex/tectonic.
    - Works great locally (you already tested `pandoc … --pdf-engine=xelatex`)
    - On Streamlit Cloud (no TeX/Pandoc), we simply won’t enable the PDF button.
    """
    if not _pandoc_available():
        return None
    try:
        md_bytes = _export_chats_to_markdown(project_name, rows, for_pdf=True)
        with tempfile.TemporaryDirectory() as td:
            md_path = Path(td) / "chats.md"
            pdf_path = Path(td) / "out.pdf"
            md_path.write_bytes(md_bytes)
            engine = _pandoc_engine() or "xelatex"
            args = [
                _which("pandoc"),
                str(md_path),
                "-o", str(pdf_path),
                "--standalone",
                "--from", "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex",
                "--pdf-engine", engine,
                # keep geometry but avoid hardcoding fonts that may not exist on Windows
                "-V", "geometry:margin=1in",
            ]
            proc = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                st.error("Pandoc PDF export failed.\n\n```\n" + (proc.stderr or b"").decode("utf-8","ignore")[-1200:] + "\n```")
                return None
            return pdf_path.read_bytes()
    except Exception as e:
        st.error(f"Pandoc PDF export failed.\n\n```\n{e}\n```")
        return None

def _export_chats_to_pdf_wkhtml(project_name: str, rows: List[Dict[str, Any]], *, inject_katex: bool) -> Optional[bytes]:
    """Experimental: HTML → PDF using wkhtmltopdf. Good when pandoc/texlive are unavailable.
       For math, KaTeX injection is optional and requires network access."""
    if not _have_wkhtmltopdf():
        return None
    try:
        # Auto-enable KaTeX if math markup is present anywhere in the selection.
        if not inject_katex:
            try:
                all_md = "\n\n".join([(r.get("prompt") or "") + "\n" + (r.get("answer") or "") for r in rows])
            except Exception:
                all_md = ""
            if _md_has_math(all_md):
                inject_katex = True
        html_doc = _build_export_html(project_name, rows, include_katex=inject_katex)
        with tempfile.TemporaryDirectory() as td:
            html_path = Path(td) / "export.html"
            pdf_path  = Path(td) / "export.pdf"
            html_path.write_text(html_doc, encoding="utf-8")
            cmd = [
                shutil.which("wkhtmltopdf"),
                "--enable-local-file-access",
                "--print-media-type",
                "--dpi", os.environ.get("TARKASH_WKHTML_DPI", "144"),
                "--encoding", "utf-8",
                "-q",
            ]
            # When KaTeX is injected, enable JS and wait for window.status='katex-done'
            if inject_katex:
                js_delay = os.environ.get("TARKASH_WKHTML_JS_DELAY", "1600")
                cmd += ["--enable-javascript", "--javascript-delay", js_delay, "--window-status", "katex-done"]
            cmd += [str(html_path), str(pdf_path)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return pdf_path.read_bytes()
    except Exception as e:
        logger.warning(f"wkhtmltopdf export failed: {e}")
        return None


# ---------- PDF export (selected chats) ----------

def _render_history_cards(project_name: str, *, flt: dict, manager_ui: bool = True) -> None:
    chats = list_chats(project_name, **flt)
    # Wrap history area so we can scope CSS reliably
    st.markdown('<div class="hist-root">', unsafe_allow_html=True)

    # ----------Load per-project UI (no column picker UI anymore) ----------
    try:
        _ui_path = project_paths(project_name)["ui_settings"]
        _ui_local = _read_json(_ui_path)
    except Exception:
        _ui_local = {}
        _ui_path = None

    # Always show these extra columns
    visible_extra_cols = ["mode", "model", "embedder"]

    # ---------- Header ----------
    st.markdown("### History")
    # Toolbar placeholder so it renders *after* the table (ensures single-click enable for Copy)
    st.session_state.setdefault("hist_view_ids", set())
    toolbar_ph = st.empty()

    # ---------- Build DF (ALWAYS render table right after toolbar) ----------
    df = _history_dataframe(project_name, flt=flt, show_cols=visible_extra_cols)
    # Keep a stable, 0..N-1 index so edited_rows mapping stays correct
    df = df.reset_index(drop=True)
    # Ensure the editor renders even with no rows by keeping the schema stable
    if df.empty:
        # keep only the known, desired columns (in the same order)
        expected_cols = (
            ["id"]
            + [c for c in visible_extra_cols if c in ["mode", "model", "embedder"]]
            + ["prompt", "answer", "view"]
        )
        # create any missing columns with appropriate dtypes
        for col in expected_cols:
            if col not in df.columns:
                # bool dtype for 'view', object for others keeps st.data_editor happy
                df[col] = pd.Series(dtype=("bool" if col == "view" else "object"))
        # keep column order tidy
        df = df[expected_cols]
    else:
        # ★ Critical: apply the user's *latest* click before re-rendering the editor
        df = _apply_pending_view_edits_from_editor(df, key="history_table")
    displayed_ids = df["id"].astype(int).tolist() if "id" in df.columns else [int(r["id"]) for r in chats]
    # cache for the "Select all" button on first paint
    st.session_state["_hist_last_displayed_ids"] = displayed_ids

    # Lock non-boolean columns; only 'view' is editable
    disabled_cols = ["id","prompt","answer"] + [c for c in ["mode","model","embedder"] if c in df.columns]
    colcfg: Dict[str, Any] = {
        "id": st.column_config.NumberColumn("id", help="Chat ID", width="small", format="%d"),
        "prompt": st.column_config.TextColumn("prompt", width="large", help="User prompt (truncated)"),
        "answer": st.column_config.TextColumn("answer", width="large", help="Assistant answer (truncated)"),
        "view": st.column_config.CheckboxColumn("view", width="small", help="Show details below"),
    }
    if "mode" in df.columns:
        colcfg["mode"] = st.column_config.TextColumn("mode", width="small", help="Run mode")
    if "model" in df.columns:
        colcfg["model"] = st.column_config.TextColumn("model", width="small", help="LLM model")
    if "embedder" in df.columns:
        colcfg["embedder"] = st.column_config.TextColumn("embedder", width="small", help="Embedding model")
    # desired order
    order_cols = [c for c in ["id","mode","model","embedder","prompt","answer","view"] if c in df.columns]

    df_ret = st.data_editor(
        df,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        disabled=disabled_cols,
        column_config=colcfg,
        column_order=order_cols,
        key="history_table",
    )
    _update_history_selection_from_editor(df_ret)

    # ---------- Fill toolbar *after* table has updated selection ----------
    with toolbar_ph.container():
        st.markdown('<div class="toolbar">', unsafe_allow_html=True)
        left_grp, right_grp = st.columns([7, 5], gap="small")
        with left_grp:
            cb1, cb2 = st.columns([1, 1], gap="small")
            with cb1:
                st.checkbox(
                    "Save to project (exports/)",
                    value=st.session_state.get("save_exports_to_project", True),
                    key="save_exports_to_project",
                )
            with cb2:
                st.checkbox(
                    "Inject KaTeX (wkhtmltopdf)",
                    value=st.session_state.get("inject_katex", False),
                    key="inject_katex",
                )

        def _compose_copy_payload(rows: list[dict]) -> str:
            parts: list[str] = []
            for r in rows:
                cid = r.get("id")
                ts  = r.get("ts_iso") or _human_dt(r.get("ts"))
                mode = r.get("mode",""); model = r.get("model") or ""; emb = r.get("embedder") or ""
                parts.append(f"# Chat #{cid} — {ts}\n[{mode}] model={model} · emb={emb}\n")
                parts.append("You:\n" + (r.get("prompt") or "").rstrip() + "\n")
                parts.append("Assistant:\n" + (r.get("answer") or "").rstrip())
                parts.append("\n---\n")
            return "\n".join(parts).strip()

        # ---------- Markdown → "friendlier KaTeX" preprocessing ----------
        def _normalize_latexish(md_text: str) -> str:
            """
            Make common “almost-LaTeX” patterns KaTeX-friendly.
            - Any [ ... ] segment containing a TeX command → \[ ... \]  (display)
            - Any ( ... ) segment containing a TeX command → \( ... \)  (inline)
            - Collapse doubled backslashes (\\alpha → \alpha) inside math only.
            Conservative: only transforms when a backslash-command exists.
            """
            def _clean_math(s: str) -> str:
                # turn \\alpha → \alpha, \\sin → \sin (only before a letter)
                return re.sub(r"\\\\([A-Za-z])", r"\\\1", s)

            text = md_text

            # 1) Bracketed display math anywhere in a line (not just whole line)
            #    Example: "... [ \\cos^2 A + \\sin^2 A = 1 ]."
            def _bracket_any_repl(m: re.Match) -> str:
                inner = m.group(1)
                if re.search(r"\\[A-Za-z]+", inner):
                    # → \[ … \]  (single backslashes)
                    return "\\[" + _clean_math(inner).strip() + "\\]"
                return m.group(0)
            text = re.sub(r"\[\s*([^\[\]\n]*\\[A-Za-z][^\[\]\n]*)\s*\]", _bracket_any_repl, text)

            # 2) Parenthetical inline math anywhere in a line
            #    Example: "Solve for (\\sin A) when ..."
            def _paren_any_repl(m: re.Match) -> str:
                inner = m.group(1)
                if re.search(r"\\[A-Za-z]+", inner):
                    # → \( … \)  (single backslashes)
                    return "\\(" + _clean_math(inner).strip() + "\\)"
                return m.group(0)
            text = re.sub(r"\(\s*([^()\n]*\\[A-Za-z][^()\n]*)\s*\)", _paren_any_repl, text)

            # 3) As a final pass, convert lines that are *only* bracket-math into $$ blocks
            #    (helps spacing/centering when the equation sits alone on a line)
            def _block_repl(m: re.Match) -> str:
                inner = m.group(1)
                if re.search(r"\\[A-Za-z]+", inner):
                    return "$$\n" + _clean_math(inner).strip() + "\n$$"
                return m.group(0)
            text = re.sub(r"^[ \t]*\[\s*(.+?)\s*\][ \t]*$", _block_repl, text, flags=re.MULTILINE)

            return text

        # ---------- HTML builder for PDF (Chromium/wkhtml) ----------
        def _compose_html_doc(project_name: str, rows: List[Dict[str, Any]], inject_katex: bool = True) -> str:
            md = _compose_copy_payload(rows)
            md = _normalize_latexish(md)
            # Convert Markdown → HTML (with tables & fenced code)
            html_body = markdown.markdown(
                md,
                extensions=[
                    TableExtension(),
                    FencedCodeExtension(),
                    CodeHiliteExtension(linenums=False, guess_lang=True, noclasses=True),
                ],
                output_format="html5",
            )
            # Minimal print CSS for nice PDF
            css = """
                @page { size: A4; margin: 18mm 16mm 18mm 16mm; }
                body { font: 12pt -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji", sans-serif; color:#111; line-height:1.35; }
                h1,h2,h3{ margin: 1.2em 0 .4em; page-break-after: avoid; }
                p, li { orphans: 3; widows: 3; }
                pre code, code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 10pt; }
                pre { background: #f6f8fa; padding: 10px 12px; border-radius: 8px; overflow-x: auto; page-break-inside: avoid; }
                table { width: 100%; border-collapse: collapse; margin: .8em 0; page-break-inside: avoid; }
                th,td { border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }
                hr { border: 0; border-top: 1px solid #e5e7eb; margin: 18px 0; }
                .header { font-size: 13pt; font-weight: 600; margin-bottom: 12px; }
                .footer { font-size: 10pt; color: #666; margin-top: 24px; }
            """
            # KaTeX (auto-render) — robust math without LaTeX toolchain
            katex = ""
            if inject_katex:
                katex = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/mhchem.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "$", right: "$", display: false},
      {left: "\\\\(", right: "\\\\)", display: false},
      {left: "\\\\[", right: "\\\\]", display: true}
    ],
    throwOnError: false,
    strict: "ignore"
  });
});
</script>
"""
            title = f"Tarkash — {project_name}"
            return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>{css}</style>
  {katex}
</head>
<body>
  <div class="header">{title}</div>
  {html_body}
  <div class="footer">Generated by Tarkash</div>
</body>
</html>"""

        # ---------- Chromium print-to-PDF (preferred) ----------
        def _export_chats_to_pdf_chromium(project_name: str, rows: List[Dict[str, Any]], inject_katex: bool = True) -> Optional[bytes]:
            """Render Markdown→HTML→PDF using headless Chromium (Playwright)."""
            if sync_playwright is None:
                return None
            html = _compose_html_doc(project_name, rows, inject_katex=inject_katex)
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                    page = browser.new_page()
                    # Load HTML and give KaTeX/fonts a brief moment to settle.
                    page.set_content(html, wait_until="domcontentloaded")
                    try:
                        page.wait_for_function(
                            "document.fonts ? document.fonts.status === 'loaded' : true",
                            timeout=5000,
                        )
                    except Exception:
                        pass
                    page.wait_for_timeout(400)
                    # Footer with page numbers; no header (to avoid duplicate title)
                    footer = """
                    <style>
                      .f { font-size:8px; color:#666; width:100%; padding:0 12mm; }
                    </style>
                    <div class="f" style="text-align:center;">
                      Page <span class="pageNumber"></span> / <span class="totalPages"></span>
                    </div>
                    """
                    pdf_bytes = page.pdf(
                        format="A4",
                        print_background=True,
                        margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                        display_header_footer=True,
                        header_template="<div></div>",
                        footer_template=footer,
                    )
                    browser.close()
                    return pdf_bytes
            except Exception as e:
                logger.warning(f"Chromium PDF export failed: {e}")
                return None
            
        def _available_pdf_paths() -> list[tuple[str, str]]:
            """Return a list of (key, label) for available PDF export engines."""
            opts: list[tuple[str, str]] = []
            if _pandoc_available():       opts.append(("pandoc",   "Pandoc (LaTeX)"))
            if sync_playwright is not None: opts.append(("chromium", "Chromium (Playwright)"))
            if _have_wkhtmltopdf():       opts.append(("wkhtml",   "wkhtmltopdf"))
            if _have_weasyprint():        opts.append(("weasy",    "WeasyPrint"))
            return opts

        def _export_chats_to_pdf_weasy(project_name: str, rows: List[Dict[str, Any]]) -> Optional[bytes]:
            """Render PDF via WeasyPrint explicitly (HTML -> PDF)."""
            if not _have_weasyprint():
                return None
            from weasyprint import HTML  # type: ignore
            html_doc = _build_export_html(project_name, rows, include_katex=False)
            return HTML(string=html_doc).write_pdf()


        def _export_chats_to_pdf_auto(project_name: str, rows: List[Dict[str, Any]]) -> Optional[bytes]:
            """
            Build a PDF using the best available engine in this order:
            1) Chromium (Playwright) + KaTeX (best HTML/CSS support, robust math)
            2) Pandoc + (xelatex|tectonic)  (best for heavy LaTeX)
            3) wkhtmltopdf (with optional KaTeX)
            4) Pure-Python fallback (WeasyPrint → xhtml2pdf → PyMuPDF)
            """
            inject_katex = bool(st.session_state.get("inject_katex", False))
            # 1) Chromium
            pdf = _export_chats_to_pdf_chromium(project_name, rows, inject_katex=inject_katex)
            if pdf:
                return pdf
            # 2) Pandoc
            pdf = _export_chats_to_pdf_pandoc(project_name, rows)
            if pdf:
                return pdf
            # 3) wkhtmltopdf
            pdf = _export_chats_to_pdf_wkhtml(project_name, rows, inject_katex=inject_katex)
            if pdf:
                return pdf
            # 4) pure-Python fallback
            return _export_chats_to_pdf_rich(project_name, rows)

        def _export_pdf_by_path(path_key: str, project_name: str, rows: List[Dict[str, Any]], *, inject_katex: bool = False) -> Optional[bytes]:
            """Switchboard for specific engines: pandoc|chromium|wkhtml|weasy|auto"""
            if not rows:
                return None
            key = (path_key or "").strip().lower()
            if key == "pandoc":
                return _export_chats_to_pdf_pandoc(project_name, rows)
            if key == "chromium":
                return _export_chats_to_pdf_chromium(project_name, rows, inject_katex=inject_katex)
            if key == "wkhtml":
                return _export_chats_to_pdf_wkhtml(project_name, rows, inject_katex=inject_katex)
            if key == "weasy":
                return _export_chats_to_pdf_weasy(project_name, rows)
            return _export_chats_to_pdf_auto(project_name, rows)

        def _send_telegram_channel(text: str, *, caption: str = "Tarkash share") -> tuple[bool, str]:
            """Send text to a Telegram channel using Bot API.
            Uses env TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
            For long texts, uploads as a .txt document (no length cap).
            Returns (ok, info)."""
            token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
            chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
            if not token or not chat_id:
                return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
            api = f"https://api.telegram.org/bot{token}"
            try:
                if len(text) <= 3500:
                    r = requests.post(f"{api}/sendMessage", data={"chat_id": chat_id, "text": text})
                else:
                    bio = BytesIO(text.encode("utf-8"))
                    bio.name = "tarkash_share.txt"
                    r = requests.post(
                        f"{api}/sendDocument",
                        data={"chat_id": chat_id, "caption": caption},
                        files={"document": ("tarkash_share.txt", bio, "text/plain")},
                    )
                ok = False
                info = ""
                try:
                    j = r.json()
                    ok = bool(j.get("ok"))
                    info = j.get("description", "") or ""
                except Exception:
                    info = r.text
                return (True, "ok") if ok else (False, info or "Telegram API error")
            except Exception as e:
                return False, str(e)

        def _send_telegram_document(data: bytes, *, filename: str = "tarkash_share.pdf", caption: str = "Tarkash share (PDF)") -> tuple[bool, str]:
            """Upload a document (PDF) to the Telegram channel via Bot API."""
            token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
            chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
            if not token or not chat_id:
                return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
            api = f"https://api.telegram.org/bot{token}"
            try:
                bio = BytesIO(data)
                bio.name = filename
                r = requests.post(
                    f"{api}/sendDocument",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"document": (filename, bio, "application/pdf")},
                )
                ok = False
                info = ""
                try:
                    j = r.json()
                    ok = bool(j.get("ok"))
                    info = j.get("description", "") or ""
                except Exception:
                    info = r.text
                return (True, "ok") if ok else (False, info or "Telegram API error")
            except Exception as e:
                return False, str(e)

        with right_grp:
            # Trimmed toolbar: Select all | Clear | Copy | Delete | Telegram
            bsel, bclr, bcpy, bdel, btgsend = st.columns([1.25, 1, 1, 1, 1], gap="small")

            # ---- Selection intelligence (based on current view) ----
            _displayed_ids: list[int] = list(map(int, st.session_state.get("_hist_last_displayed_ids", []) or []))
            _selected_ids: set[int] = set(map(int, st.session_state.get("hist_view_ids", set()) or set()))
            _selected_in_view: set[int] = _selected_ids.intersection(set(_displayed_ids))
            _any_selected: bool = len(_selected_in_view) > 0
            _all_selected: bool = (len(_displayed_ids) > 0) and (len(_selected_in_view) == len(_displayed_ids))
            _has_rows: bool = len(_displayed_ids) > 0
            with bsel:
                if st.button(
                    "Select all",
                    use_container_width=True,
                    key="hist_select_all_top",
                    disabled=(not _has_rows or _all_selected),
                ):
                    ids = st.session_state.get("_hist_last_displayed_ids", [])
                    st.session_state["hist_view_ids"] = set(map(int, ids))
                    _rerun()
            with bclr:
                if st.button(
                    "Clear",
                    use_container_width=True,
                    key="hist_clear_selection_top",
                    disabled=(not _any_selected),
                ):
                    st.session_state["hist_view_ids"] = set()
                    _rerun()
            with bcpy:
                sel_ids  = {int(i) for i in st.session_state.get("hist_view_ids", set())}
                sel_rows = [r for r in chats if int(r.get("id")) in sel_ids]
                payload  = _compose_copy_payload(sel_rows) if sel_rows else ""
                b64      = base64.b64encode(payload.encode("utf-8")).decode("ascii")
                disabled = "disabled" if not sel_rows else ""
                tooltip  = f"Copy {len(sel_rows)} selected chat(s)" if sel_rows else "Copy selected chat(s)"
                components.html(
                    f"""
                    <div style="width:100%">
                      <button class="hist-copy-btn" title="{tooltip}" {disabled}
                        onclick="(function(btn){{
                          if (btn.hasAttribute('disabled')) return;
                          const b64 = '{b64}';
                          function b64ToUtf8(s){{
                            try {{
                              const bin = atob(s);
                              const bytes = new Uint8Array(bin.length);
                              for (let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
                              return new TextDecoder('utf-8').decode(bytes);
                            }} catch(e) {{
                              try {{ return decodeURIComponent(escape(atob(s))); }}
                              catch(_e) {{ return atob(s); }}
                            }}
                          }}
                          const txt = b64ToUtf8(b64);
                          function legacyCopy(t){{
                            const ta=document.createElement('textarea');
                            ta.value=t; ta.setAttribute('readonly','');
                            ta.style.position='fixed'; ta.style.top='-1000px';
                            document.body.appendChild(ta); ta.select();
                            try{{ document.execCommand('copy'); }}catch(_){{
                            }} document.body.removeChild(ta);
                          }}
                          const done = () => {{ btn.innerText='Copied!'; setTimeout(()=>{{btn.innerText='Copy';}}, 1200); }};
                          if (navigator.clipboard && window.isSecureContext) {{
                            navigator.clipboard.writeText(txt).then(done).catch(function(){{ legacyCopy(txt); done(); }});
                          }} else {{
                            legacyCopy(txt); done();
                          }}
                        }})(this)">
                        Copy
                      </button>
                      <style>
                        :root {{ --btn-border:#e2e8f0; --btn-bg:#ffffff; --btn-bg-hover:#f3f4f6; }}
                        html, body {{ margin:0; padding:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; font-size:16px; line-height:1.2; }}
                        .hist-copy-btn {{
                          display:inline-flex; align-items:center; justify-content:center;
                          width:100%;
                          height:46px;                 /* match Clear/Delete pill height */
                          padding:0 .75rem;            /* no vertical padding to avoid overflow */
                          font-size:.85rem;
                          border-radius:14px; border:1px solid var(--btn-border); background:var(--btn-bg);
                          white-space:nowrap; box-sizing:border-box; cursor:pointer;
                          transition: background .15s ease, transform .02s ease, opacity .15s ease;
                          -webkit-font-smoothing: antialiased;
                        }}
                        .hist-copy-btn:hover {{ background:var(--btn-bg-hover); }}
                        .hist-copy-btn:active {{ transform: translateY(2px); }}
                        .hist-copy-btn[disabled] {{ opacity:.5; cursor:not-allowed; }}
                        .hist-copy-btn:focus {{ outline:none; box-shadow:none; }}
                      </style>
                    </div>
                    """,
                    height=50,
                )
            with bdel:
                if st.button(
                    "Delete",
                    use_container_width=True,
                    key="hist_delete_bulk",
                    disabled=(not _any_selected),
                ):
                    sel_ids_list = list(st.session_state.get("hist_view_ids", set()))
                    if not sel_ids_list:
                        st.toast("Select at least one row to delete.", icon="⚠️")
                    else:
                        try:
                            for _cid in sel_ids_list:
                                delete_chat(project_name, int(_cid))
                            st.session_state["hist_view_ids"] = set()
                            st.toast(f"Deleted {len(sel_ids_list)} chat(s).", icon="🗑️")
                            _rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
            with btgsend:
                sel_ids_send  = {int(i) for i in st.session_state.get("hist_view_ids", set())}
                sel_rows_send = [r for r in chats if int(r.get("id")) in sel_ids_send]
                if st.button("Telegram", use_container_width=True, key="hist_send_tg", disabled=(not sel_rows_send)):
                    # Use the selected (or explicitly set) PDF engine for Telegram
                    path = st.session_state.get('telegram_pdf_path') or st.session_state.get('export_pdf_path') or 'pandoc'
                    pdf_bytes = _export_pdf_by_path(
                        path, project_name, sel_rows_send,
                        inject_katex=bool(st.session_state.get('inject_katex', False))
                    )
                    if pdf_bytes:
                        tsf = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                        fname = f"{_safe_name(project_name)}_chats_{tsf}.pdf"
                        ok, info = _send_telegram_document(pdf_bytes, filename=fname, caption=f"Tarkash share ({len(sel_rows_send)} chat(s))")
                    else:
                        payload_send  = _compose_copy_payload(sel_rows_send) if sel_rows_send else ""
                        ok, info = _send_telegram_channel(payload_send, caption=f"Tarkash share ({len(sel_rows_send)} chat(s))")
                    if ok:
                        st.toast(f"Sent {len(sel_rows_send)} chat(s) to Telegram", icon="📤")
                    else:
                        st.error(f"Telegram send failed: {info}")
        st.markdown('</div>', unsafe_allow_html=True)  # end .toolbar

    # ---------- Exports + downloads ----------
    if manager_ui:
        st.session_state.setdefault("hist_view_ids", set())
        sel_ids = list(st.session_state.get("hist_view_ids", set()))
        sel_rows = [r for r in chats if int(r.get("id")) in set(map(int, sel_ids))]
        disabled = (len(sel_rows) == 0)

        # ---- Unified Export Toolbar ----
        st.markdown('<div class="exports">', unsafe_allow_html=True)
        # Layout: [file type] [engine] [spacer] [Export] [↓.md] [↓PDF]
        try:
            # Move Export (col[2]) closer to the selectors by shrinking the spacer.
            bcol = st.columns([1.2, 2.0, 1.1, 6, 0.9, 0.9], gap="small", vertical_alignment="center")
        except TypeError:
            # Streamlit < 1.31 doesn't support vertical_alignment
            bcol = st.columns([1.2, 2.0, 1.1, 6, 0.9, 0.9], gap="small")

        # (1) File type selector
        with bcol[0]:
            st.session_state.setdefault("export_file_type", "PDF")
            st.selectbox(
                "Export file type",
                options=["PDF", "Markdown"],
                key="export_file_type",
                help="Choose whether to export a PDF or a Markdown file.",
            )

        # (2) PDF engine selector (shown only when File type == PDF)
        with bcol[1]:
            avail = _available_pdf_paths()
            opts   = [k for k,_ in avail]
            labels = {k: lbl for k,lbl in avail}
            default_key = st.session_state.get("export_pdf_path")
            if not default_key or default_key not in labels:
                default_key = (opts[0] if opts else "pandoc")
                st.session_state["export_pdf_path"] = default_key
            st.selectbox(
                "Export path",
                options=opts if opts else ["pandoc"],
                index=(opts.index(st.session_state["export_pdf_path"]) if opts and st.session_state["export_pdf_path"] in opts else 0),
                key="export_pdf_path",
                format_func=lambda k: labels.get(k, k),
                disabled=(st.session_state.get("export_file_type") != "PDF") or (not opts),
                help="Choose the engine for PDF export.",
            )

        # (3) Export — align with the two selects by padding for label height
        with bcol[2]:
            # Add a small spacer so the button lines up with the selectboxes (which have labels).
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            if st.button("Export", disabled=disabled, use_container_width=True, key="btn_export_unified"):
                try:
                    if st.session_state.get("export_file_type") == "Markdown":
                        md_bytes = _export_chats_to_markdown(project_name, sel_rows, for_pdf=False)
                        if md_bytes:
                            st.session_state["last_export_md"] = md_bytes
                            tsf = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                            st.session_state["last_export_md_name"] = f"{_safe_name(project_name)}_chats_{tsf}.md"
                            st.success("Markdown prepared")
                            if bool(st.session_state.get("save_exports_to_project", True)):
                                export_dir = Path.home()/".tarkash"/"projects"/_safe_name(project_name)/"exports"
                                export_dir.mkdir(parents=True, exist_ok=True)
                                (export_dir/st.session_state["last_export_md_name"]).write_bytes(md_bytes)
                                st.caption(f"Saved: `{export_dir/st.session_state['last_export_md_name']}`")
                    else:
                        path_key = st.session_state.get("export_pdf_path") or "pandoc"
                        pdf_bytes = _export_pdf_by_path(
                            path_key, project_name, sel_rows,
                            inject_katex=bool(st.session_state.get("inject_katex", False))
                        )
                        if pdf_bytes:
                            st.session_state["last_export_pdf"] = pdf_bytes
                            tsf = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                            st.session_state["last_export_pdf_name"] = f"{_safe_name(project_name)}_chats_{tsf}.pdf"
                            st.success("PDF prepared")
                            if bool(st.session_state.get("save_exports_to_project", True)):
                                export_dir = Path.home()/".tarkash"/"projects"/_safe_name(project_name)/"exports"
                                export_dir.mkdir(parents=True, exist_ok=True)
                                (export_dir/st.session_state["last_export_pdf_name"]).write_bytes(pdf_bytes)
                                st.caption(f"Saved: `{export_dir/st.session_state['last_export_pdf_name']}`")
                        else:
                            st.error("PDF export failed. Check engine availability (Pandoc/LaTeX, Playwright Chromium, wkhtmltopdf, or WeasyPrint).")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        # Spacer to push the download buttons to the far right
        with bcol[3]:
            st.write("")

        # (4) Downloads
        with bcol[4]:
            if st.session_state.get("last_export_md"):
                st.download_button("⬇ .md",
                                   data=st.session_state["last_export_md"],
                                   file_name=st.session_state["last_export_md_name"],
                                   mime="text/markdown",
                                   use_container_width=True,
                                   key="dl_hist_md")
        with bcol[5]:
            if st.session_state.get("last_export_pdf"):
                st.download_button("⬇ PDF",
                                   data=st.session_state["last_export_pdf"],
                                   file_name=st.session_state["last_export_pdf_name"],
                                   mime="application/pdf",
                                   use_container_width=True,
                                   key="dl_hist_pdf")
        st.markdown('</div>', unsafe_allow_html=True)  # close .exports

    # ---------- Rows as expanders ----------
    for row in chats:
        cid = int(row["id"])
        if cid not in st.session_state.get("hist_view_ids", set()):
            continue
        ts = row.get("ts")
        ts_iso = row.get("ts_iso") or _human_dt(ts)
        mode = row.get("mode", "")
        prompt = row.get("prompt", "")
        answer = row.get("answer", "")
        model = row.get("model") or ""
        embedder = row.get("embedder") or ""
        rag_on = "on" if row.get("rag_on") else "off"
        streaming = "on" if row.get("streaming") else "off"
        proj_path = row.get("project_path") or ""

        # Detail expander label
        label = f"#{cid} — {_shorten_one_line(prompt)}"
        with st.expander(label, expanded=False):
            # Meta
            st.caption(f"{ts_iso} · {mode} · model={model} · emb={embedder} · RAG={rag_on} · stream={streaming}")
            if proj_path:
                st.caption(f"root: {proj_path}")
            # Content
            st.markdown("**You:**")
            st.markdown(prompt)
            _md = _render_images_and_svg_from_text(answer)
            render_response_with_latex(_md)
    st.markdown("</div>", unsafe_allow_html=True)  # close .hist-root


# =============================================================================
# Page & CSS
# =============================================================================

st.set_page_config(page_title=APP_TITLE, page_icon="💎", layout="wide")

_log_env_status()

def _env_pills():
    # Compute environment once per render
    have_pandoc = _pandoc_available()
    engine = _pandoc_engine()
    pills = []
    pills.append(("Pandoc", "ok" if have_pandoc else "no",
                  f"pandoc {'+'+engine if engine else ''}" if have_pandoc else "not found"))
    pills.append(("TeX", "ok" if engine in ("xelatex","tectonic") else "no",
                  engine or "no engine"))
    pills.append(("WeasyPrint", "ok" if _have_weasyprint() else "no",
                  "weasyprint"))
    pills.append(("xhtml2pdf", "ok" if _have_xhtml2pdf() else "no", "xhtml2pdf"))
    pills.append(("PyMuPDF", "ok" if _have_pymupdf() else "no", "fitz"))
    pills.append(("wkhtmltopdf", "ok" if _have_wkhtmltopdf() else "no", "wkhtmltopdf"))
    return pills


def _inject_css():
    st.markdown(
        """
        <style>
        :root{
          --bg-soft:#f8fafc;      /* slate-50  */
          --ink:#0f172a;          /* slate-900 */
          --muted:#64748b;        /* slate-500 */
          --ring:#e2e8f0;         /* slate-200 */
          --card:#ffffff;
          --accent:#6366f1;       /* indigo-500 */
          --radius:14px;
        }
        .main > div { padding-top: .6rem; }
        h1,h2,h3{ letter-spacing:-.02em; }
        .title-card{
          background:var(--card);
          border:1px solid var(--ring);
          border-radius:var(--radius);
          padding:.9rem 1.2rem;
          box-shadow:0 1px 2px rgba(2,6,23,.05);
          display:flex; align-items:center; justify-content:space-between;
          margin-bottom:.6rem;
        }
        .brand{ display:flex; gap:.6rem; align-items:center; }
        .pills{ display:flex; gap:.4rem; flex-wrap: wrap; }
        .pill{
          border:1px solid var(--ring);
          padding:.25rem .6rem;
          border-radius:999px; font-size:.8rem; color:var(--muted); background:white;
        }
        .pill.ok{ color:#065f46; border-color:#34d399; background:#ecfdf5; }     /* teal/green */
        .pill.maybe{ color:#92400e; border-color:#fbbf24; background:#fffbeb; }  /* amber */
        .pill.no{ color:#991b1b; border-color:#fca5a5; background:#fef2f2; }     /* red */
        .card{
          background:var(--card);
          border:1px solid var(--ring);
          border-radius:var(--radius);
          padding:1rem 1.25rem;
          box-shadow:0 1px 2px rgba(2,6,23,.05);
        }
        section[data-testid="stSidebar"]{ background:var(--bg-soft); padding-top:.25rem; }
        div[data-baseweb="radio"] > div { flex-direction:row; gap:.75rem; }  /* horizontal radios */
        .stButton>button{
          border-radius:var(--radius); border:1px solid var(--ring); padding:.6rem 1rem;
        }        
        /* Sidebar buttons: same height + no wrapping so all align */
        section[data-testid="stSidebar"] .stButton>button{
          height: 44px;
          white-space: nowrap;
          text-overflow: ellipsis;
        }
        .stButton>button[kind="primary"]{ background:var(--accent); color:white; border:none; }
        .stTextArea textarea{ line-height:1.4; }
        /* Indexing action area (no border/shadow) */
        section[data-testid="stSidebar"] .index-actions{
          padding:.25rem 0 .25rem 0;
          background: transparent !important;
        }
        history-note{ font-size:.8rem; color:var(--muted); margin-top:.25rem; }
        /* Chat history cards */
        .chat-card{
          background:var(--card);
          border:1px solid var(--ring);
          border-radius:var(--radius);
          padding:.8rem 1rem;
          margin:.75rem 0;
          box-shadow:0 1px 2px rgba(2,6,23,.05);
        }
        .copy-btn{
          /* match Streamlit button look & height */
          display:inline-flex;
          align-items:center;
          justify-content:center;
          width:100%;
          height:36px;
          font-size:.85rem;
          padding:0 .6rem;
          border-radius:8px;
          border:1px solid var(--ring);
          background:white;
          cursor:pointer;
          transition: background .15s ease;
        }
        .copy-btn:hover{ background:#f3f4f6; }
        /* History area: align Copy/Delete and unify button sizes */
        .hist-root .stButton>button{
          height:36px;
          width:100%;
          border-radius:8px;
        }
        /* Tarkash glyphs (corners + sidebar) */
        .tarkash-logo{
          position: fixed; top: 6px; left: 14px; z-index: 1000;
          display:flex; align-items:center; gap:.35rem;
          font-weight:600; color:var(--ink); opacity:.95; pointer-events:none;
        }
        .tarkash-logo-right{
          position: fixed; top: 6px; right: 14px; z-index: 1000;
          display:flex; align-items:center; gap:.35rem;
          font-weight:600; color:var(--ink); opacity:.95; pointer-events:none;
        }
        .tarkash-logo svg, .tarkash-logo-right svg, .sidebar-logo svg{ width:28px; height:28px; }
        /* Make the main prompt container visually flat (no border line showing above "Assistant Response") */
        .prompt-card{
          border: 0 !important;
          box-shadow: none !important;
          padding: 0 !important;
          background: transparent !important;
          margin-bottom: .35rem !important; /* tighter gap to the next heading */
        }
        /* -------- History view polish -------- */
        .hist-root [data-testid="stDataFrame"] *{
          font-size: 15px;               /* larger grid text */
        }
        .hist-root [data-testid="stDataFrame"] th,
        .hist-root [data-testid="stDataFrame"] td{
          padding: 10px 12px !important; /* comfy row height */
        }
        .hist-root .card{
          margin-top:.75rem;
        }
        /* History header toolbar + smaller buttons */
        .hist-root .toolbar .stButton>button{
          height:46px;               /* match custom Copy pill */
          padding:0 .75rem;          /* consistent horizontal padding */
          font-size:.85rem;
          border-radius:14px;        /* same curvature as Copy */
          white-space: nowrap;          /* avoid two-line button labels */
          min-width: 96px;              /* gives 'Select all' a bit more room */
        }
        /* Smaller export buttons row */
        .hist-root .exports .stButton>button{
          height:32px;
          padding:.25rem .6rem;
          font-size:.85rem;
          border-radius:8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_inject_css()

# Title bar (env pills removed; status is logged instead)
with st.container():
    st.markdown(
        "<div class='title-card'><div class='brand'><span style='font-weight:700'>Tarkash</span></div></div>",
        unsafe_allow_html=True
    )


# Logo
st.markdown(
    """
    <div class="tarkash-logo" title="Tarkash">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="3" width="10" height="18" rx="2"></rect>
        <path d="M7.5 6 L6 7.5 L9 7.5 Z"></path>
        <path d="M11.5 5 L10 6.5 L13 6.5 Z"></path>
        <path d="M8 8 V18"></path>
        <path d="M12 7 V18"></path>
      </svg>
      <span>Tarkash</span>
    </div>
    """,
    unsafe_allow_html=True)

# Logo (top-right)
st.markdown(
    """
    <div class="tarkash-logo-right" title="Tarkash">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="3" width="10" height="18" rx="2"></rect>
        <path d="M7.5 6 L6 7.5 L9 7.5 Z"></path>
        <path d="M11.5 5 L10 6.5 L13 6.5 Z"></path>
        <path d="M8 8 V18"></path>
        <path d="M12 7 V18"></path>
      </svg>
    </div>
    """,
    unsafe_allow_html=True)

# =============================================================================
# Sidebar — grouped, minimal-scrolling
# =============================================================================

with st.sidebar:
    # Sidebar logo
    st.markdown(
        """
        <div class="sidebar-logo" title="Tarkash">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="3" y="3" width="10" height="18" rx="2"></rect>
            <path d="M7.5 6 L6 7.5 L9 7.5 Z"></path>
            <path d="M11.5 5 L10 6.5 L13 6.5 Z"></path>
            <path d="M8 8 V18"></path>
            <path d="M12 7 V18"></path>
          </svg>
          <span>Tarkash</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Configuration", anchor=False)

    # read cross-panel state early
    _mode_now = st.session_state.get("mode_radio", "Direct Chat")
    _rag_enabled = bool(st.session_state.get("rag_on", False))
    _is_direct_chat = (_mode_now == "Direct Chat")

    # Log level
    log_level = st.selectbox("Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
    logging_setup.set_level(log_level)

    # ---- Project (grouped in an expander) ----
    with st.expander("Project", expanded=False):
        # name
        global_ui = _read_json(GLOBAL_UI_SETTINGS_PATH)
        default_project_name = global_ui.get("project_name") or Path.cwd().name
        project_name = st.text_input(
            "Project name",
            value=default_project_name,
            help="Logical name shown in UI and used for per-project folders."
        )
        if project_name != global_ui.get("project_name"):
            global_ui["project_name"] = project_name
            _write_json(GLOBAL_UI_SETTINGS_PATH, global_ui)

        # scaffold + per-project files
        scaff = ensure_project_scaffold(project_name)
        per_ui_path = scaff["ui_settings"]
        per_rag_path = scaff["rag"]
        rag_index_dir = scaff["rag_index_dir"]

        # per-project UI state
        ui = _read_json(per_ui_path)

        # root
        default_root = ui.get("project_root") or str(Path.cwd())
        project_root = st.text_input(
            "Project root (--root)",
            value=default_root,
            disabled=not _rag_enabled,
            help=None if _rag_enabled else "Disabled while RAG is OFF",
        )
        # Canonicalize now to keep indexer/status keys consistent
        try:
            project_root_resolved = str(Path(project_root).expanduser().resolve())
        except Exception:
            project_root_resolved = project_root
        project_root = project_root_resolved
        ui["project_root"] = project_root_resolved

        # --- Indexation (moved out of Project) ---
        # nothing rendered here; see dedicated "Indexation" expander below

    # --- Indexation ---
    with st.expander("Indexation", expanded=False):
        # Embedding model (moved here from Models & Embeddings; fully controlled to avoid double-clicks)
        try:
            models_config_path = ui.get("models_config") or str(MODELS_JSON_PATH)
            with open(models_config_path, "r", encoding="utf-8") as f:
                _models_cfg = json.load(f)
            emb_list = _models_cfg.get("embedding_models", []) or []
            emb_names = [e["name"] for e in emb_list]
            seed_emb = (
                ui.get("embedding_model")
                or _models_cfg.get("default_embedding_model")
                or (emb_names[0] if emb_names else "")
            )
        except Exception:
            emb_list, emb_names, seed_emb = [], [], ""

        if emb_names:
            # Seed once; keep a single source of truth in session_state (no index/value on widget)
            if ("embedding_model_select" not in st.session_state) or (
                st.session_state["embedding_model_select"] not in emb_names
            ):
                st.session_state["embedding_model_select"] = (
                    seed_emb if seed_emb in emb_names else emb_names[0]
                )
            st.selectbox(
                "Embedding model (RAG / indexing)",
                emb_names,
                key="embedding_model_select",
                help="Used by the indexer and RAG retrieval.",
            )
            selected_emb = st.session_state["embedding_model_select"]
            ui["embedding_model"] = selected_emb
            st.session_state["embedding_model"] = selected_emb
            # Persist selection to rag.json immediately so indexer uses it
            try:
                cfg = load_rag(per_rag_path)
                cfg.setdefault("embedder", {})
                cfg["embedder"]["selected_name"] = selected_emb
                cfg["chroma_dir"] = str(rag_index_dir)
                save_rag(per_rag_path, cfg)
            except Exception:
                pass
        else:
            st.info("No embedding models found in models.json.")

        # Auto indexing toggle
        auto_index_flag = st.checkbox(
            "Auto indexing",
            value=bool(ui.get("rag_auto_index", True)),
            help="First run creates index; later runs update (delta).",
        )
        ui["rag_auto_index"] = auto_index_flag

        # determine current indexing status ONCE (always fetch; independent of chat mode)
        try:
            _st = index_status(project_root) or {}
            is_running = (str(_st.get("state", "")).lower() == "running")
        except Exception:
            _st, is_running = {}, False

        # ── Actions (Delta, Full, Stop). Only disable while a run is active.
        st.markdown('<div class="index-actions">', unsafe_allow_html=True)
        btn_cols = 3 if is_running else 2
        cols = st.columns(btn_cols, gap="small")

        # Delta
        with cols[0]:
            if st.button("Delta index", use_container_width=True, disabled=is_running, key="btn_delta_index"):
                with st.spinner("Delta indexing…"):
                    try:
                        cfg = load_rag(per_rag_path)
                        emb_name = ui.get("embedding_model")
                        cfg.setdefault("embedder", {})
                        if emb_name:
                            cfg["embedder"]["selected_name"] = emb_name
                        cfg["chroma_dir"] = str(rag_index_dir)
                        save_rag(per_rag_path, cfg)
                        res = delta_index(project_root, per_rag_path)
                        # Robust summary across possible keys that the indexer may return
                        files_changed = int(res.get("files_changed") or res.get("changed_files") or 0)
                        added         = int(res.get("added") or res.get("chunks_added") or 0)
                        updated       = int(res.get("updated") or res.get("chunks_updated") or res.get("modified") or 0)
                        deleted       = int(res.get("deleted") or res.get("removed") or res.get("chunks_removed") or 0)
                        st.success(
                            f"Delta index complete · added={added}, updated={updated}, deleted={deleted}, changed(existing)={files_changed}"
                        )
                        with st.expander("Delta details", expanded=False):
                            added_files   = res.get("added_files") or res.get("files_added") or []
                            changed_files = res.get("changed_files") or []
                            deleted_files = res.get("deleted_files") or res.get("files_removed") or []
                            per_file      = res.get("per_file") or res.get("file_stats")
                            if added_files:
                                st.write("Added files:", added_files)
                            if changed_files:
                                st.write("Changed files:", changed_files)
                            if deleted_files:
                                st.write("Deleted files:", deleted_files)
                            if per_file:
                                st.write("Per-file chunk counts:", per_file)
                            st.caption(res)
                    except Exception as e:
                        st.error(f"Delta index failed: {e}")

        # Full
        with cols[1]:
            if st.button("Full reindex", use_container_width=True, disabled=is_running, key="btn_full_reindex"):
                with st.spinner("Reindexing…"):
                    try:
                        cfg = load_rag(per_rag_path)
                        emb_name = ui.get("embedding_model")
                        cfg.setdefault("embedder", {})
                        if emb_name:
                            cfg["embedder"]["selected_name"] = emb_name
                        cfg["chroma_dir"] = str(rag_index_dir)
                        save_rag(per_rag_path, cfg)
                        res = full_reindex(project_root, per_rag_path)
                        added   = int(res.get("added") or res.get("chunks_added") or 0)
                        deleted = int(res.get("deleted") or res.get("removed") or res.get("chunks_removed") or 0)
                        st.success(f"Reindex complete · added={added}" + (f", deleted={deleted}" if deleted else ""))
                        st.caption(res)
                    except Exception as e:
                        st.error(f"Reindex failed: {e}")

        # Stop (only while running)
        if is_running:
            with cols[2]:
                if st.button("Stop indexing", use_container_width=True, key="btn_stop_index"):
                    try:
                        st.warning("Stop requested; workers finish current files.")
                        request_stop(project_root)
                    except Exception as e:
                        st.error(f"Failed to request stop: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Status: concise, professional summary (clean/dirty, last time, counts)
        def _fmt_idx_summary(_st: dict) -> str:
            def _ival(*keys, default=0):
                for k in keys:
                    v = _st.get(k)
                    if v is not None:
                        try: return int(v)
                        except Exception: pass
                return default
            def _b(*keys, default=False):
                for k in keys:
                    v = _st.get(k)
                    if isinstance(v, bool): return v
                    s = str(v).lower()
                    if s in ("1","true","yes","dirty"):  return True
                    if s in ("0","false","no","clean"):  return False
                return default
            def _ts(*keys):
                for k in keys:
                    v = _st.get(k)
                    if isinstance(v, (int,float)) and v>0:
                        try: return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(v)))
                        except Exception: pass
                    if isinstance(v, str) and v: return v
                return "—"
            running  = str(_st.get("state","")).lower() == "running"
            dirty    = _b("dirty")
            pf, tf   = _ival("processed_files","files_processed"), _ival("total_files","files_total", default=0)
            added    = _ival("added","files_added","chunks_added")
            changed  = _ival("files_changed","changed_files","chunks_updated","updated","modified")
            deleted  = _ival("deleted","removed","chunks_removed")
            last_ts  = _ts("last_index_ts","last_run_ts","last_ts","last_run")
            if running:
                tf = tf or 1
                return f"Indexing… {pf}/{tf} files"
            state_txt = "DIRTY" if dirty else "clean"
            parts = [f"Indexer: {state_txt}", f"last: {last_ts}"]
            if any([added, changed, deleted]):
                parts.append(f"Δ added={added}, changed={changed}, deleted={deleted}")
            return " · ".join(parts)
        try:
            st.caption(_fmt_idx_summary(_st))
        except Exception:
            st.caption("Indexer: status unavailable")

    # --- RAG settings (per project) ---
    with st.expander("RAG settings (project)", expanded=False):
        RAG_PATH = str(per_rag_path)
        try:
            cfg = load_rag(RAG_PATH)
            st.caption(f"Path: `{RAG_PATH}`")
            # make sure chroma_dir points to per-project index dir
            cfg.setdefault("chroma_dir", str(rag_index_dir))
            rag_text = st.text_area("rag.json", json.dumps(cfg, indent=2), height=320, key="rag_textarea")
            cA, cB = st.columns(2)
            with cA:
                if st.button("Save RAG settings", key="btn_save_rag"):
                    save_rag(RAG_PATH, json.loads(rag_text))
                    st.success("Saved. Re-run app to apply.")
            with cB:
                if st.button("Reload from disk", key="btn_reload_rag"):
                    _rerun()
        except Exception as e:
            st.error(f"Failed to load rag.json: {e}")

    # --- Models (LLM picker + config editor) ---
    with st.expander("Models", expanded=False):
        default_cfg = str(MODELS_JSON_PATH)
        models_config = st.text_input("Model config (--config)", value=ui.get("models_config", default_cfg))
        ui["models_config"] = models_config

        try:
            model_registry = ModelRegistry(models_config)
            models_list = model_registry.list()
            names = [m.name for m in models_list]
            # --- Controlled selectbox: no index/value; drive via session_state only ---
            seed_name = ui.get("llm_model") or model_registry.default_name or (names[0] if names else "")
            if names:
                if ("llm_model_select" not in st.session_state) or (
                    st.session_state["llm_model_select"] not in names
                ):
                    st.session_state["llm_model_select"] = (
                        seed_name if seed_name in names else names[0]
                    )
                st.selectbox("Model (--model)", names, key="llm_model_select")
                selected_model_name = st.session_state["llm_model_select"]
                ui["llm_model"] = selected_model_name
                model = model_registry.get(selected_model_name)
            else:
                # No models available; keep placeholders
                ui["llm_model"] = ""
                model = None
            st.caption(f"Provider: {model.provider}  •  Endpoint: {model.endpoint}")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()

        # ---- Inline models.json editor (moved from separate expander) ----
        st.divider()
        st.caption("**Config (models.json)**")
        st.caption(f"Path: `{models_config}`")
        _models_raw = ""
        try:
            _models_raw = Path(models_config).read_text(encoding="utf-8")
        except Exception as _e:
            st.warning(f"Could not read models config: {str(_e)}")
            _models_raw = "{}"
        models_text = st.text_area("models.json", _models_raw, height=320, key="models_textarea")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("Save models.json", key="btn_save_models_json"):
                try:
                    data = json.loads(models_text)
                    p = Path(models_config)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    st.success("Saved. Re-run the app if models changed.")
                except Exception as _e:
                    st.error(f"Invalid JSON or write failed: {str(_e)}")
        with col_m2:
            if st.button("Reload models.json", key="btn_reload_models_json"):
                _rerun()


    # --- Secrets (.env) editor ---
    with st.expander("Secrets (.env)", expanded=False):
        st.caption(f"Path: `{str(ENV_PATH)}`")

        # ---------- helpers (local scope) ----------
        def _parse_env_file(path: Path) -> Dict[str, str]:
            """Best-effort .env parser: KEY=VALUE lines, ignore comments/blank lines."""
            out: Dict[str, str] = {}
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                return out
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                # drop trailing comment when value is unquoted
                if v and not (v.startswith("'") or v.startswith('"')):
                    # allow inline comments: KEY=value # comment
                    if " #" in v:
                        v = v.split(" #", 1)[0].rstrip()
                    elif "\t#" in v:
                        v = v.split("\t#", 1)[0].rstrip()
                # strip surrounding quotes
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k:
                    out[k] = v
            return out

        def _dump_env_lines(kv: Dict[str, str]) -> str:
            header = (
                "# Tarkash .env (managed by UI)\n"
                "# Edit via the app or by hand. Lines below were generated.\n"
            )
            lines = [header]
            for k in sorted(kv.keys()):
                v = kv[k] if kv[k] is not None else ""
                # quote if contains spaces or special chars
                if re.search(r"\s|[#\"']", v):
                    v_out = json.dumps(v, ensure_ascii=False)  # safe JSON quoted
                else:
                    v_out = v
                lines.append(f"{k}={v_out}")
            return "\n".join(lines) + "\n"

        def _write_env_file(path: Path, updates: Dict[str, str], keep_existing: bool = True) -> None:
            existing = _parse_env_file(path) if (keep_existing and path.exists()) else {}
            merged = dict(existing)
            for k, v in (updates or {}).items():
                if not k:
                    continue
                if v == "" or v is None:
                    # treat empty value as 'remove' only if it exists already
                    merged.pop(k, None)
                else:
                    merged[k] = v
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_dump_env_lines(merged), encoding="utf-8")

        # ---------- discover known keys ----------
        known_keys: List[str] = []
        try:
            cfg_json = json.loads(Path(models_config).read_text(encoding="utf-8"))
            for entry in (cfg_json.get("llm_models") or []):
                if isinstance(entry, dict) and entry.get("api_key_reqd") and entry.get("api_key_env"):
                    known_keys.append(entry["api_key_env"])
            for entry in (cfg_json.get("embedding_models") or []):
                if isinstance(entry, dict) and entry.get("api_key_reqd") and entry.get("api_key_env"):
                    known_keys.append(entry["api_key_env"])
        except Exception:
            pass
        # Helpful extras commonly used by the app
        known_keys.extend(["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
        known_keys = sorted(set([k for k in known_keys if isinstance(k, str) and k.strip()]))

        # Load current values (prefer file, fall back to environment)
        file_env = _parse_env_file(ENV_PATH)
        def _get_val(k: str) -> str:
            return file_env.get(k) or os.environ.get(k) or ""

        st.caption("**Known keys from config**")
        for k in known_keys:
            c1, c2 = st.columns([2, 5])
            with c1:
                st.text_input("KEY", value=k, key=f"env_known_key_{k}", disabled=True, label_visibility="collapsed")
            with c2:
                st.text_input("VALUE", value=_get_val(k), key=f"env_known_val_{k}", type="password", label_visibility="collapsed")

        # Extra rows the user can add
        if "secret_rows" not in st.session_state:
            st.session_state["secret_rows"] = [{"k": "", "v": ""}]

        st.caption("**Add more**")
        new_rows = []
        for i, row in enumerate(st.session_state["secret_rows"]):
            c1, c2 = st.columns([2, 5])
            with c1:
                nk = st.text_input("KEY", value=row.get("k",""), key=f"env_free_key_{i}", placeholder="MY_API_KEY", label_visibility="collapsed")
            with c2:
                nv = st.text_input("VALUE", value=row.get("v",""), key=f"env_free_val_{i}", type="password", placeholder="value", label_visibility="collapsed")
            new_rows.append({"k": nk.strip(), "v": nv})
        st.session_state["secret_rows"] = new_rows

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("+ Add row", key="btn_env_add_row"):
                st.session_state["secret_rows"].append({"k": "", "v": ""})
                st.experimental_rerun()
        with col_s2:
            if st.button("Save to .env", key="btn_env_save", type="primary"):
                # Collect updates
                upd: Dict[str, str] = {}
                # known
                for k in known_keys:
                    v = st.session_state.get(f"env_known_val_{k}", "")
                    if k and v is not None:
                        upd[k] = str(v)
                # freeform (validate)
                KEY_RX = re.compile(r"^[A-Z_][A-Z0-9_]*$")
                errs = []
                for i, row in enumerate(st.session_state["secret_rows"]):
                    k = (row.get("k") or "").strip()
                    v = (row.get("v") or "").strip()
                    if not k and not v:
                        continue
                    if not KEY_RX.match(k):
                        errs.append(f"Row {i+1}: invalid KEY '{k}'. Use A–Z, 0–9, and underscores; must not start with a digit.")
                        continue
                    upd[k] = v
                if errs:
                    for e2 in errs:
                        st.error(e2)
                else:
                    try:
                        _write_env_file(ENV_PATH, upd, keep_existing=True)
                        # reload into process so the session sees new values
                        load_dotenv(dotenv_path=ENV_PATH, override=True)
                        st.success("Saved secrets to .env and reloaded environment.")
                    except Exception as _e:
                        st.error(f"Save failed: {str(_e)}")
        with col_s3:
            if st.button("Reload from disk", key="btn_env_reload"):
                _rerun()


    # --- Execution Options ---
    with st.expander("Execution Options", expanded=False):
        enable_tools = True  # always enabled
        ui["enable_tools"] = True
        enhanced = bool(RAG_AGENT_AVAILABLE)
        ui["enhanced"] = enhanced
        if not RAG_AGENT_AVAILABLE and RAG_AGENT_ERR:
            st.caption("Enhanced planner unavailable — falling back to base Agent.")

        max_iters = st.number_input(
            "Max tool iters (--max-iters)",
            1,
            20,
            int(ui.get("max_iters", 5)),
            1,
            disabled=_is_direct_chat,
            help=None if not _is_direct_chat else "Disabled in Direct Chat",
        )
        ui["max_iters"] = int(max_iters)

        analysis_only = st.checkbox(
            "Analysis-only (no writes)",
            value=bool(ui.get("analysis_only", True)),
            disabled=_is_direct_chat,
            help=None if not _is_direct_chat else "Disabled in Direct Chat",
        )
        ui["analysis_only"] = analysis_only

        callgraph_depth = st.number_input(
            "Call graph depth (0 = full)",
            0,
            10,
            int(ui.get("callgraph_depth", 3)),
            1,
            disabled=_is_direct_chat,
            help=None if not _is_direct_chat else "Disabled in Direct Chat",
        )
        ui["callgraph_depth"] = int(callgraph_depth)
        st.session_state["callgraph_depth"] = int(callgraph_depth)

    # --- History (sidebar) ---
    with st.expander("History", expanded=False):
        # Default OFF
        history_on = st.checkbox(
            "Enable history (persist chats in SQLite)",
            value=bool(ui.get("history_enabled", False)),
            help="When OFF, chats are not saved and history is hidden."
        )
        ui["history_enabled"] = history_on

        # Options visible only when enabled
        if history_on:
            st.caption("Filter")
            # Search + regex toggle
            csa, csb = st.columns([3,1])
            with csa:
                search_q = st.text_input("Search prompt/answer/mode", value=ui.get("history_search", ""))
            with csb:
                regex = st.checkbox("Regex", value=bool(ui.get("history_regex", False)))
            ui["history_search"] = search_q
            ui["history_regex"] = bool(regex)

            # Mode selector
            st.caption(" ")
            st.caption("**Time window**")

            FILTER_OPTS = {
                "All": "all",
                "Top N (most recent)": "top_n",
                "Last N (oldest first)": "last_n",
                "Past 7 days": "past_7d",
                "Past 30 days": "past_30d",
                "Date range…": "date_range",
            }
            human_to_code = {k:v for k,v in FILTER_OPTS.items()}
            code_to_human = {v:k for k,v in FILTER_OPTS.items()}
            sel = ui.get("history_filter_mode", "top_n")
            filter_label = st.selectbox(
                "Choose filter",
                list(FILTER_OPTS.keys()),
                index=list(FILTER_OPTS.values()).index(sel) if sel in FILTER_OPTS.values() else 1,
            )
            ui["history_filter_mode"] = human_to_code[filter_label]

            # N (for top/last)
            if ui["history_filter_mode"] in ("top_n", "last_n"):
                n_val = int(ui.get("history_n", 20))
                n_val = st.number_input("N", min_value=1, max_value=1000, value=n_val, step=1)
                ui["history_n"] = int(n_val)

            # Date presets don't need inputs. Custom range does:
            if ui["history_filter_mode"] == "date_range":
                import datetime as _dt
                df_str = ui.get("history_date_from")
                dt_str = ui.get("history_date_to")
                today = _dt.date.today()
                default_from = _dt.date.fromisoformat(df_str) if df_str else today
                default_to   = _dt.date.fromisoformat(dt_str) if dt_str else today
                dr = st.date_input("Date range", value=(default_from, default_to))
                if isinstance(dr, tuple) and len(dr) == 2:
                    ui["history_date_from"] = dr[0].isoformat()
                    ui["history_date_to"]   = dr[1].isoformat()

            st.divider()
            st.caption("**Conversation context**")
            ctx_on = st.toggle(
                "Use context in replies",
                value=bool(ui.get("ctx_on", False)),
                help="When on, your last N Q→A turns are prepended before the new prompt so the assistant can follow the conversation."
            )
            ui["ctx_on"] = bool(ctx_on)
            if ctx_on:
                ctx_turns = st.number_input(
                    "Context turns (Q→A pairs)",
                    min_value=1, max_value=10, value=int(ui.get("ctx_turns", 4)), step=1,
                    help="How many most recent Q→A pairs to include as context."
                )
                ui["ctx_turns"] = int(ctx_turns)

            # Optional: clear all
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Refresh", key="btn_history_refresh"):
                    _rerun()
            with c2:
                if st.button("Delete all", key="btn_history_delete_all"):
                    try:
                        dbp = _chat_db_path(project_name)
                        if dbp.exists():
                            with sqlite3.connect(dbp) as con:
                                con.execute("DELETE FROM chats WHERE project=?", (project_name,))
                                con.commit()
                        st.success("History cleared for this project.")
                    except Exception as e:
                        st.error(f"Failed to clear history: {e}")

    # Persist per-project UI + remember last project globally
    # (Moved to AFTER History controls so the latest toggle/search/N values are saved
    #  before the main pane reads them to decide whether to render history.)
    _write_json(per_ui_path, ui)
    _write_json(GLOBAL_UI_SETTINGS_PATH, {"project_name": project_name})

    # --- Build runtime (on key change) ---
    rt_key = (project_root, enhanced, True, ui.get("llm_model"))
    if "rt_key" not in st.session_state or st.session_state["rt_key"] != rt_key:
        tools = (
            EnhancedToolRegistry(project_root)
            if (enhanced and ENHANCED_TOOLS_AVAILABLE)
            else ToolRegistry(project_root)
        )
        agent = (
            ReasoningAgent(model, tools, enable_tools=True)
            if (enhanced and RAG_AGENT_AVAILABLE)
            else Agent(model, tools, enable_tools=True)
        )
        st.session_state["tools"] = tools
        st.session_state["agent"] = agent
        # Pass the UI system prompt (if ON) to Agent+RAG (ReasoningAgent) so final synthesis uses it.
        try:
            if hasattr(agent, "set_system_override"):
                _ui = _read_json(per_ui_path)
                _on = bool(_ui.get("sys_prompt_on", False))
                _txt = (_ui.get("sys_prompt_text") or "").strip() if _on else ""
                agent.set_system_override(_txt)
        except Exception as _e:
            logger.debug("ignored: could not set system_override on agent: {}", _e)

        st.session_state["rt_key"] = rt_key
        st.session_state["last_results"] = None

    if not RAG_AGENT_AVAILABLE and RAG_AGENT_ERR:
        with st.expander("Planner diagnostics", expanded=False):
            st.caption(
                "ReasoningAgent could not be imported; using base Agent with local fallback executor."
            )
            st.code(RAG_AGENT_ERR)

# =============================================================================
# Main — Mode bar, toggles, prompt, response
# =============================================================================

current_project_name_for_history = None

# Init run-state keys
for key, default in [
    ("cancel", False),
    ("progress_rows", []),
    ("running", False),
    ("show_chart_gallery", False),
    ("last_tools_used", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Defaults so History tab can render without Chat widgets
prompt: str = ""
submit: bool = False

# Compose the effective system prompt each run
def _compose_system_prompt(ui_dict: dict, mode_label: str) -> str:
    # Internal baseline (dynamic bits kept minimal & safe)
    lines = ["You are a helpful assistant."]
    if bool(st.session_state.get("rag_on", False)):
        lines.append("Use retrieved project documents when relevant; prefer code citations with file:line.")
    if mode_label in ("LLM Tools", "Agent Plan & Run") and bool(ui_dict.get("analysis_only", True)):
        lines.append("Operate in analysis-only mode unless explicitly instructed otherwise.")
    internal = "\n".join(lines).strip()
    # Optional user addendum
    user_add = (ui_dict.get("sys_prompt_text") or "").strip() if bool(ui_dict.get("sys_prompt_on")) else ""
    if user_add:
        return internal + "\n\n" + user_add
    return internal

# Helper: for adapters that only take a single user string (no system role)
def _prefix_with_system(user_prompt: str, system_text: str) -> str:
    if not (system_text and system_text.strip()):
        return user_prompt
    return f"[SYSTEM]\n{system_text.strip()}\n\n[USER]\n{user_prompt}"


# Tabs at the top; Chat holds all runtime controls, History stays clean
chat_tab, history_tab = st.tabs(["Chat", "History"])

with chat_tab:
    # ----- Mode & run toggles (Chat-only) -----
    mt_left, mt_right = st.columns([3, 2], gap="large")
    with mt_left:
        MODE_LABELS = ["Direct Chat", "LLM Tools", "Agent Plan & Run"]
        st.radio("Mode", MODE_LABELS, horizontal=True, index=0, key="mode_radio")
    with mt_right:
        col1, col2, col3 = st.columns(3)
        with col1:
            _adapter = getattr(st.session_state.get("agent"), "adapter", None)
            _has_stream = bool(getattr(_adapter, "chat_stream", None))
            _mode_now = st.session_state.get("mode_radio", "Direct Chat")
            can_stream = (_mode_now == "Direct Chat") and _has_stream
            streaming = st.toggle(
                "Streaming",
                value=can_stream,
                disabled=not can_stream,
                help=("Streaming is only available in Direct Chat"
                      if _mode_now != "Direct Chat"
                      else ("Adapter does not support streaming" if not _has_stream else None)),
            )
            st.session_state["streaming_enabled"] = bool(streaming)
        with col2:
            rag_on = st.toggle(
                "RAG",
                value=bool(st.session_state.get("rag_on", False)),
                help="Use this project's indexed documents in answers. Set project root & index in the sidebar."
            )
            st.session_state["rag_on"] = rag_on
        with col3:
            st.toggle(
                "Complex planning",
                value=bool(st.session_state.get("complex_planning", False)),
                key="complex_planning",
                disabled=(st.session_state.get("mode_radio", "Direct Chat") != "Agent Plan & Run"),
            )

    # Prompt + single primary action (Chat-only)
    st.markdown('<div class="card prompt-card">', unsafe_allow_html=True)
    prompt = st.text_area("input_prompt", height=140, placeholder="Type your instruction…")
    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        main_label = "Stop" if st.session_state.get("running") else "Submit"
        action_clicked = st.button(main_label, type="primary", use_container_width=True, key="submit_stop_btn")
    with b2:
        clear = st.button("Clear", use_container_width=True, key="btn_clear_prompt")
    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state["progress_rows"] = []
        st.session_state["last_tools_used"] = set()
        st.session_state["show_chart_gallery"] = False
        _rerun()

    # Interpret action button
    if action_clicked and st.session_state.get("running"):
        st.session_state["cancel"] = True
        st.toast("Stopping…", icon="🛑")
        submit = False
    else:
        submit = bool(action_clicked and not st.session_state.get("running"))

    # ---- System prompt (Chat-only) ----
    # Per-project persistence: read the current project's ui_settings.json
    def _current_project_name_and_paths():
        _g = _read_json(GLOBAL_UI_SETTINGS_PATH)
        name = _g.get("project_name") or Path.cwd().name
        return name, project_paths(name)
    proj_name_for_ui, _paths_for_ui = _current_project_name_and_paths()
    _ui_for_ui = _read_json(_paths_for_ui["ui_settings"])

    st.divider()
    if "sys_prompt_on" not in st.session_state:
        st.session_state["sys_prompt_on"] = bool(_ui_for_ui.get("sys_prompt_on", False))
    if "sys_prompt_text" not in st.session_state:
        st.session_state["sys_prompt_text"] = _ui_for_ui.get("sys_prompt_text", "")
    st.toggle(
        "System prompt",
        key="sys_prompt_on",
        help="When ON, the text below is sent as a system message (merged with internal system instructions).",
    )
    if st.session_state["sys_prompt_on"]:
        st.text_area(
            "system_prompt",
            key="sys_prompt_text",
            height=72,
            placeholder="e.g., Be concise; answer strictly with steps; prefer KaTeX for math…",
        )
    # Persist per-project immediately from session_state
    _ui_for_ui["sys_prompt_on"] = bool(st.session_state["sys_prompt_on"])
    _ui_for_ui["sys_prompt_text"] = st.session_state.get("sys_prompt_text", "")
    _write_json(_paths_for_ui["ui_settings"], _ui_for_ui)

    # Response area (Chat-only)
    st.markdown("### Assistant Response")
    response_area = st.empty()

# Resolve current mode for downstream logic (visible even outside Chat tab)
mode = st.session_state.get("mode_radio", "Direct Chat")

# --- transient conversation buffer (used when history is OFF) ---
if "transient_turns" not in st.session_state:
    st.session_state["transient_turns"] = []

def _push_transient_turn(prompt_text: str, answer_text: str, limit: int = 10):
    """
    Store a Q→A pair in session memory so context can work even if
    history persistence is disabled for the project.
    """
    buf = list(st.session_state.get("transient_turns", []))
    buf.append({"prompt": prompt_text, "answer": answer_text})
    st.session_state["transient_turns"] = buf[-limit:]

final_answer_text_for_history: Optional[str] = None
def _looks_like_tool_blob(text: str) -> Optional[Any]:
    if not text:
        return None
    right = text.split("->", 1)[1].strip() if "->" in text else text.strip()
    if right.startswith("```"):
        right = right.strip("` \n")
        if right.lower().startswith("json"):
            right = right[4:].strip()
    try:
        return json.loads(right)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", right, re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                import ast as _ast
                return _ast.literal_eval(candidate)
            except Exception:
                return None
    return None

def _render_call_graph_payload(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if not (("calls" in result and isinstance(result["calls"], list)) or ("dot" in result and isinstance(result["dot"], str))):
        return False
    fn = result.get("function", "<function>")
    file_ = result.get("file", "<file>")
    st.markdown("### Call graph")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**Function:** `{fn}`")
    with c2:
        st.markdown(f"**Defined in:** `{file_}`")
    calls = result.get("calls") or []
    if calls:
        st.markdown("#### Direct callees")
        rows = [{"name": c.get("name"), "defined_in": c.get("defined_in")} for c in calls]
        st.table(rows)
    try:
        if isinstance(result.get("dot"), str):
            st.graphviz_chart(result["dot"])
    except Exception:
        pass
    return True

def _maybe_render_dot_from_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    m = re.search(r"(digraph\s+G\s*\{.*?\})", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return False
    dot = m.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    dot = dot.replace("→", "->").replace("⇒", "->")
    try:
        st.graphviz_chart(dot)
        return True
    except Exception:
        return False

def _rag_summary_from_steps(steps: List[dict]) -> Optional[str]:
    try:
        rsteps = [s for s in steps if s.get("tool") == "rag_retrieve" and s.get("success")]
        if not rsteps:
            return None
        total = 0
        files = set()
        file_boost_cnt = 0
        for s in rsteps:
            result = s.get("result") or {}
            chunks = result.get("chunks") or []
            total += len(chunks)
            for c in chunks:
                md = c.get("metadata") or {}
                fp = md.get("file_path") or md.get("relpath")
                if fp:
                    files.add(fp)
            try:
                file_boost_cnt += len(result.get("filename_boosted") or [])
            except Exception:
                pass
        try:
            boosted_syms = (rsteps[-1].get("result") or {}).get("symbol_boosted") or []
        except Exception:
            boosted_syms = []
        sym = f" · sym_boost={len(boosted_syms)}" if boosted_syms else ""
        fboost = f" · file_boost={file_boost_cnt}" if file_boost_cnt else ""
        return f"RAG · retrievals={len(rsteps)} · chunks={total} · files={len(files)}{fboost}{sym}"
    except Exception:
        return None

def _rag_mismatch_from_steps(steps: List[dict]) -> Optional[dict]:
    try:
        for s in steps:
            if s.get("tool") == "rag_retrieve":
                mm = (s.get("result") or {}).get("embedder_mismatch")
                if mm:
                    return mm
    except Exception:
        pass
    return None

def _keywordize(text: str, max_terms: int = 8) -> list[str]:
    if not text:
        return []
    t = text.strip()
    phrases = [m.group(1).strip() for m in re.finditer(r'"([^"]+)"', t)]
    stop = {
        "the","and","or","a","an","to","for","of","in","on","by","with","from","at",
        "this","that","these","those","it","is","are","be","as","into","via","using",
        "make","create","build","generate","please","show","need","want","how","why",
        "plan","run","agent","llm","tools","rag","final","answer","step","steps"
    }
    tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", t)]
    keywords, seen = [], set()
    for w in tokens:
        if len(w) <= 2 or w in stop:
            continue
        if w not in seen:
            seen.add(w)
            keywords.append(w)
    queries: list[str] = []
    queries.extend(phrases[:3])
    head = keywords[:6]
    if head:
        if len(head) >= 4: queries.append(" ".join(head[:4]))
        if len(head) >= 3: queries.append(" ".join(head[:3]))
        if len(head) >= 2: queries.append(" ".join(head[:2]))
    for w in head:
        if len(queries) >= max_terms:
            break
        if w not in queries:
            queries.append(w)
    return queries[:max_terms]

def _ensure_index_ready(project_root: str, rag_path: Path, embedder_name: Optional[str], auto: bool, rag_index_dir: Path):
    """Auto index (full / delta) when RAG is ON and auto indexing is enabled."""
    if not auto:
        return
    try:
        cfg = load_rag(rag_path)
        cfg.setdefault("embedder", {})
        if embedder_name:
            cfg["embedder"]["selected_name"] = embedder_name
        cfg["chroma_dir"] = str(rag_index_dir)
        save_rag(rag_path, cfg)
    except Exception as e:
        st.warning(f"RAG config sync failed: {e}")

    try:
        exists_and_has_files = rag_index_dir.exists() and any(rag_index_dir.iterdir())
    except Exception:
        exists_and_has_files = False

    try:
        if not exists_and_has_files:
            with st.spinner("Auto-index: creating fresh index…"):
                full_reindex(project_root, rag_path)
        else:
            with st.spinner("Auto-index: updating index (delta)…"):
                delta_index(project_root, rag_path)
    except Exception as e:
        st.warning(f"Auto-index preflight failed: {e}")

# =============================================================================
# Execution — identical behaviors, reorganized
# =============================================================================

if submit:
    st.session_state["running"] = True
    st.session_state["last_tools_used"] = set()
    st.session_state["show_chart_gallery"] = False

    # Render everything from this point inside the Chat tab
    with chat_tab:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            st.stop()

    # Recompute per-project paths (sidebar variables not in scope here)
    global_ui = _read_json(GLOBAL_UI_SETTINGS_PATH)
    current_project = global_ui.get("project_name") or Path.cwd().name
    scaff = project_paths(current_project)
    current_project_name_for_history = current_project
    ui = _read_json(scaff["ui_settings"])
    project_root = ui.get("project_root") or str(Path.cwd())
    per_rag_path = scaff["rag"]
    rag_index_dir = scaff["rag_index_dir"]

    if st.session_state.get("rag_on", False):
        _ensure_index_ready(
            project_root, per_rag_path, st.session_state.get("embedding_model"), ui.get("rag_auto_index", True), rag_index_dir
        )

    agent = st.session_state["agent"]
    tools = st.session_state["tools"]

    try:
        # ---------------------- Direct Chat ----------------------
        if mode == "Direct Chat":
            cg_match = re.search(
                r"(?:call\s*graph|callgraph).*(?:of|for)\s+([A-Za-z_]\w*)\s*\(\)?", prompt, re.IGNORECASE
            )
            if cg_match:
                fn = cg_match.group(1)
                depth = st.session_state.get("callgraph_depth", 3)
                res = tools.call("call_graph_for_function", function=fn, depth=depth)
                _render_call_graph_payload(res) or st.json(res)
            else:
                # Compose system prompt (internal + user)
                eff_system = _compose_system_prompt(ui, mode)
                msgs = []
                if eff_system:
                    msgs.append({"role": "system", "content": eff_system})
                # Build conversation-aware messages (uses SQLite if history is ON,
                # otherwise falls back to a transient session buffer).
                msgs += _build_context_messages(current_project_name_for_history or current_project, ui, prompt)
                try:
                    _adapter = getattr(agent, "adapter", None)
                    if _adapter and hasattr(_adapter, "chat_stream"):
                        placeholder = response_area.empty()
                        buf = ""
                        try:
                            for chunk in _adapter.chat_stream(msgs):
                                if not isinstance(chunk, str):
                                    continue
                                buf += chunk
                                s = _normalize_math_delimiters(_fix_common_latex_typos(_sanitize_latex_text(buf)))
                                placeholder.markdown(s)
                            final_answer_text_for_history = buf
                            # Re-render fully so inline SVG / base64 images display
                            response_area.empty()
                            _md = _render_images_and_svg_from_text(buf)
                            render_response_with_latex(_md)
                            _push_transient_turn(prompt, buf)
                        except Exception as _e:
                            st.error(f"[stream error] {_e}")
                    else:
                        resp = agent.adapter.chat(msgs)
                        answer = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
                        final_answer_text_for_history = answer
                        if not _maybe_render_dot_from_text(answer):
                            # render final text into Chat tab area (with images/SVG)
                            _md = _render_images_and_svg_from_text(answer)
                            render_response_with_latex(_md)
                        _push_transient_turn(prompt, answer)
                except Exception as e:
                    st.error(f"[error] {e}")

        # ---------------------- LLM Tools ----------------------
        elif mode == "LLM Tools":
            prog_exp = st.expander("Run progress", expanded=False)  # closed by default
            prog_ph = prog_exp.empty()
            st.session_state["progress_rows"] = []

            def _progress_cb(event: str, **data):
                if st.session_state.get("cancel"):
                    return False
                import time as _t

                ts = _t.strftime("%H:%M:%S")
                icon = {
                    "start": "▶️",
                    "model_call_start": "⌛",
                    "model_call_done": "✅",
                    "tool_call_start": "🔧",
                    "tool_call_done": "✅",
                    "synthesis_start": "📝",
                    "synthesis_done": "✅",
                }.get(event, "•")
                line = f"{ts} {icon} **{event}** — " + ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
                st.session_state["progress_rows"].append(line)
                prog_ph.markdown("\n\n".join(st.session_state["progress_rows"]))

                tname = data.get("tool") or data.get("name")
                if isinstance(tname, str) and tname:
                    used = set(st.session_state.get("last_tools_used") or set())
                    used.add(tname)
                    st.session_state["last_tools_used"] = used
                return True

            try:
                # Prefix user prompt so adapters that only accept a single string still get system guidance.
                eff_system = _compose_system_prompt(ui, mode)
                prompt_for_llm = _prefix_with_system(prompt, eff_system)
                answer = agent.ask_once(
                    prompt_for_llm, max_iters=int(st.session_state.get("max_iters", 5)), progress_cb=_progress_cb
                )
            except Exception as e:
                answer = f"[error] {e}"
            finally:
                st.session_state["cancel"] = False

            parsed = _looks_like_tool_blob(answer)
            if parsed is not None:
                if isinstance(parsed, dict) and parsed.get("tool") == "call_graph_for_function":
                    parsed.setdefault("args", {}).setdefault("depth", st.session_state.get("callgraph_depth", 3))
                if isinstance(parsed, dict) and "diff" in parsed:
                    st.code(parsed.get("diff") or "(no diff)", language="diff")
                elif not _render_call_graph_payload(parsed):
                    try:
                        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                    except Exception:
                        pretty = str(parsed)
                    st.json(parsed)
                    final_answer_text_for_history = pretty
            else:
                final_answer_text_for_history = answer if isinstance(answer, str) else str(answer)
                if not _maybe_render_dot_from_text(final_answer_text_for_history):
                    # ensure rendering stays inside Chat tab (with images/SVG)
                    _md = _render_images_and_svg_from_text(final_answer_text_for_history)
                    render_response_with_latex(_md)
                _push_transient_turn(prompt, final_answer_text_for_history)

        # ---------------------- Agent Plan & Run ----------------------
        else:
            prog_exp = st.expander("Run progress", expanded=False)  # closed by default
            prog_ph = prog_exp.empty()
            st.session_state["progress_rows"] = []

            def _progress_cb(event: str, **data):
                if st.session_state.get("cancel"):
                    return False
                import time as _t

                ts = _t.strftime("%H:%M:%S")
                icon = {"plan_start": "🗺️", "step_start": "🔧", "step_done": "✅", "synthesis_start": "📝", "synthesis_done": "✅"}.get(event, "•")
                line = f"{ts} {icon} **{event}** — " + ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
                st.session_state["progress_rows"].append(line)
                prog_ph.markdown("\n\n".join(st.session_state["progress_rows"]))
                return True

            # PLAN
            try:
                if hasattr(agent, "analyze_and_plan"):
                    plan = agent.analyze_and_plan(  # type: ignore[attr-defined]
                        prompt,
                        rag_on=bool(st.session_state.get("rag_on", False)),
                    )
                else:
                    plan = []
                    # Only add retrieval steps if RAG is enabled
                    if bool(st.session_state.get("rag_on", False)):
                        for sub in _keywordize(prompt, max_terms=8):
                            plan.append(
                                {
                                    "tool": "rag_retrieve",
                                    "args": {"query": sub, "top_k": 8},
                                    "description": "Focused retrieval from sub-query",
                                    "critical": False,
                                }
                            )
                        # Broad retrieval only when RAG is ON
                        plan.append(
                            {
                                "tool": "rag_retrieve",
                                "args": {"query": prompt, "top_k": 12},
                                "description": "Broad retrieval for residual gaps",
                                "critical": False,
                            }
                        )
                    plan.append(
                        {
                            "tool": "_answer",
                            "args": {"prompt": prompt},
                            "description": "Synthesize final answer grounded in retrieved text",
                            "critical": True,
                        }
                    )
            except Exception as e:
                plan = []
                st.error(f"[plan error] {e}")

            with st.expander("Plan (steps)", expanded=False):  # closed by default
                st.code(_pretty(plan), language="json")

            # EXECUTE
            try:
                if hasattr(agent, "execute_plan"):
                    exec_json = agent.execute_plan(  # type: ignore[attr-defined]
                        plan,
                        max_iters=max_iters,
                        analysis_only=bool(st.session_state.get("analysis_only", True)),
                        progress_cb=_progress_cb,
                        rag_on=bool(st.session_state.get("rag_on", False)),
                    )
                else:
                    # Minimal local executor
                    CTX_TOOLS = {
                        "rag_retrieve","read_file","list_files","search_code","scan_relevant_files","analyze_files",
                        "edu_detect_intent","edu_similar_questions","edu_question_paper","edu_explain",
                        "edu_extract_tables","edu_build_blueprint","find_related_files","analyze_code_structure",
                        "detect_errors","call_graph_for_function","analyze_function",
                    }
                    def _ctx_defaults():
                        return {"project_root": project_root, "rag_path": str(per_rag_path)}
                    steps: list[dict] = []
                    preview = bool(st.session_state.get("analysis_only", True))
                    for i, step in enumerate(plan, 1):
                        t = step.get("tool")
                        a = dict(step.get("args") or {})
                        desc = step.get("description", "")
                        if t in EDIT_TOOLS and preview:
                            a.setdefault("dry_run", True)
                        if t == "call_graph_for_function":
                            a.setdefault("depth", st.session_state.get("callgraph_depth", 3))
                        if t in CTX_TOOLS or "project_root" in a or "rag_path" in a:
                            for k, v in _ctx_defaults().items():
                                if a.get(k) in (None, "", ".", "default"):
                                    a[k] = v
                        if t == "_answer":
                            continue
                        try:
                            res = tools.call(t, **a)
                            steps.append({"step": i, "tool": t, "args": a, "result": res, "success": True, "description": desc})
                        except Exception as e:
                            steps.append({"step": i, "tool": t, "args": a, "error": str(e), "success": False, "description": desc})
                            break
                    results_json = json.dumps(steps, indent=2)
                    synth_prompt = intents._build_synth_prompt(prompt, results_json)
                    eff_system = _compose_system_prompt(ui, mode)
                    sys_msgs = []
                    if eff_system:
                        sys_msgs.append({"role": "system", "content": eff_system})
                    sys_msgs.append({"role": "system", "content": "Be concrete, cite code (file:line)."})
                    resp = agent.adapter.chat(sys_msgs + [{"role": "user", "content": synth_prompt}])
                    exec_json = json.dumps(
                        steps
                        + [
                            {
                                "step": len(steps) + 1,
                                "tool": "_answer",
                                "args": {"prompt": prompt},
                                "result": (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", ""),
                                "success": True,
                                "description": "Synthesize final answer from prior steps",
                            }
                        ],
                        indent=2,
                    )

                try:
                    steps = json.loads(exec_json) if isinstance(exec_json, str) else (exec_json or [])
                except Exception:
                    steps = []
                finally:
                    st.session_state["cancel"] = False

                st.caption(
                    f"Run · Planner={agent.__class__.__name__} · Executor={'agent' if hasattr(agent,'execute_plan') else 'local'} · "
                    f"RAG={'on' if st.session_state.get('rag_on', True) else 'off'} · Steps={sum(1 for r in steps if r.get('success'))} ok / "
                    f"{sum(1 for r in steps if not r.get('success', False))} failed"
                )
                _rs = _rag_summary_from_steps(steps)
                if _rs:
                    st.caption(_rs)
                _mm = _rag_mismatch_from_steps(steps)
                if _mm:
                    st.warning(f"Embedder mismatch: index was built with `{_mm['indexed']}`, "
                               f"but retrieval requested `{_mm['requested']}`. {_mm.get('advice','')}")

                for rec in steps:
                    res = rec.get("result")
                    if isinstance(res, dict) and ("dot" in res or "calls" in res):
                        _render_call_graph_payload(res)
                    if isinstance(res, dict) and "diff" in res:
                        st.code(res["diff"] or "(no diff)", language="diff")

                final_answer = next(
                    (r.get("result") for r in reversed(steps) if r.get("tool") == "_answer" and r.get("success")), None
                )
                if isinstance(final_answer, str) and final_answer.strip():
                    final_answer_text_for_history = final_answer.strip()
                    # render inside Chat tab (with images/SVG)
                    _md = _render_images_and_svg_from_text(final_answer_text_for_history)
                    render_response_with_latex(_md)
                    _push_transient_turn(prompt, final_answer_text_for_history)
                else:
                    results_json = json.dumps(steps, indent=2)
                    synth_prompt = intents._build_synth_prompt(prompt, results_json)
                    eff_system = _compose_system_prompt(ui, mode)
                    sys_msgs = []
                    if eff_system:
                        sys_msgs.append({"role": "system", "content": eff_system})
                    sys_msgs.append({"role": "system", "content": "Be concrete, cite code (file:line)."})
                    resp = agent.adapter.chat(sys_msgs + [{"role": "user", "content": synth_prompt}])
                    alt = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or "(no content)"
                    final_answer_text_for_history = alt
                    # render inside Chat tab (with images/SVG)
                    _md = _render_images_and_svg_from_text(final_answer_text_for_history)
                    render_response_with_latex(_md)
                    _push_transient_turn(prompt, final_answer_text_for_history)

            except Exception as e:
                st.error(f"[error] {e}")

    except Exception as e:
        st.error(f"[fatal] {e}")
    finally:
        st.session_state["running"] = False


# =============================================================================
# Persist + render chat history (only when enabled)
# =============================================================================
try:
    # Determine active project to show history even when not submitting
    if not current_project_name_for_history:
        _g = _read_json(GLOBAL_UI_SETTINGS_PATH)
        current_project_name_for_history = _g.get("project_name") or Path.cwd().name
    # Load per-project UI to read history settings
    _scaff = project_paths(current_project_name_for_history)
    _ui = _read_json(_scaff["ui_settings"])
    _history_on = bool(_ui.get("history_enabled", False))
    _history_q = _ui.get("history_search", "")
    _history_n = int(_ui.get("history_show_n", 20))

    # Save the just-finished exchange (only if ON)
    if _history_on and submit and current_project_name_for_history and final_answer_text_for_history:
        save_chat(
            project_name=current_project_name_for_history,
            project_path=_ui.get("project_root", ""),
            mode=mode,
            prompt=prompt,
            answer=final_answer_text_for_history,
            streaming=bool(st.session_state.get("streaming_enabled", False)),
            rag_on=bool(st.session_state.get("rag_on", True)),
            model=_ui.get("llm_model"),
            embedder=_ui.get("embedding_model"),
        )
except Exception as _persist_e:
    logger.warning("chat persistence skipped: {}", _persist_e)

# Show history only inside the History tab
with history_tab:
    try:
        _scaff2 = project_paths(current_project_name_for_history or "")
        _ui2 = _read_json(_scaff2["ui_settings"]) if current_project_name_for_history else {}
        if bool(_ui2.get("history_enabled", False)):
            # Build filter dict from per-project UI
            flt = {
                "query": _ui2.get("history_search", "") or "",
                "regex": bool(_ui2.get("history_regex", False)),
                "filter_mode": _ui2.get("history_filter_mode", "top_n"),
                "n": int(_ui2.get("history_n", 20)),
                "date_from": _ui2.get("history_date_from"),
                "date_to": _ui2.get("history_date_to"),
            }
            _render_history_cards(current_project_name_for_history or "", flt=flt, manager_ui=True)
        else:
            st.caption(
                "<span class='history-note'>History is disabled. Enable it from the sidebar ▸ History to persist and view past chats.</span>",
                unsafe_allow_html=True,
            )
    except Exception as _hist_e:
        logger.warning("history render skipped: {}", _hist_e)
