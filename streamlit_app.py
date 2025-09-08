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
import html
from functools import lru_cache

import sqlite3
import base64
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from loguru import logger

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
APP_TITLE = "gemcli — Code Assistant"
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

# ========= Feature flags =========
# Disable in-app PDF export (use Markdown export + your external MD→PDF tool)
PDF_EXPORT_ENABLED = False
# Flip to True later if you want to re-enable in-app PDF rendering.

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
def _history_dataframe(project_name: str, *, flt: dict) -> pd.DataFrame:
    rows = list_chats(project_name, **flt)
    # compact snippets for the grid
    recs = []
    for r in rows:
        recs.append({
            "id": int(r["id"]),
            "prompt": _shorten_one_line(r.get("prompt",""), 110),
            "answer": _shorten_one_line(r.get("answer",""), 110),
        })
    df = pd.DataFrame(recs, columns=["id","prompt","answer"])
    # attach a 'view' flag bound to session selection
    sel = set(st.session_state.get("hist_view_ids", set()))
    df["view"] = df["id"].apply(lambda i: (i in sel))
    return df

def _update_history_selection_from_editor(df_ret: pd.DataFrame):
    ids = set(df_ret.loc[df_ret["view"] == True, "id"].tolist())
    st.session_state["hist_view_ids"] = ids

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

@lru_cache(maxsize=1)
def _pygments_formatter():
    try:
        from pygments.formatters import HtmlFormatter
        return HtmlFormatter(nowrap=False)
    except Exception:
        return None

def _pygments_css() -> str:
    fmt = _pygments_formatter()
    if not fmt:
        return ""
    try:
        return fmt.get_style_defs(".codehilite")
    except Exception:
        return ""

@lru_cache(maxsize=1)
def _md_renderer():
    """
    Build a markdown-it renderer with useful plugins and a pygments highlighter.
    Math is rendered to MathML via texmath, so WeasyPrint can print it.
    """
    from markdown_it import MarkdownIt
    from mdit_py_plugins.table import table_plugin
    from mdit_py_plugins.deflist import deflist_plugin
    from mdit_py_plugins.tasklists import tasklists_plugin
    from mdit_py_plugins.anchors import anchors_plugin
    from mdit_py_plugins.attrs import attrs_plugin
    from mdit_py_plugins.texmath import texmath_plugin
    md = MarkdownIt("commonmark", {"typographer": True})
    # Plugins
    md.use(table_plugin)
    md.use(deflist_plugin)
    md.use(tasklists_plugin, enabled=True)
    md.use(anchors_plugin, max_level=3)
    md.use(attrs_plugin)
    # Math → MathML (no JS needed)
    md.use(texmath_plugin, renderer="mathml")
    # Pygments highlighting
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


def _md_to_html(md_text: str) -> str:
    """Convert Markdown (incl. tables/code/lists) to HTML with MathML + Pygments."""
    src = _clean_markdown_for_pdf(md_text or "")
    try:
        md = _md_renderer()
        return md.render(src)
    except Exception:
        return f"<pre>{html.escape(src)}</pre>"

def _pygments_css() -> str:
    """Inline Pygments CSS for codehilite blocks (no external files, no JS)."""
    try:
        from pygments.formatters import HtmlFormatter
        return HtmlFormatter(style="default").get_style_defs(".codehilite")
    except Exception:
        return ""

def _build_export_html(project_name: str, rows: List[Dict[str, Any]]) -> str:
    """Compose a complete HTML document for selected chats (print-friendly)."""
    pyg_css = _pygments_css()
    # Print-grade CSS; WeasyPrint supports @page margin boxes + counters
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
    p {{ line-height: 1.38; margin: 6px 0; }}
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
    img {{ max-width: 100%; height: auto; }}
    /* Pygments for codehilite */
    {pyg_css}
    /* MathML tweaks (WeasyPrint) */
    math, mrow, mi, mo, mn, mfrac, msup, msub, msubsup, mtable, mtr, mtd {{
      font-family: STIXGeneral, 'DejaVu Serif', serif;
    }}
    """
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<style>{css}</style>",
        "</head><body>",
        f"<div class='title'>Tarkash — Chat Export</div>",
        f"<div class='subtitle'>{html.escape(project_name)} · generated at {now_iso}</div>",
    ]
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
        # Convert markdown → HTML
        prompt_html = _md_to_html(prompt_md)
        answer_html = _md_to_html(answer_md)
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
    html_doc = _build_export_html(project_name, rows)
    return _render_pdf_from_html(html_doc)

def _export_chats_to_markdown(project_name: str, rows: List[Dict[str, Any]]) -> bytes:
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


# ---------- PDF export (selected chats) ----------
def _export_chats_to_pdf(project_name: str, rows: List[Dict[str, Any]]) -> Optional[bytes]:
    """
    Create a neat PDF for selected chats (prompt+answer only, no UI).
    Returns PDF bytes or None if ReportLab is unavailable.
    """
    try:
        # Lazy imports so the app runs even if reportlab isn't installed yet.
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, XPreformatted, PageBreak, KeepTogether
        )
    except Exception as e:
        logger.warning("ReportLab import failed: {}", e)
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.7*cm, rightMargin=1.7*cm, topMargin=1.5*cm, bottomMargin=1.5*cm,
        title=f"Tarkash Chat Export — {project_name}"
    )
    styles = getSampleStyleSheet()
    # Base styles
    title = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=10
    )
    meta = ParagraphStyle(
        "meta", parent=styles["Normal"], fontSize=9, textColor=colors.gray
    )
    h = ParagraphStyle(
        "h", parent=styles["Heading3"], fontSize=12, spaceBefore=6, spaceAfter=4
    )
    pre = ParagraphStyle(
        "pre", parent=styles["Code"], fontName="Courier", fontSize=9, leading=12
    )

    def _on_page(canvas, _doc):
        # footer: page number right; project left
        canvas.saveState()
        w, h_page = A4
        canvas.setFont("Helvetica", 8)
        canvas.setFillGray(0.4)
        canvas.drawString(1.7*cm, 1.0*cm, f"Project: {project_name}")
        canvas.drawRightString(w - 1.7*cm, 1.0*cm, f"Page {_doc.page}")
        canvas.restoreState()

    story: List[Any] = []

    # Title
    ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    story.append(Paragraph(f"Tarkash — Chat Export", title))
    story.append(Paragraph(f"{project_name} · generated at {ts_iso}", meta))
    story.append(Spacer(1, 10))

    # Each chat section
    for r in rows:
        cid = r.get("id")
        ts = r.get("ts")
        ts_iso = r.get("ts_iso") or _human_dt(ts)
        mode = r.get("mode", "")
        model = r.get("model") or ""
        embedder = r.get("embedder") or ""
        rag_on = "on" if r.get("rag_on") else "off"
        streaming = "on" if r.get("streaming") else "off"
        prompt = (r.get("prompt") or "").rstrip()
        answer = (r.get("answer") or "").rstrip()

        header = Paragraph(f"Chat #{cid} — {ts_iso}", h)
        meta_line = Paragraph(
            f"{mode} · model={model} · emb={embedder} · RAG={rag_on} · stream={streaming}",
            meta
        )
        # Use monospaced block to preserve formatting/code fences
        you_lbl = Paragraph("<b>You:</b>", styles["Normal"])
        you_txt = XPreformatted(prompt or "(empty)", pre)
        asst_lbl = Paragraph("<b>Assistant:</b>", styles["Normal"])
        asst_txt = XPreformatted(answer or "(empty)", pre)

        block = [
            header, meta_line, Spacer(1, 6),
            you_lbl, Spacer(1, 2), you_txt, Spacer(1, 6),
            asst_lbl, Spacer(1, 2), asst_txt, Spacer(1, 12),
        ]
        story.append(KeepTogether(block))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()

def _render_history_cards(project_name: str, *, flt: dict, manager_ui: bool = True) -> None:
    chats = list_chats(project_name, **flt)
    if not chats:
        return
    st.markdown("#### History")
    # Wrap history area so we can scope CSS reliably
    st.markdown('<div class="hist-root">', unsafe_allow_html=True)

    # ---------- Data table (run FIRST so selection state is fresh) ----------
    df = _history_dataframe(project_name, flt=flt)
    df_ret = st.data_editor(
        df,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        disabled=["id","prompt","answer"],  # only 'view' is editable
        column_config={
            "id": st.column_config.NumberColumn(
                "id", help="Chat ID", width="small", format="%d"
            ),
            "prompt": st.column_config.TextColumn(
                "prompt", width="medium", help="User prompt (truncated)"
            ),
            "answer": st.column_config.TextColumn(
                "answer", width="large", help="Assistant answer (truncated)"
            ),
            "view": st.column_config.CheckboxColumn("view", width="small", help="Show details below"),
        },
        key="history_table",
    )
    _update_history_selection_from_editor(df_ret)
    displayed_ids = df_ret["id"].tolist() if "id" in df_ret.columns else [int(r["id"]) for r in chats]

    # ---------- Manager toolbar (now AFTER table, uses fresh selection) ----------
    if manager_ui:
        st.session_state.setdefault("hist_view_ids", set())
        t1, t2, t3, t4 = st.columns([1.2, 1.5, 3.6, 5])
        with t1:
            if st.button("Select All"):
                st.session_state["hist_view_ids"] = set(map(int, displayed_ids))
                _rerun()
        with t2:
            if st.button("Clear Selection"):
                st.session_state["hist_view_ids"] = set()
                _rerun()
        with t3:
            sel_ids = list(st.session_state.get("hist_view_ids", set()))
            sel_rows = [r for r in chats if int(r["id"]) in sel_ids]
            disabled = (len(sel_rows) == 0)
            # --- Exports ---
            if PDF_EXPORT_ENABLED:
                c_md, c_pdf = st.columns(2, gap="small")
                with c_md:
                    export_md = st.button("Export selected (.md)", disabled=disabled, help="Raw Markdown for external MD→PDF tooling", key="btn_export_md")
                    if export_md and not disabled:
                        try:
                            md_bytes = _export_chats_to_markdown(project_name, sel_rows)
                            st.session_state["last_export_md"] = md_bytes
                            tsf = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                            st.session_state["last_export_md_name"] = f"{_safe_name(project_name)}_chats_{tsf}.md"
                            st.success("Markdown prepared. Use the Download button →")
                        except Exception as e:
                            st.error(f"Markdown export failed: {e}")
                with c_pdf:
                    export_pdf = st.button("Export selected (PDF)", disabled=disabled, help="Render Markdown/Math to PDF server-side", key="btn_export_pdf")
                    if export_pdf and not disabled:
                        pdf_bytes = _export_chats_to_pdf_rich(project_name, sel_rows)
                        if not pdf_bytes:
                            st.error("Could not render PDF. Ensure WeasyPrint is installed, or add xhtml2pdf / PyMuPDF as fallbacks.")
                        else:
                            st.session_state["last_export_pdf"] = pdf_bytes
                            tsf = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                            st.session_state["last_export_pdf_name"] = f"{_safe_name(project_name)}_chats_{tsf}.pdf"
                            st.success("PDF prepared. Use the Download button →")
            else:
                # Markdown-only export (PDF path disabled)
                export_md = st.button("Export selected (.md)", disabled=disabled, help="Raw Markdown for external MD→PDF tooling", key="btn_export_md")
                if export_md and not disabled:
                    try:
                        md_bytes = _export_chats_to_markdown(project_name, sel_rows)
                        st.session_state["last_export_md"] = md_bytes
                        tsf = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                        st.session_state["last_export_md_name"] = f"{_safe_name(project_name)}_chats_{tsf}.md"
                        st.success("Markdown prepared. Use the Download button →")
                    except Exception as e:
                        st.error(f"Markdown export failed: {e}")
        # Downloads
        if PDF_EXPORT_ENABLED:
            dlc1, dlc2 = t4.columns(2, gap="small")
            with dlc1:
                if st.session_state.get("last_export_md"):
                    st.download_button(
                        "Download .md",
                        data=st.session_state["last_export_md"],
                        file_name=st.session_state.get("last_export_md_name", "tarkash_chats.md"),
                        mime="text/markdown",
                        key="dl_hist_md",
                    )
            with dlc2:
                if st.session_state.get("last_export_pdf"):
                    st.download_button(
                        "Download PDF",
                        data=st.session_state["last_export_pdf"],
                        file_name=st.session_state.get("last_export_pdf_name", "tarkash_chats.pdf"),
                        mime="application/pdf",
                        key="dl_hist_pdf",
                    )
        else:
            # Markdown-only download button
            if st.session_state.get("last_export_md"):
                st.download_button(
                    "Download .md",
                    data=st.session_state["last_export_md"],
                    file_name=st.session_state.get("last_export_md_name", "tarkash_chats.md"),
                    mime="text/markdown",
                    key="dl_hist_md",
                )

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
            # Actions
            ac1, ac2 = st.columns([1, 1], gap="small")
            with ac1:
                _copy_button(answer, f"copy_{cid}")
            with ac2:
                if st.button("Delete", key=f"del_{cid}", help="Delete this chat"):
                    delete_chat(project_name, cid)
                    _rerun()
            # Content
            st.markdown("**You:**")
            st.markdown(prompt)
            st.markdown("**Assistant:**")
            render_response_with_latex(answer)
    st.markdown("</div>", unsafe_allow_html=True)  # close .hist-root


# =============================================================================
# Page & CSS
# =============================================================================

st.set_page_config(page_title=APP_TITLE, page_icon="💎", layout="wide")

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
        .pill{
          border:1px solid var(--ring);
          padding:.25rem .6rem;
          border-radius:999px; font-size:.8rem; color:var(--muted); background:white;
        }
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


        /* Tarkash glyph (top-left) */
        .tarkash-logo{
          position: fixed;
          top: 6px;
          left: 14px;
          z-index: 1000;
          display:flex; align-items:center; gap:.35rem;
          font-weight:600; color:var(--ink); opacity:.95;
          pointer-events:none;
        }
        .tarkash-logo svg{ width:28px; height:28px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

_inject_css()

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
        ui["project_root"] = project_root

        # auto-index controls
        auto_index_flag = st.checkbox(
            "Auto indexing",
            value=bool(ui.get("rag_auto_index", True)),
            help="First run creates index; later runs update (delta).",
        )
        ui["rag_auto_index"] = auto_index_flag

        # determine current indexing status ONCE for both buttons & status text
        if _rag_enabled:
            try:
                _st = index_status(project_root) or {}
                is_running = (_st.get("state") == "running")
            except Exception:
                _st, is_running = {}, False
        else:
            _st, is_running = {}, False

        # actions (Delta, Full, Stop) — honor Auto indexing and running state
        # Also disable when RAG is OFF
        disabled_manual = (not _rag_enabled) or bool(auto_index_flag or is_running)

        # Place the buttons inside a borderless container and distribute evenly
        st.markdown('<div class="index-actions">', unsafe_allow_html=True)
        btn_cols = 3 if is_running else 2
        cols = st.columns(btn_cols, gap="small")

        # --- Delta index ---
        with cols[0]:
            if st.button("Delta index", use_container_width=True, disabled=disabled_manual):
                with st.spinner("Delta indexing…"):
                    try:
                        cfg = load_rag(per_rag_path)                     # Path ok
                        emb_name = ui.get("embedding_model")
                        cfg.setdefault("embedder", {})
                        if emb_name:
                            cfg["embedder"]["selected_name"] = emb_name
                        cfg["chroma_dir"] = str(rag_index_dir)            # ensure per-project index dir
                        save_rag(per_rag_path, cfg)                        # Path ok
                        res = delta_index(project_root, per_rag_path)      # Path ok
                        st.success(f"Delta index complete. Updated chunks: {res.get('added')}")
                        st.caption(res)
                    except Exception as e:
                        st.error(f"Delta index failed: {e}")

        # --- Full reindex ---
        with cols[1]:
            if st.button("Full reindex", use_container_width=True, disabled=disabled_manual):
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
                        st.success(f"Reindex complete. Added chunks: {res.get('added')}")
                        st.caption(res)
                    except Exception as e:
                        st.error(f"Reindex failed: {e}")

        # --- Stop (only while running) ---
        if is_running:
            with cols[2]:
                if st.button("Stop indexing", use_container_width=True):
                    try:
                        st.warning("Stop requested; workers finish current files.")
                        request_stop(project_root)
                    except Exception as e:
                        st.error(f"Failed to request stop: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # status
        try:
            if _st.get("dirty"):
                st.markdown(
                    "<span style='color:#c00;font-size:0.85em'>Indexing status: DIRTY</span>",
                    unsafe_allow_html=True,
                )
            if _st.get("state") == "running":
                pf, tf = int(_st.get("processed_files", 0)), int(_st.get("total_files", 0) or 1)
                st.caption(f"Indexing… {pf}/{tf} files")
            elif not disabled_manual:
                st.caption("Indexer is idle.")
        except Exception:
            pass

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
                if st.button("Save RAG settings"):
                    save_rag(RAG_PATH, json.loads(rag_text))
                    st.success("Saved. Re-run app to apply.")
            with cB:
                if st.button("Reload from disk"):
                    _rerun()
        except Exception as e:
            st.error(f"Failed to load rag.json: {e}")

    # --- Models & embeddings ---
    with st.expander("Models & Embeddings", expanded=False):
        default_cfg = str(MODELS_JSON_PATH)
        models_config = st.text_input("Model config (--config)", value=ui.get("models_config", default_cfg))
        ui["models_config"] = models_config

        try:
            model_registry = ModelRegistry(models_config)
            models_list = model_registry.list()
            names = [m.name for m in models_list]
            chosen_model_name = ui.get("llm_model") or model_registry.default_name or (names[0] if names else "")
            chosen_model_name = st.selectbox(
                "Model (--model)",
                names,
                index=names.index(chosen_model_name) if chosen_model_name in names else 0,
            )
            ui["llm_model"] = chosen_model_name
            model = model_registry.get(chosen_model_name)
            st.caption(f"Provider: {model.provider}  •  Endpoint: {model.endpoint}")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()

        # Embedding model
        try:
            with open(models_config, "r", encoding="utf-8") as f:
                _models_cfg = json.load(f)
            emb_list = _models_cfg.get("embedding_models", []) or []
            emb_names = [e["name"] for e in emb_list]
            default_emb = ui.get("embedding_model") or _models_cfg.get("default_embedding_model") or (
                emb_names[0] if emb_names else ""
            )
        except Exception:
            emb_list, emb_names, default_emb = [], [], ""

        if emb_names:
            chosen_emb = st.selectbox(
                "Embedding model (RAG / indexing)",
                emb_names,
                index=emb_names.index(default_emb) if default_emb in emb_names else 0,
                disabled=not _rag_enabled,
                help=None if _rag_enabled else "Disabled while RAG is OFF",
            )
            if _rag_enabled:
                ui["embedding_model"] = chosen_emb
                st.session_state["embedding_model"] = chosen_emb
                try:
                    cfg = load_rag(RAG_PATH)
                    cfg.setdefault("embedder", {})
                    cfg["embedder"]["selected_name"] = chosen_emb
                    cfg["chroma_dir"] = str(rag_index_dir)
                    save_rag(RAG_PATH, cfg)
                except Exception:
                    pass
        else:
            st.info("No embedding models found in models.json.")

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
                if st.button("Refresh"):
                    _rerun()
            with c2:
                if st.button("Delete all"):
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

st.write("")
top = st.container()
with top:
    left, right = st.columns([3, 2], gap="large")
    with left:
        MODE_LABELS = ["Direct Chat", "LLM Tools", "Agent Plan & Run"]
        mode = st.radio("Mode", MODE_LABELS, horizontal=True, index=0, key="mode_radio")
    with right:
        col1, col2, col3 = st.columns(3)
        with col1:
            _adapter = getattr(st.session_state.get("agent"), "adapter", None)
            _has_stream = bool(getattr(_adapter, "chat_stream", None))
            can_stream = (mode == "Direct Chat") and _has_stream
            streaming = st.toggle(
                "Streaming",
                value=can_stream,
                disabled=not can_stream,
                help=(
                    "Streaming is only available in Direct Chat"
                    if mode != "Direct Chat"
                    else ("Adapter does not support streaming" if not _has_stream else None)
                ),
            )
            st.session_state["streaming_enabled"] = bool(streaming)
        with col2:
            # Default OFF; preserve prior choice if already set
            # Label shortened to "RAG" with a minimal help tooltip explaining usage.
            rag_on = st.toggle(
                "RAG",
                value=bool(st.session_state.get("rag_on", False)),
                help="Use this project's indexed documents in answers. "
                     "Set the Project name & root in the sidebar; indexing controls live there and enable when RAG is on."
            )
            st.session_state["rag_on"] = rag_on
        with col3:
            complex_planning = st.toggle(
                "Complex planning", value=False, disabled=(mode != "Agent Plan & Run")
            )

# Prompt + single primary action
st.markdown('<div class="card prompt-card">', unsafe_allow_html=True)
prompt = st.text_area("input_prompt", height=140, placeholder="Type your instruction…")
b1, b2, _ = st.columns([1, 1, 6])
with b1:
    main_label = "Stop" if st.session_state.get("running") else "Submit"
    action_clicked = st.button(main_label, type="primary", use_container_width=True, key="submit_stop_btn")
with b2:
    clear = st.button("Clear", use_container_width=True)
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

# Tabs: keep chat UI as-is in “Chat”, move history into its own tab
chat_tab, history_tab = st.tabs(["Chat", "History"])

# --- Chat tab: current response UI as before ---
with chat_tab:
    st.markdown("### Assistant Response")
    response_area = st.empty()

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
                msgs = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
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
                            _push_transient_turn(prompt, buf)
                        except Exception as _e:
                            st.error(f"[stream error] {_e}")
                    else:
                        resp = agent.adapter.chat(msgs)
                        answer = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
                        final_answer_text_for_history = answer
                        if not _maybe_render_dot_from_text(answer):
                            # render final text into Chat tab area
                            render_response_with_latex(answer)
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
                answer = agent.ask_once(
                    prompt, max_iters=int(st.session_state.get("max_iters", 5)), progress_cb=_progress_cb
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
                    # ensure rendering stays inside Chat tab
                    render_response_with_latex(final_answer_text_for_history)
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
                    plan = agent.analyze_and_plan(prompt)  # type: ignore[attr-defined]
                else:
                    plan = []
                    for sub in _keywordize(prompt, max_terms=8):
                        plan.append(
                            {
                                "tool": "rag_retrieve",
                                "args": {"query": sub, "top_k": 8},
                                "description": "Focused retrieval from sub-query",
                                "critical": False,
                            }
                        )
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
                        max_iters=int(st.session_state.get("max_iters", 5)),
                        analysis_only=bool(st.session_state.get("analysis_only", True)),
                        progress_cb=_progress_cb,
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
                    resp = agent.adapter.chat(
                        [
                            {"role": "system", "content": "Be concrete, cite code (file:line)."},
                            {"role": "user", "content": synth_prompt},
                        ]
                    )
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
                    # render inside Chat tab
                    render_response_with_latex(final_answer_text_for_history)
                    _push_transient_turn(prompt, final_answer_text_for_history)
                else:
                    results_json = json.dumps(steps, indent=2)
                    synth_prompt = intents._build_synth_prompt(prompt, results_json)
                    resp = agent.adapter.chat(
                        [
                            {"role": "system", "content": "Be concrete, cite code (file:line)."},
                            {"role": "user", "content": synth_prompt},
                        ]
                    )
                    alt = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or "(no content)"
                    final_answer_text_for_history = alt
                    # render inside Chat tab
                    render_response_with_latex(final_answer_text_for_history)
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
