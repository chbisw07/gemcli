#!/usr/bin/env python3
# streamlit_app.py — polished, professional layout (no functionality lost)

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

# --- Logging (existing helper) ---
import logging_setup

logging_setup.configure_logging()

# --- Optional visuals ---
import graphviz  # kept for DOT graphs

# --- Config & indexing helpers (existing modules) ---
try:
    from config_home import UI_SETTINGS_PATH, GLOBAL_RAG_PATH, project_rag_dir
    from indexing.settings import load as load_rag, save as save_rag
    from indexing.indexer import full_reindex, delta_index, request_stop, index_status
except Exception as e:
    # Fallback home if imports fail (keeps app usable)
    HOME = Path(os.path.expanduser("~")) / ".gencli"
    HOME.mkdir(parents=True, exist_ok=True)
    UI_SETTINGS_PATH = HOME / "ui_settings.json"
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

# --- Load .env from repo root (keeps behavior) ---
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logger.info("Environment loaded from .env at {}", str((Path(__file__).parent / ".env").resolve()))

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

def _list_charts(root: Path, charts_subdir: str = "charts") -> list[dict]:
    base = (root / charts_subdir).resolve()
    if not base.exists():
        return []
    out = []
    for name in os.listdir(base):
        if not name.lower().endswith(".png"):
            continue
        p = base / name
        try:
            stt = p.stat()
        except Exception:
            continue
        meta = None
        j = p.with_suffix(".json")
        if j.exists():
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                meta = None
        out.append(
            {
                "path": p,
                "rel": str(p.relative_to(root)),
                "name": name,
                "mtime": stt.st_mtime,
                "size": stt.st_size,
                "meta": meta,
            }
        )
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out

def render_chart_gallery(root: Path, charts_subdir="charts"):
    items = _list_charts(root, charts_subdir)
    st.caption(f"Folder: `{charts_subdir}` · {len(items)} image(s)")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        q = st.text_input("Search name", placeholder="filter…", label_visibility="collapsed")
    with c2:
        cols = st.slider("Columns", 2, 6, 4)
    with c3:
        limit = st.selectbox("Show", [12, 24, 48, 96, 200], index=1)
    with c4:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Name A→Z", "Name Z→A"])
    if sort_by == "Oldest":
        items = list(reversed(items))
    elif sort_by == "Name A→Z":
        items = sorted(items, key=lambda d: d["name"].lower())
    elif sort_by == "Name Z→A":
        items = sorted(items, key=lambda d: d["name"].lower(), reverse=True)
    if q:
        ql = q.lower()
        items = [it for it in items if ql in it["name"].lower() or ql in it["rel"].lower()]
    if not items:
        st.info("No charts yet. Use `draw_chart_csv` or `draw_chart_data`.")
        return
    grid = [items[i : i + cols] for i in range(0, min(len(items), limit), cols)]
    for row in grid:
        cols_list = st.columns(len(row))
        for col, it in zip(cols_list, row):
            with col:
                st.image(str(it["path"]), use_container_width=True)
                st.markdown(f"**{it['name']}**")
                st.caption(f"{_human_dt(it['mtime'])} · {it['size']/1024:.1f} KB")
                with st.expander("Details", expanded=False):
                    st.code(it["rel"], language="bash")
                    if it["meta"]:
                        st.json(it["meta"])
                with open(it["path"], "rb") as fh:
                    st.download_button(
                        "Download PNG",
                        fh.read(),
                        file_name=it["name"],
                        mime="image/png",
                        use_container_width=True,
                    )

def _load_ui_settings() -> dict:
    try:
        return json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_ui_settings(d: dict) -> None:
    try:
        UI_SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save UI settings: {}", e)

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
                return ast.literal_eval(candidate)
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
    """Light sub-query builder for local plan fallback."""
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

def _ensure_index_ready(project_root: str, rag_path: str, embedder_name: Optional[str], auto: bool):
    """Auto index (full / delta) when RAG is ON and auto indexing is enabled."""
    if not auto:
        return
    try:
        cfg = load_rag(rag_path)
        cfg.setdefault("embedder", {})
        if embedder_name:
            cfg["embedder"]["selected_name"] = embedder_name
        save_rag(rag_path, cfg)
    except Exception as e:
        st.warning(f"RAG config sync failed: {e}")

    rag_dir = project_rag_dir(project_root)
    try:
        exists_and_has_files = rag_dir.exists() and any(rag_dir.iterdir())
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
        .stButton>button[kind="primary"]{ background:var(--accent); color:white; border:none; }
        .stTextArea textarea{ line-height:1.4; }

        /* --- Tarkash logo (fixed, top-left in the same row as Deploy) --- */
        .tarkash-logo{
          position: fixed;
          top: 6px;               /* aligns with Streamlit top bar */
          left: 14px;
          z-index: 1000;
          display:flex; align-items:center; gap:.35rem;
          font-weight:600; color:var(--ink); opacity:.95;
          pointer-events:none;    /* don't block clicks on the toolbar */
        }
        .tarkash-logo svg{ width:18px; height:18px; }


        </style>
        """,
        unsafe_allow_html=True,
    )

_inject_css()

# --- Small Tarkash logo in toolbar row (replaces the large title card) ---
st.markdown(
    """
    <div class="tarkash-logo" title="Tarkash">
      <!-- simple quiver-with-arrows glyph -->
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="3" width="10" height="18" rx="2"></rect>
        <path d="M7.5 6 L6 7.5 L9 7.5 Z"></path>       <!-- arrowhead 1 -->
        <path d="M11.5 5 L10 6.5 L13 6.5 Z"></path>   <!-- arrowhead 2 -->
        <path d="M8 8 V18"></path>                    <!-- shaft 1 -->
        <path d="M12 7 V18"></path>                   <!-- shaft 2 -->
      </svg>
      <span>Tarkash</span>
    </div>
    """,
    unsafe_allow_html=True)

# =============================================================================
# Sidebar — grouped, minimal-scrolling
# =============================================================================

with st.sidebar:
    st.header("Configuration", anchor=False)

    # read cross-panel state early
    _mode_now = st.session_state.get("mode_radio", "Direct Chat")
    _rag_enabled = bool(st.session_state.get("rag_on", True))
    _is_direct_chat = (_mode_now == "Direct Chat")

    # Log level
    log_level = st.selectbox("Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
    logging_setup.set_level(log_level)

    ui = _load_ui_settings()

    # --- Project section ---
    with st.expander("Project", expanded=False):
        default_root = str(Path.cwd())
        project_root = st.text_input(
            "Project root (--root)",
            value=ui.get("project_root", default_root),
            disabled=not _rag_enabled,
            help=None if _rag_enabled else "Disabled while RAG is OFF",
        )
        ui["project_root"] = project_root

        auto_index_flag = st.checkbox(
            "Auto indexing",
            value=bool(ui.get("rag_auto_index", True)),
            help="First run creates index; later runs update (delta).",
        )
        ui["rag_auto_index"] = auto_index_flag

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reindex now (full)", use_container_width=True):
                with st.spinner("Reindexing…"):
                    try:
                        cfg = load_rag(str(GLOBAL_RAG_PATH))
                        emb_name = ui.get("embedding_model")
                        cfg.setdefault("embedder", {})
                        if emb_name:
                            cfg["embedder"]["selected_name"] = emb_name
                        save_rag(str(GLOBAL_RAG_PATH), cfg)
                        res = full_reindex(project_root, str(GLOBAL_RAG_PATH))
                        st.success(f"Reindex complete. Added chunks: {res.get('added')}")
                        st.caption(res)
                    except Exception as e:
                        st.error(f"Reindex failed: {e}")
        with c2:
            if st.button("Stop indexing (graceful)", use_container_width=True):
                try:
                    st.warning("Stop requested; workers finish current files.")
                    request_stop(project_root)
                except Exception as e:
                    st.error(f"Failed to request stop: {e}")

        try:
            _st = index_status(project_root) or {}
            if _st.get("dirty"):
                st.markdown(
                    "<span style='color:#c00;font-size:0.85em'>Indexing status: DIRTY</span>",
                    unsafe_allow_html=True,
                )
            if _st.get("state") == "running":
                pf, tf = int(_st.get("processed_files", 0)), int(_st.get("total_files", 0) or 1)
                st.caption(f"Indexing… {pf}/{tf} files")
        except Exception:
            pass

    # --- RAG settings ---
    with st.expander("RAG settings (global)", expanded=False):
        RAG_PATH = str(GLOBAL_RAG_PATH)
        try:
            cfg = load_rag(RAG_PATH)
            st.caption(f"Path: `{RAG_PATH}`")
            rag_text = st.text_area("rag.json", json.dumps(cfg, indent=2), height=320, key="rag_textarea")
            cA, cB = st.columns(2)
            with cA:
                if st.button("Save RAG settings"):
                    save_rag(RAG_PATH, json.loads(rag_text))
                    st.success("Saved. Re-run app to apply.")
            with cB:
                if st.button("Reload from disk"):
                    st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to load rag.json: {e}")

    # --- Models & embeddings ---
    with st.expander("Models & Embeddings", expanded=False):
        default_cfg = str(Path("data") / "models.json")
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
                    save_rag(RAG_PATH, cfg)
                except Exception:
                    pass
        else:
            st.info("No embedding models found in models.json.")

    # --- Execution options ---
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

    _save_ui_settings(ui)

    # --- Build runtime (on key change) ---
    rt_key = (project_root, enhanced, enable_tools, chosen_model_name)
    if "rt_key" not in st.session_state or st.session_state["rt_key"] != rt_key:
        tools = (
            EnhancedToolRegistry(project_root)
            if (enhanced and ENHANCED_TOOLS_AVAILABLE)
            else ToolRegistry(project_root)
        )
        agent = (
            ReasoningAgent(model, tools, enable_tools=enable_tools)
            if (enhanced and RAG_AGENT_AVAILABLE)
            else Agent(model, tools, enable_tools=enable_tools)
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
        with col2:
            rag_on = st.toggle("RAG (use project data)", value=True)
            st.session_state["rag_on"] = rag_on
        with col3:
            complex_planning = st.toggle(
                "Complex planning", value=False, disabled=(mode != "Agent Plan & Run")
            )

# Prompt + single primary action
st.markdown('<div class="card">', unsafe_allow_html=True)
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
    st.experimental_rerun()

# Interpret action button
if action_clicked and st.session_state.get("running"):
    st.session_state["cancel"] = True
    st.toast("Stopping…", icon="🛑")
    submit = False
else:
    submit = bool(action_clicked and not st.session_state.get("running"))

st.markdown("### Assistant Response")
response_area = st.empty()

# Optional tool visibility (non-Direct modes)
try:
    agent = st.session_state["agent"]
    if ENHANCED_TOOLS_AVAILABLE and isinstance(agent, ReasoningAgent) and mode != "Direct Chat":  # type: ignore
        with st.expander("Tools visible to the model (by route)", expanded=False):
            try:
                names = agent.allowed_tool_names()
                used = set(st.session_state.get("last_tools_used") or [])
                if names:
                    lines = [(f"* {n}" if n in used else f"  {n}") for n in names]
                    st.code("\n".join(lines), language="text")
                    if used:
                        st.caption("Legend: '*' = used in this run")
                else:
                    st.code("(none)", language="text")
            except Exception:
                st.caption("Tool list unavailable for this agent.")
except Exception:
    pass

# Chart gallery
if st.session_state.get("show_chart_gallery"):
    try:
        _root = Path(getattr(st.session_state["tools"], "root", Path.cwd()))
        with st.expander("📈 Local chart gallery", expanded=False):
            render_chart_gallery(_root, charts_subdir="charts")
    except Exception as _e:
        logger.warning("Chart gallery failed: {}", _e)

# =============================================================================
# Execution — identical behaviors, reorganized
# =============================================================================

if submit:
    st.session_state["running"] = True
    st.session_state["last_tools_used"] = set()
    st.session_state["show_chart_gallery"] = False

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    if rag_on:
        _ensure_index_ready(
            project_root, str(GLOBAL_RAG_PATH), st.session_state.get("embedding_model"), auto_index_flag
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
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                try:
                    if streaming and hasattr(agent.adapter, "chat_stream"):
                        placeholder = st.empty()
                        buf = ""
                        try:
                            for chunk in agent.adapter.chat_stream(msgs):
                                if not isinstance(chunk, str):
                                    continue
                                buf += chunk
                                s = _normalize_math_delimiters(_fix_common_latex_typos(_sanitize_latex_text(buf)))
                                placeholder.markdown(s)
                        except Exception as _e:
                            st.error(f"[stream error] {_e}")
                    else:
                        resp = agent.adapter.chat(msgs)
                        answer = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
                        if not _maybe_render_dot_from_text(answer):
                            render_response_with_latex(answer)
                except Exception as e:
                    st.error(f"[error] {e}")

        # ---------------------- LLM Tools ----------------------
        elif mode == "LLM Tools":
            prog_exp = st.expander("Run progress", expanded=False)
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
                    if tname in {"draw_chart_csv", "draw_chart_data"}:
                        st.session_state["show_chart_gallery"] = True
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
                    st.json(parsed)
            else:
                if not _maybe_render_dot_from_text(answer):
                    render_response_with_latex(answer)

        # ---------------------- Agent Plan & Run ----------------------
        else:
            prog_exp = st.expander("Run progress", expanded=True)
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
                tname = data.get("tool")
                if isinstance(tname, str) and tname:
                    used = set(st.session_state.get("last_tools_used") or set())
                    used.add(tname)
                    st.session_state["last_tools_used"] = used
                    if tname in {"draw_chart_csv", "draw_chart_data"}:
                        st.session_state["show_chart_gallery"] = True
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

            with st.expander("Plan (steps)", expanded=False):
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
                    # Minimal local executor (kept from previous version)
                    CTX_TOOLS = {
                        "rag_retrieve","read_file","list_files","search_code","scan_relevant_files","analyze_files",
                        "edu_detect_intent","edu_similar_questions","edu_question_paper","edu_explain",
                        "edu_extract_tables","edu_build_blueprint","find_related_files","analyze_code_structure",
                        "detect_errors","call_graph_for_function","analyze_function",
                    }
                    def _ctx_defaults():
                        return {"project_root": project_root, "rag_path": str(GLOBAL_RAG_PATH)}
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
                    f"RAG={'on' if rag_on else 'off'} · Steps={sum(1 for r in steps if r.get('success'))} ok / "
                    f"{sum(1 for r in steps if not r.get('success', False))} failed"
                )
                _rs = _rag_summary_from_steps(steps)
                if _rs:
                    st.caption(_rs)

                try:
                    cg_steps = [s for s in steps if s.get("tool") == "call_graph_for_function" and (s.get("result") or s.get("success"))]
                    if cg_steps:
                        with st.expander("Call graph (computed)", expanded=True):
                            for s in cg_steps[-2:]:
                                res = s.get("result") or {}
                                fn = res.get("function") or s.get("args", {}).get("function") or "<function>"
                                fpath = res.get("file") or s.get("args", {}).get("file_hint") or "<file>"
                                st.markdown(f"**Function:** `{fn}`  \n**File:** `{fpath}`")
                                calls = res.get("calls") or []
                                if calls:
                                    rows = [{"name": c.get("name"), "defined_in": c.get("defined_in")} for c in calls if isinstance(c, dict)]
                                    if rows:
                                        st.table(rows)
                                edges = res.get("edges") or []
                                if edges and isinstance(edges, list):
                                    txt = ", ".join(f"{e[0]}→{e[1]}" for e in edges[:15] if isinstance(e, (list, tuple)) and len(e) >= 2)
                                    if txt:
                                        st.caption(f"Edges: {txt}")
                                if isinstance(res.get("dot"), str):
                                    try:
                                        st.graphviz_chart(res["dot"])
                                    except Exception:
                                        pass
                except Exception:
                    pass

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
                    render_response_with_latex(final_answer.strip())
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
                    render_response_with_latex(alt)

                try:
                    used_tools = {s.get("tool") for s in steps if isinstance(s, dict) and s.get("tool")}
                    if used_tools:
                        st.session_state["last_tools_used"] = set(used_tools)
                    if any(t in {"draw_chart_csv", "draw_chart_data"} for t in used_tools):
                        st.session_state["show_chart_gallery"] = True
                except Exception:
                    pass

            except Exception as e:
                st.error(f"[error] {e}")

    except Exception as e:
        st.error(f"[fatal] {e}")
    finally:
        st.session_state["running"] = False
