#!/usr/bin/env python3
# streamlit_app.py

import json
import os
import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# --- Logging (global) ---
import logging_setup  # NEW
from loguru import logger
logging_setup.configure_logging()  # default INFO (can flip via env or UI below)

import streamlit as st
import graphviz  # pip install graphviz

# --- App config helpers (global config home) ---
# See config_home.py you created:
#   - UI_SETTINGS_PATH (e.g., ~/.gencli/ui_settings.json)
#   - GLOBAL_RAG_PATH  (e.g., ~/.gencli/rag.json)
try:
    from config_home import UI_SETTINGS_PATH, GLOBAL_RAG_PATH
    from config_home import project_rag_dir, GLOBAL_RAG_PATH, UI_SETTINGS_PATH  # already present if you used my file
    from indexing.settings import load as load_rag, save as save_rag
    from indexing.indexer import full_reindex, delta_index, request_stop, index_status
except Exception as e:
    # Soft fallback if helper not yet present; keeps app runnable
    HOME = Path(os.path.expanduser("~")) / ".gencli"
    HOME.mkdir(parents=True, exist_ok=True)
    UI_SETTINGS_PATH = HOME / "ui_settings.json"
    GLOBAL_RAG_PATH = HOME / "rag.json"
    logger.warning("Config/import fallback enabled: {}", e)

# --- Import your existing modules (unchanged) ---
from models import ModelRegistry
from tools.registry import ToolRegistry
from agent import Agent

# Enhanced (optional)
try:
    from tools.enhanced_registry import EnhancedToolRegistry
    from reasoning_agent import ReasoningAgent
    ENHANCED_AVAILABLE = True
except Exception as e:
    ENHANCED_AVAILABLE = False
    EnhancedToolRegistry = None
    ReasoningAgent = None  # type: ignore
    logger.info("Enhanced toolchain unavailable: {}", e)

import intents

# Force load from gemcli/.env no matter where you launch from
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logger.info("Environment loaded from .env at {}", str((Path(__file__).parent / ".env").resolve()))

# ---------- Small helpers ----------
APP_TITLE = "gemcli — Code Assistant"

EDIT_TOOLS = {
    "replace_in_file",
    "bulk_edit",
    "format_python_files",
    "rewrite_naive_open",
    "write_file",
}


def _load_models(config_path: str):
    logger.info("Loading models from config='{}'", config_path)
    registry = ModelRegistry(config_path)  # reads models.json
    models = registry.list()
    logger.info("Models loaded: {} (default='{}')", len(models), registry.default_name)
    for m in models:
        logger.debug(
            "Model: name='{}' provider='{}' endpoint='{}' model='{}' tags={}",
            m.name, m.provider, m.endpoint, m.model, m.tags
        )
    return registry, models


def _make_tools(project_root: str, enhanced: bool):
    logger.info("Building ToolRegistry (enhanced={}): root='{}'", enhanced, project_root)
    if enhanced and ENHANCED_AVAILABLE:
        return EnhancedToolRegistry(project_root)  # wraps ToolRegistry under the hood
    return ToolRegistry(project_root)


def _make_agent(model, tools, enhanced: bool, enable_tools: bool):
    logger.info(
        "Creating agent: class='{}' tools_enabled={} enhanced={}",
        "ReasoningAgent" if (enhanced and ENHANCED_AVAILABLE) else "Agent",
        enable_tools, enhanced
    )
    if enhanced and ENHANCED_AVAILABLE:
        return ReasoningAgent(model, tools, enable_tools=enable_tools)
    return Agent(model, tools, enable_tools=enable_tools)


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def _sanitize_latex_text(text: str) -> str:
    """
    Normalize unicode super/subscripts to LaTeX-ish, but
    DO NOT modify math delimiters ($...$, $$...$$, \(...\), \[...\]).
    """
    if not isinstance(text, str) or not text:
        return text
    # Map superscripts / subscripts (common ones)
    SUP = {
        "⁰":"^{0}","¹":"^{1}","²":"^{2}","³":"^{3}","⁴":"^{4}",
        "⁵":"^{5}","⁶":"^{6}","⁷":"^{7}","⁸":"^{8}","⁹":"^{9}",
        "⁺":"^{+}","⁻":"^{-}","⁽":"^{(}","⁾":"^{)}","ⁿ":"^{n}"
    }
    SUB = {
        "₀":"_{0}","₁":"_{1}","₂":"_{2}","₃":"_{3}","₄":"_{4}",
        "₅":"_{5}","₆":"_{6}","₇":"_{7}","₈":"_{8}","₉":"_{9}",
        "₊":"_{+}","₋":"_{-}","₍":"_{(}","₎":"_{)}"
    }
    return "".join(SUP.get(ch, SUB.get(ch, ch)) for ch in text)

def _fix_common_latex_typos(text: str) -> str:
    """
    Repair frequent model artifacts so MathJax renders cleanly:
      • unicode minus/en-dash → hyphen
      • '\left$' / '\right$' → '\left(' / '\right)'
      • collapse spaces after '^' / '_' (e.g., '^ 2' → '^2')
      • specific '\left$\\frac' → '\left(\\frac', and ensure exponent braces
    """
    if not text:
        return text
    s = text
    # minus/dash normalization
    s = s.replace("−", "-").replace("–", "-")
    # illegal \left$ … \right$
    s = s.replace(r"\left$", r"\left(").replace(r"\right$", r"\right)")
    # common variant: \left$\frac …  → \left(\frac …
    s = re.sub(r"\\left\$(\\frac|\\sqrt)", r"\\left(\1", s)
    # tighten '^ 2' / '_ 3'
    s = re.sub(r"\^\s+", "^", s)
    s = re.sub(r"_\s+", "_", s)
    # ensure braces on exponents of \frac{…}{…}
    s = re.sub(r"(\\frac\{[^}]+\}\{[^}]+\})\s*\^\s*(\d+)", r"\1^{\2}", s)
    return s

def _wrap_naked_tex(text: str) -> str:
    """
    Wrap obvious TeX sequences that lack delimiters.
    Heuristics:
      • a whole short paragraph that contains TeX tokens → $$…$$
      • otherwise, short TeX-y spans in-line → $…$
    Avoids touching code fences.
    """
    if not text:
        return text
    blocks = re.split(r"(```[\s\S]*?```)", text)
    out = []
    for b in blocks:
        if b.startswith("```"):
            out.append(b); continue
        def _wrap_para(p: str) -> str:
            ps = p.strip()
            if not ps or ("$" in ps) or (r"\[" in ps) or (r"\(" in ps):
                return p
            if re.search(r"\\(frac|sqrt|sum|int|lim|tan|sin|cos|log|theta|pi|alpha|beta|gamma)", ps):
                # short “mathy” line → display; longer → inline
                return f"$${ps}$$" if len(ps.split()) <= 12 else f"${ps}$"
            return p
        out.append("\n".join(_wrap_para(x) for x in b.split("\n")))
    return "".join(out)

def _normalize_math_delimiters(text: str) -> str:
    """
    Make model output renderable by Streamlit/MathJax:
      - \\[ ... \\]  -> $$ ... $$
      - \\( ... \\)  -> $ ... $
      - [  ...  ]    -> $$ ... $$   (only if content looks like TeX)
      - (  ...  )    -> $  ...  $   (only if content looks like TeX and short)
    """
    if not text:
        return text
    s = text
    # canonical LaTeX delimiters first
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)   # \[...\] -> $$...$$
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$",   s, flags=re.DOTALL)   # \(...\) -> $...$

    # square brackets containing TeX (heuristic: starts with backslash after optional space)
    def _bracket_to_display(m):
        inner = m.group(1)
        if re.match(r"^\s*\\[A-Za-z]", inner):
            return "$$" + inner.strip() + "$$"
        return "[" + inner + "]"
    s = re.sub(r"\[(.*?)\]", _bracket_to_display, s, flags=re.DOTALL)

    # parentheses containing TeX (heuristic: starts with backslash; keep short to avoid over-capture)
    def _paren_to_inline(m):
        inner = m.group(1)
        if len(inner) <= 180 and re.match(r"^\s*\\[A-Za-z]", inner):
            return "$" + inner.strip() + "$"
        return "(" + inner + ")"
    s = re.sub(r"\((.*?)\)", _paren_to_inline, s, flags=re.DOTALL)
    return s

# NOTE: Removed the KaTeX injector; Streamlit ships MathJax by default.

def render_response_with_latex(text: str):
    """
    Minimal, robust renderer:
      • Let Streamlit markdown render normally.
      • KaTeX auto-render turns $...$, $$...$$, \\(\\), \\[\\] into math (inline/block) without extra newlines.
      • Additionally, if a display block is multi-line, show a LaTeX code block beneath for readability.
    """
    import re
    from textwrap import dedent
    if not isinstance(text, str) or not text.strip():
        return
    s = _sanitize_latex_text(dedent(text))
    s = _fix_common_latex_typos(s)
    s = _normalize_math_delimiters(s)
    # 1) Render all content as normal markdown (Streamlit's MathJax handles $ / $$)
    st.markdown(s)
    # 2) For multi-line display equations, show LaTeX source as a code block (optional)
    for m in re.finditer(r"\$\$([\s\S]*?)\$\$", s, flags=re.DOTALL):
        expr = m.group(1) or m.group(2) or ""
        if ("\n" in expr) or ("\\\\" in expr) or ("\\begin{" in expr):
            st.code(expr.strip(), language="latex")

def _ensure_index_ready(project_root: str, rag_path: str, embedder_name: Optional[str], auto: bool):
    """
    If auto is True:
      - Mirror selected embedder into rag.json
      - If no per-project RAG store exists or is empty -> full_reindex
      - Else -> delta_index
    """
    if not auto:
        logger.info("Auto-index disabled; skipping index preflight")
        return

    logger.info("Auto-index preflight: project_root='{}' rag_path='{}' embedder='{}'", project_root, rag_path, embedder_name)

    # 1) Mirror embedder into rag.json so indexer resolves the same embedder the UI shows
    try:
        cfg = load_rag(rag_path)
        cfg.setdefault("embedder", {})
        if embedder_name:
            cfg["embedder"]["selected_name"] = embedder_name
        save_rag(rag_path, cfg)
        logger.info("Mirrored embedder into rag.json: selected_name='{}'", embedder_name)
    except Exception as e:
        logger.error("RAG config sync failed: {}", e)
        st.warning(f"RAG config sync failed: {e}")

    # 2) Decide: full vs delta based on per-project RAG dir contents
    rag_dir = project_rag_dir(project_root)
    try:
        exists_and_has_files = rag_dir.exists() and any(rag_dir.iterdir())
        logger.debug("RAG dir='{}' exists_and_has_files={}", str(rag_dir), exists_and_has_files)
    except Exception as e:
        exists_and_has_files = False
        logger.warning("Failed to inspect RAG dir='{}': {}", str(rag_dir), e)

    try:
        if not exists_and_has_files:
            logger.info("Auto-index → full_reindex() starting…")
            with st.spinner("Auto-index: creating fresh index…"):
                res = full_reindex(project_root, rag_path)
            logger.info("Auto-index full completed: {}", res)
            st.caption({"auto_index_full": res})
        else:
            logger.info("Auto-index → delta_index() starting…")
            with st.spinner("Auto-index: updating index (delta)…"):
                res = delta_index(project_root, rag_path)
            logger.info("Auto-index delta completed: {}", res)
            st.caption({"auto_index_delta": res})
    except Exception as e:
        logger.exception("Auto-index preflight failed: {}", e)
        st.warning(f"Auto-index preflight failed: {e}")

def _extract_diffs(tool_result: Any) -> List[Dict[str, str]]:
    """Normalize various tool outputs to a list of {path, diff} for preview."""
    diffs: List[Dict[str, str]] = []
    if isinstance(tool_result, dict):
        # format_python_files returns {path: {"diff": "..."}}
        if tool_result and all(
            isinstance(v, dict) and "diff" in v for v in tool_result.values()
        ):
            for path, meta in tool_result.items():
                diffs.append({"path": path, "diff": meta.get("diff", "")})
        # replace_in_file style
        elif "diff" in tool_result:
            diffs.append(
                {"path": tool_result.get("path", ""), "diff": tool_result.get("diff", "")}
            )
    elif isinstance(tool_result, list):
        # bulk_edit returns list of dicts with {path, diff}
        for item in tool_result:
            if isinstance(item, dict) and "diff" in item:
                diffs.append({"path": item.get("path", ""), "diff": item.get("diff", "")})
    logger.debug("Extracted {} diff(s) from tool result", len(diffs))
    return diffs


def _looks_like_tool_blob(text: str):
    """
    Try hard to extract a structured object (dict/list) from assistant text:
      • accepts 'toolname -> {...}' or plain JSON
      • tolerates code fences and leading/trailing prose
      • falls back to ast.literal_eval on Python reprs (single quotes/None/True/False)
    Returns parsed object or None.
    """
    if not text:
        return None

    # 1) If the model prefixed with "tool -> { ... }", strip the left part
    right = text.split("->", 1)[1].strip() if "->" in text else text.strip()

    # 2) Strip code fences if present
    if right.startswith("```"):
        right = right.strip("` \n")
        if right.lower().startswith("json"):
            right = right[4:].strip()

    # 3) Try direct JSON
    try:
        obj = json.loads(right)
        logger.debug("Parsed tool blob via json.loads")
        return obj
    except Exception:
        pass

    # 4) Try to locate the first {...} or [...] region inside the text
    m = re.search(r"(\{.*\}|\[.*\])", right, re.DOTALL)
    if m:
        candidate = m.group(1)

        # 4a) JSON first
        try:
            obj = json.loads(candidate)
            logger.debug("Parsed embedded JSON block from assistant content")
            return obj
        except Exception:
            pass

        # 4b) Last resort: Python literal (handles single quotes/None/etc.)
        try:
            obj = ast.literal_eval(candidate)
            logger.debug("Parsed tool blob via ast.literal_eval fallback")
            return obj
        except Exception:
            logger.debug("Failed to parse assistant content as tool blob")
            return None

    return None


def _maybe_render_dot_from_text(text: str) -> bool:
    """
    If the assistant returned only text, try to find a Graphviz DOT block
    (digraph G { ... }), normalize it, and render it. Returns True if rendered.
    """
    if not isinstance(text, str):
        return False

    # Find a DOT block (tolerate newlines and extra prose)
    m = re.search(r"(digraph\s+G\s*\{.*?\})", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return False

    dot = m.group(1)

    # Unescape common JSON escapes if the DOT came from a JSON string
    dot = dot.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')

    # Normalize any Unicode arrows that sometimes sneak in
    dot = dot.replace("→", "->").replace("⇒", "->")

    try:
        st.graphviz_chart(dot)  # streamlit renderer
        logger.debug("Rendered DOT graph from assistant text")
        return True
    except Exception as e:
        logger.warning("Graph render failed from text: {}", e)
        return False


def _maybe_render_graph(result: Any):
    """
    If the tool result contains a Graphviz 'dot' string, render it.
    """
    try:
        if isinstance(result, dict) and "dot" in result and isinstance(result["dot"], str):
            st.markdown("### Call Graph")
            st.graphviz_chart(result["dot"])
            logger.debug("Rendered DOT graph from tool result")
    except Exception as e:
        logger.warning("Graph render failed from tool result: {}", e)


# Nice visualizer for call-graph payloads
def _render_call_graph_payload(result: Any) -> bool:
    """
    If 'result' looks like output from call_graph_for_function, render a polished view
    and return True. Otherwise return False so caller can fall back to default.
    """
    if not isinstance(result, dict):
        return False
    if not (
        ("calls" in result and isinstance(result["calls"], list))
        or ("dot" in result and isinstance(result["dot"], str))
    ):
        return False

    # Header
    fn = result.get("function", "<function>")
    file_ = result.get("file", "<file>")
    st.markdown("### Call graph")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**Function:** `{fn}`")
    with c2:
        st.markdown(f"**Defined in:** `{file_}`")

    # Calls table
    calls = result.get("calls") or []
    if calls:
        st.markdown("#### Direct callees")
        rows = [{"name": c.get("name"), "defined_in": c.get("defined_in")} for c in calls]
        st.table(rows)
        logger.debug("Rendered call graph table with {} rows", len(rows))

    # Graph (DOT)
    _maybe_render_graph(result)
    return True


# ---------- UI settings (global) ----------
def _load_ui_settings() -> dict:
    try:
        d = json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
        logger.debug("Loaded UI settings from '{}'", str(UI_SETTINGS_PATH))
        return d
    except Exception as e:
        logger.info("UI settings load failed (will use defaults): {}", e)
        return {}


def _save_ui_settings(d: dict) -> None:
    try:
        UI_SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
        logger.debug("Saved UI settings to '{}'", str(UI_SETTINGS_PATH))
    except Exception as e:
        logger.warning("Failed to save UI settings: {}", e)


# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
logger.info("Streamlit app started: {}", APP_TITLE)

with st.sidebar:
    st.subheader("Configuration")

    # ---- Log level control (UI) ----
    log_level = st.selectbox("Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0, help="Affects both console and file sinks.")
    logging_setup.set_level(log_level)
    logger.info("Log level set via UI → {}", log_level)

    ui = _load_ui_settings()

    # Paths
    default_root = str(Path.cwd().parent / "scv2" / "backend")
    project_root = st.text_input("Project root (--root)", value=default_root)
    logger.info("Project root selected: '{}'", project_root)

    # -----------------------------
    # RAG / Indexing controls (global rag.json)
    # -----------------------------
    from indexing.settings import load as load_rag, save as save_rag
    from indexing.indexer import full_reindex, delta_index
    from indexing.retriever import retrieve

    RAG_PATH = str(GLOBAL_RAG_PATH)  # global (until per-project rag.json is introduced)
    auto_index_flag = st.checkbox(
        "Auto indexing",
        value=bool(ui.get("rag_auto_index", True)),
        help="Index on first use; delta thereafter.",
        key="rag_auto_index",
    )
    ui["rag_auto_index"] = st.session_state["rag_auto_index"]
    logger.info("Auto indexing: {}", bool(ui["rag_auto_index"]))

    # Graceful STOP control
    if st.button("Stop indexing (graceful)"):
        try:
            res = request_stop(project_root)
            st.warning("Stop requested. Current workers will finish; no new files will start.")
            logger.warning("User requested STOP: {}", res)
        except Exception as e:
            st.error(f"Failed to request stop: {e}")

    # (Optional) show last known status / dirty flag
    try:
        _st = index_status(project_root) or {}
        if _st.get("dirty"):
            st.markdown("<span style='color:#c00;font-size:0.85em'>Indexing status: DIRTY</span>", unsafe_allow_html=True)
        if _st.get("state") == "running":
            pf, tf = int(_st.get("processed_files", 0)), int(_st.get("total_files", 0) or 1)
            st.caption(f"Indexing… {pf}/{tf} files")
    except Exception as e:
        logger.debug("index_status read failed: {}", e)

    # Manual full reindex
    if st.button("Reindex now (full)"):
        with st.spinner("Reindexing…"):
            try:
                # Mirror current embedder choice into rag.json before indexing
                cfg = load_rag(RAG_PATH)
                emb_name = ui.get("embedding_model")
                cfg.setdefault("embedder", {})
                if emb_name:
                    cfg["embedder"]["selected_name"] = emb_name
                save_rag(RAG_PATH, cfg)
                logger.info("Manual full reindex triggered (embedder='{}')", emb_name)

                res = full_reindex(project_root, RAG_PATH)
                if res.get("dirty"):
                    st.markdown("<span style='color:#c00;font-size:0.85em'>Indexing finished early (DIRTY). Partial index is usable.</span>", unsafe_allow_html=True)
                else:
                    st.success(f"Reindex complete. Added chunks: {res.get('added')}")
                st.caption(res)
                logger.info("Manual full reindex completed: {}", res)
            except Exception as e:
                st.error(f"Reindex failed: {e}")
                logger.exception("Manual full reindex failed: {}", e)

    # Edit RAG settings (global)
    with st.expander("Edit RAG settings (global)", expanded=False):
        try:
            cfg = load_rag(RAG_PATH)
            logger.debug("Loaded rag.json from '{}'", RAG_PATH)
        except Exception as e:
            cfg = None
            st.error(f"Failed to load rag.json: {e}")
            logger.error("Failed to load rag.json: {}", e)
        if cfg is not None:
            st.caption(f"Path: `{RAG_PATH}`")
            rag_text = st.text_area(
                "rag.json", json.dumps(cfg, indent=2), height=320, key="rag_textarea"
            )
            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("Save RAG settings"):
                    try:
                        save_rag(RAG_PATH, json.loads(rag_text))
                        st.success("Saved. Re-run app to apply.")
                        logger.info("rag.json saved")
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                        logger.error("rag.json save failed: {}", e)
            with colB:
                if st.button("Reload from disk"):
                    logger.info("Rerun requested (reload rag.json)")
                    st.experimental_rerun()

    # Model config file
    default_cfg = str(Path("data") / "models.json")
    models_config = st.text_input("Model config (--config)", value=default_cfg)
    logger.info("Model config path selected: '{}'", models_config)

    # Load LLM models (from models.json) and select one
    try:
        model_registry, models_list = _load_models(models_config)
        model_names = [m.name for m in models_list]
        default_model_name = (
            ui.get("llm_model")
            or model_registry.default_name
            or (model_names[0] if model_names else "")
        )
        chosen_model_name = st.selectbox(
            "Model (--model)",
            options=model_names,
            index=model_names.index(default_model_name)
            if default_model_name in model_names
            else 0,
        )
        ui["llm_model"] = chosen_model_name
        model = model_registry.get(chosen_model_name)
        st.caption(f"Provider: {model.provider}  •  Endpoint: {model.endpoint}")
        logger.info(
            "Model selected: name='{}' provider='{}' endpoint='{}' resolved='{}'",
            model.name, model.provider, model.endpoint, model.resolved_model()
        )
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        logger.exception("Failed to load models: {}", e)
        st.stop()

    # Embedding selector (from data/models.json → embedding_models)
    try:
        with open(models_config, "r", encoding="utf-8") as _f:
            _models_cfg = json.load(_f)
        emb_list = _models_cfg.get("embedding_models", []) or []
        emb_names = [e["name"] for e in emb_list]
        default_emb_name = (
            ui.get("embedding_model")
            or _models_cfg.get("default_embedding_model")
            or (emb_names[0] if emb_names else "")
        )
    except Exception as e:
        emb_list, emb_names, default_emb_name = [], [], ""
        logger.warning("Embedding models not found in '{}': {}", models_config, e)

    if emb_names:
        chosen_emb_name = st.selectbox(
            "Embedding model (RAG / indexing)",
            options=emb_names,
            index=emb_names.index(default_emb_name)
            if default_emb_name in emb_names
            else 0,
            help="Used by Chroma via OpenAI-compatible /v1/embeddings.",
        )
        ui["embedding_model"] = chosen_emb_name

        # Also mirror into rag.json immediately so indexer sees it even without reindex click
        try:
            cfg = load_rag(RAG_PATH)
            cfg.setdefault("embedder", {})
            cfg["embedder"]["selected_name"] = chosen_emb_name
            save_rag(RAG_PATH, cfg)
            logger.info("Embedder selected: '{}'", chosen_emb_name)
            try:
                emb_entry = next(e for e in emb_list if e["name"] == chosen_emb_name)
                logger.info(
                    "Embedder details: provider='{}' endpoint='{}'",
                    emb_entry.get('provider','?'), emb_entry.get('endpoint','?')
                )
            except Exception:
                pass
        except Exception as _e:
            logger.error("Failed to mirror embedder into rag.json: {}", _e)
            pass
    else:
        st.info("No embedding models found in models.json.")
        logger.info("No embedding models available in config")

    # Flags
    enable_tools = st.checkbox("Enable tools (default on)", value=True)
    enhanced = st.checkbox(
        "Enhanced mode (planning, extra tools)",
        value=True if ENHANCED_AVAILABLE else False,
        disabled=not ENHANCED_AVAILABLE,
    )
    plan_mode = st.checkbox(
        "Use planning for complex tasks",
        value=False,
        disabled=not ENHANCED_AVAILABLE,
    )
    logger.info("Flags: enable_tools={} enhanced={} plan_mode={}", enable_tools, enhanced, plan_mode)

    max_iters = st.number_input(
        "Max tool iters (--max-iters)",
        min_value=1,
        max_value=20,
        value=5 if ENHANCED_AVAILABLE else 3,
        step=1,
    )
    analysis_only = st.checkbox("Analysis-only (no writes)", value=bool(ui.get("analysis_only", True)))
    ui["analysis_only"] = analysis_only
    logger.info("Execution params: max_iters={} analysis_only={}", max_iters, analysis_only)

    callgraph_depth = st.number_input(
        "Call graph depth (0 = full)", min_value=0, max_value=10, value=int(ui.get("callgraph_depth", 3)), step=1
    )
    st.session_state["callgraph_depth"] = callgraph_depth
    ui["callgraph_depth"] = callgraph_depth
    logger.info("Call graph depth set to {}", callgraph_depth)

    # Persist UI choices
    _save_ui_settings(ui)

    # Build runtime objects (once)
    rt_key = (project_root, enhanced, enable_tools, chosen_model_name)
    if "rt_built" not in st.session_state or st.session_state.get("rt_key") != rt_key:
        logger.info("Building runtime objects (new session key={})", rt_key)
        tools = _make_tools(project_root, enhanced)
        agent = _make_agent(model, tools, enhanced, enable_tools)
        st.session_state["tools"] = tools
        st.session_state["agent"] = agent
        st.session_state["rt_key"] = rt_key
        st.session_state["last_plan"] = None
        st.session_state["last_results"] = None
        st.session_state["last_edit_payloads"] = []  # for Apply step
        st.session_state["log_lines"] = []

st.write("")

# ------------ Prompt & controls ------------
colL, colR = st.columns([2, 1], gap="large")

with colL:
    prompt = st.text_area(
        "Your high-level instruction / prompt",
        height=140,
        placeholder="e.g., Rewrite open() usages under app/api (dry run)",
    )
    run_col1, run_col2, run_col3 = st.columns([1, 1, 1])
    with run_col1:
        do_plan = st.button("Plan (show steps)", use_container_width=True)
    with run_col2:
        do_execute = st.button("Run", use_container_width=True)
    with run_col3:
        clear_log = st.button("Clear log", use_container_width=True)

with colR:
    st.markdown("#### Options")
    # Removed the 'Force preview' UI; we infer dry-run from 'analysis_only'
    preview_only = False
    show_raw_json = st.checkbox("Show raw JSON/tool outputs", value=False)

# ------------ Log window ------------
st.markdown("### Activity / Thinking")
if clear_log:
    st.session_state["log_lines"] = []
    logger.info("UI log cleared by user")
log_container = st.container(border=True)
with log_container:
    for line in st.session_state["log_lines"]:
        st.markdown(line)


def log(line: str):
    st.session_state["log_lines"].append(line)
    with log_container:
        st.markdown(line)
    # Mirror to Loguru as well:
    logger.info("[UI] {}", line)


# ------------ Planning ------------
agent = st.session_state["agent"]
tools = st.session_state["tools"]


def render_plan(plan_steps: List[Dict[str, Any]]):
    st.markdown("### Execution Plan")
    if not plan_steps:
        st.info("No plan.")
        logger.info("No plan generated")
        return
    logger.info("Rendering plan with {} step(s)", len(plan_steps))
    for i, step in enumerate(plan_steps, 1):
        with st.expander(
            f"Step {i}: {step.get('tool','(no tool)')} — {step.get('description','')}",
            expanded=False,
        ):
            st.code(_pretty(step), language="json")


def execute_plan(plan_steps: List[Dict[str, Any]], analysis_only: bool = False):
    """
    Execute planned steps and collect previews/diffs for edits.
    """
    results = []
    applied_payloads = []  # track {tool,args} for potential Apply
    discovered_files: List[str] = []  # accumulate file paths from discovery steps
    effective_preview = preview_only or analysis_only  # force dry-run if analysis_only
    logger.info("Executing plan: steps={} analysis_only={} preview={}", len(plan_steps or []), analysis_only, effective_preview)

    # --- execution-time web fallback (UI runner) ---
    def _grounding_signal_from(out: Any) -> dict:
        pdf = tot = 0
        chunks = out.get("chunks") if isinstance(out, dict) else None
        if isinstance(chunks, list):
            for c in chunks:
                md = (c.get("metadata") or {}) if isinstance(c, dict) else {}
                if isinstance(md, dict):
                    fp = (md.get("file_path") or "").lower()
                    ct = (md.get("chunk_type") or "").lower()
                    if fp.endswith(".pdf") or ct.startswith("pdf"):
                        pdf += 1
                    tot += 1
        ratio = (pdf / max(1, tot))
        return {"pdf_chunks": pdf, "total_chunks": tot, "pdf_ratio": ratio}
    _web_fetch_batch: List[Dict[str, Any]] = []
    _MIN_CHUNKS, _MIN_RATIO = 6, 0.60
    _web_fallback_done = False

    for i, step in enumerate(plan_steps, 1):
        if "error" in step:
            msg = f"❌ Plan step {i} error: {step['error']}"
            log(msg)
            logger.error("Plan step {} error: {}", i, step["error"])
            results.append({"step": i, "error": step["error"], "success": False})
            continue

        tool_name = step.get("tool")
        args = dict(step.get("args", {}))
        desc = step.get("description", "")
        logger.info("Plan step {}: tool='{}' desc='{}' args={}", i, tool_name, desc, args)

        # If we already fetched web pages via automatic fallback and this is an edu_* tool,
        # forward them so edu_tools can append web context *after* PDF snippets.
        if isinstance(tool_name, str) and tool_name.startswith("edu_"):
            try:
                if _web_fetch_batch and "web_fetch_batch" not in args:
                    args["web_fetch_batch"] = _web_fetch_batch
            except Exception:
                pass

        # Impose preview for edit-capable tools
        if effective_preview and tool_name in EDIT_TOOLS:
            if tool_name == "replace_in_file":
                args.setdefault("dry_run", True)
                args.setdefault("backup", True)
            elif tool_name == "bulk_edit":
                args.setdefault("dry_run", True)
                args.setdefault("backup", True)
            elif tool_name == "format_python_files":
                args.setdefault("dry_run", True)
            elif tool_name == "rewrite_naive_open":
                args.setdefault("dry_run", True)
                args.setdefault("backup", True)
            elif tool_name == "write_file":
                pass
            if analysis_only:
                args["_analysis_only_blocked"] = True
            logger.debug("Preview enforced for edit tool='{}' args={}", tool_name, args)

        try:
            log(f"🛠️ Executing step {i}: **{tool_name}** with args `{args}`")

            # Inject UI depth into call_graph_for_function
            if tool_name == "call_graph_for_function":
                args.setdefault("depth", st.session_state.get("callgraph_depth", 3))
                logger.debug("call_graph depth set to {}", args["depth"])

            # Opportunistic binding for analyze/detect_errors
            if tool_name in {"analyze_code_structure", "detect_errors"}:
                p = args.get("path")
                if isinstance(p, str) and (
                    not p
                    or "path/to" in p
                    or p.strip() in {"<>", "<file>", "<path>", "<BIND:search_code.file0>"}
                ):
                    if discovered_files:
                        args["path"] = discovered_files[0]
                        logger.debug("Bound analyze/detect path to '{}'", args["path"])

            res = tools.call(tool_name, **args)
            results.append(
                {
                    "step": i,
                    "tool": tool_name,
                    "args": args,
                    "result": res,
                    "success": True,
                    "description": desc,
                }
            )
            logger.info("Tool '{}' executed OK (step {})", tool_name, i)

            # Collect discovered files from search results (best-effort)
            if tool_name == "search_code" and isinstance(res, list):
                for r in res:
                    f = r.get("file") if isinstance(r, dict) else None
                    if isinstance(f, str):
                        discovered_files.append(f)
                logger.debug("Discovered {} file(s) from search_code", len(discovered_files))

            # Record applicable edits (for later Apply), but not in analysis-only mode
            if tool_name in EDIT_TOOLS and not analysis_only:
                applied_payloads.append({"tool": tool_name, "args": args})

            # Stream diffs if present
            diffs = _extract_diffs(res)
            for d in diffs:
                st.markdown(f"**File:** `{d['path']}`")
                st.code(d["diff"] or "(no diff)", language="diff")

            # Visualize call-graph payloads cleanly (if present)
            _render_call_graph_payload(res)
            _maybe_render_graph(res)

            # --- Automatic web fallback (once) when RAG grounding looks weak ---
            if (not _web_fallback_done) and isinstance(res, dict) and isinstance(res.get("chunks"), list):
                sig = _grounding_signal_from(res)
                weak = (sig["total_chunks"] < _MIN_CHUNKS) or (sig["pdf_ratio"] < _MIN_RATIO)
                is_edu = isinstance(tool_name, str) and tool_name.startswith("edu_")
                # Confirm web tools are registered in this ToolRegistry
                available = set(getattr(tools, "tools", {}).keys()) if hasattr(tools, "tools") else set()
                web_ok = {"web_search", "web_fetch"}.issubset(available)
                if weak and web_ok and not is_edu:
                    # Use the step query/topic if available, otherwise fall back to the global prompt
                    q = (args.get("query") or args.get("topic") or desc or prompt or "").strip()
                    if q:
                        try:
                            ws = tools.call("web_search", query=q, max_results=5)
                            results.append({
                                "step": f"{i}.a",
                                "tool": "web_search",
                                "description": f"Automatic web fallback for weak grounding (pdf_ratio={sig['pdf_ratio']:.2f}, total_chunks={sig['total_chunks']})",
                                "args": {"query": q, "max_results": 5},
                                "result": ws, "success": True
                            })
                            urls = [r.get("url") for r in (ws.get("results") or []) if isinstance(r, dict)]
                            urls = [u for u in urls if u][:2]
                            for u in urls:
                                wf = tools.call("web_fetch", url=u, max_chars=60000)
                                _web_fetch_batch.append(wf)
                                results.append({
                                    "step": f"{i}.b", "tool": "web_fetch",
                                    "description": "Fetch page for fallback grounding",
                                    "args": {"url": u, "max_chars": 60000},
                                    "result": wf, "success": True
                                })
                            logger.info("UI execute_plan: auto web fallback injected urls={}", urls)
                            _web_fallback_done = True
                        except Exception as _e:
                            logger.warning("UI execute_plan: web fallback failed (ignored): {}", _e)

            if show_raw_json:
                with st.expander(f"Raw result (step {i})"):
                    st.code(_pretty(res), language="json")

        except Exception as e:
            log(f"❌ Step {i} failed: {e}")
            logger.exception("Step {} failed: {}", i, e)
            results.append(
                {
                    "step": i,
                    "tool": tool_name,
                    "args": args,
                    "error": str(e),
                    "success": False,
                    "description": desc,
                }
            )

    st.session_state["last_results"] = results
    st.session_state["last_edit_payloads"] = applied_payloads
    logger.info("Plan execution finished: steps={} success={} fail={}",
                len(results),
                sum(1 for r in results if r.get("success")),
                sum(1 for r in results if not r.get("success")))
    return results


# ------------ Main actions ------------
if do_execute:
    if prompt.strip():
        logger.info("Run clicked: prompt='{}' plan_mode={} enhanced={} analysis_only={}",
                    prompt.strip()[:200], bool(st.session_state.get("rag_auto_index", True)) and False,  # placeholder
                    ENHANCED_AVAILABLE, bool(st.session_state.get("analysis_only", True)))
        # Before planning/asking, make sure index is ready if user enabled auto indexing
        try:
            _ensure_index_ready(
                project_root=project_root,
                rag_path=str(GLOBAL_RAG_PATH),
                embedder_name=st.session_state.get("embedding_model"),
                auto=bool(st.session_state.get("rag_auto_index", True)),
            )
        except Exception as e:
            logger.warning("Auto-index preflight skipped due to error: {}", e)
            st.warning(f"Auto-index preflight skipped due to error: {e}")

        # Two modes:
        # 1) Enhanced + plan_mode → plan then execute plan (step-by-step visibility)
        # 2) Standard/simple → intercept call-graph queries, otherwise ask_once
        if plan_mode and ENHANCED_AVAILABLE and isinstance(agent, ReasoningAgent):  # type: ignore
            if not st.session_state.get("last_plan"):
                log("🧭 Creating plan (no existing plan found)...")
                logger.info("Planning…")
                st.session_state["last_plan"] = agent.analyze_and_plan(prompt)
            render_plan(st.session_state["last_plan"])
            st.markdown("---")
            # --- Router debug (ReasoningAgent only) ---
            try:
                if ENHANCED_AVAILABLE and isinstance(agent, ReasoningAgent):  # type: ignore
                    info = agent.router_info() or {}
                    with st.expander("Router debug", expanded=False):
                        st.markdown(f"**Route:** `{info.get('route', '?')}`")
                        scores = info.get("scores") or {}
                        if scores:
                            st.table([{"bucket": k, "score": round(v, 3)} for k, v in scores.items()])
                        topk = info.get("top_k")
                        if topk is not None:
                            st.caption(f"Top-K: {topk}")
            except Exception as _e:
                logger.warning("Router debug panel failed: {}", _e)

            try:
                if ENHANCED_AVAILABLE and isinstance(agent, ReasoningAgent):  # type: ignore
                    with st.expander("Tools visible to the model (by route)", expanded=False):
                        names = agent.allowed_tool_names()  # route-aware, sorted
                        if names:
                            st.code("\n".join(names), language="text")
                        else:
                            st.caption("No tools (unlikely).")
            except Exception as _e:
                logger.warning("Toolbox panel failed: {}", _e)

            log("🚀 Executing plan...")
            logger.info("Executing planned steps…")
            results = execute_plan(st.session_state["last_plan"], analysis_only=analysis_only)

            # ---------- Final Report (natural language) ----------
            report_text = None
            try:
                results_json = json.dumps(results, indent=2)
                synth_prompt = intents._build_synth_prompt(prompt, results_json)
                logger.debug("Synth prompt intent='{}'", intents._detect_intent(prompt))
                resp = agent.adapter.chat([
                    {"role": "system", "content": "Be concrete, cite code (file:line) with short excerpts. No tool calls."},
                    {"role": "user", "content": synth_prompt},
                ])
                report_text = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
                logger.info("Final report synthesized (chars={})", len(report_text or ""))
            except Exception as e:
                report_text = f"(Failed to synthesize final report: {e})"
                logger.error("Final report synthesis failed: {}", e)

            if isinstance(report_text, str) and report_text.strip():
                st.markdown("### Final Report")
                render_response_with_latex(report_text.strip())

        else:
            # 🔎 Intercept single-turn call-graph requests so we can honor UI depth
            cg_match = re.search(
                r"(?:call\s*graph|callgraph).*(?:of|for)\s+([A-Za-z_]\w*)\s*\(\)?",
                prompt,
                re.IGNORECASE,
            )
            if cg_match:
                fn = cg_match.group(1)
                depth = st.session_state.get("callgraph_depth", 3)
                log(f"🔎 Direct call graph for `{fn}` (depth={depth})")
                logger.info("Direct call_graph_for_function: function='{}' depth={}", fn, depth)
                try:
                    res = tools.call("call_graph_for_function", function=fn, depth=depth)
                    st.markdown("### Assistant Response")
                    _render_call_graph_payload(res) or _maybe_render_graph(res)
                    if show_raw_json:
                        with st.expander("Raw tool JSON"):
                            st.code(_pretty(res), language="json")
                except Exception as e:
                    st.error(f"call_graph_for_function failed: {e}")
                    logger.exception("call_graph_for_function failed: {}", e)
                st.stop()  # prevent falling through to ask_once

            # 💬 Fallback: standard single-turn ask
            log("💬 Running single-turn ask (agent.ask_once)...")
            logger.info("ask_once starting… model='{}'", model.resolved_model())
            try:
                if ENHANCED_AVAILABLE and isinstance(agent, ReasoningAgent) and plan_mode:
                    answer = agent.ask_with_planning(
                        prompt, max_iters=int(max_iters), analysis_only=analysis_only
                    )
                else:
                    answer = agent.ask_once(prompt, max_iters=int(max_iters))
            except Exception as e:
                logger.exception("ask_once failed: {}", e)
                answer = f"[error] {e}"
            st.markdown("### Assistant Response")

            parsed = _looks_like_tool_blob(answer)
            if parsed is not None:
                logger.debug("Assistant responded with tool JSON; rendering specialized view")
                # If it's a tool call for call_graph_for_function, inject UI depth before rendering
                if isinstance(parsed, dict) and parsed.get("tool") == "call_graph_for_function":
                    parsed.setdefault("args", {}).setdefault(
                        "depth", st.session_state.get("callgraph_depth", 3)
                    )

                diffs = _extract_diffs(parsed)
                if diffs:
                    st.markdown("### Preview (diff)")
                    for d in diffs:
                        st.markdown(f"**File:** `{d['path']}`")
                        st.code(d["diff"] or "(no diff)", language="diff")

                if not _render_call_graph_payload(parsed):
                    _maybe_render_graph(parsed)

                if show_raw_json:
                    with st.expander("Raw tool JSON"):
                        st.code(_pretty(parsed), language="json")
            else:
                if not _maybe_render_dot_from_text(answer):
                    render_response_with_latex(answer)
                if show_raw_json:
                    with st.expander("Raw text"):
                        st.code(answer)
    else:
        st.warning("Please enter a prompt.")
        logger.info("Run clicked without a prompt")

# (end)
