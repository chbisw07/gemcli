#!/usr/bin/env python3
# streamlit_app.py

import json
import os
import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

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
    from indexing.indexer import full_reindex, delta_index
except Exception:
    # Soft fallback if helper not yet present; keeps app runnable
    HOME = Path(os.path.expanduser("~")) / ".gencli"
    HOME.mkdir(parents=True, exist_ok=True)
    UI_SETTINGS_PATH = HOME / "ui_settings.json"
    GLOBAL_RAG_PATH = HOME / "rag.json"

# --- Import your existing modules (unchanged) ---
from models import ModelRegistry  # :contentReference[oaicite:4]{index=4}
from tools.registry import ToolRegistry
from agent import Agent

# Enhanced (optional)
try:
    from tools.enhanced_registry import EnhancedToolRegistry  # :contentReference[oaicite:5]{index=5}
    from reasoning_agent import ReasoningAgent
    ENHANCED_AVAILABLE = True
except Exception:
    ENHANCED_AVAILABLE = False
    EnhancedToolRegistry = None
    ReasoningAgent = None  # type: ignore

import intents

# Force load from gemcli/.env no matter where you launch from
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

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
    registry = ModelRegistry(config_path)  # reads models.json
    return registry, registry.list()


def _make_tools(project_root: str, enhanced: bool):
    if enhanced and ENHANCED_AVAILABLE:
        return EnhancedToolRegistry(project_root)  # wraps ToolRegistry under the hood
    return ToolRegistry(project_root)


def _make_agent(model, tools, enhanced: bool, enable_tools: bool):
    if enhanced and ENHANCED_AVAILABLE:
        return ReasoningAgent(model, tools, enable_tools=enable_tools)
    return Agent(model, tools, enable_tools=enable_tools)


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def _ensure_index_ready(project_root: str, rag_path: str, embedder_name: Optional[str], auto: bool):
    """
    If auto is True:
      - Mirror selected embedder into rag.json
      - If no per-project RAG store exists or is empty -> full_reindex
      - Else -> delta_index
    """
    if not auto:
        return

    # 1) Mirror embedder into rag.json so indexer resolves the same embedder the UI shows
    try:
        cfg = load_rag(rag_path)
        cfg.setdefault("embedder", {})
        if embedder_name:
            cfg["embedder"]["selected_name"] = embedder_name
        save_rag(rag_path, cfg)
    except Exception as e:
        st.warning(f"RAG config sync failed: {e}")

    # 2) Decide: full vs delta based on per-project RAG dir contents
    rag_dir = project_rag_dir(project_root)
    try:
        exists_and_has_files = rag_dir.exists() and any(rag_dir.iterdir())
    except Exception:
        exists_and_has_files = False

    try:
        if not exists_and_has_files:
            with st.spinner("Auto-index: creating fresh index…"):
                res = full_reindex(project_root, rag_path)
                st.caption({"auto_index_full": res})
        else:
            with st.spinner("Auto-index: updating index (delta)…"):
                res = delta_index(project_root, rag_path)
                st.caption({"auto_index_delta": res})
    except Exception as e:
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
        return json.loads(right)
    except Exception:
        pass

    # 4) Try to locate the first {...} or [...] region inside the text
    m = re.search(r"(\{.*\}|\[.*\])", right, re.DOTALL)
    if m:
        candidate = m.group(1)

        # 4a) JSON first
        try:
            return json.loads(candidate)
        except Exception:
            pass

        # 4b) Last resort: Python literal (handles single quotes/None/etc.)
        try:
            return ast.literal_eval(candidate)
        except Exception:
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
        return True
    except Exception:
        return False


def _maybe_render_graph(result: Any):
    """
    If the tool result contains a Graphviz 'dot' string, render it.
    """
    try:
        if isinstance(result, dict) and "dot" in result and isinstance(result["dot"], str):
            st.markdown("### Call Graph")
            st.graphviz_chart(result["dot"])
    except Exception as e:
        st.warning(f"Graph render failed: {e}")


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

    # Graph (DOT)
    _maybe_render_graph(result)
    return True


# ---------- UI settings (global) ----------
def _load_ui_settings() -> dict:
    try:
        return json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ui_settings(d: dict) -> None:
    try:
        UI_SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Configuration")
    ui = _load_ui_settings()

    # Paths
    default_root = str(Path.cwd().parent / "scv2" / "backend")
    project_root = st.text_input("Project root (--root)", value=default_root)

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

                res = full_reindex(project_root, RAG_PATH)
                st.success(f"Reindex complete. Added chunks: {res.get('added')}")
                st.caption(res)
            except Exception as e:
                st.error(f"Reindex failed: {e}")

    # Edit RAG settings (global)
    with st.expander("Edit RAG settings (global)", expanded=False):
        try:
            cfg = load_rag(RAG_PATH)
        except Exception as e:
            cfg = None
            st.error(f"Failed to load rag.json: {e}")
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
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
            with colB:
                if st.button("Reload from disk"):
                    st.experimental_rerun()

    # Model config file
    default_cfg = str(Path("data") / "models.json")
    models_config = st.text_input("Model config (--config)", value=default_cfg)

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
    except Exception as e:
        st.error(f"Failed to load models: {e}")
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
    except Exception:
        emb_list, emb_names, default_emb_name = [], [], ""

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
        except Exception as _e:
            pass

        # Quick details
        try:
            emb_entry = next(e for e in emb_list if e["name"] == chosen_emb_name)
            st.caption(
                f"Embedder → Provider: {emb_entry.get('provider','?')} • Endpoint: {emb_entry.get('endpoint','?')}"
            )
        except Exception:
            pass
    else:
        st.info("No embedding models found in models.json.")

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
    max_iters = st.number_input(
        "Max tool iters (--max-iters)",
        min_value=1,
        max_value=20,
        value=5 if ENHANCED_AVAILABLE else 3,
        step=1,
    )
    analysis_only = st.checkbox("Analysis-only (no writes)", value=bool(ui.get("analysis_only", True)))
    ui["analysis_only"] = analysis_only

    callgraph_depth = st.number_input(
        "Call graph depth (0 = full)", min_value=0, max_value=10, value=int(ui.get("callgraph_depth", 3)), step=1
    )
    st.session_state["callgraph_depth"] = callgraph_depth
    ui["callgraph_depth"] = callgraph_depth

    # Persist UI choices
    _save_ui_settings(ui)

    # Build runtime objects (once)
    rt_key = (project_root, enhanced, enable_tools, chosen_model_name)
    if "rt_built" not in st.session_state or st.session_state.get("rt_key") != rt_key:
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
    preview_only = st.checkbox("Force preview (dry run) where applicable", value=True)
    show_raw_json = st.checkbox("Show raw JSON/tool outputs", value=False)

# ------------ Log window ------------
st.markdown("### Activity / Thinking")
if clear_log:
    st.session_state["log_lines"] = []
log_container = st.container(border=True)
with log_container:
    for line in st.session_state["log_lines"]:
        st.markdown(line)


def log(line: str):
    st.session_state["log_lines"].append(line)
    with log_container:
        st.markdown(line)


# ------------ Planning ------------
agent = st.session_state["agent"]
tools = st.session_state["tools"]


def render_plan(plan_steps: List[Dict[str, Any]]):
    st.markdown("### Execution Plan")
    if not plan_steps:
        st.info("No plan.")
        return
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

    for i, step in enumerate(plan_steps, 1):
        if "error" in step:
            log(f"❌ Plan step {i} error: {step['error']}")
            results.append({"step": i, "error": step["error"], "success": False})
            continue

        tool_name = step.get("tool")
        args = dict(step.get("args", {}))
        desc = step.get("description", "")

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

        try:
            log(f"🛠️ Executing step {i}: **{tool_name}** with args `{args}`")

            # Inject UI depth into call_graph_for_function
            if tool_name == "call_graph_for_function":
                args.setdefault("depth", st.session_state.get("callgraph_depth", 3))

            # Opportunistic binding: if analyze/detect_errors has a placeholder path, patch it
            if tool_name in {"analyze_code_structure", "detect_errors"}:
                p = args.get("path")
                if isinstance(p, str) and (
                    not p
                    or "path/to" in p
                    or p.strip() in {"<>", "<file>", "<path>", "<BIND:search_code.file0>"}
                ):
                    if discovered_files:
                        args["path"] = discovered_files[0]

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

            # Collect discovered files from search results (best-effort)
            if tool_name == "search_code" and isinstance(res, list):
                for r in res:
                    f = r.get("file") if isinstance(r, dict) else None
                    if isinstance(f, str):
                        discovered_files.append(f)

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

            if show_raw_json:
                with st.expander(f"Raw result (step {i})"):
                    st.code(_pretty(res), language="json")

        except Exception as e:
            log(f"❌ Step {i} failed: {e}")
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
    return results


# ------------ Main actions ------------
if do_execute:
    if prompt.strip():
        # Before planning/asking, make sure index is ready if user enabled auto indexing
        try:
            # Use your selected project root and global rag.json (for now)
            _ensure_index_ready(
                project_root=project_root,
                rag_path=str(GLOBAL_RAG_PATH),
                embedder_name=st.session_state.get("embedding_model"),
                auto=bool(st.session_state.get("rag_auto_index", True)),
            )
        except Exception as e:
            st.warning(f"Auto-index preflight skipped due to error: {e}")

        # Two modes:
        # 1) Enhanced + plan_mode → plan then execute plan (step-by-step visibility)
        # 2) Standard/simple → intercept call-graph queries, otherwise ask_once
        if plan_mode and ENHANCED_AVAILABLE and isinstance(agent, ReasoningAgent):  # type: ignore
            if not st.session_state.get("last_plan"):
                log("🧭 Creating plan (no existing plan found)...")
                st.session_state["last_plan"] = agent.analyze_and_plan(prompt)
            render_plan(st.session_state["last_plan"])
            st.markdown("---")
            log("🚀 Executing plan...")
            results = execute_plan(st.session_state["last_plan"], analysis_only=analysis_only)

            # ---------- Final Report (natural language) ----------
            # Prefer agent-side summarization if you later add it; otherwise synthesize here
            report_text = None
            try:
                # synth_prompt = (
                #     "Synthesize a concise, actionable report from these execution results.\n"
                #     "Focus strictly on:\n"
                #     "1) Root cause(s)\n"
                #     "2) Supporting evidence with file/line where possible\n"
                #     "3) Risks/unknowns\n"
                #     "4) Recommended fix steps (do NOT write or apply code)\n\n"
                #     f"RESULTS JSON:\n{json.dumps(results, indent=2)}"
                # )

                results_json = json.dumps(results, indent=2)
                synth_prompt = intents._build_synth_prompt(prompt, results_json)
                resp = agent.adapter.chat([
                    {"role": "system", "content": "Be concrete, cite code (file:line) with short excerpts. No tool calls."},
                    {"role": "user", "content": synth_prompt},
                ])
                report_text = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()

                # snips_text = "\n\n".join(
                #     f"{s['path']}:{s.get('lineno','?')}-{s.get('end_lineno','?')}\n{s['text']}"
                #     for s in top_snippets  # however you collect them
                # )
                # synth_prompt = intents._build_synth_prompt(prompt, results_json, extra_context=snips_text)

                # # Use the model adapter directly to ensure plain text (OpenAI-compat) :contentReference[oaicite:6]{index=6}
                # resp = agent.adapter.chat(
                #     [
                #         {
                #             "role": "system",
                #             "content": "You are a precise code analyst. Be concrete and terse.",
                #         },
                #         {"role": "user", "content": synth_prompt},
                #     ]
                # )
                # report_text = (
                #     (resp.get("choices") or [{}])[0].get("message", {}) or {}
                # ).get("content", "")
            except Exception as e:
                report_text = f"(Failed to synthesize final report: {e})"

            if isinstance(report_text, str) and report_text.strip():
                st.markdown("### Final Report")
                st.write(report_text.strip())

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
                try:
                    res = tools.call("call_graph_for_function", function=fn, depth=depth)
                    st.markdown("### Assistant Response")
                    _render_call_graph_payload(res) or _maybe_render_graph(res)
                    if show_raw_json:
                        with st.expander("Raw tool JSON"):
                            st.code(_pretty(res), language="json")
                except Exception as e:
                    st.error(f"call_graph_for_function failed: {e}")
                st.stop()  # prevent falling through to ask_once

            # 💬 Fallback: standard single-turn ask
            log("💬 Running single-turn ask (agent.ask_once)...")
            if ENHANCED_AVAILABLE and isinstance(agent, ReasoningAgent) and plan_mode:
                answer = agent.ask_with_planning(
                    prompt, max_iters=int(max_iters), analysis_only=analysis_only
                )
            else:
                answer = agent.ask_once(prompt, max_iters=int(max_iters))
            st.markdown("### Assistant Response")

            parsed = _looks_like_tool_blob(answer)
            if parsed is not None:
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
                    st.write(answer)
                if show_raw_json:
                    with st.expander("Raw text"):
                        st.code(answer)
    else:
        st.warning("Please enter a prompt.")

# (end)
