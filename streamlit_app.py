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
import graphviz   # pip install graphviz


# --- Import your existing modules (unchanged) ---
from models import ModelRegistry                         # :contentReference[oaicite:0]{index=0}
from tools.registry import ToolRegistry                  # :contentReference[oaicite:1]{index=1}
from agent import Agent                                  # :contentReference[oaicite:2]{index=2}

# Enhanced (optional)
try:
    from tools.enhanced_registry import EnhancedToolRegistry  # :contentReference[oaicite:3]{index=3}
    from reasoning_agent import ReasoningAgent                # :contentReference[oaicite:4]{index=4}
    ENHANCED_AVAILABLE = True
except Exception:
    ENHANCED_AVAILABLE = False
    EnhancedToolRegistry = None
    ReasoningAgent = None  # type: ignore

# Force load from gemcli/.env no matter where you launch from
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------- Small helpers ----------
APP_TITLE = "gemcli — Code Assistant (Streamlit UI)"

EDIT_TOOLS = {"replace_in_file", "bulk_edit", "format_python_files", "rewrite_naive_open", "write_file"}

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

def _extract_diffs(tool_result: Any) -> List[Dict[str, str]]:
    """Normalize various tool outputs to a list of {path, diff} for preview."""
    diffs: List[Dict[str, str]] = []
    if isinstance(tool_result, dict):
        # format_python_files returns {path: {"diff": "..."}}
        if all(isinstance(v, dict) and "diff" in v for v in tool_result.values()):
            for path, meta in tool_result.items():
                diffs.append({"path": path, "diff": meta.get("diff", "")})
        # replace_in_file style
        elif "diff" in tool_result:
            diffs.append({"path": tool_result.get("path", ""), "diff": tool_result.get("diff", "")})
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

# NEW: nice visualizer for call-graph payloads
def _render_call_graph_payload(result: Any) -> bool:
    """
    If 'result' looks like output from call_graph_for_function, render a polished view
    and return True. Otherwise return False so caller can fall back to default.
    """
    if not isinstance(result, dict):
        return False
    if not (("calls" in result and isinstance(result["calls"], list)) or ("dot" in result and isinstance(result["dot"], str))):
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

    # Calls table (project-local, already filtered if your tool does it)
    calls = result.get("calls") or []
    if calls:
        st.markdown("#### Direct callees")
        # show only concise columns
        rows = [{"name": c.get("name"), "defined_in": c.get("defined_in")} for c in calls]
        st.table(rows)

    # Graph (DOT)
    _maybe_render_graph(result)
    return True

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Configuration")

    # Paths
    default_root = str(Path.cwd().parent / "scv2" / "backend")
    project_root = st.text_input("Project root (--root)", value=default_root)

    # Model config file
    default_cfg = str(Path("data") / "models.json")
    models_config = st.text_input("Model config (--config)", value=default_cfg)

    # Load models
    try:
        model_registry, models_list = _load_models(models_config)
        model_names = [m.name for m in models_list]
        default_model_name = model_registry.default_name or (model_names[0] if model_names else "")
        chosen_model_name = st.selectbox("Model (--model)", options=model_names, index=model_names.index(default_model_name) if default_model_name in model_names else 0)
        model = model_registry.get(chosen_model_name)
        st.caption(f"Provider: {model.provider}  •  Endpoint: {model.endpoint}")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # Flags
    enable_tools = st.checkbox("Enable tools (default on)", value=True)
    enhanced = st.checkbox("Enhanced mode (planning, extra tools)", value=True if ENHANCED_AVAILABLE else False, disabled=not ENHANCED_AVAILABLE)
    plan_mode = st.checkbox("Use planning for complex tasks", value=False, disabled=not ENHANCED_AVAILABLE)
    max_iters = st.number_input("Max tool iters (--max-iters)", min_value=1, max_value=20, value=5 if ENHANCED_AVAILABLE else 3, step=1)
    callgraph_depth = st.number_input(
        "Call graph depth (0 = full)", min_value=0, max_value=10, value=1, step=1
    )
    st.session_state["callgraph_depth"] = callgraph_depth

    # Build runtime objects (once)
    if "rt_built" not in st.session_state or st.session_state.get("rt_key") != (project_root, enhanced, enable_tools, chosen_model_name):
        tools = _make_tools(project_root, enhanced)
        agent = _make_agent(model, tools, enhanced, enable_tools)
        st.session_state["tools"] = tools
        st.session_state["agent"] = agent
        st.session_state["rt_key"] = (project_root, enhanced, enable_tools, chosen_model_name)
        st.session_state["last_plan"] = None
        st.session_state["last_results"] = None
        st.session_state["last_edit_payloads"] = []  # for Apply step
        st.session_state["log_lines"] = []

st.write("")

# ------------ Prompt & controls ------------
colL, colR = st.columns([2, 1], gap="large")

with colL:
    prompt = st.text_area("Your high-level instruction / prompt", height=140, placeholder="e.g., Rewrite open() usages under app/api (dry run)")
    run_col1, run_col2, run_col3 = st.columns([1,1,1])
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
log_box = st.empty()
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
        with st.expander(f"Step {i}: {step.get('tool','(no tool)')} — {step.get('description','')}", expanded=False):
            st.code(_pretty(step), language="json")

def execute_plan(plan_steps: List[Dict[str, Any]]):
    """
    Execute planned steps and collect previews/diffs for edits.
    """
    results = []
    applied_payloads = []  # track {tool,args} for potentially-applicable edits

    for i, step in enumerate(plan_steps, 1):
        if "error" in step:
            log(f"❌ Plan step {i} error: {step['error']}")
            results.append({"step": i, "error": step["error"], "success": False})
            continue

        tool_name = step.get("tool")
        args = dict(step.get("args", {}))
        desc = step.get("description", "")

        # Impose preview_only when tool is an edit-capable one
        if preview_only and tool_name in EDIT_TOOLS:
            # normalize common dry_run switches
            if tool_name == "replace_in_file":
                args.setdefault("dry_run", True)
                args.setdefault("backup", True)
            elif tool_name == "bulk_edit":
                # bulk_edit(dry_run=True) applies to all inner edits
                args.setdefault("dry_run", True)
                args.setdefault("backup", True)
            elif tool_name == "format_python_files":
                args.setdefault("dry_run", True)
            elif tool_name == "rewrite_naive_open":
                args.setdefault("dry_run", True)
                args.setdefault("backup", True)
            elif tool_name == "write_file":
                # write_file has no diff; still allow preview by setting overwrite=False when file exists
                pass

        try:
            log(f"🛠️ Executing step {i}: **{tool_name}** with args `{args}`")
            # Inject UI depth into call_graph_for_function
            if tool_name == "call_graph_for_function":
                args.setdefault("depth", st.session_state.get("callgraph_depth", 1))
            
            res = tools.call(tool_name, **args)
            results.append({"step": i, "tool": tool_name, "args": args, "result": res, "success": True, "description": desc})

            # If this step is an edit-capable tool and we ran in preview, record payload for later Apply
            if tool_name in EDIT_TOOLS:
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
            results.append({"step": i, "tool": tool_name, "args": args, "error": str(e), "success": False, "description": desc})
            # If step is critical by plan contract, could break; we keep going to surface more info.

    st.session_state["last_results"] = results
    st.session_state["last_edit_payloads"] = applied_payloads
    return results

# --- Buttons actions ---
if do_plan:
    if not ENHANCED_AVAILABLE or not isinstance(agent, ReasoningAgent):  # type: ignore
        st.warning("Enhanced planning is unavailable. Enable Enhanced mode or install the enhanced modules.")
    else:
        try:
            log("🧭 Creating plan...")
            plan = agent.analyze_and_plan(prompt or "")
            st.session_state["last_plan"] = plan
            render_plan(plan)
        except Exception as e:
            st.error(f"Planning failed: {e}")

if do_execute:
    if prompt.strip():
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
            execute_plan(st.session_state["last_plan"])
        else:
            # 🔎 Intercept single-turn call-graph requests so we can honor UI depth
            cg_match = re.search(r"(?:call\s*graph|callgraph).*(?:of|for)\s+([A-Za-z_]\w*)\s*\(\)?", prompt, re.IGNORECASE)
            if cg_match:
                fn = cg_match.group(1)
                depth = st.session_state.get("callgraph_depth", 1)
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
            answer = agent.ask_once(prompt, max_iters=int(max_iters))
            st.markdown("### Assistant Response")

            # Try to extract tool JSON and render diffs/graphs
            parsed = _looks_like_tool_blob(answer)
            if parsed is not None:
                # If it's a tool call for call_graph_for_function, inject UI depth before rendering
                if isinstance(parsed, dict) and parsed.get("tool") == "call_graph_for_function":
                    parsed.setdefault("args", {}).setdefault("depth", st.session_state.get("callgraph_depth", 1))

                diffs = _extract_diffs(parsed)
                if diffs:
                    st.markdown("### Preview (diff)")
                    for d in diffs:
                        st.markdown(f"**File:** `{d['path']}`")
                        st.code(d["diff"] or "(no diff)", language="diff")

                # Prefer the polished call-graph renderer; fall back to DOT if needed
                if not _render_call_graph_payload(parsed):
                    _maybe_render_graph(parsed)

                if show_raw_json:
                    with st.expander("Raw tool JSON"):
                        st.code(_pretty(parsed), language="json")
            else:
                # No structured JSON — try to render a DOT block directly from text
                if not _maybe_render_dot_from_text(answer):
                    st.write(answer)
                if show_raw_json:
                    with st.expander("Raw text"):
                        st.code(answer)
    else:
        st.warning("Please enter a prompt.")

# ------------ Apply changes ------------
st.markdown("---")
st.markdown("### Approve & Apply Changes")
st.caption("This will re-run edit tools with `dry_run=False`. Only enable after reviewing diffs above.")

apply_cols = st.columns([1,1,3])
with apply_cols[0]:
    do_apply = st.button("Apply all pending edits", type="primary", use_container_width=True)
with apply_cols[1]:
    do_refresh = st.button("Refresh file list", use_container_width=True)

if do_apply:
    payloads = st.session_state.get("last_edit_payloads", [])
    if not payloads:
        st.info("No pending preview edits to apply.")
    else:
        applied = []
        for p in payloads:
            name = p["tool"]
            args = dict(p["args"])
            # flip preview → apply
            if "dry_run" in args:
                args["dry_run"] = False
            try:
                log(f"✅ Applying {name} with args: {args}")
                res = tools.call(name, **args)  # write happens here via ToolRegistry  :contentReference[oaicite:9]{index=9}
                applied.append({"tool": name, "args": args, "result": res})
            except Exception as e:
                applied.append({"tool": name, "args": args, "error": str(e)})
                log(f"❌ Apply {name} failed: {e}")
        with st.expander("Apply Results", expanded=True):
            st.code(_pretty(applied), language="json")

if do_refresh:
    # Example: show Python files under project root quickly
    try:
        py_files = tools.call("list_files", subdir="", exts=[".py"])
        with st.expander("Python files under project root", expanded=False):
            st.code(_pretty(py_files), language="json")
    except Exception as e:
        st.error(f"Refresh failed: {e}")

# Footer notes / provenance
with st.expander("About this UI / Integration details", expanded=False):
    st.markdown(
        """
- Uses your existing **Agent** loop (OpenAI-compatible tool-calls + textual JSON fallback) for single-turn runs. :contentReference[oaicite:10]{index=10}
- Loads models from your **models.json** via **ModelRegistry**. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
- Exposes every tool registered in **ToolRegistry** and (optionally) **EnhancedToolRegistry**. :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}
- In Enhanced + Plan mode, uses **ReasoningAgent.analyze_and_plan()** then executes steps, collecting diffs. :contentReference[oaicite:15]{index=15}
- The agent talks to your OpenAI-compatible adapter for LLMs/endpoints. :contentReference[oaicite:16]{index=16}
        """
    )
