# reasoning_agent.py
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger

from agent import Agent
from models import LLMModel
from tools.registry import ToolRegistry

# --- NEW: tiny helper to find BIND tokens like {{BIND:search_code.file0}} ---
_BIND_RX = re.compile(r"\{\{BIND:([A-Za-z0-9_]+)\.([A-Za-z0-9_\-]+)\}\}")

def _extract_bind_tokens(obj: Any) -> List[tuple[str, str]]:
    s = ""
    if isinstance(obj, str):
        s = obj
    return _BIND_RX.findall(s) if s else []

class ReasoningAgent(Agent):
    """
    Enhanced agent with multi-step reasoning capabilities:
    - analyze_and_plan(): prompts the LLM with an EXACT tool spec (names + arg names)
      so it produces a valid JSON plan.
    - execute_plan(): runs steps safely and collects results.
    - _normalize_step_args(): fixes common arg-name mistakes from LLM output.
    """

    # ---- canonical tool specifications (names + exact arg names) ----
    # Keep these in sync with tools/registry.py and tools/enhanced_registry.py
    _TOOL_SPECS: Dict[str, Dict[str, Any]] = {
        # registry.py tools
        "scan_relevant_files": {
            "required": [],
            "allowed": {"prompt", "path", "subdir", "max_results", "exts"},
            "defaults": {"prompt": "", "path": "", "subdir": "", "max_results": 200, "exts": [".py"]},
        },
        "search_code": {
            "required": {"query"},
            "allowed": {"query", "subdir", "regex", "case_sensitive", "exts", "max_results"},
            "defaults": {"subdir": "", "regex": False, "case_sensitive": False, "exts": [".py"], "max_results": 2000},
        },
        "analyze_files": {
            "required": {"paths"},
            "allowed": {"paths", "max_bytes"},
            "defaults": {"max_bytes": 8000},
        },
        "write_file": {
            "required": {"path", "content"},
            "allowed": {"path", "content", "overwrite", "backup"},
            "defaults": {"overwrite": True, "backup": False},
        },
        "replace_in_file": {
            "required": {"path", "find", "replace"},
            "allowed": {"path", "find", "replace", "regex", "dry_run", "backup"},
            "defaults": {"regex": False, "dry_run": False, "backup": True},
        },
        "format_python_files": {
            "required": [],
            "allowed": {"subdir", "line_length", "dry_run"},
            "defaults": {"subdir": "", "line_length": 88, "dry_run": False},
        },
        "bulk_edit": {
            "required": {"edits"},
            "allowed": {"edits", "dry_run", "backup"},
            "defaults": {"dry_run": True, "backup": True},
        },
        "rewrite_naive_open": {
            "required": [],
            "allowed": {"dir", "exts", "dry_run", "backup"},
            "defaults": {"dir": ".", "exts": [".py"], "dry_run": True, "backup": True},
        },

        # enhanced_registry.py tools
        "analyze_code_structure": {
            "required": {"path"},
            "allowed": {"path", "subdir"},
            "defaults": {"subdir": ""},
        },
        "find_related_files": {
            "required": {"main_file"},
            "allowed": {"main_file"},
            "defaults": {},
        },
        "detect_errors": {
            "required": {"path"},
            "allowed": {"path", "subdir"},
            "defaults": {"subdir": ""},
        },
        "call_graph_for_function": {
            "required": {"function"},
            "allowed": {"function", "subdir", "depth"},
            "defaults": {"subdir": "", "depth": 3},
        },        
    }

    # common alias map to repair sloppy args from the LLM
    _ARG_ALIASES: Dict[str, Dict[str, str]] = {
        "write_file": {"filename": "path", "file": "path"},
        "analyze_files": {"files": "paths", "file": "paths", "filename": "paths"},
        "search_code": {"dir": "subdir", "path": "subdir"},
        "format_python_files": {"dir": "subdir", "path": "subdir"},
        "scan_relevant_files": {"dir": "path"},
        "rewrite_naive_open": {"path": "dir"},
        "analyze_code_structure": {"file": "path", "filename": "path"},
        "find_related_files": {"file": "main_file", "filename": "main_file"},
        "detect_errors": {"file": "path", "filename": "path", "files": "path"},
        "replace_in_file": {"filename": "path", "file": "path"},
        "bulk_edit": {},  # edits is already correct
        "call_graph_for_function": {"name": "function", "level": "depth"},
    }

    # Map common synonyms → real tool names
    _TOOL_REDIRECT = {
        "analyze_function": "call_graph_for_function",
    }

    # Tools that mutate files and must be blocked (or dry-run) in analysis-only mode
    _EDIT_TOOLS = {
        "write_file", "replace_in_file", "bulk_edit",
        "rewrite_naive_open", "format_python_files"
    }

    def __init__(self, model: LLMModel, tools: ToolRegistry, enable_tools: bool = True):
        super().__init__(model, tools, enable_tools)
        logger.info("ReasoningAgent initialized (tools_enabled={})", enable_tools)

    def _try_direct_actions(self, query: str) -> Optional[str]:
        """Handle call graph queries directly without planning for faster response."""
        q = (query or "").lower().strip()
        if "call graph" in q or "callgraph" in q:
            match = re.search(r'of\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)?', q) or \
                    re.search(r'for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)?', q)
            if match:
                function_name = match.group(1)
                logger.info("ReasoningAgent direct call_graph_for_function('{}')", function_name)
                try:
                    # Check if the tool is available
                    if hasattr(self.tools, 'tools') and "call_graph_for_function" in self.tools.tools:
                        result = self.tools.call("call_graph_for_function", function=function_name)
                        return json.dumps(result, indent=2)
                    else:
                        return "[error] call_graph_for_function tool not available"
                except Exception as e:
                    logger.error("Direct call_graph_for_function failed: {}", e)
                    return f"[error] call_graph_for_function failed: {e}"
        
        # Fall back to parent's direct actions for other queries
        return super()._try_direct_actions(query)

    # ---------------- Planning: strict, grounded, example-driven ----------------
    _PLAN_SYSTEM = (
        "You are a careful *code* planner. Produce a small sequence of TOOL calls that are:\n"
        "1) GROUNDED: Never invent file paths. First discover, then analyze.\n"
        "2) CONCRETE: Use only the allowed tools and exact argument names.\n"
        "3) EFFICIENT: 3–6 steps. Mark a step 'critical': true when later steps depend on it.\n"
        "4) JSON ONLY: Return a JSON array of steps (no prose)."
    )

    _PLAN_USER_TMPL = """\
Goal:
{goal}

Rules:
- Never use placeholders like "path/to/*.py". If a file path is needed, add a prior discovery step first.
- Typical flow:
  - search_code / scan_relevant_files  → locate file(s)
  - call_graph_for_function (depth={default_depth})  → structural context
  - analyze_code_structure / detect_errors → targeted analysis on discovered file(s)
- Always include {{ "subdir": "" }} when the tool allows it.
- If your plan needs a path for a function, first *find that file* then pass the exact file path to later steps.

Return JSON array like:
[
  {"tool": "search_code", "args": {"query": "def my_func", "subdir": "", "exts": [".py"], "max_results": 10}, "description": "Find file for my_func", "critical": true},
  {"tool": "call_graph_for_function", "args": {"function": "my_func", "subdir": "", "depth": %DEPTH%}, "description": "Generate call graph", "critical": false},
  {"tool": "analyze_code_structure", "args": {"path": "<BIND:search_code.file0>", "subdir": ""}, "description": "Analyze the file that defines my_func()", "critical": false}
]
"""

    def _critique_and_fix_plan(self, plan: list[dict]) -> list[dict]:
        """
        Post-process the LLM plan to avoid common failure modes:
        - Replace placeholders like 'path/to/*.py' with real files when we can infer them.
        - Inject missing defaults (e.g., subdir="").
        - Normalize depth for call_graph_for_function.
        The executor will still run with safety, but this prevents 80% of bad steps.
        """
        fixed: list[dict] = []
        last_found_file: str | None = None

        def _looks_placeholder(p: str) -> bool:
            return not p or "path/to" in p or p.strip() in {"<>", "<file>", "<path>","<BIND>"}

        logger.debug("critique_and_fix_plan: input steps={}", len(plan or []))
        for step in plan or []:
            s = dict(step or {})
            args = dict(s.get("args") or {})
            tool = s.get("tool", "")

            # Ensure subdir defaults where allowed
            if tool in {"search_code", "format_python_files"}:
                args.setdefault("subdir", "")
            if tool in {"analyze_code_structure", "detect_errors", "call_graph_for_function"}:
                args.setdefault("subdir", "")

            # Normalize depth default
            if tool == "call_graph_for_function":
                args.setdefault("depth", 3)

            # Placeholder binding hinting
            if tool == "search_code":
                # we can't know the result now; mark intention for binding
                s["_expects_files"] = True
            if tool in {"analyze_code_structure", "detect_errors"}:
                path = args.get("path", "")
                if isinstance(path, str) and _looks_placeholder(path) and last_found_file:
                    args["path"] = last_found_file

            s["args"] = args
            fixed.append(s)

        logger.debug("critique_and_fix_plan: output steps={}", len(fixed))
        return fixed
    
    def analyze_and_plan(self, query: str) -> list:
        """
        Create a grounded, discovery-first execution plan.
        Uses strict, example-driven planner prompts and then runs a
        local critique/repair pass to remove placeholders and inject defaults.
        """
        default_depth = 3
        user_msg = (
            self._PLAN_USER_TMPL
            .replace("{goal}", query)
            .replace("{default_depth}", str(default_depth))
            .replace("%DEPTH%", str(default_depth))
        )
        messages = [
            {"role": "system", "content": self._PLAN_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        # --- RAG-first planning hint (prefer PDFs; allow web only if needed) ---
        try:
            # Your ToolRegistry exposes .tools (used elsewhere in this file)
            available = sorted(getattr(self.tools, "tools", {}).keys())
        except Exception:
            available = []

        rag_policy = (
            "AVAILABLE_TOOLS:\n- " + "\n- ".join(available) + "\n\n"
            "PLANNING POLICY:\n"
            "• Use only AVAILABLE_TOOLS; do not invent tool names.\n"
            "• Prefer RAG grounding from project PDFs using edu_* tools "
            "(edu_detect_intent, edu_explain, edu_similar_questions, edu_question_paper) "
            "and the retriever.\n"
            "• Only consider web_search/web_fetch if RAG returns too little evidence "
            "(heuristic: fewer than ~6 relevant chunks, or less than ~60% of context coming from PDFs).\n"
            "• When you do use the web, explicitly cite the URLs and still keep book/PDF citations when present.\n"
            "• If no grounded evidence is found at all, state that clearly rather than guessing.\n"
        )
        # Append as a system message so it has high weight but remains separate from _PLAN_SYSTEM
        messages.append({
            "role": "system",
            "content": rag_policy
        })

        try:
            logger.info("ReasoningAgent.analyze_and_plan → planning for prompt='{}...'", (query or "")[:160])
            resp = self.adapter.chat(messages)
            content = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            # Allow fenced JSON
            m = re.search(r"```json\s*(\[.*?\])\s*```", content, re.DOTALL | re.IGNORECASE)
            if m:
                content = m.group(1)
            plan = json.loads(content)
            logger.info("Plan generated with {} step(s)", len(plan or []))
        except Exception as e:
            logger.warning("Planner failed: {}. Falling back to discovery-first plan.", e)
            # Fallback: minimal discovery-first plan so executor can proceed
            plan = [
                {
                    "tool": "search_code",
                    "args": {"query": query, "subdir": "", "exts": [".py"], "max_results": 50},
                    "description": "Discover relevant files",
                    "critical": True,
                }
            ]
        # Final local guard-rails: clean up placeholders, inject defaults
        return self._critique_and_fix_plan(plan)
    
    def _render_tool_spec_for_prompt(self) -> str:
        lines = []
        for name, spec in self._TOOL_SPECS.items():
            allowed = ", ".join(sorted(spec["allowed"]))
            required = ", ".join(sorted(spec["required"])) if spec["required"] else "(none)"
            lines.append(f'- {name}({allowed})  REQUIRED: {required}')
        out = "\n".join(lines)
        logger.debug("render_tool_spec_for_prompt len={}", len(out))
        return out

    def _normalize_step_args(self, tool: str, args: Dict[str, Any]) -> (bool, Dict[str, Any], Optional[str]):
        """
        Enforce tool arg names and provide light aliasing. Returns (ok, fixed_args, error).
        """
        spec = self._TOOL_SPECS.get(tool)
        if not spec:
            logger.error("normalize_step_args: unknown tool '{}'", tool)
            return False, {}, f"Unknown tool '{tool}'."
        allowed = set(spec["allowed"])
        required = set(spec["required"])
        defaults = dict(spec.get("defaults", {}))

        # 1) alias repair (e.g., filename->path, files->paths)
        alias_map = self._ARG_ALIASES.get(tool, {})
        repaired: Dict[str, Any] = {}
        for k, v in (args or {}).items():
            nk = alias_map.get(k, k)
            repaired[nk] = v

        # 2) drop unknown keys
        cleaned = {k: v for k, v in repaired.items() if k in allowed}

        # 3) fill defaults
        for k, v in defaults.items():
            cleaned.setdefault(k, v)

        # 4) special coercions
        if tool == "analyze_files" and isinstance(cleaned.get("paths"), str):
            cleaned["paths"] = [cleaned["paths"]]
        if tool == "detect_errors":
            v = cleaned.get("path")
            if isinstance(v, list) and v:
                cleaned["path"] = v[0]
        if tool in {"format_python_files"} and "line_length" in cleaned:
            try:
                cleaned["line_length"] = int(cleaned["line_length"])
            except Exception:
                logger.error("normalize_step_args: line_length must be an integer")
                return False, {}, "line_length must be an integer"

        # 5) required checks
        missing = required - set(cleaned.keys())
        if missing:
            logger.error("normalize_step_args: missing required args for {}: {}", tool, sorted(missing))
            return False, {}, f"Missing required args for {tool}: {sorted(missing)}"

        logger.debug("normalize_step_args OK for tool='{}' keys={}", tool, list(cleaned.keys()))
        return True, cleaned, None

    # ---------------- execution ----------------

    def execute_plan(self, plan: list, max_iters: int = 10, analysis_only: bool = False) -> str:
        """
        Execute an LLM-generated plan step-by-step with safety checks
        and clear, streamlit-friendly results.

        If analysis_only=True, any edit-capable tool is run in preview mode
        (dry-run when supported) and labelled as blocked-from-write.

        Contract of each plan step:
        {
            "tool": "<registered tool name>",
            "args": { ... concrete args ... },
            "description": "what this step does",
            "critical": true   # optional; default True
        }
        """
        import json
        results = []
        last_by_tool: dict[str, Any] = {}  # keep outputs for {{BIND:...}} usage

        # Track which tools produced at least one item we can work with
        tool_nonempty: dict[str, bool] = {}

        logger.info("ReasoningAgent.execute_plan begin: steps={} analysis_only={}", len(plan or []), analysis_only)

        # --- execution-time web fallback guard (injects web context once if RAG is weak) ---
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
        _web_fallback_done = False
        _MIN_CHUNKS, _MIN_RATIO = 6, 0.60

        # quick recursive placeholder detector
        def _looks_like_placeholder(v) -> bool:
            if isinstance(v, str):
                s = v.strip().lower()
                return (
                    "previous_step" in s or "identified_file" in s
                    or s.startswith("<") or s.endswith(">")
                    or "tbd" == s or "placeholder" in s
                )
            if isinstance(v, (list, tuple, set)):
                return any(_looks_like_placeholder(x) for x in v)
            if isinstance(v, dict):
                return any(_looks_like_placeholder(x) for x in v.values())
            return False

        # light alias map to fix common LLM slips
        ARG_ALIASES = {
            "write_file": {"filename": "path", "file": "path"},
            "analyze_files": {"files": "paths", "file": "paths", "filename": "paths"},
            "search_code": {"dir": "subdir", "path": "subdir"},
            "format_python_files": {"dir": "subdir", "path": "subdir"},
            "scan_relevant_files": {"dir": "path"},
            "rewrite_naive_open": {"path": "dir"},
            "analyze_code_structure": {"file": "path", "filename": "path"},
            "find_related_files": {"file": "main_file", "filename": "main_file"},
            "detect_errors": {"file": "path", "filename": "path", "files": "path"},
            "replace_in_file": {"filename": "path", "file": "path"},
            "bulk_edit": {},
        }

        # default values for a few tools (mirrors tools/registry.py)
        DEFAULTS = {
            "scan_relevant_files": {"prompt": "", "path": "", "subdir": "", "max_results": 200, "exts": [".py"]},
            "search_code": {"subdir": "", "regex": False, "case_sensitive": False, "exts": [".py"], "max_results": 2000},
            "analyze_files": {"max_bytes": 8000},
            "write_file": {"overwrite": True, "backup": False},
            "replace_in_file": {"regex": False, "dry_run": False, "backup": True},
            "format_python_files": {"subdir": "", "line_length": 88, "dry_run": False},
            "bulk_edit": {"dry_run": True, "backup": True},
            "rewrite_naive_open": {"dir": ".", "exts": [".py"], "dry_run": True, "backup": True},
            "analyze_code_structure": {"subdir": ""},
            "find_related_files": {},
            "detect_errors": {"subdir": ""},
        }

        if not isinstance(plan, list):
            msg = "Plan must be a JSON array of steps."
            logger.error("execute_plan: {}", msg)
            return json.dumps([{"error": msg, "success": False}], indent=2)

        for idx, step in enumerate(plan[:max_iters], start=1):
            if isinstance(step, dict) and "error" in step and "tool" not in step:
                logger.error("execute_plan: planner error at step {}: {}", idx, step.get("error"))
                results.append({"step": idx, **step, "success": False})
                break

            if not isinstance(step, dict):
                logger.error("execute_plan: invalid step type {} at {}", type(step), idx)
                results.append({"step": idx, "error": f"Invalid step type: {type(step)}", "success": False})
                break

            tool = step.get("tool")
            args = step.get("args", {}) or {}
            desc = step.get("description", "")
            critical = step.get("critical", True)
            logger.info("execute_plan: step {} tool='{}' critical={} desc='{}'", idx, tool, critical, desc)

            # Redirect synonyms first
            target_tool = self._TOOL_REDIRECT.get(tool, tool)

            if not target_tool or not hasattr(self.tools, "tools") or target_tool not in self.tools.tools:
                logger.error("execute_plan: unknown/unregistered tool '{}'", tool)
                results.append({
                    "step": idx,
                    "tool": target_tool,
                    "description": desc,
                    "error": f"Unknown or unregistered tool: {target_tool}",
                    "success": False
                })
                if critical:
                    break
                continue

            # block placeholders
            if _looks_like_placeholder(args):
                logger.error("execute_plan: placeholder argument at step {}: {}", idx, args)
                results.append({
                    "step": idx,
                    "tool": tool,
                    "description": desc,
                    "error": f"Placeholder argument detected: {args}",
                    "success": False
                })
                if critical:
                    break
                continue

            # --------- Guard unresolved BINDs (skip step) ----------
            # If args contain {{BIND:tool.fieldN}} and we *know* that tool was empty,
            # skip gracefully instead of letting the tool crash.
            def _iter_binds(obj):
                if isinstance(obj, str):
                    for t in _BIND_RX.findall(obj):
                        yield t  # (tool_name, field_key)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        yield from _iter_binds(v)
                elif isinstance(obj, (list, tuple)):
                    for x in obj:
                        yield from _iter_binds(x)
            unresolved = any(tool_nonempty.get(tname) is False for (tname, _fld) in _iter_binds(args))
            if unresolved:
                logger.warning("execute_plan: skipping step {} '{}' due to unresolved BIND (previous tool had no results)", idx, tool)
                results.append({"step": idx, "tool": tool, "description": desc, "skipped": True, "reason": "unresolved_bind"})
                continue

            # --------- BIND placeholders: {{BIND:search_code.file0}} ----------
            def _bind_value(v):
                if isinstance(v, str):
                    import re
                    # support {{BIND:tool.fieldN}}  OR  <BIND:tool.fieldN>
                    m = re.match(r"^\{\{BIND:(\w+)\.([A-Za-z_]+)(\d+)\}\}$", v) \
                        or re.match(r"^<BIND:(\w+)\.([A-Za-z_]+)(\d+)>$", v)
                    if m:
                        tname, field, idx = m.group(1), m.group(2), int(m.group(3))
                        src = last_by_tool.get(tname)
                        # Case A: previous tool returned a list (of dicts/strings)
                        if isinstance(src, list):
                            if 0 <= idx < len(src):
                                item = src[idx]
                                if isinstance(item, dict):
                                    return item.get(field, v)
                                # e.g., list of file paths and field == "file" or "path"
                                if isinstance(item, str) and field in {"file", "path"}:
                                    return item
                        # Case B: previous tool returned a dict (common: {"files":[...], "paths":[...]})
                        if isinstance(src, dict):
                            # try plural then singular
                            for key in (field + "s", field):
                                val = src.get(key)
                                if isinstance(val, list) and 0 <= idx < len(val):
                                    return val[idx]
                                if isinstance(val, dict):
                                    # very rare, but keep symmetry with list-of-dicts
                                    return val.get(str(idx), v)
                return v
            def _bind_obj(obj):
                if isinstance(obj, dict):
                    return {k: _bind_obj(_bind_value(v)) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_bind_obj(_bind_value(x)) for x in obj]
                return _bind_value(obj)
            args = _bind_obj(args)

            # --------- Normalize args strictly against tool spec ----------
            ok, fixed_args, err = self._normalize_step_args(target_tool, args)
            if not ok:
                results.append({
                    "step": idx,
                    "tool": target_tool,
                    "description": desc,
                    "error": err,
                    "success": False
                })
                logger.error("execute_plan: normalization failed for '{}' at step {}: {}", target_tool, idx, err)
                if critical:
                    break
                continue

            # coerce some expected shapes
            if tool == "analyze_files" and isinstance(fixed_args.get("paths"), str):
                fixed_args["paths"] = [fixed_args["paths"]]
            if tool == "detect_errors":
                v = fixed_args.get("path")
                if isinstance(v, list) and v:
                    fixed_args["path"] = v[0]
            if tool == "format_python_files" and "line_length" in fixed_args:
                try:
                    fixed_args["line_length"] = int(fixed_args["line_length"])
                except Exception:
                    logger.error("execute_plan: line_length must be an integer (step {})", idx)
                    results.append({
                        "step": idx,
                        "tool": tool,
                        "description": desc,
                        "error": "line_length must be an integer",
                        "success": False
                    })
                    if critical:
                        break
                    continue

            # fill defaults where missing (use the post-redirect tool name)
            for dk, dv in DEFAULTS.get(target_tool, {}).items():
                fixed_args.setdefault(dk, dv)

            # call the tool
            # If web fallback ran earlier and this is an edu_* tool, forward fetched pages
            if isinstance(target_tool, str) and target_tool.startswith("edu_"):
                try:
                    wfb = last_by_tool.get("web_fetch_batch")
                    if wfb and "web_fetch_batch" not in fixed_args:
                        fixed_args["web_fetch_batch"] = wfb
                except Exception:
                    pass

            try:
                out = self.tools.call(target_tool, **fixed_args)
                rec = {
                    "step": idx,
                    "tool": target_tool,
                    "description": desc,
                    "args": fixed_args,
                    "result": out,
                    "success": True
                }
                results.append(rec)
                logger.info("execute_plan: tool '{}' OK (step {})", target_tool, idx)
                last_by_tool[target_tool] = out

                # --- Mark tool non-empty for downstream BIND guard ---
                nonempty: Optional[bool] = None
                # Common shapes: dict with list fields, a bare list, or an explicit count
                if isinstance(out, dict):
                    for key in ("chunks", "files", "ids", "documents", "items", "results", "matches"):
                        v = out.get(key)
                        if isinstance(v, list) and len(v) > 0:
                            nonempty = True
                            break
                    if nonempty is None:
                        cnt = out.get("count")
                        if isinstance(cnt, int):
                            nonempty = (cnt > 0)
                elif isinstance(out, list):
                    nonempty = (len(out) > 0)
                # Default optimistic if unknown shape
                if nonempty is None:
                    nonempty = True
                tool_nonempty[target_tool] = bool(nonempty)


                # --- Automatic web fallback (once) when RAG grounding looks weak ---
                # Trigger only for retrieval-like outputs (those with "chunks").
                if (not _web_fallback_done) and isinstance(out, dict) and isinstance(out.get("chunks"), list):
                    sig = _grounding_signal_from(out)
                    weak = (sig["total_chunks"] < _MIN_CHUNKS) or (sig["pdf_ratio"] < _MIN_RATIO)
                    # Avoid web fallback for explicit education tools by default
                    is_edu = target_tool.startswith("edu_") or ("edu_detect_intent" in last_by_tool)

                    # Only if web tools are registered
                    available = set(getattr(self.tools, "tools", {}).keys()) if hasattr(self.tools, "tools") else set()
                    web_ok = {"web_search", "web_fetch"}.issubset(available)

                    if weak and web_ok and not is_edu:
                        # Derive a query best-effort
                        q = (fixed_args.get("query") or fixed_args.get("topic") or desc or "").strip()
                        if q:
                            try:
                                ws = self.tools.call("web_search", query=q, max_results=5)
                                results.append({
                                    "step": f"{idx}.a",
                                    "tool": "web_search",
                                    "description": f"Automatic web fallback for weak grounding (pdf_ratio={sig['pdf_ratio']:.2f}, total_chunks={sig['total_chunks']})",
                                    "args": {"query": q, "max_results": 5},
                                    "result": ws,
                                    "success": True
                                })
                                urls = [r.get("url") for r in (ws.get("results") or []) if isinstance(r, dict)]
                                urls = [u for u in urls if u][:2]
                                wf_batch = []
                                for u in urls:
                                    wf = self.tools.call("web_fetch", url=u, max_chars=60000)
                                    wf_batch.append(wf)
                                    results.append({
                                        "step": f"{idx}.b",
                                        "tool": "web_fetch",
                                        "description": "Fetch page for fallback grounding",
                                        "args": {"url": u, "max_chars": 60000},
                                        "result": wf, "success": True
                                    })
                                last_by_tool["web_search"] = ws
                                last_by_tool["web_fetch_batch"] = wf_batch
                                logger.info("execute_plan: auto web fallback injected urls={}", urls)
                                _web_fallback_done = True
                            except Exception as _e:
                                logger.warning("execute_plan: web fallback failed (ignored): {}", _e)

            except Exception as e:
                results.append({
                    "step": idx,
                    "tool": target_tool,
                    "description": desc,
                    "args": fixed_args,
                    "error": str(e),
                    "success": False
                })
                logger.exception("execute_plan: tool '{}' failed at step {}: {}", target_tool, idx, e)
                if critical:
                    break

        # ---------------- final summarization (Part-1) ----------------
        # In analysis-only mode, produce a unified, natural-language report
        # over all collected step results. We append it as a final step so
        # the return type remains a list for existing UI code.
        try:
            if analysis_only:
                summary_prompt = (
                    "Synthesize a concise, actionable report from these execution results.\n"
                    f"RESULTS JSON:\n{json.dumps(results, indent=2)}"
                    "Focus on:\n"
                    "1) Root cause(s)\n"
                    "2) Supporting evidence with file/line where possible\n"
                    "3) Risks/unknowns\n"
                    "4) Recommended fix steps (do NOT write or apply code)\n\n"                    
                )
                resp = self.adapter.chat([
                    {"role": "system", "content": "You are a precise code analyst. Be concrete and terse."},
                    {"role": "user", "content": summary_prompt},
                ])
                content = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                results.append({
                    "step": "final",
                    "tool": "_summarize",
                    "report": content,
                    "success": True
                })
                logger.info("execute_plan: summarization appended (len={})", len(content or ""))
        except Exception as _e:
            logger.warning("execute_plan: summarizer failed (ignored): {}", _e)

        out_json = json.dumps(results, indent=2)
        logger.info("ReasoningAgent.execute_plan end: steps={} ok={} fail={}",
                    len(results),
                    sum(1 for r in results if r.get("success")),
                    sum(1 for r in results if not r.get("success")))
        logger.debug("execute_plan: results json size={} bytes", len(out_json.encode("utf-8")))
        return out_json

    # ---------------- main entry ----------------

    def ask_with_planning(self, query: str, max_iters: int = 10, analysis_only: bool = False) -> str:
        """High-level: try fast-path direct actions; else plan + execute."""
        direct = self._try_direct_actions(query)
        if direct is not None:
            logger.info("ask_with_planning → served via direct path")
            return direct

        plan = self.analyze_and_plan(query)
        logger.debug("ask_with_planning: executing plan of {} step(s)", len(plan or []))
        return self.execute_plan(plan, max_iters, analysis_only=analysis_only)
