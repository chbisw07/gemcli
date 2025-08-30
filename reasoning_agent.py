# reasoning_agent.py
import json
import re
from typing import List, Dict, Any, Optional

from agent import Agent
from models import LLMModel
from tools.registry import ToolRegistry


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
            "allowed": {"prompt", "path", "max_results", "exts"},
            "defaults": {"prompt": "", "path": "", "max_results": 200, "exts": [".py"]},
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
            "allowed": {"path"},
            "defaults": {},
        },
        "find_related_files": {
            "required": {"main_file"},
            "allowed": {"main_file"},
            "defaults": {},
        },
        "detect_errors": {
            "required": {"path"},
            "allowed": {"path"},
            "defaults": {},
        },
        "call_graph_for_function": {
            "required": {"function"},
            "allowed": {"function", "subdir", "depth"},
            "defaults": {"subdir": "", "depth": 1},
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

    def __init__(self, model: LLMModel, tools: ToolRegistry, enable_tools: bool = True):
        super().__init__(model, tools, enable_tools)

    def _try_direct_actions(self, query: str) -> Optional[str]:
        """
        Handle call graph queries directly without planning for faster response.
        """
        q = query.lower().strip()
        # Check for call graph intent
        if "call graph" in q or "callgraph" in q:
            # Extract function name using regex
            match = re.search(r'of\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)?', q)
            if not match:
                # Try alternative patterns
                match = re.search(r'for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)?', q)
            if match:
                function_name = match.group(1)
                try:
                    # Check if the tool is available
                    if hasattr(self.tools, 'tools') and "call_graph_for_function" in self.tools.tools:
                        result = self.tools.call("call_graph_for_function", function=function_name)
                        return json.dumps(result, indent=2)
                    else:
                        return "[error] call_graph_for_function tool not available"
                except Exception as e:
                    return f"[error] call_graph_for_function failed: {e}"
        
        # Fall back to parent's direct actions for other queries
        return super()._try_direct_actions(query)

    # ---------------- planning ----------------

    def analyze_and_plan(self, query: str) -> list:
        """
        Analyze the query and create a step-by-step execution plan with multiple steps.
        Forces use of valid tool names/args and bans placeholders.
        """
        planning_prompt = f"""
    You are an expert coding assistant. Create a concrete execution plan for this request:

    REQUEST:
    {query}

    TOOLS available (with exact arg names):
    - scan_relevant_files(prompt, path="", max_results=200, exts=[".py"])
    - search_code(query, subdir="", regex=False, case_sensitive=False, exts=[".py"], max_results=2000)
    - analyze_files(paths, max_bytes=8000)
    - write_file(path, content, overwrite=True, backup=False)
    - replace_in_file(path, find, replace, regex=False, dry_run=False, backup=True)
    - format_python_files(subdir="", line_length=88, dry_run=False)
    - bulk_edit(edits, dry_run=True, backup=True)
    - rewrite_naive_open(dir=".", exts=[".py"], dry_run=True, backup=True)
    - analyze_code_structure(path)
    - find_related_files(main_file)
    - detect_errors(path)
    - call_graph_for_function(function, subdir="", depth=1)

    OUTPUT FORMAT (strict JSON only):
    [
    {{
        "tool": "<tool name from list>",
        "args": {{ ... valid args with real values ... }},
        "description": "what this step accomplishes",
        "critical": true
    }},
    ...
    ]

    RULES:
    - Use ONLY the arg names above.
    - DO NOT invent placeholders like "identified_file_from_previous_step".
    - If you don't know a filename yet, add a discovery step (scan_relevant_files or search_code) to obtain it.
    - Always produce concrete arg values (strings, lists) that the tool can execute immediately.
    """

        messages = [
            {"role": "system", "content": "You are a careful planner. Only output valid JSON following the rules."},
            {"role": "user", "content": planning_prompt},
        ]

        try:
            resp = self.adapter.chat(messages)
            raw = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "[]")
            # Strip markdown if needed
            m = re.search(r"```json\s*(\[.*?\])\s*```", raw, re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)
            plan = json.loads(raw)
            return plan
        except Exception as e:
            return [{"error": f"Failed to create plan: {str(e)}"}]


    def _render_tool_spec_for_prompt(self) -> str:
        lines = []
        for name, spec in self._TOOL_SPECS.items():
            allowed = ", ".join(sorted(spec["allowed"]))
            required = ", ".join(sorted(spec["required"])) if spec["required"] else "(none)"
            lines.append(f'- {name}({allowed})  REQUIRED: {required}')
        return "\n".join(lines)

    def _normalize_step_args(self, tool: str, args: Dict[str, Any]) -> (bool, Dict[str, Any], Optional[str]):
        """
        Enforce tool arg names and provide light aliasing. Returns (ok, fixed_args, error).
        """
        spec = self._TOOL_SPECS.get(tool)
        if not spec:
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
        if tool == "analyze_files":
            # paths must be a list[str]
            if "paths" in cleaned and isinstance(cleaned["paths"], str):
                cleaned["paths"] = [cleaned["paths"]]
        if tool == "detect_errors":
            # ensure 'path' is scalar (pick first if list was provided)
            v = cleaned.get("path")
            if isinstance(v, list) and v:
                cleaned["path"] = v[0]
        if tool in {"format_python_files"}:
            # guard line_length int
            if "line_length" in cleaned:
                try:
                    cleaned["line_length"] = int(cleaned["line_length"])
                except Exception:
                    return False, {}, "line_length must be an integer"

        # 5) required checks
        missing = required - set(cleaned.keys())
        if missing:
            return False, {}, f"Missing required args for {tool}: {sorted(missing)}"

        return True, cleaned, None

    # ---------------- execution ----------------

    def _looks_like_placeholder(val: Any) -> bool:
        if isinstance(val, str):
            return ("previous_step" in val.lower()) or val.startswith("<") or "identified" in val.lower()
        return False

    def execute_plan(self, plan: list, max_iters: int = 10) -> str:
        """
        Execute an LLM-generated plan step-by-step with safety checks
        and clear, streamlit-friendly results.

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
            "bulk_edit": {},  # edits already correct
        }

        # default values for a few tools (mirrors tools/registry.py)
        DEFAULTS = {
            "scan_relevant_files": {"prompt": "", "path": "", "max_results": 200, "exts": [".py"]},
            "search_code": {"subdir": "", "regex": False, "case_sensitive": False, "exts": [".py"], "max_results": 2000},
            "analyze_files": {"max_bytes": 8000},
            "write_file": {"overwrite": True, "backup": False},
            "replace_in_file": {"regex": False, "dry_run": False, "backup": True},
            "format_python_files": {"subdir": "", "line_length": 88, "dry_run": False},
            "bulk_edit": {"dry_run": True, "backup": True},
            "rewrite_naive_open": {"dir": ".", "exts": [".py"], "dry_run": True, "backup": True},
            "analyze_code_structure": {},
            "find_related_files": {},
            "detect_errors": {},
        }

        if not isinstance(plan, list):
            return json.dumps(
                [{"error": "Plan must be a JSON array of steps.", "success": False}],
                indent=2
            )

        for idx, step in enumerate(plan[:max_iters], start=1):
            # pass through pre-existing planner errors
            if isinstance(step, dict) and "error" in step and "tool" not in step:
                results.append({"step": idx, **step, "success": False})
                # if a pure planning error appears here, treat as critical and stop
                break

            # validate basic shape
            if not isinstance(step, dict):
                results.append({"step": idx, "error": f"Invalid step type: {type(step)}", "success": False})
                break

            tool = step.get("tool")
            args = step.get("args", {}) or {}
            desc = step.get("description", "")
            critical = step.get("critical", True)

            # tool must exist in the registry
            if not tool or not hasattr(self.tools, "tools") or tool not in self.tools.tools:
                results.append({
                    "step": idx,
                    "tool": tool,
                    "description": desc,
                    "error": f"Unknown or unregistered tool: {tool}",
                    "success": False
                })
                if critical:
                    break
                continue

            # block placeholders
            if _looks_like_placeholder(args):
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

            # apply simple alias fixes
            alias_map = ARG_ALIASES.get(tool, {})
            fixed_args = {}
            for k, v in args.items():
                k2 = alias_map.get(k, k)
                fixed_args[k2] = v

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

            # fill defaults where missing
            for dk, dv in DEFAULTS.get(tool, {}).items():
                fixed_args.setdefault(dk, dv)

            # call the tool
            try:
                out = self.tools.call(tool, **fixed_args)  # executes registered function  
                results.append({
                    "step": idx,
                    "tool": tool,
                    "description": desc,
                    "args": fixed_args,
                    "result": out,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "step": idx,
                    "tool": tool,
                    "description": desc,
                    "args": fixed_args,
                    "error": str(e),
                    "success": False
                })
                if critical:
                    break

        return json.dumps(results, indent=2)


    # ---------------- main entry ----------------

    def ask_with_planning(self, query: str, max_iters: int = 10) -> str:
        """
        High-level: try fast-path direct actions; else plan + execute.
        """
        direct = self._try_direct_actions(query)
        if direct is not None:
            return direct

        plan = self.analyze_and_plan(query)
        return self.execute_plan(plan, max_iters)