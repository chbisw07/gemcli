# tools/tool_schema.py
from __future__ import annotations

from typing import Dict, Any, Tuple, List, Set
import json
import re

# ---------- Canonical tool specs ----------
# Keep this in sync with tools/registry.py (names/args). Defaults are used by planners/executors.
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    # Retrieval
    "rag_retrieve": {
        "required": {"query"},
        "allowed": {"query", "top_k", "where", "min_score", "rag_path", "project_root"},
        "defaults": {},
    },
    # Files + Code helpers
    "read_file": {
        "required": {"path"},
        "allowed": {"path", "subdir"},
        "defaults": {"subdir": ""},
    },
    "list_files": {
        "required": set(),
        "allowed": {"subdir", "exts", "ignore_globs"},
        "defaults": {"subdir": "", "exts": [".py"]},
    },
    "scan_relevant_files": {
        "required": set(),
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
        "required": set(),
        "allowed": {"subdir", "line_length", "dry_run"},
        "defaults": {"subdir": "", "line_length": 88, "dry_run": False},
    },
    "bulk_edit": {
        "required": {"edits"},
        "allowed": {"edits", "dry_run", "backup"},
        "defaults": {"dry_run": True, "backup": True},
    },
    "rewrite_naive_open": {
        "required": set(),
        "allowed": {"dir", "exts", "dry_run", "backup", "aggressive"},
        "defaults": {"dir": ".", "exts": [".py"], "dry_run": True, "backup": True, "aggressive": False},
    },
    # Web + Edu tools
    "web_search": {
        "required": {"query"},
        "allowed": {"query", "max_results", "site", "recency_days"},
        "defaults": {"max_results": 5},
    },
    "web_fetch": {
        "required": {"url"},
        "allowed": {"url", "max_chars"},
        "defaults": {"max_chars": 20000},
    },
    "edu_detect_intent": {
        "required": {"prompt"},
        "allowed": {"prompt"},
        "defaults": {},
    },
    "edu_similar_questions": {
        "required": {"project_root", "rag_path"},
        "allowed": {"project_root", "rag_path", "topic", "chapter", "count", "difficulty", "scope", "top_k"},
        "defaults": {"count": 10},
    },
    "edu_question_paper": {
        "required": {"project_root", "rag_path"},
        "allowed": {"project_root", "rag_path", "chapter", "topics", "count", "mix", "difficulty", "scope", "top_k"},
        "defaults": {"count": 20},
    },
    "edu_explain": {
        "required": {"project_root", "rag_path", "question_or_topic"},
        "allowed": {"project_root", "rag_path", "question_or_topic", "scope", "top_k"},
        "defaults": {},
    },
    "edu_extract_tables": {
        "required": {"project_root", "rag_path"},
        "allowed": {"project_root", "rag_path", "chapter", "topic", "scope", "top_k"},
        "defaults": {},
    },
    "edu_build_blueprint": {
        "required": {"subject", "klass"},
        "allowed": {"subject", "klass", "board", "filename_hint", "session"},
        "defaults": {"board": "CBSE", "session": ""},
    },
    "edu_generate_paper": {
        "required": {"subject", "klass"},
        "allowed": {
            "subject", "klass", "weightage", "total_questions", "total_marks",
            "include_solutions", "board", "seed", "filename_hint"
        },
        "defaults": {"include_solutions": True, "board": "CBSE"},
    },
}

# Tools that are safe anywhere (document or code flows)
UNIVERSAL_TOOLS: Set[str] = {
    "rag_retrieve", "read_file", "list_files", "analyze_files", "web_search", "web_fetch",
    "edu_detect_intent", "edu_explain", "edu_extract_tables", "edu_similar_questions", "edu_question_paper",
    "edu_build_blueprint", "edu_generate_paper",
}


# Code-heavy tools (mutate or inspect codebase)
CODE_TOOLS: Set[str] = {
    "search_code", "scan_relevant_files", "write_file", "replace_in_file",
    "bulk_edit", "format_python_files", "rewrite_naive_open",
}

# Document-heavy tools
DOC_TOOLS: Set[str] = {
    "rag_retrieve", "web_search", "web_fetch",
    "edu_detect_intent", "edu_similar_questions", "edu_question_paper", "edu_explain", "edu_extract_tables",
    "edu_build_blueprint", "edu_generate_paper",
}



def allowlist_for_route(route: str) -> Set[str]:
    """
    Determine which tools to expose for a given route.
    """
    route = (route or "document").lower()
    if route == "code":
        return UNIVERSAL_TOOLS | CODE_TOOLS
    if route == "hybrid":
        return UNIVERSAL_TOOLS | CODE_TOOLS | DOC_TOOLS
    if route == "tabular":
        # Favor document tools; universal still allowed.
        return UNIVERSAL_TOOLS | DOC_TOOLS
    return UNIVERSAL_TOOLS | DOC_TOOLS  # document default


def normalize_args(tool: str, args: Dict[str, Any], specs: Dict[str, Dict[str, Any]] = TOOL_SPECS) -> Tuple[bool, Dict[str, Any], str | None]:
    """
    Fill defaults, drop extraneous keys, check required.
    """
    spec = specs.get(tool)
    if not spec:
        return False, args, f"Unknown tool: {tool}"

    fixed = dict(args or {})
    # defaults
    for k, v in (spec.get("defaults") or {}).items():
        fixed.setdefault(k, v)
    # required
    req = spec.get("required") or set()
    missing = [r for r in req if r not in fixed]
    if missing:
        return False, fixed, f"Missing required arg(s): {', '.join(missing)}"
    # allowed filter
    allowed = spec.get("allowed") or set()
    for k in list(fixed.keys()):
        if k not in allowed:
            fixed.pop(k, None)
    return True, fixed, None


# -------- Optional helpers used by planners/executors --------

_BIND_RX = re.compile(r"\{\{BIND:([A-Za-z0-9_]+)\.([A-Za-z0-9_\-]+)\}\}")

def apply_bindings(args: Dict[str, Any], last_by_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve placeholders like '{{BIND:search_code.paths}}' based on previous tool outputs.
    """

    def _subst(s: str) -> str:
        def _rep(m: re.Match[str]) -> str:
            t, field = m.group(1), m.group(2)
            src = last_by_tool.get(t)
            if src is None:
                return ""
            if isinstance(src, dict) and field in src:
                val = src.get(field)
            elif isinstance(src, dict) and field == "paths":
                res = src.get("result", src)
                if isinstance(res, list):
                    val = [h.get("file") for h in res if isinstance(h, dict) and h.get("file")]
                else:
                    val = None
            else:
                val = None
            if isinstance(val, (list, dict)):
                return json.dumps(val)
            return str(val or "")
        return _BIND_RX.sub(_rep, s)

    def _walk(o: Any):
        if isinstance(o, str):
            return _subst(o)
        if isinstance(o, list):
            return [_walk(x) for x in o]
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        return o

    return _walk(args)


def looks_like_placeholder(val: Any) -> bool:
    if not isinstance(val, (str, list, dict)):
        return False
    s = json.dumps(val) if not isinstance(val, str) else val
    return any(tok in s for tok in ["<path>", "<file>", "path/to", "<BIND>"])


def openai_schemas_from_specs(specs: Dict[str, Dict[str, Any]] = TOOL_SPECS) -> List[dict]:
    """
    Convert TOOL_SPECS into OpenAI function schemas. Excludes names starting with '_'.
    NOTE: This is optional—your registry already exports its own schemas, which should be preferred
    for model-facing tool exposure. This function is mainly for static checks or alternate adapters.
    """
    schemas: List[dict] = []
    # Generic property types (keep simple and permissive)
    generic_type = {"type": "string"}  # models don't need strict typing for discovery
    for name, spec in specs.items():
        if name.startswith("_"):
            continue
        props = {k: generic_type for k in (spec.get("allowed") or set())}
        req = list(spec.get("required") or [])
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool: {name}",
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": req,
                    },
                },
            }
        )
    return schemas
