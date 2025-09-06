# tools/enhanced_registry.py
from __future__ import annotations

import ast
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
from loguru import logger
from logging_decorators import log_call

class EnhancedToolRegistry:
    """
    Enhanced tool registry with additional code analysis capabilities
    """
    def __init__(self, project_root: str):
        from tools.registry import ToolRegistry
        self.tool_registry = ToolRegistry(project_root)
        self.root = self.tool_registry.root
        self.tools = self.tool_registry.tools
        logger.info("EnhancedToolRegistry init → root='{}' tools={}", str(self.root), len(self.tools))
        self._register_enhanced_tools()

    # ---------- Robust path resolving (search from project root) ----------
    def _rel(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.root))

    @lru_cache(maxsize=4096)
    def _all_files_named(self, name: str) -> list[Path]:
        matches = [p for p in self.root.rglob(name)]
        logger.debug("_all_files_named('{}') → {} match(es)", name, len(matches))
        return matches

    @lru_cache(maxsize=1)
    def _all_py_files(self) -> list[Path]:
        files = [p for p in self.root.rglob("*.py")]
        logger.debug("_all_py_files() → {} file(s)", len(files))
        return files

    def _resolve_path(self, path: str, subdir: str = "") -> Optional[str]:
        """
        Resolve 'path' relative to project root, with best-effort search.
        Strategy:
          1) Try subdir/path if subdir is provided.
          2) Try root/path as-is.
          3) If only a filename was given, rglob by filename.
          4) As a fallback, match any file whose tail endswith the provided path.
        Returns a *relative* path from self.root, or None if not found.
        """
        if not path:
            logger.debug("_resolve_path: empty path")
            return None
        path = str(Path(path))  # normalize separators
        # 1) subdir + path
        if subdir:
            cand = (self.root / subdir / path)
            if cand.exists():
                rel = self._rel(cand)
                logger.debug("_resolve_path: subdir hit → {}", rel)
                return rel

        # 2) root + path
        cand = (self.root / path)
        if cand.exists():
            rel = self._rel(cand)
            logger.debug("_resolve_path: direct hit → {}", rel)
            return rel

        name = Path(path).name
        matches = self._all_files_named(name)
        if matches:
            # prefer matches under subdir, else the shortest relative path
            if subdir:
                under = [p for p in matches if (self.root / subdir) in p.parents]
                if under:
                    rel = self._rel(under[0])
                    logger.debug("_resolve_path: filename hit under subdir → {}", rel)
                    return rel
            best = sorted(matches, key=lambda p: len(self._rel(p)))[0]
            rel = self._rel(best)
            logger.debug("_resolve_path: filename hit → {}", rel)
            return rel

        # 4) endswith match among python files (e.g., "app/rag/retriever.py")
        tail = path.replace("\\", "/")
        for p in self._all_py_files():
            if str(p).replace("\\", "/").endswith(tail):
                rel = self._rel(p)
                logger.debug("_resolve_path: endswith hit → {}", rel)
                return rel

        logger.debug("_resolve_path: not found path='{}' subdir='{}'", path, subdir)
        return None

    def call(self, name: str, **kwargs):
        """Delegate to the underlying tool registry"""
        logger.info("EnhancedToolRegistry.call('{}') args_keys={}", name, list(kwargs.keys()))
        return self.tool_registry.call(name, **kwargs)

    def list_tools(self) -> Dict[str, Any]:
        """
        Return OpenAI-style tool schemas. Ensure our enhanced tools — in particular
        `call_graph_for_function` — are advertised even if the base registry hasn't
        defined a schema for them.
        """
        base = dict(getattr(self.tool_registry, "schemas", {}) or {})
        if "call_graph_for_function" not in base and "call_graph_for_function" in self.tools:
            base["call_graph_for_function"] = {
                "name": "call_graph_for_function",
                "description": "Build a compact call graph for a function (uses AST).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function": {"type": "string", "description": "Function name to analyze"},
                        "file_hint": {"type": "string", "description": "Filename hint (e.g., retriever.py)"},
                        "project_root": {"type": "string", "description": "Project root; defaults to the app root"},
                        "subdir": {"type": "string", "description": "Optional subdir to constrain the search"},
                        "depth": {"type": "integer", "description": "Traversal depth (1=direct calls)"},
                        "project_only": {"type": "boolean", "description": "Skip external/builtins if True"},
                        "filter_noise": {"type": "boolean", "description": "Drop trivial/builtin calls"},
                    },
                    "required": ["function"],
                },
            }
        return base

    def _register_enhanced_tools(self):
        """Register enhanced code analysis tools"""
        logger.info("Registering enhanced tools…")

        def analyze_code_structure(path: str, subdir: str = ""):
            """Analyze code structure and dependencies"""
            resolved = self._resolve_path(path, subdir=subdir or "")
            if not resolved:
                msg = {"error": f"File not found under {self.root}: {path}", "path": path, "subdir": subdir}
                logger.error("analyze_code_structure: {}", msg["error"])
                return msg
            p = self.root / resolved
            logger.debug("analyze_code_structure: file='{}'", resolved)

            content = p.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(content)
                analysis = {'imports': [], 'functions': [], 'classes': [], 'dependencies': set()}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            analysis['imports'].append(n.name)
                            analysis['dependencies'].add(n.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis['imports'].append(node.module)
                            analysis['dependencies'].add(node.module.split('.')[0])
                    elif isinstance(node, ast.FunctionDef):
                        analysis['functions'].append({
                            'name': node.name, 'line': node.lineno, 'args': [arg.arg for arg in node.args.args]
                        })
                    elif isinstance(node, ast.ClassDef):
                        analysis['classes'].append({'name': node.name, 'line': node.lineno})
                analysis['dependencies'] = list(analysis['dependencies'])
                logger.info("analyze_code_structure: imports={} funcs={} classes={}",
                            len(analysis['imports']), len(analysis['functions']), len(analysis['classes']))
                return analysis
            except Exception as e:
                logger.exception("analyze_code_structure failed: {}", e)
                return {'error': str(e)}

        def find_related_files(main_file: str) -> list:
            """Find files related to the given file through imports"""
            try:
                analysis = analyze_code_structure(main_file)
                dependencies = analysis.get('dependencies', [])
                logger.info("find_related_files: deps={} for '{}'", len(dependencies), main_file)

                related_files = []
                for dep in dependencies:
                    # Look for files that might implement this dependency
                    search_results = self.call('search_code', query=dep, regex=False, case_sensitive=False)
                    logger.debug("find_related_files: dep='{}' hits={}", dep, len(search_results))
                    for result in search_results:
                        file_path = result['file']
                        if file_path not in related_files and file_path != main_file:
                            related_files.append(file_path)
                logger.info("find_related_files: related={}", len(related_files))
                return related_files
            except Exception as e:
                logger.exception("find_related_files failed: {}", e)
                return [{'error': str(e)}]

        def detect_errors(path: str, subdir: str = "") -> list:
            """Static analysis to detect potential errors"""
            resolved = self._resolve_path(path, subdir=subdir or "")
            if not resolved:
                msg = {"error": f"File not found under {self.root}: {path}", "path": path, "subdir": subdir}
                logger.error("detect_errors: {}", msg["error"])
                return msg
            p = self.root / resolved
            content = p.read_text(encoding="utf-8", errors="ignore")
            errors = []
            # Check for common issues

            if 'open(' in content and 'with open(' not in content:
                errors.append({'type': 'potential_resource_leak',
                               'message': 'File opened without context manager',
                               'line': None})
            # Check for broad exception handling
            if 'except:' in content or 'except Exception:' in content:
                errors.append({'type': 'broad_exception',
                               'message': 'Overly broad exception handling',
                               'line': None})
            logger.info("detect_errors: {} issue(s) in '{}'", len(errors), resolved)
            return errors

        def _find_function_definition(root: Path, func_name: str, subdir: str = "") -> Optional[str]:
            base = root / (subdir or "")
            for p in sorted(base.rglob("*.py")):
                try:
                    src = p.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == func_name:
                            rel = str(p.relative_to(root))
                            logger.debug("_find_function_definition: '{}' → {}", func_name, rel)
                            return rel
                except Exception:
                    continue
            logger.debug("_find_function_definition: '{}' not found", func_name)
            return None

        def _calls_in_function(src: str, func_name: str) -> List[str]:
            calls = []
            try:
                tree = ast.parse(src)
            except Exception:
                logger.debug("_calls_in_function: parse failed for '{}'", func_name)
                return calls

            class _Visitor(ast.NodeVisitor):
                def __init__(self):
                    self.in_target = False

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    if node.name == func_name:
                        self.in_target = True
                        self.generic_visit(node)
                        self.in_target = False
                    else:
                        # don't dive into other defs at top level
                        for n in node.body:
                            self.visit(n)

                def visit_Call(self, node: ast.Call):
                    if self.in_target:
                        # fn name can be Name or Attribute (e.g., mod.fn)
                        name = None
                        if isinstance(node.func, ast.Name):
                            name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            # grab the last attribute part
                            name = node.func.attr
                        if name:
                            calls.append(name)
                    # continue traversal
                    self.generic_visit(node)

            _Visitor().visit(tree)
            logger.debug("_calls_in_function('{}'): {} call(s)", func_name, len(calls))
            return calls

        def _dot_from_edges(center: str, edges: List[tuple]) -> str:
            # Graphviz DOT
            # center node is the queried function (boxed), others are ellipses
            lines = ["digraph G {", '  rankdir=LR;', '  node [shape=ellipse, fontsize=12];',
                     f'  "{center}" [shape=box, style=filled, fillcolor="#eef"];']
            seen = set()
            for u, v in edges:
                edge = (u, v)
                if edge in seen:
                    continue
                seen.add(edge)
                lines.append(f'  "{u}" -> "{v}";')
            lines.append("}")
            return "\n".join(lines)

        def call_graph_for_function(function: str, subdir: str = "",
                                    project_only: bool = True,
                                    filter_noise: bool = True,
                                    depth: int = 3, **_ignore) -> dict:
            """
            Build a call graph rooted at `function`.
            - locate the file that defines `function`
            - parse the function body to collect calls
            - recurse into callees up to `depth` levels
            depth=1 → only direct calls
            depth=N → recurse N levels
            depth=0 → unlimited (until no new project-local callees found)
            """
            logger.info("call_graph_for_function('{}', subdir='{}', depth={})", function, subdir, depth)
            visited = set()
            edges = []
            resolved_all = {}

            def expand(func: str, current_depth: int):
                if func in visited:
                    return
                if depth != 0 and current_depth > depth:
                    return
                visited.add(func)

                def_file = _find_function_definition(self.root, func, subdir=subdir)
                if not def_file:
                    return
                p = self.root / def_file
                try:
                    src = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    return

                callees = _calls_in_function(src, func)
                resolved = []
                for cal in sorted(set(callees)):
                    found_path = _find_function_definition(self.root, cal, subdir=subdir)
                    resolved.append({"name": cal, "defined_in": found_path})

                if filter_noise:
                    import builtins as _py_builtins
                    builtins_set = set(dir(_py_builtins))
                    noisy = {
                        "get", "values", "items", "keys", "extend", "append", "sort",
                        "split", "strip", "isalpha", "lower", "upper", "format",
                        "info", "warning", "debug", "error", "exception", "print",
                        "group", "groups",
                    }
                    filtered = []
                    for c in resolved:
                        name = c["name"]
                        if project_only and not c["defined_in"]:
                            continue
                        if name in noisy or name in builtins_set:
                            if not c["defined_in"]:
                                continue
                        filtered.append(c)
                    resolved = filtered

                resolved_all[func] = resolved
                for c in resolved:
                    edges.append((func, c["name"]))
                    expand(c["name"], current_depth + 1)

            expand(function, 1)
            dot = _dot_from_edges(function, edges)
            logger.info("call_graph_for_function: nodes={} edges={}", len(resolved_all), len(edges))
            return {
                "function": function,
                "file": _find_function_definition(self.root, function, subdir=subdir),
                "calls": resolved_all.get(function, []),
                "edges": edges,
                "dot": dot,
            }

        # Optional alias to match common planner wording
        def analyze_function(function: str, subdir: str = "", depth: int = 3, **_ignore):
            return call_graph_for_function(function=function, subdir=subdir, depth=depth)

        # Register enhanced tools
        self.tools["analyze_code_structure"]  = log_call("analyze_code_structure")(analyze_code_structure)
        self.tools["find_related_files"]      = log_call("find_related_files")(find_related_files)
        self.tools["detect_errors"]           = log_call("detect_errors")(detect_errors)
        self.tools["call_graph_for_function"] = log_call("call_graph_for_function", slow_ms=1500)(call_graph_for_function)
        self.tools["analyze_function"]        = log_call("analyze_function")(analyze_function)

        logger.info("Enhanced tools registered: {}", list(self.tools.keys()))
