# tools/enhanced_registry.py
import ast
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional

class EnhancedToolRegistry:
    """
    Enhanced tool registry with additional code analysis capabilities
    """
    def __init__(self, project_root: str):
        from tools.registry import ToolRegistry
        self.tool_registry = ToolRegistry(project_root)
        self.root = self.tool_registry.root
        self.tools = self.tool_registry.tools
        self._register_enhanced_tools()


    # ---------- Robust path resolving (search from project root) ----------
    def _rel(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.root))

    @lru_cache(maxsize=4096)
    def _all_files_named(self, name: str) -> list[Path]:
        return [p for p in self.root.rglob(name)]

    @lru_cache(maxsize=1)
    def _all_py_files(self) -> list[Path]:
        return [p for p in self.root.rglob("*.py")]

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
            return None
        path = str(Path(path))  # normalize separators
        # 1) subdir + path
        if subdir:
            cand = (self.root / subdir / path)
            if cand.exists():
                return self._rel(cand)
        # 2) root + path
        cand = (self.root / path)
        if cand.exists():
            return self._rel(cand)
        # 3) filename match
        name = Path(path).name
        matches = self._all_files_named(name)
        if matches:
            # prefer matches under subdir, else the shortest relative path
            if subdir:
                under = [p for p in matches if (self.root / subdir) in p.parents]
                if under:
                    return self._rel(under[0])
            best = sorted(matches, key=lambda p: len(self._rel(p)))[0]
            return self._rel(best)
        # 4) endswith match among python files (e.g., "app/rag/retriever.py")
        tail = path.replace("\\", "/")
        for p in self._all_py_files():
            if str(p).replace("\\", "/").endswith(tail):
                return self._rel(p)
        return None
        
    def call(self, name: str, **kwargs):
        """Delegate to the underlying tool registry"""
        return self.tool_registry.call(name, **kwargs)
    
    def _register_enhanced_tools(self):
        """Register enhanced code analysis tools"""
        import ast
        
        def analyze_code_structure(path: str, subdir: str = ""):
            """Analyze code structure and dependencies"""
            resolved = self._resolve_path(path, subdir=subdir or "")
            if not resolved:
                return {"error": f"File not found under {self.root}: {path}", "path": path, "subdir": subdir}
            p = self.root / resolved
            
            content = p.read_text()
            try:
                tree = ast.parse(content)
                analysis = {
                    'imports': [],
                    'functions': [],
                    'classes': [],
                    'dependencies': set()
                }
                
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
                            'name': node.name,
                            'line': node.lineno,
                            'args': [arg.arg for arg in node.args.args]
                        })
                    elif isinstance(node, ast.ClassDef):
                        analysis['classes'].append({
                            'name': node.name,
                            'line': node.lineno
                        })
                
                analysis['dependencies'] = list(analysis['dependencies'])
                return analysis
            except Exception as e:
                return {'error': str(e)}
        
        def find_related_files(main_file: str) -> list:
            """Find files related to the given file through imports"""
            try:
                analysis = analyze_code_structure(main_file)
                dependencies = analysis.get('dependencies', [])
                
                related_files = []
                for dep in dependencies:
                    # Look for files that might implement this dependency
                    search_results = self.call('search_code', query=dep, regex=False, case_sensitive=False)
                    
                    for result in search_results:
                        file_path = result['file']
                        if file_path not in related_files and file_path != main_file:
                            related_files.append(file_path)
                
                return related_files
            except Exception as e:
                return [{'error': str(e)}]
        
        def detect_errors(path: str) -> list:
            """Static analysis to detect potential errors"""
            resolved = self._resolve_path(path, subdir=subdir)
            if not resolved:
                return {"error": f"File not found under {self.root}: {path}", "path": path, "subdir": subdir}
            p = self.root / resolved

            content = p.read_text()
            errors = []
            
            # Check for common issues
            if 'open(' in content and 'with open(' not in content:
                errors.append({
                    'type': 'potential_resource_leak',
                    'message': 'File opened without context manager',
                    'line': None  # Would need more sophisticated parsing
                })
            
            # Check for broad exception handling
            if 'except:' in content or 'except Exception:' in content:
                errors.append({
                    'type': 'broad_exception',
                    'message': 'Overly broad exception handling',
                    'line': None
                })
            
            return errors
        
        def _find_function_definition(root: Path, func_name: str, subdir: str = "") -> Optional[str]:
            base = root / (subdir or "")
            for p in sorted(base.rglob("*.py")):
                try:
                    src = p.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == func_name:
                            return str(p.relative_to(root))
                except Exception:
                    continue
            return None

        def _calls_in_function(src: str, func_name: str) -> List[str]:
            calls = []
            try:
                tree = ast.parse(src)
            except Exception:
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
                            depth: int = 3) -> dict:
            """
            Build a call graph rooted at `function`:
            - locate the file that defines `function`
            - parse the function body to collect calls
            - recurse into callees up to `depth` levels
            depth=1 → only direct calls
            depth=N → recurse N levels
            depth=0 → unlimited (until no new project-local callees found)
            """
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
            return {
                "function": function,
                "file": _find_function_definition(self.root, function, subdir=subdir),
                "calls": resolved_all.get(function, []),
                "edges": edges,
                "dot": dot,
            }

        # Register enhanced tools
        self.tools["analyze_code_structure"] = analyze_code_structure
        self.tools["find_related_files"] = find_related_files
        self.tools["detect_errors"] = detect_errors

        self.tools["call_graph_for_function"] = call_graph_for_function
