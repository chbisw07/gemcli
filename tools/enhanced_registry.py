# tools/enhanced_registry.py
import ast
import re
from pathlib import Path
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
    
    def call(self, name: str, **kwargs):
        """Delegate to the underlying tool registry"""
        return self.tool_registry.call(name, **kwargs)
    
    def _register_enhanced_tools(self):
        """Register enhanced code analysis tools"""
        import ast
        
        def analyze_code_structure(path: str) -> dict:
            """Analyze code structure and dependencies"""
            p = self.root / path
            if not p.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
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
            p = self.root / path
            if not p.exists():
                return [{'error': f"File not found: {path}"}]
            
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
                                    filter_noise: bool = True) -> dict:
            """
            Build a call graph rooted at `function`:
            - locate the file that defines `function`
            - parse the function body to collect direct calls
            - try to resolve where those callees are defined
            Returns: {
              "function": "<name>",
              "file": "<relative path>",
              "calls": [{"name": "...", "defined_in": "<path>|None"}],
              "edges": [["function", "callee"], ...],
              "dot": "graphviz DOT string"
            }
            """
            # 1) find definition file
            def_file = _find_function_definition(self.root, function, subdir=subdir)
            if not def_file:
                return {"error": f"Could not find definition for function '{function}'", "function": function}

            p = self.root / def_file
            src = p.read_text(encoding="utf-8", errors="ignore")

            # 2) collect direct calls inside the function body
            callees = _calls_in_function(src, function)

            # 3) resolve where callees are defined (best-effort scan by name)
            resolved = []
            for cal in sorted(set(callees)):
                found_path = _find_function_definition(self.root, cal, subdir=subdir)
                resolved.append({"name": cal, "defined_in": found_path})

            # 3b) optionally filter out noise:
            if filter_noise:
                import builtins as _py_builtins
                builtins_set = set(dir(_py_builtins))
                # common method / logging / regex names we often don't want as “edges”
                noisy = {
                    # containers / dicts
                    "get", "values", "items", "keys", "extend", "append", "sort",
                    # strings
                    "split", "strip", "isalpha", "lower", "upper", "format",
                    # logging / prints
                    "info", "warning", "debug", "error", "exception", "print",
                    # regex match objects
                    "group", "groups",
                }
                filtered = []
                for c in resolved:
                    name = c["name"]
                    if project_only and not c["defined_in"]:
                        # keep ONLY project-local functions that we could resolve
                        continue
                    if name in noisy or name in builtins_set:
                        # drop builtins & common method noise
                        if not c["defined_in"]:
                            continue
                    filtered.append(c)
                resolved = filtered

            # 4) edges and DOT
            edges = [(function, c["name"]) for c in resolved]
            dot = _dot_from_edges(function, edges)

            return {
                "function": function,
                "file": def_file,
                "calls": resolved,
                "edges": edges,
                "dot": dot,
            }
        
        # Register enhanced tools
        self.tools["analyze_code_structure"] = analyze_code_structure
        self.tools["find_related_files"] = find_related_files
        self.tools["detect_errors"] = detect_errors

        self.tools["call_graph_for_function"] = call_graph_for_function
