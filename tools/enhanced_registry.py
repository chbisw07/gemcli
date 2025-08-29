# tools/enhanced_registry.py
from typing import List, Dict, Any
import ast

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
        
        # Register enhanced tools
        self.tools["analyze_code_structure"] = analyze_code_structure
        self.tools["find_related_files"] = find_related_files
        self.tools["detect_errors"] = detect_errors