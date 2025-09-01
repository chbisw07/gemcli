import ast, os, hashlib

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def chunk(file_path: str, cfg: dict):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    chunks = []

    # module chunk
    chunks.append({
        "id": _sha(file_path + ":module"),
        "document": src[: cfg["chunking"]["py"]["max_body_chars"]],
        "metadata": {
            "file_path": os.path.relpath(file_path),
            "symbol_kind": "module",
            "symbol_name": os.path.basename(file_path),
            "qualified_name": os.path.relpath(file_path),
            "start_line": 1,
            "end_line": len(lines),
        }
    })

    class V(ast.NodeVisitor):
        def __init__(self): self.stack=[]
        def _emit(self, node, kind, qname):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            body = "\n".join(lines[start-1:end])
            doc = ast.get_docstring(node) or ""
            text = f"{kind} {qname}\n\nDocstring:\n{doc}\n\nBody:\n{body}"
            chunks.append({
                "id": _sha(file_path + ":" + qname),
                "document": text[: cfg["chunking"]["py"]["max_body_chars"]],
                "metadata": {
                    "file_path": os.path.relpath(file_path),
                    "symbol_kind": kind,
                    "symbol_name": getattr(node, "name", qname.split(".")[-1]),
                    "qualified_name": f"{os.path.relpath(file_path)}:{qname}",
                    "start_line": start,
                    "end_line": end,
                }
            })
        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self._emit(node, "class", ".".join(self.stack))
            self.generic_visit(node)
            self.stack.pop()
        def visit_FunctionDef(self, node):
            kind = "method" if self.stack else "function"
            self.stack.append(node.name)
            self._emit(node, kind, ".".join(self.stack))
            self.stack.pop()
        def visit_AsyncFunctionDef(self, node): self.visit_FunctionDef(node)

    V().visit(tree)
    return chunks
