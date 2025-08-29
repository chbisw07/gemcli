# tools/registry.py
import re
import difflib
import json
import shutil
from pathlib import Path
from typing import Optional, Any, Callable, Iterable, List, Dict

# Optional: direct command parser hook (safe to keep even if unused)
try:
    from tools.direct_parser import try_direct_actions
except Exception:  # pragma: no cover
    def try_direct_actions(*_, **__):
        return None


class ToolRegistry:
    """
    Simple in-process tool registry with a few repo-aware helpers.
    The public surface:
      - self.tools[name] = callable
      - call(name, **kwargs) -> Any
      - schemas_openai() -> list[dict]  (OpenAI function-calling schema)
    """
    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        self.tools: Dict[str, Callable[..., Any]] = {}
        self._register_builtin_tools()

    # ---------------- registration & dispatch ----------------

    def _register(self, name: str, fn: Callable):
        self.tools[name] = fn

    def call(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' is not registered.")
        return self.tools[name](**kwargs)

    def try_direct(self, query: str) -> Optional[dict]:
        return try_direct_actions(query, self.root)

    # -------------------- built-in tools ---------------------

    def _register_builtin_tools(self):
        # ---------- read_file ----------
        def read_file(path: str) -> str:
            p = self.root / path
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Not found: {path}")
            return p.read_text(encoding="utf-8", errors="ignore")

        # ---------- list_files ----------
        def list_files(
            subdir: str = "",
            exts: Iterable[str] | None = None,
            ignore_globs: Iterable[str] | None = None,
        ) -> List[str]:
            """
            List files under `subdir`, optionally filtered by file suffix and ignore globs.
            exts: [".py", ".js"]  (defaults to [".py"])
            ignore_globs: ["*/node_modules/*", "*/.venv/*"]
            """
            target = self.root / (subdir or "")
            if not target.exists():
                return []
            if exts is None:
                exts = [".py"]
            exts = tuple(exts)
            ignore_globs = tuple(ignore_globs or [])
            out: List[str] = []
            for p in target.rglob("*"):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(self.root))
                if ignore_globs and any(p.match(g) or Path(rel).match(g) for g in ignore_globs):
                    continue
                if p.suffix in exts:
                    out.append(rel)
            return sorted(out)

        # ---------- search_code ----------
        def search_code(
            query: str,
            subdir: str = "",
            regex: bool = False,
            case_sensitive: bool = False,
            exts: Iterable[str] | None = None,
            max_results: int = 2000,
        ) -> List[dict]:
            """
            Search for a string/regex across files (default: .py) and return line hits.
            """
            target = self.root / (subdir or "")
            if not target.exists():
                return []
            if exts is None:
                exts = [".py"]
            exts = tuple(exts)
            flags = 0 if case_sensitive else re.IGNORECASE
            pat = re.compile(query, flags) if regex else None
            results: List[dict] = []
            for path in target.rglob("*"):
                if not path.is_file() or path.suffix not in exts:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if regex:
                        if pat.search(line):
                            results.append({"file": str(path.relative_to(self.root)), "line": i, "text": line.rstrip()})
                    else:
                        if (query if case_sensitive else query.lower()) in (line if case_sensitive else line.lower()):
                            results.append({"file": str(path.relative_to(self.root)), "line": i, "text": line.rstrip()})
                    if len(results) >= max_results:
                        return results
            return results

        # ---------- write_file ----------
        def write_file(
            path: str,
            content: str,
            overwrite: bool = True,
            backup: bool = False,
        ) -> dict:
            """
            Write text to a file. If backup=True and file exists, create <file>.bak first.
            """
            p = self.root / path
            p.parent.mkdir(parents=True, exist_ok=True)
            existed = p.exists()
            if existed and not overwrite:
                raise FileExistsError(f"File exists: {path}")
            if existed and backup:
                bak = p.with_suffix(p.suffix + ".bak")
                shutil.copyfile(p, bak)
            p.write_text(content, encoding="utf-8")
            return {
                "path": str(p.relative_to(self.root)),
                "bytes_written": len(content.encode("utf-8")),
                "overwrote": existed,
                "backup": bool(existed and backup),
            }

        # ---------- replace_in_file ----------
        def replace_in_file(
            path: str,
            find: str,
            replace: str,
            regex: bool = False,
            dry_run: bool = False,
            backup: bool = True,
        ) -> dict:
            """
            Find/replace in a single file, return count + unified diff. Optionally write.
            """
            p = self.root / path
            if not p.exists():
                raise FileNotFoundError(f"Not found: {path}")
            original = p.read_text(encoding="utf-8", errors="ignore")
            pattern = find if regex else re.escape(find)
            matches = list(re.finditer(pattern, original))
            new_text = re.sub(pattern, replace, original)
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=path,
                    tofile=path,
                )
            )
            if not dry_run and matches:
                if backup:
                    bak = p.with_suffix(p.suffix + ".bak")
                    bak.write_text(original, encoding="utf-8")
                p.write_text(new_text, encoding="utf-8")
            return {
                "path": str(p.relative_to(self.root)),
                "replacements": len(matches),
                "dry_run": bool(dry_run),
                "diff": diff,
            }

        # ---------- format_python_files ----------
        def format_python_files(
            subdir: str = "",
            line_length: int = 88,
            dry_run: bool = False,
        ) -> dict:
            """
            Run Black in-memory and write changes (unless dry_run). Requires 'black' installed.
            Returns {relative_path: {"diff": "..."} } for changed files.
            """
            try:
                import black  # type: ignore
            except Exception:
                raise RuntimeError("Black is not installed. `pip install black` to use format_python_files.")
            target = self.root / (subdir or "")
            if not target.exists():
                return {}
            changed: Dict[str, dict] = {}
            mode = black.Mode(line_length=line_length)
            for path in sorted(target.rglob("*.py")):
                try:
                    orig = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                try:
                    formatted = black.format_file_contents(orig, fast=True, mode=mode)
                except black.NothingChanged:
                    continue
                if formatted != orig:
                    if not dry_run:
                        path.write_text(formatted, encoding="utf-8")
                    rel = str(path.relative_to(self.root))
                    changed[rel] = {
                        "diff": "".join(
                            difflib.unified_diff(
                                orig.splitlines(keepends=True),
                                formatted.splitlines(keepends=True),
                                fromfile=rel,
                                tofile=rel,
                            )
                        )
                    }
            return changed

        # ---------- scan_relevant_files ----------
        def scan_relevant_files(
            prompt: str = "",
            path: str = "",
            max_results: int = 200,
            exts: Iterable[str] | None = None,
        ) -> List[dict]:
            """
            Heuristic scan: score files by whether prompt terms appear in path/content.
            """
            base = self.root / (path or "")
            if not base.exists():
                return []
            if exts is None:
                exts = [".py"]
            exts = tuple(exts)
            terms = [t.lower() for t in re.split(r"[^A-Za-z0-9_.:/-]+", prompt or "") if t]
            hits: List[dict] = []
            for p in base.rglob("*"):
                if not p.is_file() or p.suffix not in exts:
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore").lower()
                except Exception:
                    continue
                score = 0
                hay = (str(p.relative_to(self.root)).lower() + " " + text)
                for t in terms:
                    if t and t in hay:
                        score += 1
                if score > 0:
                    hits.append({"path": str(p.relative_to(self.root)), "score": score})
            hits.sort(key=lambda x: (-x["score"], x["path"]))
            return hits[:max_results]

        # ---------- analyze_files ----------
        def analyze_files(paths: List[str], max_bytes: int = 8000) -> List[dict]:
            """
            Lightweight analysis of specific files: size, line count, head/tail snippets.
            """
            out: List[dict] = []
            for rel in paths or []:
                p = self.root / rel
                if not p.exists() or not p.is_file():
                    out.append({"path": rel, "error": "not found"})
                    continue
                data = p.read_text(encoding="utf-8", errors="ignore")
                lines = data.splitlines()
                head = "\n".join(lines[: min(20, len(lines))])
                tail = "\n".join(lines[max(0, len(lines) - 20) :])
                out.append(
                    {
                        "path": rel,
                        "bytes": len(data.encode("utf-8")),
                        "lines": len(lines),
                        "head": head[: max_bytes // 2],
                        "tail": tail[: max_bytes // 2],
                    }
                )
            return out

        # ---------- bulk_edit ----------
        def bulk_edit(edits: List[dict], dry_run: bool = True, backup: bool = True) -> List[dict]:
            """
            Apply multiple find/replace edits.
            edits = [
              {"path":"a.py","find":"foo","replace":"bar","regex":false},
              ...
            ]
            Returns a list of {path, replacements, dry_run, diff, [error]}.
            """
            results: List[dict] = []
            for e in edits or []:
                rel = e.get("path")
                if not rel:
                    results.append({"path": rel or "", "error": "missing path"})
                    continue
                try:
                    res = replace_in_file(
                        path=rel,
                        find=e.get("find", ""),
                        replace=e.get("replace", ""),
                        regex=bool(e.get("regex", False)),
                        dry_run=dry_run,
                        backup=backup,
                    )
                    results.append(res)
                except Exception as ex:
                    results.append({"path": rel, "error": str(ex), "dry_run": dry_run})
            return results

        # ---------- rewrite_naive_open ----------
        def rewrite_naive_open(
            dir: str = ".",
            exts: Iterable[str] | None = None,
            dry_run: bool = True,
            backup: bool = True,
            aggressive: bool = False,
        ) -> List[dict]:
            """
            Convert simple `open()` usages into `with open()` blocks.
            - Detects <var> = open(...) and <var>.close()
            - Handles trailing comments and whitespace
            - If aggressive=True, will rewrite even without a matching `.close()`
            """
            if exts is None:
                exts = [".py"]
            exts = tuple(exts)
            base = self.root / (dir or ".")
            if not base.exists():
                return []

            results: List[dict] = []
            # Regex to catch '<var> = open(...)' allowing inline comments
            open_re = re.compile(
                r"""^\s*(?P<var>[A-Za-z_]\w*)\s*=\s*open\((?P<args>.+)\)\s*(#.*)?$"""
            )
            close_re_tpl = r"""^\s*{var}\.close\(\)\s*(#.*)?$"""

            for p in sorted(base.rglob("*")):
                if not p.is_file() or p.suffix not in exts:
                    continue
                original = p.read_text(encoding="utf-8", errors="ignore")
                lines = original.splitlines(keepends=True)

                i = 0
                changed = False
                while i < len(lines):
                    m = open_re.match(lines[i])
                    if not m:
                        i += 1
                        continue
                    var = m.group("var")
                    args = m.group("args")
                    # Look ahead for matching '<var>.close()'
                    j = i + 1
                    close_re = re.compile(close_re_tpl.format(var=re.escape(var)))
                    close_idx = -1
                    while j < len(lines):
                        if close_re.match(lines[j]):
                            close_idx = j
                            break
                        j += 1
                    if close_idx == -1 and not aggressive:
                        i += 1
                        continue  # skip if not aggressive
                    if close_idx == -1 and aggressive:
                        close_idx = i  # treat immediate close

                    indent = re.match(r"^(\s*)", lines[i]).group(1)
                    with_line = f"{indent}with open({args}) as {var}:\n"
                    body = []
                    for k in range(i + 1, close_idx):
                        body_line = lines[k]
                        if body_line.strip():
                            body.append(indent + "    " + body_line.lstrip())
                        else:
                            body.append(body_line)
                    new_chunk = [with_line] + body
                    lines[i : close_idx + 1] = new_chunk
                    changed = True
                    i += len(new_chunk)
                if not changed:
                    continue

                new_text = "".join(lines)
                diff = "".join(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        new_text.splitlines(keepends=True),
                        fromfile=str(p.relative_to(self.root)),
                        tofile=str(p.relative_to(self.root)),
                    )
                )
                if not dry_run:
                    if backup:
                        bak = p.with_suffix(p.suffix + ".bak")
                        bak.write_text(original, encoding="utf-8")
                    p.write_text(new_text, encoding="utf-8")
                results.append(
                    {
                        "path": str(p.relative_to(self.root)),
                        "replacements": 1,  # at least one with-block introduced; you could count more precisely if desired
                        "dry_run": bool(dry_run),
                        "diff": diff,
                    }
                )
            return results

        # -------------------- register everything --------------------

        self._register("read_file", read_file)
        self._register("list_files", list_files)
        self._register("search_code", search_code)
        self._register("write_file", write_file)
        self._register("replace_in_file", replace_in_file)
        self._register("format_python_files", format_python_files)
        self._register("scan_relevant_files", scan_relevant_files)
        self._register("analyze_files", analyze_files)
        self._register("bulk_edit", bulk_edit)
        self._register("rewrite_naive_open", rewrite_naive_open)

    # --------------- OpenAI tool schema (function-calling) ---------------

    def schemas_openai(self) -> List[dict]:
        """
        Advertise tools in OpenAI function-calling format so compatible models
        can emit native tool_calls. This mirrors names/params in this registry.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a text file at a relative path.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files under a subdir filtered by extensions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subdir": {"type": "string"},
                            "exts": {"type": "array", "items": {"type": "string"}},
                            "ignore_globs": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search files for a string/regex and return file/line hits.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "subdir": {"type": "string"},
                            "regex": {"type": "boolean"},
                            "case_sensitive": {"type": "boolean"},
                            "exts": {"type": "array", "items": {"type": "string"}},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file (optionally backup and/or refuse overwrite).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "overwrite": {"type": "boolean"},
                            "backup": {"type": "boolean"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "replace_in_file",
                    "description": "Find/replace in a file; emit unified diff; optionally write.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                            "regex": {"type": "boolean"},
                            "dry_run": {"type": "boolean"},
                            "backup": {"type": "boolean"},
                        },
                        "required": ["path", "find", "replace"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "format_python_files",
                    "description": "Run Black formatter over Python files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subdir": {"type": "string"},
                            "line_length": {"type": "integer"},
                            "dry_run": {"type": "boolean"},
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scan_relevant_files",
                    "description": "Heuristically score files relevant to a prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "path": {"type": "string"},
                            "max_results": {"type": "integer"},
                            "exts": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_files",
                    "description": "Summarize basic stats and small snippets for specific files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {"type": "array", "items": {"type": "string"}},
                            "max_bytes": {"type": "integer"},
                        },
                        "required": ["paths"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bulk_edit",
                    "description": "Apply multiple find/replace edits across files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edits": {"type": "array", "items": {"type": "object"}},
                            "dry_run": {"type": "boolean"},
                            "backup": {"type": "boolean"},
                        },
                        "required": ["edits"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rewrite_naive_open",
                    "description": "Convert simple open()/close() pairs into with-open blocks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir": {"type": "string"},
                            "exts": {"type": "array", "items": {"type": "string"}},
                            "dry_run": {"type": "boolean"},
                            "backup": {"type": "boolean"},
                        },
                        "required": [],
                    },
                },
            },
        ]
