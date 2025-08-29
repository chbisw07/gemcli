#!/usr/bin/env python3
"""
Gemini‑style Code Assistant (gemcli)

Fresh rewrite — Aug 28, 2025

Features
- Works with OpenAI and OpenAI‑compatible local servers (LM Studio, Ollama, etc.).
- Optional Google Gemini support if `google.genai` is installed.
- Native OpenAI tool calling when available; smart fallback to a JSON‑Tool protocol.
- Robust error messages + automatic fallback from /chat/completions → /responses on HTTP 400.
- Auto‑loads `.env` (OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.) if python‑dotenv is installed.
- Minimal but useful repo tools: read_file, list_files, search_code.

Usage
  pip install requests python-dotenv
  python gemcli.py --config ./model_config.json --root /path/to/repo --model gpt-4o
  python gemcli.py --config ./model_config.json --root . --model deepseek-coder-6.7b-instruct
  python gemcli.py -c model_config.json -r . -m "Find places that open files but never close them"

Notes
- For LM Studio/Ollama, run their OpenAI-compatible server (e.g., http://localhost:1234/v1) and ensure the chosen
  model is available there. No API key required for local servers.
- For OpenAI, export OPENAI_API_KEY in your shell or place it in a `.env` file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:
    print("Please `pip install requests`.")
    sys.exit(1)

# Optional .env auto-load
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    _HAVE_DOTENV = True
except Exception:
    _HAVE_DOTENV = False

# ---------------- Config ----------------

@dataclass
class LLMModel:
    name: str
    provider: str
    endpoint: str
    model: Optional[str] = None
    temperature: float = 0.2
    system_prompt: str = "You are a helpful coding assistant."
    max_context_tokens: int = 8192
    api_key_reqd: bool = False
    api_key_env: Optional[str] = None
    tags: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMModel":
        return cls(
            name=d.get("name"),
            provider=d.get("provider", "openai"),
            endpoint=d.get("endpoint", "https://api.openai.com/v1"),
            model=d.get("model"),
            temperature=float(d.get("temperature", 0.2)),
            system_prompt=d.get("system_prompt", "You are a helpful coding assistant."),
            max_context_tokens=int(d.get("max_context_tokens", 8192)),
            api_key_reqd=bool(d.get("api_key_reqd", False)),
            api_key_env=d.get("api_key_env"),
            tags=d.get("tags", []),
        )

    def resolved_model(self) -> str:
        return self.model or self.name

    def is_openai_compat(self) -> bool:
        prov = (self.provider or "").lower()
        host = (self.endpoint or "").lower()
        return (
            prov in {"openai", "deepseek", "mistral", "qwen", "anyscale", "groq"}
            or host.startswith("http://")
            or host.startswith("https://")
        )

    def is_gemini(self) -> bool:
        return (self.provider or "").lower() in {"google", "gemini"}

    def supports_thinking(self) -> bool:
        name = (self.name or "").lower()
        tags = [t.lower() for t in (self.tags or [])]
        sys_p = (self.system_prompt or "").lower()
        return (
            "think" in name or "reason" in name or
            any(t in {"thinking", "reasoning", "chain-of-thought"} for t in tags) or
            "reason" in sys_p
        )


@dataclass
class ModelConfig:
    default_model_name: str
    llm_models: Dict[str, LLMModel]

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        data = json.loads(path.read_text())
        models: Dict[str, LLMModel] = {}
        for m in data.get("llm_models", []):
            mm = LLMModel.from_dict(m)
            models[mm.name] = mm
        default_name = data.get("default_llm_model") or next(iter(models), "gpt-4o")
        return cls(default_model_name=default_name, llm_models=models)

    def pick_best_tool_thinker(self) -> LLMModel:
        cands = [m for m in self.llm_models.values() if m.supports_thinking() and (m.is_openai_compat() or m.is_gemini())]
        if cands:
            # Prefer OpenAI-compatible for native tools first
            for m in cands:
                if m.is_openai_compat():
                    return m
            return cands[0]
        return self.llm_models.get(self.default_model_name) or list(self.llm_models.values())[0]


# ---------------- Tools ----------------

class ToolError(Exception):
    pass

class ToolRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._tools: Dict[str, Dict[str, Any]] = {
            "read_file": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a text file relative to project root and return its content (truncated).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "max_bytes": {"type": "integer", "default": 20000},
                                "encoding": {"type": "string", "default": "utf-8"}
                            },
                            "required": ["path"]
                        },
                    },
                },
                "impl": self._read_file,
            },
            "list_files": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": "List files under a directory filtered by extensions.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "dir": {"type": "string", "default": "."},
                                "exts": {"type": "array", "items": {"type": "string"}, "default": [".py", ".js", ".jsx", ".md"]},
                                "max_results": {"type": "integer", "default": 1000}
                            },
                        },
                    },
                },
                "impl": self._list_files,
            },
            "search_code": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "description": "Search for a substring in code files under project root and return matches (path,line,text).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "exts": {"type": "array", "items": {"type": "string"}, "default": [".py", ".js", ".jsx", ".md"]},
                                "max_results": {"type": "integer", "default": 50}
                            },
                            "required": ["query"]
                        },
                    },
                },
                "impl": self._search_code,
            },
            # ---------- NEW: Auto-format Python files with Black ----------
            "format_python_files": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "format_python_files",
                        "description": "Format Python files using Black (if installed). Accepts a list of paths or a directory scan.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "paths": {"type": "array", "items": {"type": "string"}, "description": "Specific .py files to format. If omitted, scans dir/exts."},
                                "dir": {"type": "string", "default": "."},
                                "exts": {"type": "array", "items": {"type": "string"}, "default": [".py"]},
                                "line_length": {"type": "integer", "default": 88},
                                "preview": {"type": "boolean", "default": False, "description": "If true, do not write; return diffs only."}
                            }
                        }
                    }
                },
                "impl": self._format_python_files,
            },
            # ---------- NEW: Rewrite naive open() patterns into with open(...) as f ----------
            "rewrite_naive_open": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "rewrite_naive_open",
                        "description": "Conservatively rewrite common unsafe open() usages into context managers. Supports single-line read()/write() chains and simple two-line patterns. Returns unified diffs. Backups are created by default.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "dir": {"type": "string", "default": "."},
                                "exts": {"type": "array", "items": {"type": "string"}, "default": [".py"]},
                                "dry_run": {"type": "boolean", "default": True},
                                "backup": {"type": "boolean", "default": True},
                                "max_files": {"type": "integer", "default": 200}
                            }
                        }
                    }
                },
                "impl": self._rewrite_naive_open,
            },
            "scan_relevant_files": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "scan_relevant_files",
                        "description": "Scan codebase for files relevant to a natural-language prompt by matching keywords.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "dir": {"type": "string", "default": "."},
                                "exts": {"type": "array", "items": {"type": "string"}, "default": [".py", ".js", ".jsx", ".md"]},
                                "case_sensitive": {"type": "boolean", "default": False},
                                "max_files": {"type": "integer", "default": 40}
                            },
                            "required": ["prompt"]
                        }
                    }
                },
                "impl": self._scan_relevant_files,
            },
            "analyze_files": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "analyze_files",
                        "description": "Analyze files: line count, has TODO/FIXME, json.loads, open(), etc.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "paths": {"type": "array", "items": {"type": "string"}},
                                "max_bytes": {"type": "integer", "default": 20000}
                            },
                            "required": ["paths"]
                        }
                    }
                },
                "impl": self._analyze_files,
            },
            "write_file": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Create or overwrite a text file. Optionally back up the old version.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "overwrite": {"type": "boolean", "default": False},
                                "backup": {"type": "boolean", "default": True}
                            },
                            "required": ["path", "content"]
                        }
                    }
                },
                "impl": self._write_file,
            },
            "replace_in_file": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "replace_in_file",
                        "description": "Replace text in a single file using literal or regex find.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "find": {"type": "string"},
                                "replace": {"type": "string"},
                                "regex": {"type": "boolean", "default": False},
                                "count": {"type": "integer", "default": 0},
                                "dry_run": {"type": "boolean", "default": False},
                                "backup": {"type": "boolean", "default": True}
                            },
                            "required": ["path", "find", "replace"]
                        }
                    }
                },
                "impl": self._replace_in_file,
            },
            "bulk_edit": {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "bulk_edit",
                        "description": "Apply multiple edits across files.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "edits": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "path": {"type": "string"},
                                            "find": {"type": "string"},
                                            "replace": {"type": "string"},
                                            "regex": {"type": "boolean", "default": False},
                                            "count": {"type": "integer", "default": 0}
                                        },
                                        "required": ["path", "find", "replace"]
                                    }
                                },
                                "dry_run": {"type": "boolean", "default": False},
                                "backup": {"type": "boolean", "default": True}
                            },
                            "required": ["edits"]
                        }
                    }
                },
                "impl": self._bulk_edit,
            },
        }

    # Tool impls
    def _read_file(self, path: str, max_bytes: int = 20000, encoding: str = "utf-8") -> Dict[str, Any]:
        fp = (self.root / path).resolve()
        if not str(fp).startswith(str(self.root)):
            raise ToolError("Path traversal blocked.")
        if not fp.exists() or not fp.is_file():
            raise ToolError(f"File not found: {path}")
        data = fp.read_bytes()[: max(100, min(max_bytes, 2_000_000))]
        try:
            text = data.decode(encoding, errors="replace")
        except Exception as e:
            raise ToolError(f"Decoding failed: {e}")
        return {"path": str(fp.relative_to(self.root)), "bytes": len(data), "content": text}

    def _list_files(self, dir: str = ".", exts: Optional[List[str]] = None, max_results: int = 1000) -> Dict[str, Any]:
        exts = exts or [".py", ".js", ".jsx", ".md"]
        base = (self.root / dir).resolve()
        if not str(base).startswith(str(self.root)):
            raise ToolError("Path traversal blocked.")
        out: List[str] = []
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in exts:
                out.append(str(p.relative_to(self.root)))
                if len(out) >= max_results:
                    break
        return {"dir": str(base.relative_to(self.root)), "count": len(out), "files": out}

    def _search_code(self, query: str, exts: Optional[List[str]] = None, max_results: int = 50) -> Dict[str, Any]:
        exts = exts or [".py", ".js", ".jsx", ".md"]
        matches: List[Dict[str, Any]] = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix not in exts:
                continue
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), start=1):
                    if query in line:
                        matches.append({"path": str(p.relative_to(self.root)), "line": i, "text": line.strip()})
                        if len(matches) >= max_results:
                            return {"query": query, "matches": matches}
            except Exception:
                continue
        return {"query": query, "matches": matches}

    # ---------- NEW: Tool implementations ----------
    def _format_python_files(
        self,
        paths: Optional[List[str]] = None,
        dir: str = ".",
        exts: Optional[List[str]] = None,
        line_length: int = 88,
        preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Format Python files using Black if available. Returns per-file status and optional diffs.
        """
        exts = exts or [".py"]
        # Collect files
        files: List[Path] = []
        if paths:
            for rel in paths:
                p = (self.root / rel).resolve()
                if str(p).startswith(str(self.root)) and p.exists() and p.suffix in exts:
                    files.append(p)
        else:
            base = (self.root / dir).resolve()
            if not str(base).startswith(str(self.root)):
                raise ToolError("Path traversal blocked.")
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in exts:
                    files.append(p)

        try:
            import black  # type: ignore
            from black.mode import Mode  # type: ignore
        except Exception:
            return {"error": "Black is not installed. Run `pip install black`.", "files": [str(f.relative_to(self.root)) for f in files]}

        results: List[Dict[str, Any]] = []
        mode = Mode(line_length=line_length)
        for fp in files:
            before = fp.read_text(encoding="utf-8", errors="replace")
            try:
                after = black.format_file_contents(before, fast=False, mode=mode)  # type: ignore
            except black.NothingChanged:  # type: ignore
                results.append({"path": str(fp.relative_to(self.root)), "changed": False})
                continue
            diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=str(fp), tofile=str(fp), lineterm=""))
            if not preview:
                fp.write_text(after, encoding="utf-8", errors="replace")
            results.append({"path": str(fp.relative_to(self.root)), "changed": True, "diff": diff if preview else None})
        return {"formatted": results, "preview": preview}

    def _rewrite_naive_open(
        self,
        dir: str = ".",
        exts: Optional[List[str]] = None,
        dry_run: bool = True,
        backup: bool = True,
        max_files: int = 200,
    ) -> Dict[str, Any]:
        """
        Conservatively rewrites a few common unsafe open() patterns into context managers.
        Supported transforms:
          1) data = open(PATH,...).read()              -> with open(PATH, ...) as _f: data = _f.read()
          2) open(PATH,...).write(EXPR)                -> with open(PATH, ...) as _f: _f.write(EXPR)
          3) f = open(PATH, ...) ; data = f.read()     -> with open(PATH, ...) as f: data = f.read()
          4) f = open(PATH, ...) ; f.write(EXPR)       -> with open(PATH, ...) as f: f.write(EXPR)
        We intentionally avoid complex control-flow rewrites.
        """
        exts = exts or [".py"]
        base = (self.root / dir).resolve()
        if not str(base).startswith(str(self.root)):
            raise ToolError("Path traversal blocked.")

        # Regex patterns
        # 1) single-line read chain: data = open(...).read()
        pat_chain_read = re.compile(r'^(\s*)([A-Za-z_]\w*)\s*=\s*open\((.+?)\)\.read\(\)\s*$')
        # 2) single-line write call: open(...).write(...)
        pat_chain_write = re.compile(r'^(\s*)open\((.+?)\)\.write\((.+?)\)\s*$')
        # 3/4) two-line assignment then immediate read/write
        pat_assign_open = re.compile(r'^(\s*)([A-Za-z_]\w*)\s*=\s*open\((.+?)\)\s*$')
        pat_read_assign  = re.compile(r'^(\s*)([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\.read\(\)\s*$')  # data = f.read()
        pat_write_call   = re.compile(r'^(\s*)([A-Za-z_]\w*)\.write\((.+?)\)\s*$')                 # f.write(x)

        results: List[Dict[str, Any]] = []
        processed = 0

        for fp in base.rglob("*"):
            if processed >= max_files:
                break
            if not fp.is_file() or fp.suffix not in exts:
                continue
            try:
                before = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = before.splitlines()
            changed = False
            i = 0
            new_lines: List[str] = []
            while i < len(lines):
                line = lines[i]

                m = pat_chain_read.match(line)
                if m:
                    indent, lhs, args = m.groups()
                    new_lines.append(f"{indent}with open({args}) as _f:")
                    new_lines.append(f"{indent}    {lhs} = _f.read()")
                    changed = True
                    i += 1
                    continue

                m = pat_chain_write.match(line)
                if m:
                    indent, args, expr = m.groups()
                    new_lines.append(f"{indent}with open({args}) as _f:")
                    new_lines.append(f"{indent}    _f.write({expr})")
                    changed = True
                    i += 1
                    continue

                m = pat_assign_open.match(line)
                if m and i + 1 < len(lines):
                    indent, var, args = m.groups()
                    next_line = lines[i + 1]
                    mr = pat_read_assign.match(next_line)   # data = f.read()
                    mw = pat_write_call.match(next_line)    # f.write(x)
                    if mr and mr.group(3) == var:
                        # two-line: f = open(...); data = f.read()
                        _, lhs, _ = mr.groups()
                        new_lines.append(f"{indent}with open({args}) as {var}:")
                        new_lines.append(f"{indent}    {lhs} = {var}.read()")
                        changed = True
                        i += 2
                        # optionally skip a trailing 'var.close()'
                        if i < len(lines) and re.match(r'^\s*' + re.escape(var) + r'\.close\(\)\s*$', lines[i]):
                            i += 1
                        continue
                    if mw and mw.group(2) == var:
                        # two-line: f = open(...); f.write(x)
                        _, _, expr = mw.groups()
                        new_lines.append(f"{indent}with open({args}) as {var}:")
                        new_lines.append(f"{indent}    {var}.write({expr})")
                        changed = True
                        i += 2
                        if i < len(lines) and re.match(r'^\s*' + re.escape(var) + r'\.close\(\)\s*$', lines[i]):
                            i += 1
                        continue

                # default: keep line
                new_lines.append(line)
                i += 1

            if changed:
                after = "\n".join(new_lines)
                diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=str(fp), tofile=str(fp), lineterm=""))
                if not dry_run:
                    if backup:
                        try:
                            (fp.with_suffix(fp.suffix + ".bak")).write_text(before, encoding="utf-8", errors="replace")
                        except Exception:
                            pass
                    fp.write_text(after, encoding="utf-8", errors="replace")
                results.append({"path": str(fp.relative_to(self.root)), "changed": True, "dry_run": dry_run, "diff": diff})
                processed += 1
            else:
                results.append({"path": str(fp.relative_to(self.root)), "changed": False})

        return {"rewrites": results, "dry_run": dry_run}

    # ---------- New tool implementations ----------
    def _scan_relevant_files(
        self,
        prompt: str,
        dir: str = ".",
        exts: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        case_sensitive: bool = False,
        max_files: int = 40,
        context_lines: int = 2,
    ) -> Dict[str, Any]:
        """
        Simple keyword scanner:
        - derives keywords from prompt (unless provided),
        - ranks files by total keyword hits,
        - returns line snippets around matches.
        """
        exts = exts or [".py", ".js", ".ts", ".jsx", ".tsx"]
        base = (self.root / dir).resolve()
        if not str(base).startswith(str(self.root)):
            raise ToolError("Path traversal blocked.")
        if keywords is None or len(keywords) == 0:
            # derive tokens: keep words of length >= 3, strip common stopwords
            toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", prompt)
            stop = {"the","and","for","with","this","that","into","from","your","you","are","will","files","file","code","project","scan","search"}
            keywords = [t for t in toks if len(t) >= 3 and t.lower() not in stop]
            if not keywords:
                keywords = toks or [prompt]
        flags = 0 if case_sensitive else re.IGNORECASE
        patterns = [re.compile(re.escape(k), flags) for k in keywords]

        results: List[Dict[str, Any]] = []
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in exts:
                continue
            try:
                lines = p.read_text(errors="ignore").splitlines()
            except Exception:
                continue
            hits: List[Dict[str, Any]] = []
            score = 0
            for i, line in enumerate(lines, start=1):
                matched = False
                for pat in patterns:
                    if pat.search(line):
                        matched = True
                        score += 1
                if matched:
                    start = max(1, i - context_lines)
                    end = min(len(lines), i + context_lines)
                    snippet = "\n".join(f"{ln}:{lines[ln-1]}" for ln in range(start, end + 1))
                    hits.append({"line": i, "snippet": snippet})
            if score > 0:
                results.append({"path": str(p.relative_to(self.root)), "score": score, "hits": hits})
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"prompt": prompt, "keywords": keywords, "matches": results[:max_files]}

    def _analyze_files(self, paths: List[str], max_bytes: int = 20000, encoding: str = "utf-8") -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for rel in paths:
            fp = (self.root / rel).resolve()
            if not str(fp).startswith(str(self.root)) or not fp.exists() or not fp.is_file():
                out.append({"path": rel, "error": "not found"})
                continue
            data = fp.read_bytes()[: max(100, min(max_bytes, 2_000_000))]
            text = data.decode(encoding, errors="replace")
            lines = text.splitlines()
            info = {
                "path": str(fp.relative_to(self.root)),
                "bytes": len(data),
                "lines": len(lines),
                "has_todo": any("TODO" in l or "FIXME" in l for l in lines),
                "has_json_loads": any("json.loads" in l for l in lines),
                "has_open_calls": any("open(" in l for l in lines),
                "preview_head": "\n".join(lines[:20]),
            }
            out.append(info)
        return {"analysis": out}

    def _write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = False,
        create_parents: bool = True,
        backup: bool = True,
    ) -> Dict[str, Any]:
        fp = (self.root / path).resolve()
        if not str(fp).startswith(str(self.root)):
            raise ToolError("Path traversal blocked.")
        if fp.exists() and not overwrite:
            raise ToolError(f"Refusing to overwrite existing file without overwrite=true: {path}")
        if create_parents:
            fp.parent.mkdir(parents=True, exist_ok=True)
        if backup and fp.exists():
            bak = fp.with_suffix(fp.suffix + ".bak")
            try:
                bak.write_bytes(fp.read_bytes())
            except Exception:
                pass
        data = content.encode(encoding, errors="replace")
        fp.write_bytes(data)
        return {"path": str(fp.relative_to(self.root)), "bytes_written": len(data), "overwrote": fp.exists()}

    def _replace_in_file(
        self,
        path: str,
        find: str,
        replace: str,
        regex: bool = False,
        count: int = 0,
        encoding: str = "utf-8",
        dry_run: bool = False,
        backup: bool = True,
    ) -> Dict[str, Any]:
        fp = (self.root / path).resolve()
        if not str(fp).startswith(str(self.root)) or not fp.exists() or not fp.is_file():
            raise ToolError(f"File not found: {path}")
        before = fp.read_text(encoding=encoding, errors="replace")
        if regex:
            new_text, n = re.subn(find, replace, before, count=0 if count == 0 else count, flags=0)
        else:
            n = before.count(find) if count == 0 else min(count, before.count(find))
            new_text = before.replace(find, replace, n)
        diff = "\n".join(difflib.unified_diff(before.splitlines(), new_text.splitlines(), fromfile=path, tofile=path, lineterm=""))
        if not dry_run:
            if backup:
                try:
                    (fp.with_suffix(fp.suffix + ".bak")).write_text(before, encoding=encoding, errors="replace")
                except Exception:
                    pass
            fp.write_text(new_text, encoding=encoding, errors="replace")
        return {"path": str(fp.relative_to(self.root)), "replacements": n, "dry_run": dry_run, "diff": diff}

    def _bulk_edit(self, edits: List[Dict[str, Any]], dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for e in edits:
            res = self._replace_in_file(
                path=e.get("path", ""),
                find=e.get("find", ""),
                replace=e.get("replace", ""),
                regex=bool(e.get("regex", False)),
                count=int(e.get("count", 0)),
                encoding=e.get("encoding", "utf-8"),
                dry_run=dry_run,
                backup=backup,
            )
            results.append(res)
        total = sum(r.get("replacements", 0) for r in results)
        return {"edits": results, "total_replacements": total, "dry_run": dry_run}
    
    # Registry helpers
    def schemas_openai(self) -> List[Dict[str, Any]]:
        return [v["schema"] for v in self._tools.values()]

    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]["impl"](**kwargs)


# ---------------- LLM Adapters ----------------

class LLMAdapter:
    def __init__(self, model: LLMModel, tools: ToolRegistry):
        self.model = model
        self.tools = tools

    def chat(self, messages: List[Dict[str, str]], tool_choice_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def tool_loop(
        self,
        messages: List[Dict[str, str]],
        max_iters: int = 3,
        tool_choice_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Single source of truth for tool execution:
        - Calls chat()
        - If assistant returns tool_calls: append assistant msg, execute tools, append tool replies, loop.
        - If assistant returns normal content: return it.
        - Never emits a 'tool' message unless the immediately previous assistant message had tool_calls.
        """
        for _ in range(max_iters):
            resp = self.chat(messages, tool_choice_override=tool_choice_override)
            choice = (resp.get("choices") or [{}])[0]
            assistant_msg = choice.get("message", {}) or {}

            tool_calls = assistant_msg.get("tool_calls") or []
            content = assistant_msg.get("content") or ""

            # Always append the assistant message we just got (if it has tool_calls or content)
            if tool_calls or content:
                messages.append(assistant_msg)

            # Case 1: tool calls present -> execute each, append tool replies, then continue
            if tool_calls:
                for tc in tool_calls:
                    fn = (tc.get("function") or {}).get("name")
                    arg_str = (tc.get("function") or {}).get("arguments") or "{}"
                    try:
                        args = json.loads(arg_str)
                    except Exception:
                        args = {}

                    try:
                        result = self.tools.call(fn, **args)
                        tool_reply = {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": fn,
                            "content": json.dumps(result),
                        }
                    except Exception as e:
                        tool_reply = {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": fn,
                            "content": json.dumps({"error": str(e)}),
                        }

                    # IMPORTANT: 'tool' reply immediately follows the assistant message with tool_calls
                    messages.append(tool_reply)

                # After appending all tool results, subsequent iterations should not force a tool choice
                tool_choice_override = None
                continue

            # Case 2: no tool calls -> if model gave content, return it; if not, give a minimal fallback
            if content:
                return content

            return "(no content)"

        return "(no content)"


class OpenAICompatAdapter(LLMAdapter):
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key_env = self.model.api_key_env or ("OPENAI_API_KEY" if "api.openai.com" in self.model.endpoint else None)
        key = os.getenv(key_env) if key_env else os.getenv("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _chat_via_responses(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Fallback path using /v1/responses.

        - Uses the proper part type "input_text" (not "text").
        - DOES NOT send tools; Responses tool-calls require a different loop.
        - Returns an OpenAI-chat-like structure: {"choices": [{"message": {"content": "..."} }]}
        """
        url = self.model.endpoint.rstrip("/") + "/responses"

        def to_parts(msg: Dict[str, str]) -> List[Dict[str, str]]:
            return [{"type": "input_text", "text": msg.get("content", "")}]

        payload = {
            "model": self.model.resolved_model(),
            "input": [
                {"role": m["role"], "content": to_parts(m)}
                for m in messages
                if m.get("role") in ("system", "user", "assistant")
            ],
        }

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=90)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            body = resp.text if hasattr(resp, "text") else ""
            raise RuntimeError(f"Responses API error {resp.status_code}: {body}") from e

        data = resp.json()
        # Prefer 'output_text', then walk 'output[].content[].text'
        content = data.get("output_text")
        if not (isinstance(content, str) and content.strip()):
            content = "(no content)"
            for item in data.get("output") or []:
                for part in (item.get("content") or []):
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        content = t
                        break
                if content != "(no content)":
                    break

        return {"choices": [{"message": {"content": content}}]}

    def chat(self, messages: List[Dict[str, str]], tool_choice_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.model.endpoint.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model.resolved_model(),
            "messages": messages,
            "temperature": self.model.temperature,
            "stream": False,
            "tools": self.tools.schemas_openai(),
            "tool_choice": tool_choice_override or "auto",
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=90)
        try:
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            server_text = getattr(resp, "text", "")
            if status == 401:
                raise RuntimeError(
                    "401 Unauthorized. Load OPENAI_API_KEY or choose a local model."
                ) from e
            if status == 400:
                has_tools = bool(self.tools.schemas_openai())
                if has_tools:
                    # Show the error rather than silently degrading tool behavior
                    raise RuntimeError(f"OpenAI 400 via /chat/completions (tools enabled): {server_text}") from e
                # No tools -> your existing /responses fallback may be used here if you want
                raise RuntimeError(f"OpenAI 400 via /chat/completions: {server_text}") from e
            raise RuntimeError(f"OpenAI error {status}: {server_text}") from e

class GeminiAdapter(LLMAdapter):
    def __init__(self, model: LLMModel, tools: ToolRegistry):
        super().__init__(model, tools)
        self._sdk = None
        try:
            from google import genai  # type: ignore
            self._sdk = genai.Client(api_key=os.getenv(self.model.api_key_env or "GEMINI_API_KEY"))
        except Exception:
            self._sdk = None

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if self._sdk is None:
            # Minimal fallback without tools
            user_text = "".join([m.get("content", "") for m in messages if m.get("role") == "user"]) or messages[-1].get("content", "")
            return {"choices": [{"message": {"content": f"[Gemini fallback] {user_text[:2000]}"}}]}
        # With SDK; basic generate_content (no function-calling bridge in this minimal version)
        sys_prompt = next((m["content"] for m in messages if m.get("role") == "system"), self.model.system_prompt)
        history = [{"role": m["role"], "parts": [m.get("content", "")]}
                   for m in messages if m.get("role") in ("user", "assistant")]
        resp = self._sdk.models.generate_content(model=self.model.resolved_model(), contents=history, config={"system_instruction": sys_prompt})
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        return {"choices": [{"message": {"content": text}}]}


# ---------------- Agent ----------------

class Agent:
    _RE_QUOTED = r"'([^']*)'|\"([^\"]*)\""   # matches 'x' or "x"

    def __init__(self, model: LLMModel, tools: ToolRegistry, enable_tools: bool = True):
        self.model = model
        self.tools = tools
        self.enable_tools = enable_tools

        if model.is_openai_compat():
            self.adapter: LLMAdapter = OpenAICompatAdapter(model, tools)
        elif model.is_gemini():
            self.adapter = GeminiAdapter(model, tools)
        else:
            self.adapter = OpenAICompatAdapter(model, tools)

    def system_prompt(self) -> str:
        prompt = self.model.system_prompt or "You are a helpful coding assistant."
        if self.model.supports_thinking():
            prompt += "\nThink step-by-step internally, then provide a concise final answer."
        if self.enable_tools:
            prompt += (
                "\nYou have tools for scanning, analyzing and editing code:"
                "\n- scan_relevant_files(prompt)->candidates"
                "\n- analyze_files(paths)->light analysis"
                "\n- replace_in_file / bulk_edit / write_file -> apply fixes"
                "\nWhen asked to find or fix code, prefer this flow: scan_relevant_files → analyze_files → propose & apply edits (dry_run first). "
                "NEVER print big file dumps; summarize and use the tools. If native tool-calls are unavailable, respond with a single JSON object: {\"tool\":\"name\",\"arguments\":{...}}."
            )
        return prompt

    def ask_once(self, query: str, max_iters: int = 3) -> str:
        direct = self._try_direct_actions(query)
        if direct is not None:
            return direct
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": query},
        ]

        if self.enable_tools:
            tool_choice_override = self._route_tool_call(query)

            if tool_choice_override:
                try:
                    # Step 1: force a tool call
                    resp = self.adapter.chat(messages, tool_choice_override=tool_choice_override)
                    msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
                    tool_calls = msg.get("tool_calls") or []

                    if tool_calls:
                        # ✅ Correct order: first the assistant message with tool_calls...
                        messages.append(msg)

                        # ...then each matching tool reply with the same tool_call_id
                        for tc in tool_calls:
                            fn = (tc.get("function") or {}).get("name")
                            args_str = (tc.get("function") or {}).get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                            except Exception:
                                args = {}
                            try:
                                result = self.tools.call(fn, **args)
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id"),
                                    "name": fn,
                                    "content": json.dumps(result),
                                }
                            except Exception as e:
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id"),
                                    "name": fn,
                                    "content": json.dumps({"error": str(e)}),
                                }
                            messages.append(tool_msg)

                        # Continue loop now that tool outputs are in context
                        return self.adapter.tool_loop(messages, max_iters=max_iters - 1)

                    # No tool_calls: if the model gave normal content, just return it
                    content = msg.get("content")
                    if content:
                        return content

                except Exception:
                    # If anything goes wrong here, fall back to the regular tool loop
                    pass

            # Let the model decide tool use
            return self.adapter.tool_loop(messages, max_iters=max_iters)

        # Tools disabled → normal chat
        resp = self.adapter.chat(messages)
        return (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")


    def _route_tool_call(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Heuristic router that maps a natural-language prompt to the most likely tool.
        It only returns a tool_choice override; the model will still supply arguments.
        Priority is important: we match more specific intents first.
        """
        q = (query or "").lower()

        def choose(name: str) -> Dict[str, Any]:
            return {"type": "function", "function": {"name": name}}

        # --- Highly specific/intentional actions first ---
        # Edits / replacements
        if any(k in q for k in ("bulk edit", "bulk change", "multiple files", "across the repo", "across the project")):
            return choose("bulk_edit")
        if any(k in q for k in ("replace", "substitute", "find and replace", "search and replace", "swap text")):
            return choose("replace_in_file")
        if any(k in q for k in ("rewrite open(", "fix open()", "context manager", "with open", "file leak", "not close", "don’t close", "doesn't close", "doesnt close")):
            return choose("rewrite_naive_open")

        # Write/create files
        if any(k in q for k in ("write file", "create file", "new file", "make file", "save file")):
            return choose("write_file")

        # Formatting / cleanup
        if any(k in q for k in ("format python", "run black", "reformat", "auto format", "auto-format", "code style", "pep8")):
            return choose("format_python_files")

        # Scan → Analyze flow
        if any(k in q for k in ("scan for", "find relevant files", "which files are relevant", "search repo for", "locate files", "discover files")):
            return choose("scan_relevant_files")
        if any(k in q for k in ("analyze files", "analyze these files", "summarize files", "quick analysis", "file signals", "show preview", "line counts")):
            return choose("analyze_files")

        # --- Core browsing/file operations ---
        # Listing
        if ("list" in q or "show" in q) and ("file" in q or "files" in q):
            # ex: "list all python files in app/api"
            return choose("list_files")

        # Searching
        if any(k in q for k in ("search", "find", "grep", "look for", "scan")) and any(k in q for k in ("code", "function", "class", "def ", ".py", ".js", ".ts", ".jsx", ".tsx")):
            return choose("search_code")

        # Reading (prefer when an explicit filename/path is mentioned)
        if any(k in q for k in ("read", "open", "show contents", "view file", "print file")):
            return choose("read_file")

        # Default: no override (let the model decide / general chat)
        return None

    def repl(self, max_iters: int = 3):
        tool_names = list(self.tools._tools.keys()) if self.enable_tools else []
        print(f"Model: {self.model.name} ({self.model.provider})  |  Tools: {', '.join(tool_names) if tool_names else 'disabled'}")
        print("Enter your question (Ctrl+C to exit)")
        while True:
            try:
                q = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("bye!")
                break
            if not q:
                continue
            try:
                ans = self.ask_once(q, max_iters=max_iters)
            except Exception as e:
                ans = f"[error] {e}"
            print(f"assistant> {ans}")

    def _unquote(self, m) -> str:
        return m.group(1) if m.group(1) is not None else m.group(2)

    def _first(self, *vals) -> str:
        """Return the first non-None value as a string ('' if all None)."""
        for v in vals:
            if v is not None:
                return str(v)
        return ""

    def _parse_replace(self, q: str):
        """
        Parse commands like:
          Replace 'hello' with 'hi' in app/api/tmp/hello.txt
          Replace "hello" with "hi" in app/api/tmp/hello.txt (dry run)
        Returns {"tool":"replace_in_file","args":{...}} or None.
        """
        pat = re.compile(
            r"""(?ix)
            \breplace\s+
              (?:'(?P<find_sq>[^']*)'|"(?P<find_dq>[^"]*)")   # find
            \s+with\s+
              (?:'(?P<repl_sq>[^']*)'|"(?P<repl_dq>[^"]*)")   # replace
            \s+in\s+
              (?P<path>\S+)
            (?:\s*\(\s*(?P<dry>dry\s*run)\s*\))?             # optional (dry run)
            """,
        )
        m = pat.search(q)
        if not m:
            return None
        find_val = self._first(m.group("find_sq"), m.group("find_dq"))
        repl_val = self._first(m.group("repl_sq"), m.group("repl_dq"))
        path_val = m.group("path")
        dry = bool(m.group("dry"))
        return {
            "tool": "replace_in_file",
            "args": {
                "path": path_val,
                "find": find_val,
                "replace": repl_val,
                "dry_run": dry,
                "backup": True,
            },
        }

    def _parse_writefile(self, q: str):
        """
        Parse commands like:
          Create a temp helper file under app/api/tmp/hello.txt
          Write a file at app/api/tmp/hello.txt with 'Hello, world!'
        Returns {"tool":"write_file","args":{...}} or None.
        """
        pat = re.compile(
            r"""(?ix)
            \b(?:create|write)\b .*? \bfile\b .*?
            (?:under|at|in)\s+(?P<path>\S+)
            (?: .*? (?:with|containing)\s+
                (?:'(?P<text_sq>[^']*)'|"(?P<text_dq>[^"]*)")
            )?
            """,
        )
        m = pat.search(q)
        if not m:
            return None
        path = m.group("path")
        content = self._first(m.group("text_sq"), m.group("text_dq"))
        if content == "":
            content = "Hello, world!\n"
        return {
            "tool": "write_file",
            "args": {
                "path": path,
                "content": content,
                "overwrite": False,
                "backup": True,
            },
        }

    def _try_direct_actions(self, query: str):
        """
        Deterministic fast-paths: execute certain commands directly (no LLM).
        Falls back to normal flow when no parser matches.
        """
        q = query.strip()

        # 1) Replace in file: Replace 'A' with 'B' in PATH [(dry run)]
        rep = self._parse_replace(q)
        if rep:
            try:
                res = self.tools.call(rep["tool"], **rep["args"])
                return f"{rep['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                return f"[error] {rep['tool']} failed: {e}"

        # 2) Write/Create file: Create/Write file ... at/under PATH [with 'TEXT']
        wf = self._parse_writefile(q)
        if wf:
            try:
                res = self.tools.call(wf["tool"], **wf["args"])
                return f"{wf['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                return f"[error] {wf['tool']} failed: {e}"

        # No deterministic command matched
        return None
    
    def _flag(self, q: str, *phrases: str) -> bool:
        """Return True if any phrase appears (case-insensitive) in the query."""
        ql = q.lower()
        return any(p.lower() in ql for p in phrases)

    def _extract_json_block(self, q: str) -> Optional[str]:
        """
        Extract a JSON block either from a fenced block ```json ... ``` or the first {...} array/object.
        Returns the raw JSON string or None.
        """
        # fenced ```json ... ```
        m = re.search(r"```json\s*(?P<body>\{.*?\}|\[.*?\])\s*```", q, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group("body")
        # first object/array literal
        m2 = re.search(r"(?P<body>\{.*\}|\[.*\])", q, flags=re.DOTALL)
        if m2:
            return m2.group("body")
        return None

    def _parse_bool_flag(self, q: str, default: bool, *aliases: str) -> bool:
        """
        Scan for explicit boolean in text "(preview|no preview)", "(dry run|apply)", etc.
        If not found, return default.
        """
        ql = q.lower()
        yes = {"true", "yes", "y", "on", "enable", "enabled"}
        no  = {"false", "no", "n", "off", "disable", "disabled"}
        # Generic “(preview) / (dry run)” patterns
        for a in aliases:
            # (alias) => true
            if re.search(rf"\(\s*{re.escape(a)}\s*\)", ql):
                return True
            # (no alias) => false
            if re.search(rf"\(\s*no\s+{re.escape(a)}\s*\)", ql):
                return False
            # alias=true/false
            m = re.search(rf"{re.escape(a)}\s*=\s*(\w+)", ql)
            if m:
                return m.group(1) in yes
        return default

    def _parse_format_python(self, q: str):
        """
        Parse: 'Format Python files under <dir> (preview)' or
            'Format Python files in <dir> line length 100'
        Returns {"tool":"format_python_files","args":{...}} or None.
        """
        m = re.search(r"(?:format|reformat|run\s+black).*(?:under|in)\s+(?P<dir>\S+)", q, flags=re.IGNORECASE)
        if not m:
            return None
        dir_val = m.group("dir")
        # line length (optional)
        mlen = re.search(r"(?:line\s*length|ll)\s*(?P<ll>\d{2,3})", q, flags=re.IGNORECASE)
        ll = int(mlen.group("ll")) if mlen else 88
        preview = self._parse_bool_flag(q, default=True, *("preview",))
        return {
            "tool": "format_python_files",
            "args": {"dir": dir_val, "exts": [".py"], "line_length": ll, "preview": preview}
        }

    def _parse_rewrite_open(self, q: str):
        """
        Parse: 'Rewrite open() usages [in|under] <dir> (dry run)'
        Returns {"tool":"rewrite_naive_open","args":{...}} or None.
        """
        if not self._flag(q, "rewrite open", "fix open()", "with open", "context manager"):
            return None
        m = re.search(r"(?:in|under)\s+(?P<dir>\S+)", q, flags=re.IGNORECASE)
        dir_val = m.group("dir") if m else "."
        dry = self._parse_bool_flag(q, default=True, *("dry run", "preview"))
        return {
            "tool": "rewrite_naive_open",
            "args": {"dir": dir_val, "exts": [".py"], "dry_run": dry, "backup": True}
        }

    def _parse_bulk_edit(self, q: str):
        """
        Accepts JSON for edits, e.g.:
        bulk edit (dry run)
        [
            {"path":"app/api/tmp/hello.txt","find":"hi","replace":"hello"},
            {"path":"app/api/chat.py","find":"APIRouter","replace":"APIRouter"}
        ]
        Or inline: bulk edit in app/api (dry run) find 'A' replace 'B' ext .py
        Prefer JSON; fall back to simple inline for one edit if no JSON present.
        """
        if not self._flag(q, "bulk edit", "bulk change", "multiple files", "across the repo", "across the project"):
            return None

        # 1) JSON-first path
        raw = self._extract_json_block(q)
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "edits" in payload:
                    edits = payload["edits"]
                else:
                    edits = payload if isinstance(payload, list) else []
            except Exception:
                edits = []
        else:
            edits = []

        # 2) If no JSON edits, try parsing a minimal inline single-edit
        #    bulk edit in DIR find 'A' replace 'B' ext .py (dry run)
        if not edits:
            m = re.search(
                r"bulk\s+edit(?:\s+in\s+(?P<dir>\S+))?.*?"
                r"find\s+(?:'(?P<find_sq>[^']*)'|\"(?P<find_dq>[^\"]*)\").*?"
                r"replace\s+(?:'(?P<repl_sq>[^']*)'|\"(?P<repl_dq>[^\"]*)\")"
                r"(?:.*?\bext\s+(?P<ext>\.\w+))?",
                q,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m:
                find_val = self._first(m.group("find_sq"), m.group("find_dq"))
                repl_val = self._first(m.group("repl_sq"), m.group("repl_dq"))
                # If a dir was given, we’ll expand to files later (caller can ask scan first).
                # Here we build a trivial single-edit example expecting an exact file path later.
                # If you want, you can extend this to enumerate files in dir matching ext.
                # For now, we require a 'path' to be present in JSON or user will run scan first.
                # To avoid a no-op, skip creating an edit without a specific path.
                pass  # leave 'edits' empty if no explicit path is provided

        # 3) Flags
        dry = self._parse_bool_flag(q, default=True, *("dry run", "preview"))

        if not edits:
            # Nothing to do; ask user to provide JSON or run scan first
            return {
                "tool": "bulk_edit",
                "args": {"edits": [], "dry_run": True, "backup": True},
                "warning": "No edits parsed. Provide JSON `edits` or specify explicit file paths."
            }

        return {"tool": "bulk_edit", "args": {"edits": edits, "dry_run": dry, "backup": True}}

# ---------------- CLI ----------------

def load_config(path: Path) -> ModelConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return ModelConfig.load(path)


def find_model(cfg: ModelConfig, name: Optional[str]) -> LLMModel:
    if name:
        if name not in cfg.llm_models:
            raise ValueError(f"Model '{name}' not in config. Available: {', '.join(cfg.llm_models.keys())}")
        return cfg.llm_models[name]
    return cfg.pick_best_tool_thinker()


def main():
    ap = argparse.ArgumentParser(description="Gemini-like CLI code assistant with tools & local model support")
    ap.add_argument("-c", "--config", type=Path, default=Path("model_config.json"), help="Path to model_config.json")
    ap.add_argument("-r", "--root", type=Path, default=Path.cwd(), help="Project root for code tools")
    ap.add_argument("-m", "--message", type=str, help="One-shot question (non-interactive)")
    ap.add_argument("--model", type=str, help="Model name from model_config.json")
    ap.add_argument("--env-file", type=Path, help="Optional .env file to load before running")
    ap.add_argument("--no-tools", action="store_true", help="Disable tool calling")
    ap.add_argument("--max-iters", type=int, default=3, help="Max tool iterations")
    args = ap.parse_args()

    # .env
    if _HAVE_DOTENV:
        if args.env_file:
            load_dotenv(args.env_file)
        else:
            load_dotenv(find_dotenv(usecwd=True))

    cfg = load_config(args.config)
    model = find_model(cfg, args.model)

    tools = ToolRegistry(args.root)
    agent = Agent(model, tools, enable_tools=not args.no_tools)

    if args.message:
        try:
            print(agent.ask_once(args.message, max_iters=args.max_iters))
        except Exception as e:
            print(f"[error] {e}")
    else:
        agent.repl(max_iters=args.max_iters)


if __name__ == "__main__":
    main()
