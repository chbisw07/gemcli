# =========================
# tools/registry.py  (full file)
# =========================
from __future__ import annotations

import re
import difflib
import json
import shutil
from pathlib import Path
from functools import lru_cache
from typing import Optional, Any, Callable, Iterable, List, Dict
from loguru import logger
from logging_decorators import log_call

# Optional: direct command parser hook (safe to keep even if unused)
try:
    from tools.direct_parser import try_direct_actions
except Exception:  # pragma: no cover
    def try_direct_actions(*_, **__):
        return None

from tools.edu_intents import detect_edu_intent
from tools.edu_tools import (
    edu_similar_questions, edu_question_paper, edu_explain, edu_extract_tables
)
from tools.web_tools import web_search, web_fetch
# NEW — blueprint & generation helpers
from tools.edu_blueprint import build_blueprint as _bp_build, save_blueprint as _bp_save, load_blueprint as _bp_load
from tools.edu_parsers import parse_weightage_request as _parse_weightage
from tools.edu_generator import collect_exemplars as _collect_exemplars, build_generation_pack as _build_pack
from tools.charting import (
    draw_chart_from_csv as _chart_from_csv,
    draw_chart_from_json_records as _chart_from_json,
)
from indexing.chunkers.py_ast import build_call_graph

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
        self.schemas: Dict[str, Any] = {}
        logger.info("ToolRegistry init → root='{}'", str(self.root))
        self._register_builtin_tools()
        logger.info("ToolRegistry ready with {} tool(s)", len(self.tools))

    # ----- same resolver pattern as Enhanced registry -----
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
        if not path:
            return None
        path = str(Path(path))
        if subdir:
            cand = (self.root / subdir / path)
            if cand.exists():
                rel = self._rel(cand)
                logger.debug("_resolve_path: subdir hit → {}", rel)
                return rel
        cand = (self.root / path)
        if cand.exists():
            rel = self._rel(cand)
            logger.debug("_resolve_path: direct hit → {}", rel)
            return rel
        name = Path(path).name
        matches = self._all_files_named(name)
        if matches:
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
        tail = path.replace("\\", "/")
        for p in self._all_py_files():
            if str(p).replace("\\", "/").endswith(tail):
                rel = self._rel(p)
                logger.debug("_resolve_path: endswith hit → {}", rel)
                return rel
        logger.debug("_resolve_path: not found '{}'", path)
        return None

    # ---------------- registration & dispatch ----------------

    def _register(self, name: str, fn: Callable):
        # optional per-tool summaries for cleaner logs
        summarize = None
        if name == "search_code":
            summarize = lambda ret: {"hits": len(ret) if isinstance(ret, list) else None}
        elif name == "replace_in_file":
            summarize = lambda ret: {"replacements": ret.get("replacements")} if isinstance(ret, dict) else {}
        elif name == "format_python_files":
            summarize = lambda ret: {"changed_files": len(ret.get("changed", {}))} if isinstance(ret, dict) else {}

        wrapped = log_call(
            name,
            slow_ms=1000,
            redact={"api_key", "authorization"},
            summarize=summarize
        )(fn)
        self.tools[name] = wrapped
        logger.debug("Registered tool '{}'", name)

    def call(self, name: str, **kwargs):
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        fn = self.tools[name]
        # Safely drop unexpected kwargs so tools don't crash on extra params.
        # Unwrap decorated callables to inspect real signature.
        try:
            import inspect
            orig = fn
            while hasattr(orig, "__wrapped__"):
                orig = orig.__wrapped__  # type: ignore[attr-defined]
            sig = inspect.signature(orig)
            accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            filtered = kwargs if accepts_var_kw else {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            filtered = kwargs
        logger.info("→ %s(%s)", name, ", ".join(f"{k}={v!r}" for k, v in filtered.items()))
        return fn(**filtered)

    def try_direct(self, query: str) -> Optional[dict]:
        res = try_direct_actions(query, self.root)
        if res:
            logger.info("try_direct matched → {}", res.get("tool"))
        else:
            logger.debug("try_direct: no match")
        return res

    # -------------------- built-in tools ---------------------

    def _register_builtin_tools(self):
        # ---------- rag_retrieve ----------
        def rag_retrieve(
            query: str,
            top_k: int | None = None,
            where: Dict[str, Any] | None = None,
            min_score: float | None = None,
            rag_path: str | None = None,
            project_root: str | None = None,
            enable_filename_boost: bool | None = None,
        ) -> Dict[str, Any]:
            """
            Retrieve top chunks from the project's Chroma index, with optional metadata filters.
            Defaults:
              - project_root: this registry's root
              - rag_path: config_home.GLOBAL_RAG_PATH
            Returns: {"chunks":[{id, document, metadata, score, distance}], "top_k":int, ...}
            """
            try:
                from indexing.retriever import retrieve as _retrieve
                from config_home import GLOBAL_RAG_PATH
            except Exception as e:  # pragma: no cover
                logger.error("rag_retrieve import failed: {}", e)
                raise
            pr = project_root or str(self.root)
            rp = rag_path or str(GLOBAL_RAG_PATH)
            res = _retrieve(project_root=pr, rag_path=rp, query=query, k=top_k, where=where, min_score=min_score, enable_filename_boost=enable_filename_boost)
            logger.info("rag_retrieve: query='{}...' hits={} where_keys={} threshold={}", (query or "")[:80], len(res.get("chunks") or []), list((where or {}).keys()), min_score)
            return res

        # ---------- read_file ----------
        @log_call
        def read_file(path: str, subdir: str = "") -> str:
            """
            Read a text file.
            - Accepts optional `subdir` so the planner can pass it.
            - Uses the existing resolver to be flexible with partial paths / filenames.
            """
            # Try flexible resolution first
            resolved = self._resolve_path(path, subdir=subdir) or path
            p = (self.root / resolved).resolve()
            if not p.exists() or not p.is_file():
                logger.error("read_file: not found '{}' (resolved from path='{}', subdir='{}')", str(p), path, subdir)
                raise FileNotFoundError(f"Not found: {path}")
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except UnicodeDecodeError:
                # Fall back to binary read -> decode best effort
                text = p.read_bytes().decode("utf-8", errors="ignore")
            logger.info("read_file: '{}' bytes={}", self._rel(p), len(text.encode("utf-8")))
            return text

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
                logger.debug("list_files: subdir not found '{}'", subdir)
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
            logger.info("list_files: subdir='{}' exts={} → {}", subdir, list(exts), len(out))
            return sorted(out)

        # ---------- search_code ----------
        def search_code(
            query: str,
            subdir: str = "",
            regex: bool = False,
            case_sensitive: bool = False,
            exts: Iterable[str] | None = None,
            max_results: int = 2000,
            **_: Any,
        ) -> List[dict]:
            """
            Search for a string/regex across files (default: .py) and return line hits.
            """
            target = self.root / (subdir or "")
            if not target.exists():
                logger.debug("search_code: subdir not found '{}'", subdir)
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
                        logger.info("search_code: capped at max_results={} (more hits exist)", max_results)
                        return results
            logger.info("search_code: '{}' hits={} subdir='{}' regex={} case_sensitive={}",
                        query, len(results), subdir, regex, case_sensitive)
            return results

        def call_graph_for_function(function: str, file_hint: str | None = None, project_root: str | None = None) -> Dict[str, Any]:
            """
            Build a compact call graph for `function`. If `file_hint` is provided, we scope search to that file.
            """
            root = Path(project_root or self.root)
            return build_call_graph(function=function, file_hint=file_hint, project_root=root)

        self.tools["call_graph_for_function"] = call_graph_for_function

        # (If you publish OpenAI tool schemas here, add:)
        self.schemas["call_graph_for_function"] = {
            "name": "call_graph_for_function",
            "description": "Build a compact call graph for a function (uses AST).",
            "parameters": {
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "file_hint": {"type": "string"},
                    "project_root": {"type": "string"},
                },
                "required": ["function"]
            }
        }

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
            resolved = self._resolve_path(path) or path
            p = self.root / resolved
            p.parent.mkdir(parents=True, exist_ok=True)
            existed = p.exists()
            if existed and not overwrite:
                logger.error("write_file: exists and overwrite=False '{}'", path)
                raise FileExistsError(f"File exists: {path}")
            if existed and backup:
                bak = p.with_suffix(p.suffix + ".bak")
                shutil.copyfile(p, bak)
            p.write_text(content, encoding="utf-8")
            res = {
                "path": str(p.relative_to(self.root)),
                "bytes_written": len(content.encode("utf-8")),
                "overwrote": existed,
                "backup": bool(existed and backup),
            }
            logger.info("write_file: {}", res)
            return res

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
            resolved = self._resolve_path(path)
            if not resolved:
                msg = {"error": f"File not found under {self.root}: {path}", "path": path}
                logger.error("replace_in_file: {}", msg["error"])
                return msg
            p = self.root / resolved
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
            res = {
                "path": str(p.relative_to(self.root)),
                "replacements": len(matches),
                "dry_run": bool(dry_run),
                "diff": diff,
            }
            logger.info("replace_in_file: path='{}' replacements={} dry_run={}", res["path"], res["replacements"], res["dry_run"])
            return res

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
                logger.error("format_python_files: Black not installed")
                raise RuntimeError("Black is not installed. `pip install black` to use format_python_files.")
            target = self.root / (subdir or "")
            if not target.exists():
                logger.debug("format_python_files: subdir not found '{}'", subdir)
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
            logger.info("format_python_files: changed={} dry_run={} subdir='{}'", len(changed), dry_run, subdir)
            return changed

        # ---------- scan_relevant_files ----------
        def scan_relevant_files(
            prompt: str = "",
            path: str = "",
            subdir: str = "",
            max_results: int = 200,
            exts: Iterable[str] | None = None,
        ) -> List[dict]:
            """
            Heuristic scan: score files by whether prompt terms appear in path/content.
            """
            # Resolve path relative to project root (works for file OR directory)
            resolved = self._resolve_path(path, subdir=subdir or "") if path else None
            base = (self.root / resolved) if resolved else (self.root / (subdir or ""))
            if not base.exists():
                logger.debug("scan_relevant_files: base not found path='{}' subdir='{}'", path, subdir)
                return []
            if exts is None:
                exts = [".py"]
            exts = tuple(exts)
            terms = [t.lower() for t in re.split(r"[^A-Za-z0-9_.:/-]+", prompt or "") if t]
            hits: List[dict] = []
            def _score_file(p: Path):
                if not p.is_file() or p.suffix not in exts:
                    return
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore").lower()
                except Exception:
                    return
                score = 0
                hay = (str(p.relative_to(self.root)).lower() + " " + text)
                for t in terms:
                    if t and t in hay:
                        score += 1
                if score > 0:
                    hits.append({"path": str(p.relative_to(self.root)), "score": score})

            if base.is_file():
                _score_file(base)
            else:
                for p in base.rglob("*"):
                    _score_file(p)

            hits.sort(key=lambda x: (-x["score"], x["path"]))
            out = hits[:max_results]
            logger.info("scan_relevant_files: terms={} hits={} path='{}' subdir='{}'",
                        len(terms), len(out), path, subdir)
            return out

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
            logger.info("analyze_files: {} file(s)", len(out))
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
            ok = fail = 0
            for e in edits or []:
                rel = e.get("path")
                if not rel:
                    results.append({"path": rel or "", "error": "missing path"})
                    fail += 1
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
                    ok += 1 if "error" not in res else 0
                    fail += 1 if "error" in res else 0
                except Exception as ex:
                    results.append({"path": rel, "error": str(ex), "dry_run": dry_run})
                    fail += 1
            logger.info("bulk_edit: edits={} ok={} fail={} dry_run={}", len(edits or []), ok, fail, dry_run)
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
                logger.debug("rewrite_naive_open: base not found '{}'", dir)
                return []

            results: List[dict] = []
            open_re = re.compile(r"""^\s*(?P<var>[A-Za-z_]\w*)\s*=\s*open\((?P<args>.+)\)\s*(#.*)?$""")
            close_re_tpl = r"""^\s*{var}\.close\(\)\s*(#.*)?$"""

            changed_files = 0
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
                        "replacements": 1,  # at least one with-block introduced; could count precisely
                        "dry_run": bool(dry_run),
                        "diff": diff,
                    }
                )
                changed_files += 1
            logger.info("rewrite_naive_open: changed_files={} dir='{}' dry_run={}", changed_files, dir, dry_run)
            return results

        # ---------- edu_build_blueprint ----------
        def edu_build_blueprint(
            subject: str,
            klass: int,
            board: str = "CBSE",
            filename_hint: str | None = None,
            session: str = "",
        ) -> dict:
            """
            Parse 'General Instructions' + section rules from indexed SQPs/syllabi and save a blueprint JSON.
            Returns: {"blueprint": {...}, "saved_to": "<relpath>"}
            """
            try:
                bp = _bp_build(
                    project_root=str(self.root),
                    subject=subject,
                    klass=int(klass),
                    board=board,
                    filename_hint=filename_hint,
                    session=session,
                )
                p = _bp_save(str(self.root), bp)
                return {"blueprint": bp.to_dict(), "saved_to": str(p.relative_to(self.root))}
            except Exception as e:
                logger.exception("edu_build_blueprint failed: {}", e)
                raise

        # ---------- edu_generate_paper ----------
        def edu_generate_paper(
            subject: str,
            klass: int,
            weightage: str | None = None,
            total_questions: int | None = None,
            total_marks: int | None = None,
            include_solutions: bool = True,
            board: str = "CBSE",
            seed: int | None = None,
            filename_hint: str | None = None,
        ) -> dict:
            """
            Produce a generation pack (system/user messages) for a new paper + optional solutions,
            honoring a blueprint and a free-text weightage request (e.g., '.25/.25/.50 short/medium/long').
            Returns: {
              "blueprint": {...},
              "distribution": {...},
              "generation_pack": {"system": "...", "user": "..."},
              "seed": <int or null>,
              "suggested_messages": [...]
            }
            """
            try:
                # Load or build blueprint
                bp = _bp_load(str(self.root), board=board, klass=int(klass), subject=subject)
                if bp is None:
                    bp = _bp_build(
                        project_root=str(self.root),
                        subject=subject,
                        klass=int(klass),
                        board=board,
                        filename_hint=filename_hint,
                    )
                    _bp_save(str(self.root), bp)
                bp_dict = bp.to_dict()

                # Parse the weightage spec (or mirror blueprint to total_questions if provided)
                dist = _parse_weightage(
                    request_text=(weightage or ""),
                    blueprint=bp_dict,
                    total_questions=total_questions,
                    total_marks=total_marks,
                )

                # Pull a few exemplars to steer style (kept short)
                exemplars = _collect_exemplars(
                    project_root=str(self.root),
                    subject=subject,
                    klass=int(klass),
                    per_type=3,
                )

                # Build the final prompt pack
                pack = _build_pack(
                    blueprint=bp_dict,
                    distribution=dist,
                    subject=subject,
                    klass=int(klass),
                    board=board,
                    include_solutions=include_solutions,
                    seed=seed,
                    exemplars=exemplars,
                )

                suggested = [
                    {"role": "system", "content": pack["system"]},
                    {"role": "user", "content": pack["user"]},
                ]
                return {
                    "blueprint": bp_dict,
                    "distribution": dist,
                    "generation_pack": {"system": pack["system"], "user": pack["user"]},
                    "seed": pack.get("seed"),
                    "suggested_messages": suggested,
                }
            except Exception as e:
                logger.exception("edu_generate_paper failed: {}", e)
                raise

                # ---------- charting (matplotlib) ----------
        def draw_chart_csv(
            csv_path: str,
            kind: str = "line",
            x: str | None = None,
            y: str | None = None,          # comma-separated list or single
            parse_dates: bool = True,
            resample: str | None = None,   # "D" | "W" | "M"
            agg: str | None = "mean",
            title: str | None = None,
            out_dir: str = "charts",
            width: int = 1200,
            height: int = 800,
            dpi: int = 144,
            grid: bool = True,
        ) -> dict:
            """
            Draw a chart from a CSV located under project root. Saves PNG locally.
            """
            # Resolve path relative to project root
            resolved = self._resolve_path(csv_path) or csv_path
            p = (self.root / resolved).resolve()
            return _chart_from_csv(
                p, kind=kind, x=x, y=y, parse_dates=parse_dates, resample=resample, agg=agg,
                title=title, out_dir=(self.root / out_dir), width=width, height=height, dpi=dpi, grid=grid
            )

        def draw_chart_data(
            data_json: str,
            kind: str = "line",
            x: str | None = None,
            y: str | None = None,
            parse_dates: bool = True,
            resample: str | None = None,
            agg: str | None = "mean",
            title: str | None = None,
            out_dir: str = "charts",
            width: int = 1200,
            height: int = 800,
            dpi: int = 144,
            grid: bool = True,
        ) -> dict:
            """
            Draw a chart from inline JSON records: '[{"date":"...", "value": ...}, ...]'.
            """
            return _chart_from_json(
                data_json, kind=kind, x=x, y=y, parse_dates=parse_dates, resample=resample, agg=agg,
                title=title, out_dir=(self.root / out_dir), width=width, height=height, dpi=dpi, grid=grid
            )

        self._register("draw_chart_csv", draw_chart_csv)
        self._register("draw_chart_data", draw_chart_data)

        # -------------------- register everything --------------------
        self._register("rag_retrieve", rag_retrieve)
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
        # Education-focused tools
        self._register("edu_detect_intent", detect_edu_intent)
        self._register("edu_similar_questions", edu_similar_questions)
        self._register("edu_question_paper", edu_question_paper)
        self._register("edu_explain", edu_explain)
        self._register("edu_extract_tables", edu_extract_tables)
        self._register("web_search", web_search)
        self._register("web_fetch", web_fetch)
        self._register("edu_build_blueprint", edu_build_blueprint)
        self._register("edu_generate_paper", edu_generate_paper)


    # --------------- OpenAI tool schema (function-calling) ---------------

    def schemas_openai(self) -> List[dict]:
        """Expose tools for OpenAI function-calling models."""
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "rag_retrieve",
                    "description": "Retrieve top chunks from the project's vector index (with optional filter).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "where": {"type": "object"},
                            "min_score": {"type": "number"},
                            "rag_path": {"type": "string"},
                            "project_root": {"type": "string"},
                            "enable_filename_boost": {"type": "boolean"}
                        },
                        "required": ["query"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a text file at a relative path (optional subdir).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "subdir": {"type": "string"}
                        },
                        "required": ["path"]
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
                        "properties": {"subdir": {"type": "string"}, "line_length": {"type": "integer"}, "dry_run": {"type": "boolean"}},
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
                        "properties": {"prompt": {"type": "string"}, "path": {"type": "string"}, "subdir": {"type": "string"},
                                       "max_results": {"type": "integer"}, "exts": {"type": "array", "items": {"type": "string"}}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_files",
                    "description": "Summarize basic stats and small snippets for specific files.",
                    "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}, "max_bytes": {"type": "integer"}}, "required": ["paths"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bulk_edit",
                    "description": "Apply multiple find/replace edits across files.",
                    "parameters": {"type": "object", "properties": {"edits": {"type": "array", "items": {"type": "object"}}, "dry_run": {"type": "boolean"}, "backup": {"type": "boolean"}}, "required": ["edits"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rewrite_naive_open",
                    "description": "Convert simple open()/close() pairs into with-open blocks.",
                    "parameters": {"type": "object", "properties": {"dir": {"type": "string"}, "exts": {"type": "array", "items": {"type": "string"}}, "dry_run": {"type": "boolean"}, "backup": {"type": "boolean"}}, "required": []},
                },
            },
        ]
        schemas += [
            {
                "type": "function",
                "function": {
                    "name": "edu_detect_intent",
                    "description": "Detect education-specific intent and slots from a user prompt.",
                    "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edu_similar_questions",
                    "description": "Build generation messages for book-style similar questions.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "project_root": {"type": "string"},
                        "rag_path": {"type": "string"},
                        "topic": {"type": "string"},
                        "chapter": {"type": "string"},
                        "count": {"type": "integer"},
                        "difficulty": {"type": "string"},
                        "scope": {"type": "string"},
                        "top_k": {"type": "integer"}
                    },
                    "required": ["project_root", "rag_path"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edu_question_paper",
                    "description": "Build generation messages for a full question paper with solutions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_root": {"type": "string"},
                            "rag_path": {"type": "string"},
                            "chapter": {"type": "string"},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "count": {"type": "integer"},
                            "mix": {"type": "object"},
                            "difficulty": {"type": "string"},
                            "scope": {"type": "string"},
                            "top_k": {"type": "integer"}
                        },
                        "required": ["project_root", "rag_path"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edu_explain",
                    "description": "Build messages to explain a topic/question grounded in the book excerpts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_root": {"type": "string"},
                            "rag_path": {"type": "string"},
                            "question_or_topic": {"type": "string"},
                            "scope": {"type": "string"},
                            "top_k": {"type": "integer"}
                        },
                        "required": ["project_root", "rag_path", "question_or_topic"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edu_extract_tables",
                    "description": "Retrieve table chunks (no model call).",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "project_root": {"type": "string"},
                        "rag_path": {"type": "string"},
                        "chapter": {"type": "string"},
                        "topic": {"type": "string"},
                        "scope": {"type": "string"},
                        "top_k": {"type": "integer"}
                    },
                    "required": ["project_root", "rag_path"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edu_build_blueprint",
                    "description": "Extract and save a blueprint JSON from indexed SQPs/syllabi for {board, class, subject}.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "klass": {"type": "integer"},
                            "board": {"type": "string"},
                            "filename_hint": {"type": "string"},
                            "session": {"type": "string"}
                        },
                        "required": ["subject", "klass"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edu_generate_paper",
                    "description": "Build a generation pack (system/user messages) for a new paper with optional solutions, honoring a blueprint and weightage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "klass": {"type": "integer"},
                            "weightage": {"type": "string"},
                            "total_questions": {"type": "integer"},
                            "total_marks": {"type": "integer"},
                            "include_solutions": {"type": "boolean"},
                            "board": {"type": "string"},
                            "seed": {"type": "integer"},
                            "filename_hint": {"type": "string"}
                        },
                        "required": ["subject", "klass"]
                    },
                },
            },
        ]
        schemas += [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for pages related to a query (DuckDuckGo).",
                    "parameters": {"type":"object","properties":{
                        "query":{"type":"string"},
                        "max_results":{"type":"integer"},
                        "site":{"type":"string"},
                        "recency_days":{"type":"integer"}
                    },"required":["query"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract main text from a web page for grounding.",
                    "parameters": {"type":"object","properties":{
                        "url":{"type":"string"},
                        "max_chars":{"type":"integer"}
                    },"required":["url"]}
                }
            },
        ]
        schemas += [
            {
                "type": "function",
                "function": {
                    "name": "draw_chart_csv",
                    "description": "Create a chart (PNG) from a CSV using matplotlib. Saves locally and returns the file path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "csv_path": {"type": "string"},
                            "kind": {"type": "string", "description": "line|bar|scatter|hist"},
                            "x": {"type": "string"},
                            "y": {"type": "string"},
                            "parse_dates": {"type": "boolean"},
                            "resample": {"type": "string", "description": "D|W|M"},
                            "agg": {"type": "string", "description": "mean|sum|min|max|median"},
                            "title": {"type": "string"},
                            "out_dir": {"type": "string"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                            "dpi": {"type": "integer"},
                            "grid": {"type": "boolean"}
                        },
                        "required": ["csv_path"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "draw_chart_data",
                    "description": "Create a chart (PNG) from inline JSON records using matplotlib. Saves locally and returns the file path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_json": {"type": "string", "description": "JSON array of records"},
                            "kind": {"type": "string", "description": "line|bar|scatter|hist"},
                            "x": {"type": "string"},
                            "y": {"type": "string"},
                            "parse_dates": {"type": "boolean"},
                            "resample": {"type": "string", "description": "D|W|M"},
                            "agg": {"type": "string", "description": "mean|sum|min|max|median"},
                            "title": {"type": "string"},
                            "out_dir": {"type": "string"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                            "dpi": {"type": "integer"},
                            "grid": {"type": "boolean"}
                        },
                        "required": ["data_json"]
                    },
                },
            },
        ]
        
        logger.debug("schemas_openai: {} tool schema(s)", len(schemas))
        return schemas
