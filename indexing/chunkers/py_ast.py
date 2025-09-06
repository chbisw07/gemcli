# indexing/chunkers/py_ast.py
from __future__ import annotations

import os
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger


def _iter_py_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]

def _best_file(root: Path, hint: Optional[str]) -> Optional[Path]:
    if not hint:
        return None
    # try exact relpath first, then basename match
    cand = root / hint
    if cand.exists():
        return cand
    hits = [p for p in _iter_py_files(root) if p.name == Path(hint).name]
    return hits[0] if hits else None

def build_call_graph(function: str, file_hint: Optional[str], project_root: str | Path) -> Dict:
    """
    Build a lightweight intra-file call graph for `function`.
    Returns: {
      "function": str, "file": str,
      "edges": [ [caller, callee], ... ],
      "calls": [ {"name": str, "lineno": int}, ... ],
      "summary": str
    }
    """
    root = Path(project_root)
    target = _best_file(root, file_hint)
    files = [target] if target else _iter_py_files(root)
    func_def: Optional[ast.FunctionDef | ast.AsyncFunctionDef] = None
    src_file: Optional[Path] = None

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            mod = ast.parse(text, filename=str(f))
        except Exception:
            continue
        for node in ast.walk(mod):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
                func_def, src_file = node, f
                break
        if func_def:
            break

    edges: List[Tuple[str, str]] = []
    calls: List[Dict] = []
    if func_def and src_file:
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call):
                callee = None
                fn = node.func
                if isinstance(fn, ast.Name):
                    callee = fn.id
                elif isinstance(fn, ast.Attribute):
                    callee = fn.attr
                if callee:
                    edges.append((function, callee))
                    calls.append({"name": callee, "lineno": getattr(node, "lineno", None)})

    summary = ""
    if src_file:
        summary = f"{function} in {src_file.as_posix()}: calls " + (
            ", ".join(sorted({c['name'] for c in calls})) if calls else "no functions"
        )

    # normalize edge format to lists for JSON safety
    edge_list = [[u, v] for (u, v) in edges[:64]]
    return {
        "function": function,
        "file": src_file.as_posix() if src_file else "",
        "edges": edge_list,
        "calls": calls[:128],
        "summary": summary,
    }

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _get(cfg: dict, *path, default=None):
    """Safe nested get: _get(cfg, "chunking", "code", "max_lines", default=120)."""
    cur = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_lines(file_path: str) -> List[str]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().splitlines()
    except Exception as e:
        logger.warning("py_ast.chunk: failed to read '%s': %s", file_path, e)
        return []


def _node_span(node: ast.AST, total_lines: int, fallback_end: Optional[int] = None) -> Tuple[int, int]:
    """
    Return 1-based (start_line, end_line) for an AST node.
    Prefers node.end_lineno when available; otherwise uses the provided fallback_end.
    """
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", None)
    if end is None:
        end = fallback_end if isinstance(fallback_end, int) and fallback_end >= start else total_lines
    # Clamp
    start = max(1, min(start, total_lines))
    end = max(start, min(end, total_lines))
    return start, end


def _collect_defs(tree: ast.AST, total_lines: int) -> List[Tuple[str, int, int]]:
    """
    Collect top-level function/class defs and class methods as (label, start, end).
    Label examples: 'def foo', 'class Bar', 'Bar.method'.
    """
    spans: List[Tuple[str, int, int]] = []

    # Top-level defs/classes
    module_body = getattr(tree, "body", []) or []
    for i, node in enumerate(module_body):
        nxt = module_body[i + 1] if (i + 1) < len(module_body) else None
        nxt_start = getattr(nxt, "lineno", None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _node_span(node, total_lines, fallback_end=(nxt_start - 1) if nxt_start else None)
            spans.append((f"def {node.name}", start, end))

        elif isinstance(node, ast.ClassDef):
            c_start, c_end = _node_span(node, total_lines, fallback_end=(nxt_start - 1) if nxt_start else None)
            spans.append((f"class {node.name}", c_start, c_end))

            # Methods inside the class
            body = getattr(node, "body", []) or []
            for j, m in enumerate(body):
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Fallback: limit by next sibling or class end
                    nxt2 = body[j + 1] if (j + 1) < len(body) else None
                    nxt2_start = getattr(nxt2, "lineno", None)
                    m_start, m_end = _node_span(m, total_lines, fallback_end=(nxt2_start - 1) if nxt2_start else c_end)
                    spans.append((f"{node.name}.{m.name}", m_start, m_end))

    # Sort and merge small gaps/overlaps lightly
    spans.sort(key=lambda x: (x[1], x[2]))
    merged: List[Tuple[str, int, int]] = []
    for label, s, e in spans:
        if not merged:
            merged.append((label, s, e))
            continue
        prev_label, ps, pe = merged[-1]
        if s <= pe + 1 and (e - ps) <= 2000:  # simple coalescing rule for near-adjacent blocks
            merged[-1] = (f"{prev_label}; {label}", ps, max(pe, e))
        else:
            merged.append((label, s, e))
    return merged


def _window(lines: List[str], start_line: int, end_line: int, max_lines: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Break a (start_line, end_line) segment into windows (1-based inclusive line numbers).
    """
    out: List[Tuple[int, int]] = []
    step = max(1, max_lines - overlap)
    cur = start_line
    while cur <= end_line:
        last = min(end_line, cur + max_lines - 1)
        out.append((cur, last))
        if last == end_line:
            break
        cur = last - overlap + 1
    return out


def _make_chunk(rel: str, base: str, ext: str, window_lines: List[str], line_start: int, line_end: int,
                label: str, preview_chars: int) -> Dict[str, Any]:
    text = "\n".join(window_lines)
    meta = {
        "file_path": rel,
        "relpath": rel,
        "file_name": base,
        "file_ext": ext,
        "profile": "code",
        "chunk_type": "code_block",
        "name": f"{base} :: {label} :: L{line_start}-{line_end}",
        "line_start": line_start,
        "line_end": line_end,
        "lang": "python",
        "symbols": [label] if label else [],
    }
    if preview_chars and preview_chars > 0:
        meta["preview"] = text[:preview_chars]
    return {
        "id": _sha(f"{rel}:{line_start}:{line_end}:{label}"),
        "document": text,
        "metadata": meta,
    }


def chunk(file_path: str, cfg: dict) -> List[Dict[str, Any]]:
    """
    Chunk a Python source file using AST-aware boundaries (functions/classes/methods),
    with a fallback to fixed-size line windows.

    Metadata fields:
      - file_path, relpath, file_name, file_ext
      - profile: "code"
      - chunk_type: "code_block"
      - name: "<file> :: <symbol> :: L<start>-<end>"
      - line_start, line_end (1-based, inclusive)
      - lang: "python"
      - symbols: [ "Class.method" | "def foo" | ... ]
      - preview: optional first N chars (controlled by settings.metadata.store_preview_chars)
    """
    logger.debug("py_ast.chunk: begin '%s'", file_path)

    lines = _read_lines(file_path)
    if not lines:
        return []

    # Settings with safe fallbacks
    max_lines = int(_get(cfg, "chunking", "code", "max_lines", default=120) or 120)
    overlap = int(_get(cfg, "chunking", "code", "overlap_lines", default=20) or 20)
    preview_chars = int(_get(cfg, "metadata", "store_preview_chars", default=0) or 0)

    rel = os.path.relpath(file_path)
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1].lower()
    total = len(lines)

    # Parse AST and collect spans; if parsing fails, fall back to whole-file windowing
    spans: List[Tuple[str, int, int]] = []
    try:
        tree = ast.parse("\n".join(lines), filename=base)
        spans = _collect_defs(tree, total)
    except Exception as e:
        logger.warning("py_ast.chunk: AST parse failed for '%s': %s (falling back)", rel, e)
        spans = []

    chunks: List[Dict[str, Any]] = []

    if spans:
        # Window each span if needed
        for label, s, e in spans:
            if e < s:
                continue
            length = e - s + 1
            if length <= max_lines:
                windowed = [(s, e)]
            else:
                windowed = _window(lines, s, e, max_lines, overlap)
            for ws, we in windowed:
                segment = lines[ws - 1: we]
                chunks.append(_make_chunk(rel, base, ext, segment, ws, we, label, preview_chars))

        # Include any uncovered regions (e.g., module docstring, imports) via coarse windows
        covered = [False] * (total + 1)
        for _, s, e in spans:
            for i in range(max(1, s), min(total, e) + 1):
                covered[i] = True
        i = 1
        while i <= total:
            if covered[i]:
                i += 1
                continue
            j = i
            while j <= total and not covered[j]:
                j += 1
            # [i, j-1] is an uncovered region
            for ws, we in _window(lines, i, j - 1, max_lines, overlap):
                segment = lines[ws - 1: we]
                chunks.append(_make_chunk(rel, base, ext, segment, ws, we, "module", preview_chars))
            i = j
    else:
        # No AST spans; do whole-file windowing
        for ws, we in _window(lines, 1, total, max_lines, overlap):
            segment = lines[ws - 1: we]
            chunks.append(_make_chunk(rel, base, ext, segment, ws, we, "module", preview_chars))

    logger.debug(
        "py_ast.chunk: done file='%s' lines=%d chunks=%d max_lines=%d overlap=%d",
        rel, total, len(chunks), max_lines, overlap
    )
    return chunks

__all__ = ["build_call_graph"]
