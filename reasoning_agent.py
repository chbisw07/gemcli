# =========================
# reasoning_agent.py  (full file)
# =========================
from __future__ import annotations

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from agent import Agent
from models import LLMModel
from tools.registry import ToolRegistry
from config_home import project_rag_json

# NEW: central router + canonical tool specs/helpers
from indexing.router import route_query, compose_agent_system_prompt
# Try to import the Agent+RAG system prompt composer; fall back to a local template if not present
try:
    from indexing.router import compose_agent_system_prompt, COV_MIN  # project-level router.py
except Exception:  # pragma: no cover
    compose_agent_system_prompt = None  # type: ignore
    COV_MIN = 4
    _AGENT_RAG_SYS_PROMPT = """You are an Educational Agent. Your job is to (1) understand the user’s task,
(2) fetch supporting material with RAG, (3) produce a clean, study-ready output.

Rules:
- Prefer RAG. If RAG returns fewer than {COV_MIN} passages or low scores, expand queries; if still weak, use web tools.
- Math: use \\( inline \\) and \\[ display \\]. Do not use $…$.
- Cite sources inline as [S1], [S2] and include a “Sources” section mapping these tags.
- When user asks for questions: include “Problems” then “Solutions” (if requested). Respect requested counts, difficulty, subject/chapter filters.
- Never invent facts; if not found, say “Not found” and still produce practice items (clearly marked).
"""
from tools.tool_schema import (
    TOOL_SPECS,
    normalize_args,
    allowlist_for_route,
    apply_bindings,
    looks_like_placeholder,
    openai_schemas_from_specs
)

# Tools that can modify files (we enforce dry-run in analysis_only mode)
_EDIT_TOOLS = {"write_file", "replace_in_file", "bulk_edit", "rewrite_naive_open", "format_python_files"}


class ReasoningAgent(Agent):
    """
    Router-first agent that:
      1) calls the centralized router (RAG-backed) to infer route (code/doc/hybrid/tabular),
      2) builds a compact plan (no model planning required),
      3) executes with guardrails (args validation, bindings, allowlist gating),
      4) synthesizes a final answer grounded in retrieved chunks (and code hits when applicable).
    Returns a JSON string of step results (compatible with the Streamlit UI).
    """

    # canonical tool specs (imported)
    _TOOL_SPECS: Dict[str, Dict[str, Any]] = TOOL_SPECS

    def __init__(self, model: LLMModel, tools: ToolRegistry, enable_tools: bool = True):
        super().__init__(model, tools, enable_tools)
        logger.info("ReasoningAgent (router-first) initialized: tools_enabled={}", enable_tools)
        self._last_router: Dict[str, Any] = {}
        self.system_override: str = ""

    # ---------------- public: router info for UI ----------------
    def router_info(self) -> Dict[str, Any]:
        return dict(self._last_router or {})


    def set_system_override(self, text: str) -> None:
        """Set a user-provided system prompt addendum (from UI)."""
        try:
            self.system_override = (text or "").strip()
        except Exception:
            self.system_override = ""

    # ---------------- routing + planning ----------------

    # ---- Generic keyword/phrase extractor → focused sub-queries (domain-agnostic) ----
    def _keywordize(self, text: str, max_terms: int = 8) -> list[str]:
        """
        Turn an arbitrary task prompt into a handful of retrieval queries:
        - preserve quoted phrases, if any
        - extract keywords (drop tiny/stop words), keep order & de-dupe
        - return a small list: phrases first, then 2–4 term combos, then single terms
        """
        if not text:
            return []
        t = text.strip()
        import re
        phrases = [m.group(1).strip() for m in re.finditer(r'"([^"]+)"', t)]
        # Drop glue words so we don't query "you", "assistant", etc.
        stop = {
            "the","and","or","a","an","to","for","of","in","on","by","with","from","at",
            "this","that","these","those","it","is","are","be","as","into","via","using",
            "make","create","build","generate","please","show","need","want","how","why",
            "plan","run","agent","llm","tools","rag","final","answer","step","steps",
            "you","your","we","assistant","first","then","finish","only","order","constraints",
            # metadata-ish tokens that are not meaningful queries on their own
            "start_line","end_line","line_start","line_end",
            # drop generic/noisy tokens that add no retrieval signal
            "code","scv2",
            # remaining low-signal tokens we saw producing 0 hits
            "backend","stack"
        }
        tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", t)]

        # Promote "codey" tokens (func names, .py) to the front
        keywords, seen, codey = [], set(), []
        for w in tokens:
            if len(w) <= 2 or w in stop:
                continue
            if w not in seen:
                seen.add(w)
                if "_" in w or w.endswith("py"):
                    codey.append(w)
                else:
                    keywords.append(w)

        queries: list[str] = []
        queries.extend(phrases[:3])  # keep a few phrases if present
        head = (codey + keywords)[:6]
        if head:
            if len(head) >= 4: queries.append(" ".join(head[:4]))
            if len(head) >= 3: queries.append(" ".join(head[:3]))
            if len(head) >= 2: queries.append(" ".join(head[:2]))
        for w in head:
            if len(queries) >= max_terms: break
            if w not in queries: queries.append(w)
        return queries[:max_terms]

    def _route_and_seed(self, prompt: str, rag_on: bool = True) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
        """
        Use the centralized router to decide a route and fetch initial chunks.
        Returns (route, rag_result, scores).
        """
        # When RAG is OFF, skip router+retriever entirely.
        if not rag_on:
            info = {"route": "direct", "scores": {"code": 0, "document": 0, "tabular": 0}, "top_k": 0, "chunks": []}
        else:
            try:
                info = route_query(project_root=str(self.tools.root), query=prompt)
            except Exception as e:
                logger.warning("router route_query failed: {}", e)
                # Prefer hybrid so code tools are available even if routing hiccups
                info = {"route": "hybrid", "scores": {"code": 0, "document": 0, "tabular": 0}, "top_k": 0, "chunks": []}

        self._last_router = {
            "route": info.get("route"),
            "scores": info.get("scores"),
            "top_k": info.get("top_k"),
            "prompt": prompt,  # keep to mine filenames later
            "auto_relaxed": info.get("auto_relaxed"),
        }
        rag = {"chunks": info.get("chunks") or [], "top_k": info.get("top_k")}
        logger.info("router: route={} scores={} top_k={}", self._last_router["route"], self._last_router["scores"], rag["top_k"])
        return self._last_router["route"], rag, self._last_router["scores"]

    def analyze_and_plan(self, query: str, *, rag_on: bool = True) -> List[Dict[str, Any]]:
        """
        Build a *small*, deterministic plan (no LLM planning).
        Steps are later filtered by a per-route allowlist.
        Now: early broad-but-scoped RAG, then focused RAG sub-queries, then (for code/hybrid) call-graph → _answer.
        """
        route, _rag, _scores = self._route_and_seed(query, rag_on=rag_on)

        # ---- CHART SHORT-CIRCUIT: route chart prompts straight to chart tools ----
        low = (query or "").lower()

        def _extract_inline_json_array(txt: str) -> Optional[str]:
            # Grab the first JSON array of objects: [ { ... }, ... ]
            m = re.search(r"\[\s*\{.*?\}\s*\]", txt, flags=re.S)
            return m.group(0) if m else None

        def _csv_in_prompt(txt: str) -> Optional[str]:
            # Match a CSV path token anywhere in the prompt
            m = re.search(r"([^\s'\"<>]+\.csv)", txt, flags=re.I)
            return m.group(1) if m else None

        is_charty = any(w in low for w in ("chart", "plot", "graph", "draw"))

        # JSON → draw_chart_data
        if is_charty:
            json_blob = _extract_inline_json_array(query)
            if json_blob:
                # Heuristic column guesses
                x_guess = "date" if ("date" in low or re.search(r"\"date\"\s*:", json_blob, flags=re.I)) else None
                y_guess = None
                for cand in ("price", "close", "value", "y"):
                    if cand in low or re.search(fr"\"{cand}\"\s*:", json_blob, flags=re.I):
                        y_guess = cand
                        break

                return [{
                    "tool": "draw_chart_data",
                    "args": {
                        "data_json": json_blob,
                        "x": x_guess,
                        "y": y_guess,
                        "kind": "line",
                        "title": "Chart from JSON",
                    },
                    "description": "Render chart from inline JSON records (matplotlib)",
                    "critical": True,
                }]

            # CSV → draw_chart_csv
            csv_path = _csv_in_prompt(query)
            if csv_path:
                resample = "W" if "week" in low else ("M" if "month" in low else None)
                x_guess = "Date" if "date" in low else None
                y_guess = None
                for cand in ("Close", "Price", "Value"):
                    if cand.lower() in low:
                        y_guess = cand
                        break

                return [{
                    "tool": "draw_chart_csv",
                    "args": {
                        "csv_path": csv_path,
                        "x": x_guess,
                        "y": y_guess,
                        "resample": resample,
                        "kind": "line",
                        "title": "Chart from CSV",
                    },
                    "description": "Render chart from CSV (matplotlib)",
                    "critical": True,
                }]
        
        plan: List[Dict[str, Any]] = []

        if rag_on:
            # Early broad-but-scoped retrieval to prime good hits/caching
            plan.append({
                "tool": "rag_retrieve",
                "args": {"query": query, "top_k": 12, "enable_filename_boost": True},
                "description": "Broad retrieval for residual gaps (scoped)",
                "critical": False,
            })

            # Focused RAG sub-queries (function/file oriented)
            for sub in self._keywordize(query, max_terms=8):
                plan.append({
                    "tool": "rag_retrieve",
                    "args": {"query": sub, "top_k": 8, "enable_filename_boost": True},
                    "description": "Focused retrieval from sub-query",
                    "critical": False,
                })

        # Extract function-like targets for call-graph & code search signals
        func_targets: List[str] = []
        try:
            raw_funcs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", query or "")
            seen = set()
            for f in raw_funcs:
                if f.endswith("_retrieve") or f.startswith("calculate_"):
                    if f not in seen:
                        seen.add(f)
                        func_targets.append(f)
        except Exception:
            pass
        # Optional filename hint from the prompt (first .py mentioned)
        file_hints = re.findall(r"\b([A-Za-z0-9_\-]+\.py)\b", query or "") or []
        default_hint = file_hints[0] if file_hints else ""
        def _hint_for(fn: str) -> str:
            fn_l = (fn or "").lower()
            if fn_l.endswith("_retrieve") or fn_l.startswith("calculate_"):
                return "retriever.py"
            if "sanitize" in fn_l or "upsert" in fn_l:
                return "indexer.py"
            return default_hint or "retriever.py"


        # ---- Detect function-like targets for call-graph injection ----
        low = (query or "").lower()
        func_targets: List[str] = []
        try:
            # capture snake_case / camelCase names; keep unique order
            raw_funcs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", query or "")
            seen = set()
            for f in raw_funcs:
                if f.endswith("_retrieve") or f.startswith("calculate_"):
                    if f not in seen:
                        seen.add(f)
                        func_targets.append(f)
        except Exception:
            pass
        # optional filename hint from the prompt (e.g., rag/retriever.py)
        file_hints = re.findall(r"\b([A-Za-z0-9_\-]+\.py)\b", query or "") or []
        file_hint = file_hints[0] if file_hints else ""

        if route == "code":
            plan += [
                {
                    "tool": "scan_relevant_files",
                    "args": {"prompt": query, "subdir": "", "exts": [".py"], "max_results": 50},
                    "description": "Heuristically list likely files",
                    "critical": False,
                },
            ]
            # NEW: auto call-graph steps when function-like tokens are present
            if func_targets:
                for fn in func_targets[:2]:
                    plan.append({
                        "tool": "call_graph_for_function",
                        "args": {"function": fn, "file_hint": _hint_for(fn)},
                        "description": f"Build call graph for {fn}",
                        "critical": False,
                    })
            plan += [
                {
                    "tool": "search_code",
                    # use compact code tokens so we actually get line-precise hits
                    "args": {"query": " ".join(func_targets) or " ".join(self._keywordize(query, 4)), "subdir": "", "exts": [".py"], "max_results": 120},
                    "description": "Find matching lines in codebase (symbol tokens)",
                    "critical": False,
                },
                {
                    "tool": "_answer",
                    "args": {"prompt": query},
                    "description": "Synthesize answer using RAG + code hits",
                    "critical": True,
                },
            ]
        elif route == "hybrid":
            plan += [
                {
                    "tool": "search_code",
                    "args": {"query": " ".join(func_targets) or " ".join(self._keywordize(query, 4)), "subdir": "", "exts": [".py"], "max_results": 120},
                    "description": "Include code hits (symbol tokens) in addition to docs",
                    "critical": False,
                },
            ]
            # NEW: auto call-graph steps for hybrid as well
            if func_targets:
                for fn in func_targets[:2]:
                    plan.append({
                        "tool": "call_graph_for_function",
                        "args": {"function": fn, "file_hint": _hint_for(fn)},
                        "description": f"Build call graph for {fn}",
                        "critical": False,
                    })
            plan += [
                {
                    "tool": "_answer",
                    "args": {"prompt": query},
                    "description": "Synthesize answer using doc + code context",
                    "critical": True,
                },
            ]
        else:  # document / tabular
            plan += [
                {
                    "tool": "_answer",
                    "args": {"prompt": query},
                    "description": "Answer grounded in retrieved documents",
                    "critical": True,
                }
            ]

        # Enforce route-specific tool allowlist up front
        allow = allowlist_for_route(route)
        filtered: List[Dict[str, Any]] = []
        for step in plan:
            t = step.get("tool")
            # Keep _answer always; also keep call_graph in code/hybrid even if not in allowlist
            if t == "_answer" or t in allow or (t == "call_graph_for_function" and route in ("code", "hybrid")):
                filtered.append(step)
            else:
                logger.debug("analyze_and_plan: dropping tool '{}' not allowed for route '{}'", t, route)

        logger.debug("analyze_and_plan: route={} steps={} (filtered from {})", route, len(filtered), len(plan))
        return filtered

    # ---------------- executor ----------------

    def _normalize_step_args(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Normalize/validate args against canonical TOOL_SPECS."""
        ok, fixed, err = normalize_args(tool, args, specs=self._TOOL_SPECS)
        return ok, fixed, err

    def _apply_bindings(self, args: Dict[str, Any], last_by_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve {{BIND:tool.field}} placeholders using central helper."""
        return apply_bindings(args, last_by_tool)

    def _looks_like_placeholder(self, val: Any) -> bool:
        return looks_like_placeholder(val)

    def _synthesize_answer(self, prompt: str, last_by_tool: Dict[str, Any], system_override: Optional[str] = None, rag_on: bool = True) -> str:
        """
        Final model call using whatever context we have gathered.
        Builds a compact, auditable context block:
        - RAG evidence (deduped) with [file Ls–Le score=…] headers
        - Code search hits grouped by file (first 10 lines/file)
        - Call-graph summary (if available)
        - Chart artifacts produced by draw_chart_* tools
        Then asks the model to synthesize a precise answer using those citations.
        """
        def _label_for_chunk(c: Dict[str, Any]) -> str:
            md = c.get("metadata") or {}
            tag = (md.get("file_path") or md.get("relpath") or md.get("file_name") or "?")
            extra = []
            # page number (for PDFs)
            if md.get("page_number"):
                extra.append(f"p.{md.get('page_number')}")
            # chunk type (code_block, pdf_page, table, etc.)
            if md.get("chunk_type"):
                extra.append(md.get("chunk_type"))
            # lines (from AST-aware chunker)
            try:
                sline = md.get("line_start") or md.get("start_line")
                eline = md.get("line_end")   or md.get("end_line") or md.get("end")
                if sline is not None or eline is not None:
                    extra.append(f"L{(sline if sline is not None else '?')}–{(eline if eline is not None else '?')}")
            except Exception:
                pass
            # retrieval score (cosine or sim proxy)
            try:
                sc = c.get("score", None)
                if isinstance(sc, (int, float)):
                    extra.append(f"score={sc:.3f}")
            except Exception:
                pass
            return f"[{tag}{' ' + ', '.join(extra) if extra else ''}]"

        # -------- Collect RAG evidence (across multiple rag steps; dedupe by id) --------
        rag = last_by_tool.get("rag_retrieve") or {}
        chunks = [] if not rag_on else list(rag.get("chunks") or [])
        total_chunks = len(chunks)
        # some executors keep only last step; in ours we already merge — keep defensive anyway
        seen_ids, evidence_blocks = set(), []
        for c in chunks[:128]:  # hard cap to avoid giant prompts
            cid = c.get("id") or (c.get("metadata") or {}).get("id")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            label = _label_for_chunk(c)
            body = (c.get("document") or "").strip()
            if not body:
                continue
            evidence_blocks.append(f"{label}\n{body}")
        rag_context = "\n\n---\n\n".join(evidence_blocks)

        # -------- Collect code search hits (grouped by file) --------
        code_hits_section = ""
        try:
            hits = last_by_tool.get("search_code")
            # our executor stores 'out' (a list) for non web_* tools
            hit_list = hits if isinstance(hits, list) else (hits.get("result") if isinstance(hits, dict) else None)
            if isinstance(hit_list, list) and hit_list:
                per_file: Dict[str, List[str]] = {}
                for h in hit_list:
                    rel = h.get("file"); line = h.get("line"); txt = h.get("text")
                    if not rel: 
                        continue
                    per_file.setdefault(rel, []).append(f"{line}: {txt}")
                parts = []
                for rel, lines in per_file.items():
                    parts.append(f"## {rel}\n" + "\n".join(lines[:10]))
                code_hits_section = "\n\n".join(parts[:8])  # up to 8 files
        except Exception:
            pass

        # -------- Surface call-graph output (latest invocation) --------
        callgraph_section = ""
        try:
            cg = last_by_tool.get("call_graph_for_function")
            if isinstance(cg, dict) and (cg.get("edges") or cg.get("calls")):
                fn = cg.get("function") or "<function>"
                fpath = cg.get("file") or "<file>"
                edges = cg.get("edges") or []
                # compact textual summary of first few edges
                edge_txt = "; ".join([f"{u} → {v}" for (u, v) in edges[:12]])
                calls = cg.get("calls") or []
                direct_txt = ", ".join(sorted({c.get("name") for c in calls if isinstance(c, dict) and c.get("name")}))[:400]
                lines = [f"Function: {fn}", f"Defined in: {fpath}"]
                if direct_txt:
                    lines.append(f"Direct callees: {direct_txt}")
                if edge_txt:
                    lines.append(f"Edges: {edge_txt}")
                callgraph_section = "\n".join(lines)
        except Exception:
            pass

        # -------- Chart artifacts (paths) --------
        artifact_lines = []
        for tname in ("draw_chart_data", "draw_chart_csv"):
            res = last_by_tool.get(tname)
            if isinstance(res, dict) and res.get("image_path"):
                artifact_lines.append(f"Chart: `{res['image_path']}`")
        artifacts_section = "\n".join(artifact_lines)

        # -------- Build the final prompt --------
        context_sections = []
        if artifacts_section:
            context_sections.append(artifacts_section)
        if rag_on and rag_context:
            context_sections.append("### RAG evidence\n" + rag_context)
        if code_hits_section:
            context_sections.append("### Code search hits\n" + code_hits_section)
        if callgraph_section:
            context_sections.append("### Call graph\n" + callgraph_section)

        context = ("\n\n".join(context_sections)).strip()
        # -------- Educational Agent+RAG system prompt (with graceful fallback) --------
        # If a composer is available (router.compose_agent_system_prompt), use it.
        # Otherwise, use a local educational template + the previous safety rails.
        override = (system_override or self._system_override or self.system_override)
        if compose_agent_system_prompt:
            system_msg = compose_agent_system_prompt(system_override=override, cov_min=COV_MIN)
        else:
            system_msg = _AGENT_RAG_SYS_PROMPT.format(COV_MIN=COV_MIN).strip()
            if override and override.strip():
                system_msg += "\n\n" + override.strip()
            # keep the existing evidence-usage guardrails as a suffix (only if RAG is ON)
            if rag_on:
                system_msg += (
                    "\n\nUse the provided EVIDENCE only; when citing, rely on the bracket headers already shown "
                    "([file Ls–Le score=…] or '## path:line'). Keep changes minimal and concrete."
                )
        # If RAG is sparse, transparently allow generative fill to meet the user's request.
        if rag_on and total_chunks < 10:
            system_msg += (
                "\n\n[Note] Retrieved evidence is sparse; if citations are insufficient, "
                "proceed to generate the requested output using domain knowledge and clear assumptions. "
                "Be transparent about any gaps and prefer grounding in the snippets that were found."
            )
        # Surface router auto-relax info (if any) to the model as context
        if rag_on:
            try:
                ar = self._last_router.get("auto_relaxed")
                if ar:
                    system_msg += "\n(Info: retrieval threshold was auto-relaxed to increase coverage.)"
            except Exception:
                pass
        if rag_on:
            user_msg = (
                f"QUESTION:\n{prompt}\n\n"
                f"EVIDENCE:\n{context if context else '(no retrieved context)'}\n\n"
                "INSTRUCTIONS:\n"
                "- Be concise and specific.\n"
                "- Prefer exact code citations from the headers above over paraphrase.\n"
                "- When proposing code changes, include filenames and unified diff hunks.\n"
                "- If the evidence is insufficient for a claim, say so and list what extra you would need."
            )
        else:
            # RAG off: no evidence block; ask for a direct, well-structured answer.
            user_msg = (
                f"QUESTION:\n{prompt}\n\n"
                "INSTRUCTIONS:\n"
                "- Produce a complete, well-structured answer.\n"
                "- Do not include a Supporting Evidence section.\n"
                "- If assumptions are required, state them briefly."
            )
        try:
            resp = self.adapter.chat([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])
            content = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or "(no content)"
        except Exception as e:
            logger.exception("final LLM call failed: {}", e)
            return f"[error] {e}"

        # -------- Footer: symbol hits + grounding summary (auditable) --------
        try:
            sym = (rag.get("symbol_boosted") or [])
            footer = []
            if sym:
                footer.append("Symbol hits: " + ", ".join(sym[:8]))
            # tiny grounding summary by file
            by_file: Dict[str, float] = {}
            for c in chunks[:40]:
                md = c.get("metadata") or {}
                key = (md.get("file_path") or md.get("relpath") or "?")
                sc = c.get("score")
                try:
                    val = float(sc) if sc is not None else 0.0
                except Exception:
                    val = 0.0
                by_file[key] = by_file.get(key, 0.0) + val
            if by_file:
                top = sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)[:5]
                footer.append("Grounding: " + "; ".join(f"{k}={v:.3f}" for k, v in top))
            if footer:
                content = f"{content}\n\n---\n" + "\n".join(footer)
        except Exception:
            pass

        return content

    # --- add this helper inside class ReasoningAgent (anywhere above execute_plan) ---
    def _apply_arg_aliases(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rename common alias parameters to the canonical names expected by TOOL_SPECS.
        Non-destructive: only maps if the canonical key isn't already present.
        """
        alias_map = {
            "edu_explain": {
                "text": "question_or_topic",
                "query": "question_or_topic",
            },
            "rag_retrieve": {
                "k": "top_k",
            },
        }
        out = dict(args or {})
        for src, dst in alias_map.get(tool, {}).items():
            if src in out and dst not in out:
                out[dst] = out.pop(src)
        return out

    def execute_plan(self, plan: List[Dict[str, Any]], max_iters: int = 10, analysis_only: bool = False, progress_cb=None, rag_on: bool = True) -> str:
        results: List[Dict[str, Any]] = []
        last_by_tool: Dict[str, Any] = {}
        logger.info("execute_plan: steps={} analysis_only={}", len(plan or []), analysis_only)
        def _emit(event: str, **data):
            try:
                if progress_cb is None: return True
                cont = progress_cb(event, **data)
                return (cont is not False)
            except Exception: return True
        if not _emit("plan_start", steps=len(plan or [])): return json.dumps(results)

        # Per-route allowlist (double-check at execution time)
        route = (self._last_router.get("route") or "document").lower()
        allow = allowlist_for_route(route)

        # Tools that should always receive project_root/rag_path defaults if missing
        _CTX_TOOLS = {
            "rag_retrieve","read_file","list_files","search_code","scan_relevant_files","analyze_files",
            "edu_detect_intent","edu_similar_questions","edu_question_paper","edu_explain",
            "edu_extract_tables","edu_build_blueprint","find_related_files","analyze_code_structure",
            "detect_errors","call_graph_for_function","analyze_function"
        }

        for idx, step in enumerate(plan or [], start=1):
            if not _emit("step_start", step=idx, tool=step.get("tool"), description=step.get("description")): break
            tool = step.get("tool")
            args = dict(step.get("args") or {})
            desc = step.get("description") or ""
            critical = bool(step.get("critical", True))

            # Hard guard: with RAG OFF, never execute rag tools even if a plan slipped one in.
            if not rag_on and tool in ("rag_retrieve", "route_query"):
                results.append({"step": idx, "tool": tool, "description": desc, "error": "RAG is disabled; skipping retrieval step.", "success": False})
                _emit("step_done", step=idx, tool=tool, ok=False, error="RAG disabled")
                continue

            # Enforce tool allowlist (virtual tool _answer is always allowed).
            # Keep call-graph in code/hybrid even if not listed in allowlist.
            if tool != "_answer" and tool not in allow and not (tool == "call_graph_for_function" and route in ("code","hybrid")):
                msg = f"Tool '{tool}' not allowed for route '{route}'"
                results.append({"step": idx, "tool": tool, "description": desc, "error": msg, "success": False})
                if critical:
                    break
                continue
            
            # Normalize aliases first (e.g., text/query -> question_or_topic)
            args = self._apply_arg_aliases(tool, args)

            # Normalize + validate
            ok, fixed_args, err = self._normalize_step_args(tool, args)
            if not ok:
                results.append({"step": idx, "tool": tool, "description": desc, "error": err, "success": False})
                if critical:
                    break
                else:
                    continue

            # Dry-run guard for editing tools (if analysis_only)
            if analysis_only and tool in _EDIT_TOOLS:
                fixed_args = dict(fixed_args)
                if "dry_run" in (self._TOOL_SPECS.get(tool) or {}).get("allowed", set()):
                    fixed_args["dry_run"] = True

            # Resolve BIND placeholders against earlier outputs
            fixed_args = self._apply_bindings(fixed_args, last_by_tool)

            # Block obvious placeholders
            if self._looks_like_placeholder(fixed_args):
                results.append({"step": idx, "tool": tool, "description": desc, "error": f"Placeholder args: {fixed_args}", "success": False})
                if critical:
                    break
                else:
                    continue

            # Virtual tool: _answer
            if tool == "_answer":
                _emit("synthesis_start")
                out_text = self._synthesize_answer(
                    fixed_args.get("prompt", ""), last_by_tool, system_override=self._system_override, rag_on=rag_on
                )
                rec = {"step": idx, "tool": tool, "description": desc, "args": fixed_args, "result": out_text, "success": True}
                results.append(rec)
                last_by_tool[tool] = rec
                _emit("synthesis_done")
                continue

            # Real tool
            try:
                # Ensure RAG context for tools that need it
                if tool in _CTX_TOOLS:
                    if fixed_args.get("project_root") in (None,"",".","default"):
                        fixed_args["project_root"] = str(getattr(self.tools, "root", ""))
                    if fixed_args.get("rag_path") in (None,"",".","default"):
                        # Use the authoritative per-project rag.json (not the app template)
                        try:
                            fixed_args["rag_path"] = str(project_rag_json(project_root=str(getattr(self.tools, "root", ""))))
                        except Exception:
                            # leave unset: retriever will infer from project_root
                            pass
                    # Gentler retrieval threshold; the index + query models mismatch slightly
                    if tool == "rag_retrieve" and fixed_args.get("min_score") in (None, "", 0):
                        fixed_args["min_score"] = 0.35
                    # If a filename is mentioned, add a Chroma-valid filter (no $contains)
                    if tool == "rag_retrieve" and not fixed_args.get("where"):
                        import re as _re
                        _prompt = (self._last_router or {}).get("prompt", "") + " " + str(fixed_args.get("query",""))
                        names = _re.findall(r"\b([A-Za-z0-9_\-]+\.(?:py|pdf|md|txt))\b", _prompt)
                        if names:
                            root = str(getattr(self.tools, "root", "")).rstrip("/\\")
                            # Try exact relpaths and absolute paths (Chroma supports $in/$eq)
                            rels = [f"rag/{n}" for n in names] + names
                            abss = [f"{root}/{p}" for p in rels]
                            fixed_args["where"] = {
                                "$or": [
                                    {"relpath": {"$in": rels}},
                                    {"file_path": {"$in": abss}},
                                ]
                            }

                    # If not explicitly set by the caller, encourage filename-aware boosting
                    if tool == "rag_retrieve" and "enable_filename_boost" not in fixed_args:
                        fixed_args["enable_filename_boost"] = True

                out = self.tools.call(tool, **fixed_args)

                # RAG retry if empty: try a simplified query once
                if tool == "rag_retrieve" and isinstance(out, dict) and not (out.get("chunks") or []):
                    import re as _re
                    q = fixed_args.get("query","")
                    # simplify query once if first retrieval came back empty
                    simple = " ".join(_re.findall(r"[A-Za-z0-9]+", q)[:12])
                    if simple and simple != q:
                        try:
                            out2 = self.tools.call(tool, **{**fixed_args, "query": simple})
                            if out2 and (out2.get("chunks") or []):
                                out = out2
                        except Exception:
                            pass

                rec = {"step": idx, "tool": tool, "description": desc, "args": fixed_args, "result": out, "success": True}
                results.append(rec)
                _emit("step_done", step=idx, tool=tool, ok=True,
                      summary=(f"chunks={len((out or {}).get('chunks', []))}" if tool=="rag_retrieve" and isinstance(out, dict) else None))
                # For web_* tools, keep the whole record (args+result) for downstream BINDs
                if tool == "rag_retrieve":
                    # Accumulate chunks across multiple retrieval steps
                    prior = last_by_tool.get("rag_retrieve")
                    if isinstance(prior, dict):
                        merged = dict(prior)
                        merged["chunks"] = (prior.get("chunks") or []) + (out.get("chunks") or [])
                        last_by_tool["rag_retrieve"] = merged
                    else:
                        last_by_tool["rag_retrieve"] = out
                else:
                    last_by_tool[tool] = rec if tool.startswith("web_") else out

                logger.info("execute_plan: '{}' OK", tool)
            except Exception as e:
                results.append({"step": idx, "tool": tool, "description": desc, "error": str(e), "success": False})
                _emit("step_done", step=idx, tool=tool, ok=False, error=str(e))
                logger.exception("execute_plan: '{}' failed: {}", tool, e)
                if critical:
                    break

        out_json = json.dumps(results, indent=2)
        logger.info(
            "execute_plan: done ok={} fail={} steps={}",
            sum(1 for r in results if r.get("success")),
            sum(1 for r in results if not r.get("success")),
            len(results),
        )
        return out_json

    def openai_tool_schemas(self, route: str | None = None) -> list[dict]:
        """
        Return OpenAI function/tool schemas filtered by the router's route.
        Use this if you want to give the model a smaller, route-aware toolset.
        """
        r = (route or (self._last_router.get("route") if isinstance(self._last_router, dict) else None) or "document").lower()
        allow = allowlist_for_route(r)
        specs = {k: v for k, v in TOOL_SPECS.items() if k in allow and not k.startswith("_")}
        return openai_schemas_from_specs(specs)

    def allowed_tool_names(self, route: str | None = None) -> list[str]:
        """
        Convenience for UIs: which tools are visible for this route?
        """
        r = (route or (self._last_router.get("route") if isinstance(self._last_router, dict) else None) or "document").lower()
        return sorted(t for t in allowlist_for_route(r) if not t.startswith("_"))

    # ---------------- entrypoint ----------------

    def ask_with_planning(self, query: str, max_iters: int = 10, analysis_only: bool = False,
                          system_override: Optional[str] = None, rag_on: bool = True) -> str:
        """
        Entry point for Agent Plan & Run.
        `system_override` lets the UI pass a user/system specialization for the educational agent.
        """
        self._system_override = (system_override or "").strip() or None
        plan = self.analyze_and_plan(query, rag_on=rag_on)
        return self.execute_plan(plan, max_iters=max_iters, analysis_only=analysis_only, rag_on=rag_on)
