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

# NEW: central router + canonical tool specs/helpers
from indexing.router import route_query
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

    # ---------------- public: router info for UI ----------------
    def router_info(self) -> Dict[str, Any]:
        return dict(self._last_router or {})

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
        stop = {
            "the","and","or","a","an","to","for","of","in","on","by","with","from","at",
            "this","that","these","those","it","is","are","be","as","into","via","using",
            "make","create","build","generate","please","show","need","want","how","why",
            "plan","run","agent","llm","tools","rag","final","answer","step","steps"
        }
        tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", t)]
        keywords, seen = [], set()
        for w in tokens:
            if len(w) <= 2 or w in stop:
                continue
            if w not in seen:
                seen.add(w)
                keywords.append(w)
        queries: list[str] = []
        queries.extend(phrases[:3])  # keep a few phrases if present
        head = keywords[:6]
        if head:
            if len(head) >= 4: queries.append(" ".join(head[:4]))
            if len(head) >= 3: queries.append(" ".join(head[:3]))
            if len(head) >= 2: queries.append(" ".join(head[:2]))
        for w in head:
            if len(queries) >= max_terms: break
            if w not in queries: queries.append(w)
        return queries[:max_terms]

    def _route_and_seed(self, prompt: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
        """
        Use the centralized router to decide a route and fetch initial chunks.
        Returns (route, rag_result, scores).
        """
        try:
            info = route_query(project_root=str(self.tools.root), query=prompt)
        except Exception as e:
            logger.warning("router route_query failed: {}", e)
            info = {"route": "document", "scores": {"code": 0, "document": 0, "tabular": 0}, "top_k": 0, "chunks": []}

        self._last_router = {"route": info.get("route"), "scores": info.get("scores"), "top_k": info.get("top_k")}
        rag = {"chunks": info.get("chunks") or [], "top_k": info.get("top_k")}
        logger.info("router: route={} scores={} top_k={}", self._last_router["route"], self._last_router["scores"], rag["top_k"])
        return self._last_router["route"], rag, self._last_router["scores"]

    def analyze_and_plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Build a *small*, deterministic plan (no LLM planning).
        Steps are later filtered by a per-route allowlist.
        Now: fan out several focused RAG sub-queries, then one broad RAG, then route steps → _answer.
        """
        route, _rag, _scores = self._route_and_seed(query)

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

        # Focused RAG sub-queries (domain-agnostic)
        for sub in self._keywordize(query, max_terms=8):
            plan.append({
                "tool": "rag_retrieve",
                "args": {"query": sub, "top_k": 8},
                "description": "Focused retrieval from sub-query",
                "critical": False,
            })
        # One broad retrieval for residual gaps
        plan.append({
            "tool": "rag_retrieve",
            "args": {"query": query, "top_k": 12},
            "description": "Broad retrieval for any residual gaps",
            "critical": False,
        })

        if route == "code":
            plan += [
                {
                    "tool": "scan_relevant_files",
                    "args": {"prompt": query, "subdir": "", "exts": [".py"], "max_results": 50},
                    "description": "Heuristically list likely files",
                    "critical": False,
                },
                {
                    "tool": "search_code",
                    "args": {"query": query, "subdir": "", "exts": [".py"], "max_results": 200},
                    "description": "Find matching lines in codebase",
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
                    "args": {"query": query, "subdir": "", "exts": [".py"], "max_results": 120},
                    "description": "Include code hits in addition to docs",
                    "critical": False,
                },
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
            if t == "_answer" or t in allow:
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

    def _synthesize_answer(self, prompt: str, last_by_tool: Dict[str, Any]) -> str:
        """Final model call using any context we have gathered."""
        # Gather doc snippets (aggregate chunks if multiple rag_retrieve calls ran)
        docs: List[str] = []
        rag = last_by_tool.get("rag_retrieve") or {}
        for c in (rag.get("chunks") or [])[:24]:
            md = c.get("metadata") or {}
            tag = (md.get("file_path") or md.get("relpath") or "?")
            extra = []
            if md.get("page_number"):
                extra.append(f"p.{md.get('page_number')}")
            if md.get("chunk_type"):
                extra.append(md.get("chunk_type"))
            label = f"[{tag}{' ' + ', '.join(extra) if extra else ''}]"
            body = c.get("document") or ""
            docs.append(f"{label}\n{body}".strip())

        # Gather code hits (search_code results)
        code_hits = last_by_tool.get("search_code")
        if isinstance(code_hits, dict) and isinstance(code_hits.get("result"), list):
            hits = code_hits["result"][:50]
            # compact format
            paths: Dict[str, List[str]] = {}
            for h in hits:
                rel = h.get("file"); line = h.get("line"); txt = h.get("text")
                if rel:
                    paths.setdefault(rel, []).append(f"{line}: {txt}")
            if paths:
                code_block = [f"## {rel}\n" + "\n".join(lines[:10]) for rel, lines in paths.items()]
                docs.append("\n\n".join(code_block))

        # -------- PATCH B: surface chart tool outputs (saved PNG path) --------
        def _tool_result(name: str):
            val = last_by_tool.get(name)
            if isinstance(val, dict) and "result" in val:
                return val["result"]
            return val

        aux_lines = []
        for tname in ("draw_chart_data", "draw_chart_csv"):
            res = _tool_result(tname)
            if isinstance(res, dict) and res.get("image_path"):
                aux_lines.append(f"Chart saved to: `{res['image_path']}`")

        if aux_lines:
            # Prepend above other context so users see the path immediately
            docs.insert(0, "\n".join(aux_lines))
        # -------- END PATCH B --------

        context = ("\n\n".join(docs)).strip()
        sys = (
            "You are a precise coding/doc assistant. Cite from the provided CONTEXT by referring to "
            "[file p.X] tags when helpful. If the context is insufficient, say so clearly and propose next steps."
        )
        user = (
            f"QUESTION:\n{prompt}\n\n"
            f"CONTEXT:\n{context if context else '(no retrieved context)'}\n\n"
            "INSTRUCTIONS:\n"
            "- Be concise and specific.\n"
            "- If code changes are needed, outline them with filenames and diff hunks.\n"
            "- If answering from PDFs, include page numbers when possible."
        )
        try:
            resp = self.adapter.chat([
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ])
            content = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            return content or "(no content)"
        except Exception as e:
            logger.exception("final LLM call failed: {}", e)
            return f"[error] {e}"

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

    def execute_plan(self, plan: List[Dict[str, Any]], max_iters: int = 10, analysis_only: bool = False) -> str:
        results: List[Dict[str, Any]] = []
        last_by_tool: Dict[str, Any] = {}
        logger.info("execute_plan: steps={} analysis_only={}", len(plan or []), analysis_only)

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
            tool = step.get("tool")
            args = dict(step.get("args") or {})
            desc = step.get("description") or ""
            critical = bool(step.get("critical", True))

            # Enforce tool allowlist (virtual tool _answer is always allowed)
            if tool != "_answer" and tool not in allow:
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
                out_text = self._synthesize_answer(fixed_args.get("prompt", ""), last_by_tool)
                rec = {"step": idx, "tool": tool, "description": desc, "args": fixed_args, "result": out_text, "success": True}
                results.append(rec)
                last_by_tool[tool] = rec
                continue

            # Real tool
            try:
                # Ensure RAG context for tools that need it
                if tool in _CTX_TOOLS:
                    if fixed_args.get("project_root") in (None,"",".","default"):
                        fixed_args["project_root"] = str(getattr(self.tools, "root", ""))
                    if fixed_args.get("rag_path") in (None,"",".","default"):
                        from config_home import GLOBAL_RAG_PATH
                        fixed_args["rag_path"] = str(GLOBAL_RAG_PATH)
                    # Gentler retrieval threshold; the index + query models mismatch slightly
                    if tool == "rag_retrieve" and fixed_args.get("min_score") in (None, "", 0):
                        fixed_args["min_score"] = 0.35
                    # If a filename is mentioned in prompt/query, add a where-filter automatically
                    if tool == "rag_retrieve" and not fixed_args.get("where"):
                        import re as _re
                        _prompt = (self._last_router or {}).get("prompt", "") + " " + str(fixed_args.get("query",""))
                        _names = _re.findall(r"\b([A-Za-z0-9_\-]+\.(?:py|pdf|md|txt))\b", _prompt)
                        if _names:
                            name = _names[0]
                            fixed_args["where"] = {"$or":[
                                {"file_path":{"$contains": name}},
                                {"relpath":{"$contains": name}},
                            ]}

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

    def ask_with_planning(self, query: str, max_iters: int = 10, analysis_only: bool = False) -> str:
        plan = self.analyze_and_plan(query)
        return self.execute_plan(plan, max_iters=max_iters, analysis_only=analysis_only)
