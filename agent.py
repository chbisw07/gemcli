# agent.py
import json
import re
from typing import List, Dict, Optional, Any

from loguru import logger

from models import LLMModel
from tools.registry import ToolRegistry
from adapters.openai_compat import OpenAICompatAdapter

# Optional Gemini adapter; safe stub if not present
try:
    from adapters.gemini import GeminiAdapter  # type: ignore
except Exception:  # pragma: no cover
    class GeminiAdapter:  # minimal stub to avoid NameError if not used
        def __init__(self, *_, **__):
            raise NotImplementedError("GeminiAdapter not implemented; please add adapters/gemini.py")


class Agent:
    _RE_QUOTED = r"'([^']*)'|\"([^\"]*)\""   # matches 'x' or "x"

    def __init__(self, model: LLMModel, tools: ToolRegistry, enable_tools: bool = True):
        self.model = model
        self.tools = tools
        self.enable_tools = enable_tools

        if model.is_openai_compat():
            self.adapter = OpenAICompatAdapter(model, tools)
            adapter_name = "OpenAICompatAdapter"
        elif model.is_gemini():
            self.adapter = GeminiAdapter(model, tools)
            adapter_name = "GeminiAdapter"
        else:
            self.adapter = OpenAICompatAdapter(model, tools)
            adapter_name = "OpenAICompatAdapter"

        logger.info(
            "Agent.__init__ → model='{}' provider='{}' endpoint='{}' tools_enabled={} adapter={}",
            model.name, model.provider, model.endpoint, enable_tools, adapter_name
        )

    # --------------------------- Prompt ---------------------------

    def system_prompt(self) -> str:
        """
        Keep the LLM in charge. If tools are available, instruct the preferred flow and
        provide a JSON fallback protocol if the model can't emit native tool_calls.
        """
        prompt = self.model.system_prompt or "You are a helpful coding assistant."
        if self.model.supports_thinking():
            prompt += "\nThink step-by-step internally, then provide a concise final answer."
            logger.debug("system_prompt: thinking mode enabled for '{}'", self.model.name)

        if self.enable_tools:
            if self.model.is_openai_compat():
                prompt += (
                    "\nYou can call tools via function calling. "
                    "When asked to find or fix code, CALL a tool instead of narrating. "
                    "Preferred flow: scan_relevant_files → analyze_files → propose & apply edits (dry_run first). "
                    "NEVER dump large file blobs; summarize and use the tools."
                )
            else:
                prompt += (
                    "\nWhen you want to act, reply with ONLY a single JSON object on one line: "
                    "{\"tool\":\"<name>\",\"arguments\":{...}}. No prose. "
                    "After the tool result is returned, continue. "
                    "Preferred flow: scan_relevant_files → analyze_files → propose & apply edits (dry_run first). "
                    "NEVER dump large file blobs; summarize and use the tools."
                )
        logger.debug("system_prompt built (len={})", len(prompt))
        return prompt

    # --------------------------- Public API ---------------------------

    def ask_once(self, query: str, max_iters: int = 3) -> str:
        """
        Single-turn ask with an internal tool loop. Keeps the LLM primary:
        - If OpenAI-native tool_calls arrive, execute them in the correct order.
        - Otherwise, if assistant content is a JSON tool request, execute it (textual protocol).
        - Deterministic parsers + search intent filter act as safety hatches.
        """
        logger.info("Agent.ask_once → prompt='{}...'", (query or "")[:200])
        # Deterministic / intent fast-paths (no LLM needed)
        direct = self._try_direct_actions(query)
        if direct is not None:
            logger.info("ask_once → satisfied via direct path")
            return direct

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": query},
        ]
        logger.debug("ask_once: tools_enabled={} max_iters={}", self.enable_tools, max_iters)

        if not self.enable_tools:
            resp = self.adapter.chat(messages)
            content = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            logger.info("ask_once (no tools) → {} chars", len(content or ""))
            return content

        tool_choice_override = self._route_tool_call(query)  # hint for first hop
        if tool_choice_override:
            logger.debug("ask_once: tool_choice_override={}", tool_choice_override)

        for it in range(max_iters):
            try:
                resp = self.adapter.chat(messages, tool_choice_override=tool_choice_override)
            except Exception as e:
                logger.exception("adapter.chat failed on iter {}: {}", it + 1, e)
                return f"[error] chat failed: {e}"

            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            logger.debug("iter {}: tool_calls={} content_len={}", it + 1, len(tool_calls), len(content or ""))

            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    fn = (tc.get("function") or {}).get("name")
                    args_str = (tc.get("function") or {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                    except Exception:
                        args = {}
                    logger.info("→ tool_call '{}' args_keys={}", fn, list(args.keys()))
                    try:
                        result = self.tools.call(fn, **args)
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": fn,
                            "content": json.dumps(result),
                        }
                        logger.info("✓ tool '{}' executed", fn)
                    except Exception as e:
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": fn,
                            "content": json.dumps({"error": str(e)}),
                        }
                        logger.error("✗ tool '{}' failed: {}", fn, e)
                    messages.append(tool_msg)

                tool_choice_override = None
                continue

            # Textual JSON tool protocol (for non-OpenAI models)
            if content and not self.model.is_openai_compat():
                ran, tool_msg = self._maybe_run_textual_tool(content)
                if ran:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(tool_msg)
                    logger.info("Executed textual tool request: {}", tool_msg.get("name"))
                    tool_choice_override = None
                    continue

            # Safety net: if model failed to call tools
            if content and self.model.is_openai_compat():
                if self._looks_like_scan_search(query) and not tool_calls:
                    try:
                        term = self._extract_search_term(query) or query.strip()
                        subdir = self._guess_subdir_from_prompt(query)
                        res = self.tools.call("search_code", query=term, subdir=subdir)
                        logger.info("search_code fallback executed (term='{}', subdir='{}')", term, subdir)
                        return f"search_code -> {json.dumps(res, indent=2)}"
                    except Exception as e:
                        logger.error("search_code fallback failed: {}", e)
                        return f"[error] search_code failed: {e}"

            # Final answer
            if content:
                ran, result_or_text = self._maybe_run_textual_tool(content)
                if ran:
                    logger.info("Final textual tool execution returned")
                    return result_or_text["content"]
                logger.info("ask_once → returning assistant content ({} chars)", len(content or ""))
                return content

            logger.debug("ask_once: no content and no tool_calls on iter {}", it + 1)

        logger.warning("ask_once → tool loop exhausted after {} iteration(s)", max_iters)
        return "[tool loop exhausted]"

    # --------------------------- JSON-tool fallback ---------------------------

    def _maybe_run_textual_tool(self, content: str):
        """If assistant content looks like {\"tool\":\"...\",\"arguments\":{...}}, execute it."""
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
                name = obj["tool"]
                args = obj.get("arguments") or {}
                logger.info("textual_tool → '{}'", name)
                result = self.tools.call(name, **args)
                return True, {
                    "role": "tool",
                    "tool_call_id": f"textual-{name}",
                    "name": name,
                    "content": json.dumps(result),
                }
        except Exception:
            pass
        return False, content

    # --------------------------- Router ---------------------------

    def _route_tool_call(self, query: str) -> Optional[Dict[str, Any]]:
        """Heuristic router that maps a natural-language prompt to the most likely tool."""
        q = (query or "").lower()

        def choose(name: str) -> Dict[str, Any]:
            return {"type": "function", "function": {"name": name}}

        # Edits / replacements
        if any(k in q for k in ("bulk edit", "bulk change", "multiple files", "across the repo", "across the project")):
            logger.debug("_route_tool_call → bulk_edit")
            return choose("bulk_edit")
        if any(k in q for k in ("replace", "substitute", "find and replace", "search and replace", "swap text")):
            logger.debug("_route_tool_call → replace_in_file")
            return choose("replace_in_file")
        if any(k in q for k in ("rewrite open(", "fix open()", "with open", "context manager", "with  open", "doesn't close", "doesnt close")):
            logger.debug("_route_tool_call → rewrite_naive_open")
            return choose("rewrite_naive_open")

        # Write/create files
        if any(k in q for k in ("write file", "create file", "new file", "make file", "save file")):
            logger.debug("_route_tool_call → write_file")
            return choose("write_file")

        # Formatting / cleanup
        if any(k in q for k in ("format python", "run black", "reformat", "auto format", "auto-format", "code style", "pep8")):
            logger.debug("_route_tool_call → format_python_files")
            return choose("format_python_files")

        # Scan → Analyze flow
        if any(k in q for k in ("scan for", "find relevant files", "which files are relevant", "search repo for", "locate files", "discover files")):
            logger.debug("_route_tool_call → scan_relevant_files")
            return choose("scan_relevant_files")
        if any(k in q for k in ("analyze files", "analyze these files", "summarize files", "quick analysis", "file signals", "show preview", "line counts")):
            logger.debug("_route_tool_call → analyze_files")
            return choose("analyze_files")

        # Listing
        if ("list" in q or "show" in q) and ("file" in q or "files" in q):
            logger.debug("_route_tool_call → list_files")
            return choose("list_files")

        # Searching
        if any(k in q for k in ("search", "find", "grep", "look for", "scan")) and any(
            k in q for k in ("code", "function", "class", "def ", ".py", ".js", ".ts", ".jsx", ".tsx")
        ):
            logger.debug("_route_tool_call → search_code")
            return choose("search_code")

        # Reading
        if any(k in q for k in ("read", "open", "show contents", "view file", "print file")):
            logger.debug("_route_tool_call → read_file")
            return choose("read_file")

        logger.debug("_route_tool_call → none")
        return None

    # --------------------------- REPL ---------------------------

    def repl(self, max_iters: int = 3):
        tool_names = list(getattr(self.tools, "tools", {}).keys()) if self.enable_tools else []
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

    # --------------------------- Helpers: search intent filter ---------------------------

    def _is_search_intent(self, q: str) -> bool:
        return bool(re.search(r"(?i)\b(search|scan|find)\b", q or ""))

    def _extract_search_term(self, q: str) -> Optional[str]:
        m = re.search(r"""['"]([^'"]+)['"]""", q or "")
        if m and m.group(1).strip():
            return m.group(1).strip()
        m = re.search(r"([A-Za-z_][A-Za-z0-9_\.\-]+)", q or "")
        if m:
            return m.group(1).strip()
        parts = re.findall(r"[A-Za-z0-9_\.\-]+", q or "")
        return parts[-1] if parts else None

    def _guess_subdir_from_prompt(self, q: str) -> str:
        root = getattr(self.tools, "root", None)
        if root is None:
            return ""
        m = re.search(r"(?i)\b(in|under|inside|within)\b\s+(.*)$", q or "")
        tail = m.group(2) if m else (q or "")
        raw_tokens = re.findall(r"[A-Za-z0-9_\-/.]+", tail.lower())
        stop = {"the", "this", "that", "these", "those", "folder", "dir", "directory",
                "module", "modules", "pkg", "package", "project", "repo", "code",
                "inside", "within", "under", "in"}
        tokens = [t.strip("/.") for t in raw_tokens if t not in stop and t.strip("/.")]
        if not tokens:
            return ""
        dirs = []
        for p in root.rglob("*"):
            if p.is_dir():
                try:
                    rel = str(p.relative_to(root))
                except Exception:
                    continue
                if rel == ".":
                    continue
                dirs.append(rel)
        best_dir = ""
        best_score = 0
        for d in dirs:
            segs = d.lower().split("/")
            score = 0
            for t in tokens:
                if not t:
                    continue
                if t in segs:
                    score += 2
                elif t in d.lower():
                    score += 1
            if score > best_score:
                best_dir, best_score = d, score
        return best_dir if best_score > 0 else ""

    # (legacy parsers unchanged)
    def _unquote(self, m) -> str:
        return m.group(1) if m.group(1) is not None else m.group(2)

    def _first(self, *vals) -> str:
        for v in vals:
            if v is not None:
                return str(v)
        return ""

    def _parse_replace(self, q: str):
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
        m = pat.search(q or "")
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
        pat = re.compile(
            r"""(?ix)
            \b(?:create|write)\b .*? \bfile\b .*?
            (?:under|at|in)\s+(?P<path>\S+)
            (?: .*? (?:with|containing)\s+
                (?:'(?P<text_sq>[^']*)'|"(?P<text_dq>[^"]*)")
            )?
            """,
        )
        m = pat.search(q or "")
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

    def _parse_search(self, q: str):
        m = re.search(
            r"(?i)\bsearch\s+for\s+(?:'(?P<sq>[^']*)'|\"(?P<dq>[^\"]*)\"|(?P<plain>\S+))\s+(?:in|under)\s+(?P<dir>\S+)",
            q or "",
        )
        if not m:
            return None
        term = m.group("sq") or m.group("dq") or m.group("plain")
        return {"tool": "search_code", "args": {"query": term, "subdir": m.group("dir")}}

    def _parse_list_files(self, q: str):
        m = re.search(r"(?i)list\s+all\s+python\s+files\s+(?:in|under)\s+(?P<dir>\S+)", q or "")
        if not m:
            return None
        return {"tool": "list_files", "args": {"subdir": m.group("dir"), "exts": [".py"]}}

    def _flag(self, q: str, *phrases: str) -> bool:
        ql = (q or "").lower()
        return any(p.lower() in ql for p in phrases)

    def _extract_json_block(self, q: str) -> Optional[str]:
        m = re.search(r"```json\s*(?P<body>\{.*?\}|\[.*?\])\s*```", q or "", flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group("body")
        m2 = re.search(r"(?P<body>\{.*\}|\[.*\])", q or "", flags=re.DOTALL)
        if m2:
            return m2.group("body")
        return None

    def _parse_bool_flag(self, q: str, default: bool, *aliases: str) -> bool:
        ql = (q or "").lower()
        yes = {"true", "yes", "y", "on", "enable", "enabled"}
        for a in aliases:
            if re.search(rf"\(\s*{re.escape(a)}\s*\)", ql):
                return True
            if re.search(rf"\(\s*no\s+{re.escape(a)}\s*\)", ql):
                return False
            m = re.search(rf"{re.escape(a)}\s*=\s*(\w+)", ql)
            if m:
                return m.group(1) in yes
        return default

    def _parse_format_python(self, q: str):
        m = re.search(r"(?:format|reformat|run\s+black).*(?:under|in)\s+(?P<dir>\S+)", q or "", flags=re.IGNORECASE)
        if not m:
            return None
        dir_val = m.group("dir")
        mlen = re.search(r"(?:line\s*length|ll)\s*(?P<ll>\d{2,3})", q or "", flags=re.IGNORECASE)
        ll = int(mlen.group("ll")) if mlen else 88
        dry = self._parse_bool_flag(q or "", True, "preview", "dry run")
        return {
            "tool": "format_python_files",
            "args": {"subdir": dir_val, "line_length": ll, "dry_run": dry},
        }

    def _parse_rewrite_open(self, q: str):
        if not self._flag(q, "rewrite open", "fix open()", "with open", "context manager"):
            return None
        m = re.search(r"(?:in|under)\s+(?P<dir>\S+)", q or "", flags=re.IGNORECASE)
        dir_val = m.group("dir") if m else "."
        dry = self._parse_bool_flag(q or "", True, "dry run", "preview")
        return {
            "tool": "rewrite_naive_open",
            "args": {"dir": dir_val, "exts": [".py"], "dry_run": dry, "backup": True},
        }

    def _parse_bulk_edit(self, q: str):
        if not self._flag(q, "bulk edit", "bulk change", "multiple files", "across the repo", "across the project"):
            return None
        raw = self._extract_json_block(q or "")
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
        dry = self._parse_bool_flag(q or "", default=True, *("dry run", "preview"))
        if not edits:
            return {
                "tool": "bulk_edit",
                "args": {"edits": [], "dry_run": True, "backup": True},
                "warning": "No edits parsed. Provide JSON `edits` or specify explicit file paths."
            }
        return {"tool": "bulk_edit", "args": {"edits": edits, "dry_run": dry, "backup": True}}

    # --------------------------- Direct executor ---------------------------

    def _try_direct_actions(self, query: str) -> Optional[str]:
        """Deterministic fast-paths & robust intent filter for search."""
        q = (query or "").strip()
        if not q:
            return None

        if self._is_search_intent(q):
            term = self._extract_search_term(q)
            if term:
                subdir = self._guess_subdir_from_prompt(q)
                try:
                    res = self.tools.call("search_code", query=term, subdir=subdir)
                    logger.info("direct_action search_code(term='{}', subdir='{}')", term, subdir)
                    return f"search_code -> {json.dumps(res, indent=2)}"
                except Exception as e:
                    logger.error("direct_action search_code failed: {}", e)
                    return f"[error] search_code failed: {e}"

        rep = self._parse_replace(q)
        if rep:
            try:
                res = self.tools.call(rep["tool"], **rep["args"])
                logger.info("direct_action replace_in_file")
                return f"{rep['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action replace_in_file failed: {}", e)
                return f"[error] {rep['tool']} failed: {e}"

        wf = self._parse_writefile(q)
        if wf:
            try:
                res = self.tools.call(wf["tool"], **wf["args"])
                logger.info("direct_action write_file")
                return f"{wf['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action write_file failed: {}", e)
                return f"[error] {wf['tool']} failed: {e}"

        lf = self._parse_list_files(q)
        if lf:
            try:
                res = self.tools.call(lf["tool"], **lf["args"])
                logger.info("direct_action list_files")
                return f"{lf['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action list_files failed: {}", e)
                return f"[error] {lf['tool']} failed: {e}"

        srch = self._parse_search(q)
        if srch:
            try:
                res = self.tools.call(srch["tool"], **srch["args"])
                logger.info("direct_action search_code (legacy)")
                return f"{srch['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action legacy search_code failed: {}", e)
                return f"[error] {srch['tool']} failed: {e}"

        fmt = self._parse_format_python(q)
        if fmt:
            try:
                res = self.tools.call(fmt["tool"], **fmt["args"])
                logger.info("direct_action format_python_files")
                return f"{fmt['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action format_python_files failed: {}", e)
                return f"[error] {fmt['tool']} failed: {e}"

        rno = self._parse_rewrite_open(q)
        if rno:
            try:
                res = self.tools.call(rno["tool"], **rno["args"])
                logger.info("direct_action rewrite_naive_open")
                return f"{rno['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action rewrite_naive_open failed: {}", e)
                return f"[error] {rno['tool']} failed: {e}"

        be = self._parse_bulk_edit(q)
        if be:
            try:
                res = self.tools.call(be["tool"], **be["args"])
                prefix = "" if "warning" not in be else "[warn] " + be["warning"] + "\n"
                logger.info("direct_action bulk_edit")
                return f"{prefix}{be['tool']} -> {json.dumps(res, indent=2)}"
            except Exception as e:
                logger.error("direct_action bulk_edit failed: {}", e)
                return f"[error] {be['tool']} failed: {e}"

        return None

    # --------------------------- Tiny nudge helpers ---------------------------

    def _looks_like_scan_search(self, q: str) -> bool:
        return bool(re.search(r"(?i)\b(scan|search|find)\b", q or ""))
