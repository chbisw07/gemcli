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
import sys
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

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:  # abstract
        raise NotImplementedError

    def tool_loop(self, messages: List[Dict[str, str]], max_iters: int = 3) -> str:
        """Runs a simple tool loop with native tool_calls or JSON-Tool fallback."""
        final_content = ""
        for _ in range(max_iters):
            resp = self.chat(messages)
            choice = (resp.get("choices") or [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])
            content = message.get("content", "")

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
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": fn,
                            "content": json.dumps(result),
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": fn,
                            "content": json.dumps({"error": str(e)}),
                        })
                continue  # ask again with tool outputs

            # If we got normal content, return it
            if content:
                final_content = content
                break

            # JSON-Tool fallback
            try:
                c = (choice.get("message") or {}).get("content", "")
                obj = json.loads(c)
                if isinstance(obj, dict) and "tool" in obj:
                    fn = obj.get("tool")
                    args = obj.get("arguments", {})
                    try:
                        result = self.tools.call(fn, **args)
                        messages.append({"role": "tool", "name": fn, "content": json.dumps(result)})
                        continue
                    except Exception as e:
                        messages.append({"role": "tool", "name": fn, "content": json.dumps({"error": str(e)})})
                        continue
            except Exception:
                pass
            break
        return final_content or "(no content)"


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
        """
        Primary OpenAI-compatible chat path using /v1/chat/completions.

        - Sends OpenAI tool schemas for native function calling.
        - On 401 → explain that a key is missing/invalid.
        - On 400 → fallback to /v1/responses (no tools) to salvage a plain-text reply.
        """
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
                    "401 Unauthorized. Load OPENAI_API_KEY from .env (or use --env-file), "
                    "or choose a local model (e.g., --model deepseek-coder-6.7b-instruct)."
                ) from e
            if status == 400:
                # Some accounts/models may balk at certain tool payloads.
                # Fall back to /responses (no tools) so the session can proceed.
                has_tools = bool(self.tools.schemas_openai())
                if has_tools:
                    raise RuntimeError(f"OpenAI 400 via /chat/completions (tools enabled): {server_text}") from e
                
                return self._chat_via_responses(messages)
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
                "\nYou have tools for reading files, listing files, and searching code. "
                "You MUST use tools instead of writing Python code when asked to list, read, or search project files. "
                "If tools are unavailable, respond with a single JSON object: {\"tool\": \"name\", \"arguments\": {...}}."
            )
        return prompt

    def ask_once(self, query: str, max_iters: int = 3) -> str:
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
                    msg = resp.get("choices", [{}])[0].get("message", {})
                    tool_calls = msg.get("tool_calls", [])

                    if tool_calls:
                        # Step 2: execute each tool
                        for tc in tool_calls:
                            fn = tc.get("function", {}).get("name")
                            args_str = tc.get("function", {}).get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                                result = self.tools.call(fn, **args)
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id"),
                                    "name": fn,
                                    "content": json.dumps(result)
                                }
                            except Exception as e:
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id"),
                                    "name": fn,
                                    "content": json.dumps({"error": str(e)})
                                }
                            messages.append(tool_msg)

                        # Step 3: continue tool loop with tool results now in context
                        messages.insert(-1, msg)  # insert assistant response before tool replies
                        return self.adapter.tool_loop(messages, max_iters=max_iters - 1)
                except Exception:
                    pass  # fallback to default tool loop if anything fails

            # Let model decide tool use
            return self.adapter.tool_loop(messages, max_iters=max_iters)

        # Tools disabled → normal chat
        resp = self.adapter.chat(messages)
        return (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")

    def _route_tool_call(self, query: str) -> Optional[Dict[str, Any]]:
        """Heuristic tool router"""
        q = query.lower()
        if "list" in q and ("python" in q or ".py" in q) and "file" in q:
            return {"type": "function", "function": {"name": "list_files"}}
        if ("search" in q or "find" in q) and any(k in q for k in ("code", "function", "def ")):
            return {"type": "function", "function": {"name": "search_code"}}
        if ("read" in q or "open" in q) and any(k in q for k in (".py", ".md", ".js", ".jsx", "file")):
            return {"type": "function", "function": {"name": "read_file"}}
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
