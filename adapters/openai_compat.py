# adapters/openai_compat.py
import os
import requests
import json
from typing import List, Dict, Optional
from models import LLMModel
from tools.registry import ToolRegistry


class OpenAICompatAdapter:
    def __init__(self, model: LLMModel, tools: ToolRegistry):
        self.model = model
        self.tools = tools

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.model.api_key_reqd:
            if not self.model.api_key_env:
                raise RuntimeError(
                    f"Model '{self.model.name}' requires API key but 'api_key_env' is not set in config."
                )
            key = os.getenv(self.model.api_key_env)
            if not key:
                raise RuntimeError(
                    f"API key env '{self.model.api_key_env}' not found in environment. "
                    f"Did you add it to your .env?"
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _tool_schemas(self):
        # If your registry exposes schemas_openai(), use it; else no tools
        if hasattr(self.tools, "schemas_openai"):
            try:
                return self.tools.schemas_openai()
            except Exception:
                return []
        return []

    def chat(self, messages: List[Dict], tool_choice_override: Optional[Dict] = None) -> Dict:
        url = (self.model.endpoint or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model.resolved_model() if hasattr(self.model, "resolved_model") else self.model.name,
            "messages": messages,
            "temperature": getattr(self.model, "temperature", 0.4),
            "stream": False,
        }
        schemas = self._tool_schemas()
        if schemas:
            payload["tools"] = schemas
            payload["tool_choice"] = tool_choice_override or "auto"

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=90)
        try:
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            server_text = getattr(resp, "text", "")
            if status == 401:
                raise RuntimeError("401 Unauthorized. Ensure the required API key env is set, or use a local model.") from e
            if status == 400 and schemas:
                raise RuntimeError(f"OpenAI 400 via /chat/completions (tools enabled): {server_text}") from e
            raise RuntimeError(f"OpenAI error {status}: {server_text}") from e

    def tool_loop(self, messages: List[Dict], max_iters: int = 3) -> str:
        for _ in range(max_iters):
            resp = self.chat(messages)
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")

            if tool_calls or content:
                messages.append(msg)

            if not tool_calls:
                return content or "(no content)"

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

        return "[tool loop exhausted]"
