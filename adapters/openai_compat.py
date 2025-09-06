# adapters/openai_compat.py
from __future__ import annotations

import os
import time
import requests
import json
from typing import List, Dict, Optional
from loguru import logger

from models import LLMModel
from tools.registry import ToolRegistry


class OpenAICompatAdapter:
    def __init__(self, model: LLMModel, tools: ToolRegistry):
        self.model = model
        self.tools = tools
        logger.info(
            "OpenAICompatAdapter init → name='{}' provider='{}' endpoint='{}' tools={}",
            model.name, model.provider, model.endpoint, len(getattr(tools, "tools", {}))
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.model.api_key_reqd:
            if not self.model.api_key_env:
                msg = f"Model '{self.model.name}' requires API key but 'api_key_env' is not set in config."
                logger.error("_headers: {}", msg)
                raise RuntimeError(msg)
            key = os.getenv(self.model.api_key_env)
            if not key:
                msg = (f"API key env '{self.model.api_key_env}' not found in environment. "
                       f"Did you add it to your .env?")
                logger.error("_headers: {}", msg)
                raise RuntimeError(msg)
            headers["Authorization"] = f"Bearer {key}"
            logger.debug("_headers: using env '{}'", self.model.api_key_env)
        return headers

    def _tool_schemas(self):
        if hasattr(self.tools, "schemas_openai"):
            try:
                schemas = self.tools.schemas_openai()
                logger.debug("_tool_schemas: {} schema(s)", len(schemas or []))
                return schemas
            except Exception as e:
                logger.warning("_tool_schemas failed: {}", e)
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

        logger.info("openai_compat.chat → url='{}' model='{}' tools={} override={}",
                    url, payload["model"], len(schemas or []), bool(tool_choice_override))
        t0 = time.time()
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=300)  # CB
        dt = (time.time() - t0) * 1000.0
        logger.info("openai_compat.chat ← status={} time_ms≈{:.0f}", resp.status_code, dt)
        try:
            resp.raise_for_status()
            data = resp.json()
            msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            logger.debug("openai_compat.chat: content_len={} tool_calls={}",
                         len(msg.get("content") or ""), len(tool_calls))
            return data
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            server_text = getattr(resp, "text", "")
            if status == 401:
                logger.error("openai_compat.chat: 401 Unauthorized")
                raise RuntimeError("401 Unauthorized. Ensure the required API key env is set, or use a local model.") from e
            if status == 400 and schemas:
                logger.error("openai_compat.chat: 400 with tools enabled: {}", server_text[:500])
                raise RuntimeError(f"OpenAI 400 via /chat/completions (tools enabled): {server_text}") from e
            logger.error("openai_compat.chat: HTTP {} {}", status, server_text[:500])
            raise RuntimeError(f"OpenAI error {status}: {server_text}") from e


    def chat_stream(self, messages: List[Dict], include_tools: bool = False, tool_choice_override: Optional[Dict] = None):
        """
        Stream chat completion tokens as plain strings.
        Yields text chunks as they arrive. Designed for Direct Chat only.
        """
        url = (self.model.endpoint or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model.resolved_model() if hasattr(self.model, "resolved_model") else self.model.name,
            "messages": messages,
            "temperature": getattr(self.model, "temperature", 0.4),
            "stream": True,
        }
        if include_tools:
            schemas = self._tool_schemas()
            if schemas:
                payload["tools"] = schemas
                payload["tool_choice"] = tool_choice_override or "auto"

        logger.info("openai_compat.chat_stream → url='{}' model='{}' include_tools={}", url, payload["model"], include_tools)
        t0 = time.time()
        resp = requests.post(url, headers=self._headers(), json=payload, stream=True, timeout=300)
        try:
            resp.raise_for_status()
        except Exception as e:
            try:
                server_text = resp.text[:500]
            except Exception:
                server_text = "<no body>"
            status = getattr(resp, "status_code", "<?>")
            logger.error("openai_compat.chat_stream ✗ status={} body≈{}", status, server_text)
            raise RuntimeError(f"OpenAI stream error {status}: {server_text}") from e

        # SSE stream lines: each 'data: {...}' or 'data: [DONE]'
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}) if "delta" in choices[0] else (choices[0].get("message") or {})
                # Only stream user-facing text; ignore tool call deltas
                text = delta.get("content")
                if text:
                    yield text
        dt = (time.time() - t0) * 1000.0
        logger.info("openai_compat.chat_stream ✓ time_ms≈{:.0f}", dt)

    def tool_loop(self, messages: List[Dict], max_iters: int = 3) -> str:
        logger.info("openai_compat.tool_loop: max_iters={}", max_iters)
        for it in range(max_iters):
            resp = self.chat(messages)
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if tool_calls or content:
                messages.append(msg)
            logger.debug("tool_loop iter {}: tool_calls={} content_len={}", it + 1, len(tool_calls), len(content or ""))

            if not tool_calls:
                logger.info("tool_loop: returning final content (len={})", len(content or ""))
                return content or "(no content)"

            for tc in tool_calls:
                fn = (tc.get("function") or {}).get("name")
                args_str = (tc.get("function") or {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                logger.info("tool_loop → call '{}' args_keys={}", fn, list(args.keys()))
                try:
                    result = self.tools.call(fn, **args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": fn,
                        "content": json.dumps(result)
                    }
                    logger.info("tool_loop ✓ '{}'", fn)
                except Exception as e:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": fn,
                        "content": json.dumps({"error": str(e)})
                    }
                    logger.error("tool_loop ✗ '{}': {}", fn, e)
                messages.append(tool_msg)

        logger.warning("tool_loop: exhausted after {} iterations", max_iters)
        return "[tool loop exhausted]"
