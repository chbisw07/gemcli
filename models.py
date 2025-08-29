# models.py
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class LLMModel:
    name: str
    provider: str
    endpoint: str
    model: Optional[str] = None
    temperature: float = 0.2
    system_prompt: str = "You are a helpful coding assistant."
    max_context_tokens: int = 8192
    # API key handling is driven entirely by config:
    api_key_reqd: bool = False               # if true, we must read api_key_env
    api_key_env: Optional[str] = None        # name of env var to read (e.g., OPENAI_API_KEY)
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
            api_key_env=d.get("api_key_env"),   # DO NOT default; trust config
            tags=d.get("tags", []),
        )

    def resolved_model(self) -> str:
        return self.model or self.name

    def is_openai_compat(self) -> bool:
        prov = (self.provider or "").lower()
        host = (self.endpoint or "").lower()
        return (
            prov in {"openai", "deepseek", "mistral", "qwen", "anyscale", "groq", "gpt-oss"}
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
            "think" in name or "reason" in name
            or any(t in {"thinking", "reasoning", "chain-of-thought"} for t in tags)
            or "reason" in sys_p
        )


class ModelRegistry:
    """
    Accepts either:
    {
      "default_llm_model": "gpt-4o",
      "llm_models": [ {...}, {...} ]
    }
    or legacy:
    {
      "models": [ {...}, {...} ]
    }
    """
    def __init__(self, config_path: str):
        self.models: Dict[str, LLMModel] = {}
        self.default_name: Optional[str] = None
        self.load(config_path)

    def load(self, path: str):
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Model config not found at: {cfg_path.resolve()}")
        data = json.loads(cfg_path.read_text())

        if isinstance(data, dict) and "llm_models" in data:
            entries = data["llm_models"]
            self.default_name = data.get("default_llm_model")
        elif isinstance(data, dict) and "models" in data:
            entries = data["models"]
            self.default_name = data.get("default_model") or None
        else:
            raise ValueError("Invalid model config: expected 'llm_models' or 'models' array.")

        if not isinstance(entries, list):
            raise ValueError("Invalid model list in config.")

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid model entry: {entry}")
            m = LLMModel.from_dict(entry)
            self.models[m.name] = m

        if not self.default_name and self.models:
            self.default_name = next(iter(self.models.keys()))

    def get(self, name: Optional[str]) -> LLMModel:
        if name:
            if name not in self.models:
                raise ValueError(f"Model '{name}' not found. Available: {', '.join(self.models.keys())}")
            return self.models[name]
        if not self.default_name:
            raise ValueError("No default model configured and none specified.")
        return self.models[self.default_name]

    def list(self):
        return list(self.models.values())
