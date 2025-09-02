# models.py
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from loguru import logger


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
        m = cls(
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
        logger.debug(
            "LLMModel.from_dict → name='{}', provider='{}', endpoint='{}', model='{}', tags={}",
            m.name, m.provider, m.endpoint, m.model, m.tags,
        )
        return m

    def resolved_model(self) -> str:
        resolved = self.model or self.name
        logger.debug("LLMModel.resolved_model → {}", resolved)
        return resolved

    def is_openai_compat(self) -> bool:
        prov = (self.provider or "").lower()
        host = (self.endpoint or "").lower()
        compat = (
            prov in {"openai", "deepseek", "mistral", "qwen", "anyscale", "groq", "gpt-oss"}
            or host.startswith("http://")
            or host.startswith("https://")
        )
        logger.debug(
            "LLMModel.is_openai_compat(name='{}') → {} (provider='{}', endpoint='{}')",
            self.name, compat, self.provider, self.endpoint,
        )
        return compat

    def is_gemini(self) -> bool:
        is_g = (self.provider or "").lower() in {"google", "gemini"}
        logger.debug("LLMModel.is_gemini(name='{}') → {}", self.name, is_g)
        return is_g

    def supports_thinking(self) -> bool:
        name = (self.name or "").lower()
        tags = [t.lower() for t in (self.tags or [])]
        sys_p = (self.system_prompt or "").lower()
        supports = (
            "think" in name or "reason" in name
            or any(t in {"thinking", "reasoning", "chain-of-thought"} for t in tags)
            or "reason" in sys_p
        )
        logger.debug("LLMModel.supports_thinking(name='{}') → {}", self.name, supports)
        return supports


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
            logger.error("Model config not found at path='{}'", str(cfg_path.resolve()))
            raise FileNotFoundError(f"Model config not found at: {cfg_path.resolve()}")

        logger.info("Loading model config from '{}'", str(cfg_path.resolve()))
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("Failed to parse model config JSON: {}", e)
            raise

        if isinstance(data, dict) and "llm_models" in data:
            entries = data["llm_models"]
            self.default_name = data.get("default_llm_model")
            cfg_style = "modern(llm_models)"
        elif isinstance(data, dict) and "models" in data:
            entries = data["models"]
            self.default_name = data.get("default_model") or None
            cfg_style = "legacy(models)"
        else:
            logger.error("Invalid model config: expected 'llm_models' or 'models' array. Keys={}", list(data.keys()))
            raise ValueError("Invalid model config: expected 'llm_models' or 'models' array.")

        if not isinstance(entries, list):
            logger.error("Invalid model list in config (entries type = {})", type(entries).__name__)
            raise ValueError("Invalid model list in config.")

        loaded = 0
        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("Skipping invalid model entry (not a dict): {}", entry)
                continue
            try:
                m = LLMModel.from_dict(entry)
                if not m.name:
                    logger.warning("Skipping model with empty name: {}", entry)
                    continue
                self.models[m.name] = m
                loaded += 1
            except Exception as e:
                logger.exception("Failed to load a model entry: {}", e)

        # Set default if not present
        if not self.default_name and self.models:
            self.default_name = next(iter(self.models.keys()))

        logger.info(
            "ModelRegistry loaded {} model(s) [{}]; default='{}'",
            loaded, cfg_style, self.default_name,
        )
        if loaded == 0:
            logger.warning("No models loaded from '{}'. Check your config.", str(cfg_path.resolve()))

    def get(self, name: Optional[str]) -> LLMModel:
        if name:
            if name not in self.models:
                logger.error("Requested model '{}' not found. Available: {}", name, ", ".join(self.models.keys()))
                raise ValueError(f"Model '{name}' not found. Available: {', '.join(self.models.keys())}")
            logger.debug("ModelRegistry.get('{}') → OK", name)
            return self.models[name]
        if not self.default_name:
            logger.error("No default model configured and none specified.")
            raise ValueError("No default model configured and none specified.")
        logger.debug("ModelRegistry.get(None) → default '{}'", self.default_name)
        return self.models[self.default_name]

    def list(self) -> List[LLMModel]:
        names = [m.name for m in self.models.values()]
        logger.debug("ModelRegistry.list() → {} model(s): {}", len(names), names)
        return list(self.models.values())
