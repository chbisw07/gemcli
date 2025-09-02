# logging_decorators.py
from __future__ import annotations
import time, functools, inspect, math
from typing import Any, Callable, Dict, Iterable
from loguru import logger

_REDACT_DEFAULT = {"api_key", "authorization", "password", "token", "secret"}

def _redact(obj: Any, redact_keys: set[str], max_len: int, max_items: int) -> Any:
    """Lightweight redaction + truncation for logs."""
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if str(k).lower() in redact_keys:
                    out[k] = "******"
                else:
                    out[k] = _redact(v, redact_keys, max_len, max_items)
            return out
        if isinstance(obj, (list, tuple, set)):
            seq = list(obj)
            cut = min(len(seq), max_items)
            trimmed = [ _redact(x, redact_keys, max_len, max_items) for x in seq[:cut] ]
            if len(seq) > cut: trimmed.append(f"... (+{len(seq)-cut} more)")
            return trimmed if not isinstance(obj, tuple) else tuple(trimmed)
        if isinstance(obj, str):
            return (obj if len(obj) <= max_len else (obj[:max_len] + f"...(+{len(obj)-max_len} chars)"))
        return obj
    except Exception:
        return "<unrepr>"

def _default_summary(ret: Any) -> Dict[str, Any]:
    """Heuristics: summarize common tool outputs."""
    try:
        if isinstance(ret, list):
            return {"items": len(ret)}
        if isinstance(ret, dict):
            # common fields
            keys = ret.keys()
            if "replacements" in ret: return {"replacements": ret["replacements"]}
            if "changed" in ret and isinstance(ret["changed"], dict): return {"changed_files": len(ret["changed"])}
            if "chunks" in ret and isinstance(ret["chunks"], list): return {"chunks": len(ret["chunks"])}
            if "files" in ret and isinstance(ret["files"], dict): return {"files": len(ret["files"]), "chunks": sum(ret["files"].values())}
            if "added" in ret and "changed_files" in ret: return {"added": ret["added"], "changed_files": len(ret["changed_files"])}
            return {"keys": list(keys)[:6]}
        return {"type": type(ret).__name__}
    except Exception:
        return {"summary": "<error>"}

def log_call(
    name: str | None = None,
    *,
    level: str = "INFO",
    slow_ms: int = 800,                # warn if slower than this
    sample: float = 1.0,               # 0<sample<=1: probabilistic sampling
    redact: Iterable[str] = _REDACT_DEFAULT,
    arg_max_len: int = 200,
    arg_max_items: int = 20,
    summarize: Callable[[Any], Dict[str, Any]] | None = None,
    include_return: bool = False,      # if True, log summarized return payload
):
    """
    Decorator to log entry/exit, args, duration, and failures.
    Use logger.opt(depth=1) to keep file:line pointing at the wrapped function.
    """
    redact_keys = {str(k).lower() for k in redact}
    summary_fn = summarize or _default_summary

    def decorator(fn: Callable):
        if getattr(fn, "__logged__", False):
            return fn  # already wrapped

        qual = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # sampling
            if sample < 1.0:
                import random
                if random.random() > sample:
                    return fn(*args, **kwargs)

            # build arg snapshot (skip self/cls)
            try:
                sig = inspect.signature(fn)
                ba = sig.bind_partial(*args, **kwargs)
                ba.apply_defaults()
                call_args = {k: v for k, v in ba.arguments.items() if k not in {"self", "cls"}}
                call_args = _redact(call_args, redact_keys, arg_max_len, arg_max_items)
            except Exception:
                call_args = "<uninspectable>"

            lg = logger.opt(depth=1)  # keep correct file:line
            lg.log(level, "→ {} args={}", qual, call_args)

            t0 = time.perf_counter()
            try:
                ret = fn(*args, **kwargs)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                summary = summary_fn(ret)
                if dur_ms >= slow_ms:
                    lg.warning("✓ {} done in {:.0f}ms (SLOW) summary={}", qual, dur_ms, summary)
                else:
                    lg.log(level, "✓ {} done in {:.0f}ms summary={}", qual, dur_ms, summary)
                if include_return:
                    lg.debug("↩ {} return={}", qual, _redact(ret, redact_keys, arg_max_len, arg_max_items))
                return ret
            except Exception as e:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                lg.exception("✗ {} failed in {:.0f}ms: {}", qual, dur_ms, e)
                raise

        wrapper.__logged__ = True
        return wrapper
    return decorator
