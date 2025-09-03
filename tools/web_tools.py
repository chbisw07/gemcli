# tools/web_tools.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from loguru import logger

_USER_AGENT = "gemcli/1.0 (+https://example.local)"

def _ddg_lib_search(q: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        from duckduckgo_search import DDGS  # type: ignore
        with DDGS() as ddgs:
            out = []
            for r in ddgs.text(q, max_results=max_results):
                if not r: continue
                out.append({"title": r.get("title") or "", "url": r.get("href") or r.get("url") or "", "snippet": r.get("body") or ""})
            return out
    except Exception as e:
        logger.debug("web_search: DDG lib missing/failed: {}", e)
        return []

def _fallback_search(q: str, max_results: int = 5) -> List[Dict[str, str]]:
    # Very naive HTML fallback (best-effort). Encourage installing duckduckgo_search.
    import requests, re
    try:
        r = requests.get("https://duckduckgo.com/html", params={"q": q}, headers={"User-Agent": _USER_AGENT}, timeout=10)
        items = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                           r.text, flags=re.S)
        out = []
        for href, title_html, snippet_html in items[:max_results]:
            # Quick strip tags
            title = re.sub("<[^>]+>", "", title_html)
            snippet = re.sub("<[^>]+>", "", snippet_html)
            out.append({"title": title, "url": href, "snippet": snippet})
        return out
    except Exception as e:
        logger.warning("web_search: fallback failed: {}", e)
        return []

def web_search(query: str, max_results: int = 5, site: str = "", recency_days: int = 0) -> Dict[str, Any]:
    q = query.strip()
    if site:
        q = f"site:{site} {q}"
    results = _ddg_lib_search(q, max_results=max_results) or _fallback_search(q, max_results=max_results)
    logger.info("web_search: q='{}' results={}", q, len(results))
    return {"query": q, "results": results, "count": len(results)}

def web_fetch(url: str, max_chars: int = 60000) -> Dict[str, Any]:
    import requests
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
    try:
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]): tag.decompose()
        title = (soup.title.string or "").strip() if soup.title else ""
        text = " ".join(soup.get_text("\n").split())
        if len(text) > max_chars:
            text = text[:max_chars]
        logger.info("web_fetch: ok url='{}' len={}", url, len(text))
        return {"url": url, "title": title, "text": text}
    except Exception as e:
        logger.warning("web_fetch: fail url='{}' err={}", url, e)
        return {"url": url, "error": str(e), "text": ""}
