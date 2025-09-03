# tools/edu_tools.py
from __future__ import annotations

import math
import json
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from logging_decorators import log_call

# Retriever import (uses your aligned collection naming)
try:
    from indexing.retriever import retrieve as rag_retrieve
except Exception:
    rag_retrieve = None  # will error at call-time with a helpful message

from indexing.pdf_utils import (
    shape_pdf_results, 
    filter_by_scope, 
    select_book_chapter,
    expand_pdf_neighbors_meta)

# --------------------------- shaping & helpers ---------------------------

def _id(c: dict) -> Optional[str]:
    return c.get("id") or c.get("chunk_id")

def _md(c: dict) -> dict:
    return c.get("metadata", {}) or {}

def _truncate_text(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else (s[:max_chars] + f"...(+{len(s)-max_chars} chars)")

# (removed: duplicate local helpers; we use the shared ones)

# --------------------------- style profiling for generation ---------------------------

def _style_profile(seed_chunks: List[dict]) -> Dict[str, Any]:
    import re
    texts = []
    for c in seed_chunks:
        doc = c.get("document") or ""
        texts.append(doc)
    joined = " \n".join(texts)
    verbs = re.findall(r"\b(explain|derive|prove|calculate|show|define|list|choose|find|evaluate|simplify)\b", joined, flags=re.I)
    has_math = any(x in joined for x in ["\\frac", "\\int", "\\sum", "$", "²", "³"])
    return {
        "avg_len": int(sum(len(t.split()) for t in texts)/max(1, len(texts))),
        "verbs": sorted(set(v.lower() for v in verbs))[:10],
        "math_style": "latex_like" if has_math else "plain",
    }

def _few_shots_from_seeds(seed_chunks: List[dict], max_examples: int = 3) -> List[Dict[str, Any]]:
    shots = []
    for c in seed_chunks[:max_examples]:
        doc = c.get("document") or ""
        cid = _id(c)
        shots.append({"id": cid, "example": _truncate_text(doc, 700)})
    return shots


# --------------------------- retrieval entry ---------------------------

def _retrieve(project_root: str, rag_path: str, query: str, top_k: int = 40) -> Dict[str, Any]:
    if rag_retrieve is None:
        raise RuntimeError("Retrieve function not available. Ensure indexing/retriever.py is present and importable.")
    return rag_retrieve(project_root=project_root, rag_path=rag_path, query=query, k=top_k)


# --------------------------- prompt builders (exact messages for OpenAICompatAdapter) ---------------------------

def _messages_for_similar_questions(topic: str, profile: dict, shots: List[dict], context_snips: List[str], count: int, difficulty: str) -> List[Dict[str, str]]:
    system = (
        "You generate SCHOOL-LEVEL practice questions in the style of provided book excerpts.\n"
        "Follow the STYLE and RULES strictly. Output JSON only."
    )
    user = {
        "style": {
            "avg_len": profile.get("avg_len"),
            "verbs": profile.get("verbs"),
            "math_style": profile.get("math_style"),
        },
        "rules": [
            "Do not copy sentences verbatim from excerpts; rephrase and vary numbers/contexts.",
            "Each MCQ has exactly one correct answer.",
            "Provide ANSWER and detailed SOLUTION for every item.",
            "Ground each item by listing 1–3 excerpt IDs from CONTEXT that inspired the item."
        ],
        "target": {
            "kind": "similar_questions",
            "topic_hint": topic,
            "count": count,
            "difficulty": difficulty
        },
        "few_shot_examples": shots,
        "CONTEXT_EXCERPTS": context_snips[:12],
        "OUTPUT_JSON_SCHEMA": {
            "items": [
                {
                    "type": "mcq|short|long",
                    "stem": "...",
                    "options": ["A","B","C","D"],  # for MCQ
                    "answer": "...",
                    "solution": "...",
                    "citations": ["<chunk_id>", "..."]
                }
            ]
        }
    }
    return [{"role": "system", "content": json.dumps({"instruction": system})},
            {"role": "user", "content": json.dumps(user)}]

def _messages_for_question_paper(mix: dict, profile: dict, shots: List[dict], context_snips: List[str], count: int, difficulty: str, chapter: str) -> List[Dict[str, str]]:
    system = (
        "You are an exam setter creating a QUESTION PAPER consistent with the style of the provided school book.\n"
        "Produce exactly the requested mix and include an answer key and solutions. Output JSON only."
    )
    user = {
        "style": profile,
        "rules": [
            "Keep variety across topics and difficulty.",
            "Do not copy verbatim; rephrase and change values/context.",
            "Provide answer key and detailed solutions.",
            "Cite 1–3 relevant excerpt IDs (chunk_ids) for each item."
        ],
        "target": {
            "kind": "question_paper",
            "chapter": chapter,
            "count": count,
            "mix": mix,               # e.g., {"mcq": 0.5, "short": 0.3, "long": 0.2}
            "difficulty": difficulty
        },
        "few_shot_examples": shots,
        "CONTEXT_EXCERPTS": context_snips[:16],
        "OUTPUT_JSON_SCHEMA": {
            "paper": {
                "items": [
                    {
                        "type": "mcq|short|long",
                        "marks": "number",
                        "stem": "...",
                        "options": ["A","B","C","D"],  # if MCQ
                        "answer": "...",
                        "solution": "...",
                        "citations": ["<chunk_id>", "..."]
                    }
                ],
                "answer_key": [{"index": 1, "answer": "C"}]
            }
        }
    }
    return [{"role": "system", "content": json.dumps({"instruction": system})},
            {"role": "user", "content": json.dumps(user)}]

def _messages_for_explain(question_or_topic: str, context_snips: List[str]) -> List[Dict[str, str]]:
    system = (
        "You are a helpful teacher. Explain clearly, step by step, using the provided excerpts as grounding. "
        "Show equations where useful. Cite chunk IDs you relied on. Output JSON with 'answer' and 'citations'."
    )
    user = {
        "question_or_topic": question_or_topic,
        "CONTEXT_EXCERPTS": context_snips[:12],
        "OUTPUT_JSON_SCHEMA": {"answer": "...", "citations": ["<chunk_id>", "..."]}
    }
    return [{"role": "system", "content": json.dumps({"instruction": system})},
            {"role": "user", "content": json.dumps(user)}]


# --------------------------- public tools ---------------------------

@log_call("edu_similar_questions", slow_ms=1500)
def edu_similar_questions(
    project_root: str,
    rag_path: str,
    topic: str = "",
    chapter: str = "",
    count: int = 20,
    difficulty: str = "mixed",
    scope: str = "chapter",
    top_k: int = 60,
    web_fetch_batch: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Build messages for OpenAICompatAdapter to generate similar questions grounded in book excerpts.
    Returns: {"messages": [...], "context": {...}, "citations": [...], "suggested_model": "<model_name>"}
    """
    query = topic or (f"chapter {chapter}" if chapter else "questions")
    res = _retrieve(project_root, rag_path, query=query, top_k=top_k)
    chunks = res.get("chunks", [])
    shaped = shape_pdf_results(chunks)

    # scope filtering (derive book/chapter from hits if not given)
    book0, ch0 = select_book_chapter(shaped)
    book = book0
    ch = chapter or ch0
    # Avoid over-filtering when book/chapter metadata isn't present yet
    scope_eff = scope if (book or ch) else "project"
    for k in shaped:
        shaped[k] = filter_by_scope(shaped[k], scope_eff, book, ch)

    # seeds for style
    seeds = (shaped["questions"] or shaped["answers"] or shaped["topics"])[:40]
    profile = _style_profile(seeds)
    shots = _few_shots_from_seeds(seeds, max_examples=3)

    # short context snips with chunk_ids for grounding
    def snip(c):
        return {"chunk_id": _id(c), "text": _truncate_text((c.get("document") or ""), 800)}
    
    # Include neighbor pages/sections for richer context (pool-only; no extra DB I/O)
    seeds_plus = expand_pdf_neighbors_meta(
        seeds, chunks,
        pages_before=1, pages_after=1,
        include_page_containers=True,
        include_paragraphs=True,
        include_sibling_paragraphs=True,
        cap_extra=6,
    )
    context_snips = [snip(c) for c in (seeds_plus + shaped["topics"][:10])]
    # --- append optional web context (PDFs first, web after) ---
    web_snips: List[Dict[str, Any]] = []
    for page in (web_fetch_batch or []):
        txt = (page.get("text") or "")
        title = (page.get("title") or page.get("url") or "")[:120]
        if txt:
            web_snips.append({"chunk_id": f"web::{title}", "text": txt[:800]})
    context_snips = context_snips + web_snips

    messages = _messages_for_similar_questions(topic=topic or "", profile=profile, shots=shots, context_snips=context_snips, count=count, difficulty=difficulty)

    # Suggested model (adjust to your default chat model)
    return {
        "messages": messages,
        "context": {"shaped": list(shaped.keys())},
        "citations": [s["chunk_id"] for s in context_snips if s.get("chunk_id")],
        "suggested_model": "default_chat"
    }


@log_call("edu_question_paper", slow_ms=2500)
def edu_question_paper(
    project_root: str,
    rag_path: str,
    chapter: str = "",
    topics: Optional[List[str]] = None,
    count: int = 30,
    mix: Optional[Dict[str, float]] = None,
    difficulty: str = "mixed",
    scope: str = "chapter",
    top_k: int = 80,
    web_fetch_batch: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Build messages for OpenAICompatAdapter to create a question paper with solutions & answer key.
    """
    q = " ".join(topics or []) if topics else f"chapter {chapter}" if chapter else "question paper"
    res = _retrieve(project_root, rag_path, query=q, top_k=top_k)
    chunks = res.get("chunks", [])
    shaped = shape_pdf_results(chunks)

    book0, ch0 = select_book_chapter(shaped)
    book = book0
    ch = chapter or ch0
    # Avoid over-filtering when book/chapter metadata isn't present yet
    scope_eff = scope if (book or ch) else "project"
    for k in shaped:
        shaped[k] = filter_by_scope(shaped[k], scope_eff, book, ch)

    seeds = (shaped["questions"] or shaped["answers"] or shaped["topics"])[:60]
    profile = _style_profile(seeds)
    shots = _few_shots_from_seeds(seeds, max_examples=3)
    def snip(c): return {"chunk_id": _id(c), "text": _truncate_text((c.get("document") or ""), 800)}
    # Include neighbor context for a more coherent paper (prev/next pages + sibling windows)
    seeds_plus = expand_pdf_neighbors_meta(
        seeds, chunks,
        pages_before=1, pages_after=1,
        include_page_containers=True,
        include_paragraphs=True,
        include_sibling_paragraphs=True,
        cap_extra=8,
    )
    context_snips = [snip(c) for c in (seeds_plus + shaped["topics"][:16])]
    # --- append optional web context (PDFs first, web after) ---
    web_snips: List[Dict[str, Any]] = []
    for page in (web_fetch_batch or []):
        txt = (page.get("text") or "")
        title = (page.get("title") or page.get("url") or "")[:120]
        if txt:
            web_snips.append({"chunk_id": f"web::{title}", "text": txt[:800]})
    context_snips = context_snips + web_snips

    mix = mix or {"mcq": 0.5, "short": 0.3, "long": 0.2}
    messages = _messages_for_question_paper(mix=mix, profile=profile, shots=shots, context_snips=context_snips, count=count, difficulty=difficulty, chapter=ch or "")

    return {
        "messages": messages,
        "context": {"mix": mix, "chapter": ch},
        "citations": [s["chunk_id"] for s in context_snips if s.get("chunk_id")],
        "suggested_model": "default_chat"
    }


@log_call("edu_explain", slow_ms=1200)
def edu_explain(
    project_root: str,
    rag_path: str,
    question_or_topic: str,
    scope: str = "chapter",
    top_k: int = 40,
    web_fetch_batch: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Build messages for a grounded explanation.
    """
    res = _retrieve(project_root, rag_path, query=question_or_topic, top_k=top_k)
    chunks = res.get("chunks", [])
    shaped = shape_pdf_results(chunks)

    book0, ch0 = select_book_chapter(shaped)
    scope_eff = scope if (book0 or ch0) else "project"
    for k in shaped:
        shaped[k] = filter_by_scope(shaped[k], scope_eff, book0, ch0)

    seeds = (shaped["topics"] or shaped["pages"] or shaped["other"])[:20]
    def snip(c): return {"chunk_id": _id(c), "text": _truncate_text((c.get("document") or ""), 900)}
    # Neighbor expansion helps catch answers that spill onto prev/next page
    seeds_plus = expand_pdf_neighbors_meta(
        seeds, chunks,
        pages_before=1, pages_after=1,
        include_page_containers=True,
        include_paragraphs=True,
        include_sibling_paragraphs=True,
        cap_extra=4,
    )
    context_snips = [snip(c) for c in seeds_plus]
    # --- append optional web context (PDFs first, web after) ---
    web_snips: List[Dict[str, Any]] = []
    for page in (web_fetch_batch or []):
        txt = (page.get("text") or "")
        title = (page.get("title") or page.get("url") or "")[:120]
        if txt:
            web_snips.append({"chunk_id": f"web::{title}", "text": txt[:800]})
    context_snips = context_snips + web_snips
    
    messages = _messages_for_explain(question_or_topic=question_or_topic, context_snips=context_snips)
    return {
        "messages": messages,
        "context": {"book": book0, "chapter": ch0},
        "citations": [s["chunk_id"] for s in context_snips if s.get("chunk_id")],
        "suggested_model": "default_chat"
    }


@log_call("edu_extract_tables", slow_ms=800)
def edu_extract_tables(
    project_root: str,
    rag_path: str,
    chapter: str = "",
    topic: str = "",
    scope: str = "chapter",
    top_k: int = 40
) -> Dict[str, Any]:
    """
    Retrieve and return table chunks (no model call; structured).
    """
    query = topic or (f"chapter {chapter}" if chapter else "tables")
    res = _retrieve(project_root, rag_path, query=query, top_k=top_k)
    chunks = res.get("chunks", [])
    shaped = shape_pdf_results(chunks)
    book0, ch0 = select_book_chapter(shaped)
    tables = filter_by_scope(shaped.get("tables", []), scope, book0, ch0)
    out = [{"chunk_id": _id(t), "page": _md(t).get("page_number"), "text": _truncate_text(t.get("document") or "", 1200)} for t in tables]
    return {"tables": out, "count": len(out), "book": book0, "chapter": ch0}
