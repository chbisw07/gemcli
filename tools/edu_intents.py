# tools/edu_intents.py
from __future__ import annotations

import re
from typing import Dict, Any
from loguru import logger
from logging_decorators import log_call


# ---- simple, high-signal router (rules-first; LLM fallback can be added later) ----

@log_call("edu_detect_intent")
def detect_edu_intent(prompt: str) -> Dict[str, Any]:
    """
    Detect education-specific intent and slots from a user prompt.

    Returns:
        {
          "intent": "similar_questions|question_paper|explain|practice_drill|structured_answer|extract_tables|unknown",
          "slots":  {...},       # normalized slots
          "confidence": "high|medium|low"
        }
    """
    p = (prompt or "").strip()
    low = p.lower()

    # defaults
    intent = "unknown"
    slots: Dict[str, Any] = {"count": None, "chapter": "", "topic": "", "difficulty": "mixed", "scope": "chapter"}
    conf = "medium"

    # counts
    m = re.search(r"\b(\d{1,3})\b", low)
    if m:
        try:
            slots["count"] = max(1, min(200, int(m.group(1))))
            if slots["count"] and re.search(r"(questions?|mcqs?)\s*(on|about|from)?\b", low):
                # treat the number as 'count'; do not infer chapter from lone trailing digits
                pass  # nothing else to do; your chapter logic already requires 'chapter|ch'            
        except Exception:
            pass

    # scope hints
    if "across book" in low or "whole book" in low or "all chapters" in low:
        slots["scope"] = "book"
    if "project" in low or "everything" in low:
        slots["scope"] = "project"

    # difficulty
    if any(k in low for k in ["easy", "beginner", "basic"]):      slots["difficulty"] = "easy"
    if any(k in low for k in ["medium", "moderate"]):             slots["difficulty"] = "medium"
    if any(k in low for k in ["hard", "challenging", "advanced"]):slots["difficulty"] = "hard"

    # chapter/topic
    m = re.search(r"(chapter|ch\.?)\s*([0-9]{1,3})", low)
    if m:
        slots["chapter"] = m.group(2)
    # grab a rough "on <topic>" / "about <topic>" phrase
    m = re.search(r"\b(?:on|about|regarding|topic|in)\s+([a-z0-9 \-+/^()]+)$", p, flags=re.I)
    if m:
        slots["topic"] = m.group(1).strip()

    # intent rules
    if any(k in low for k in ["similar questions", "more questions like", "practice questions", "generate questions"]):
        intent = "similar_questions"; conf = "high"
    elif any(k in low for k in ["question paper", "test paper", "exam paper", "mock test"]):
        intent = "question_paper"; conf = "high"
    elif any(k in low for k in ["explain", "explanation", "teach me", "how does", "why does"]):
        intent = "explain"; conf = "high"
    elif any(k in low for k in ["practice drill", "drill", "spaced repetition"]):
        intent = "practice_drill"; conf = "medium"
    elif any(k in low for k in ["answer this", "solve this", "what is the answer", "solution to"]):
        intent = "structured_answer"; conf = "medium"
    elif any(k in low for k in ["table", "tables", "extract table", "tabular"]):
        intent = "extract_tables"; conf = "high"

    result = {"intent": intent, "slots": slots, "confidence": conf}
    logger.info("edu_intents.detect → {}", result)
    return result
