# tools/edu_parsers.py
from __future__ import annotations

import math
import re
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger


# ----------------------------- type maps -----------------------------

# Normalized question types
TYPE_ALIASES = {
    "mcq": "MCQ",
    "objective": "MCQ",
    "one-mark": "MCQ",
    "vsa": "VSA",
    "very short": "VSA",
    "short": "SA",
    "sa": "SA",
    "medium": "SA",
    "la": "LA",
    "long": "LA",
    "case": "CASE_STUDY",
    "case study": "CASE_STUDY",
    "case-study": "CASE_STUDY",
}

# Default marks mapping (used if blueprint doesn’t override)
DEFAULT_MARKS_FOR_TYPE = {
    "MCQ": 1,
    "VSA": 2,
    "SA": 3,
    "LA": 5,
    "CASE_STUDY": 4,
}


def _normalize_type(token: str) -> Optional[str]:
    t = token.strip().lower()
    return TYPE_ALIASES.get(t) or t.upper() if t in {"MCQ", "VSA", "SA", "LA", "CASE_STUDY"} else None


def _marks_for_type(typ: str, blueprint: Dict[str, Any]) -> int:
    # Prefer blueprint section marks if available
    for s in blueprint.get("sections", []):
        if (s.get("type") or "").upper() == typ:
            return int(s.get("marks_each", DEFAULT_MARKS_FOR_TYPE.get(typ, 1)))
    return DEFAULT_MARKS_FOR_TYPE.get(typ, 1)


# ----------------------------- parse API -----------------------------

def parse_weightage_request(
    request_text: str,
    blueprint: Dict[str, Any],
    *,
    total_questions: Optional[int] = None,
    total_marks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Turn a free-text weightage spec into a concrete distribution plan.
    Supports:
      - "generate 10 questions with equal weightage to 1, 2 and 5 marks"
      - "50 questions with .25, .25, .50 weightage with short, medium and long answers"
      - "marks distribution: 1:0.2, 2:0.3, 5:0.5 total 40 questions"

    Returns:
      {
        "by_marks": {1: n1, 2: n2, ...},
        "by_type": {"MCQ": n, "SA": n, ...},
        "total_questions": int,
        "total_marks": int | None
      }
    """
    text = request_text.strip().lower()

    # Infer total_questions if stated
    q_num = total_questions
    m_total = total_marks

    # e.g., "generate 10 questions ..." or "total 40 questions"
    m_q1 = re.search(r"\b(\d{1,3})\s+questions?\b", text)
    if m_q1 and q_num is None:
        q_num = int(m_q1.group(1))

    # Parse explicit marks list: "equal weightage to 1, 2 and 5 marks"
    m_marks_list = re.search(r"equal\s+weightage\s+to\s+([0-9 ,and]+)\s+marks?", text)
    if m_marks_list:
        marks = re.findall(r"\d+", m_marks_list.group(1))
        marks = [int(x) for x in marks]
        if not q_num:
            q_num = sum(1 for _ in marks) * 3  # fallback: ~3 per bucket
        per = _split_even(q_num, len(marks))
        by_marks = {marks[i]: per[i] for i in range(len(marks))}
        return _complete_distribution(by_marks=by_marks, blueprint=blueprint, total_questions=q_num, total_marks=m_total)

    # Parse ratio over types: ".25, .25, .50 with short, medium and long"
    ratio_nums = [float(x) for x in re.findall(r"(?<!\d)(?:0?\.\d+|1(?:\.0)?)", text)]
    type_words = re.findall(r"(mcq|objective|very short|vsa|short|sa|medium|long|la|case(?:[- ]study)?)", text)
    types = [_normalize_type(w) for w in type_words]
    types = [t for t in types if t]

    if ratio_nums and types and len(ratio_nums) == len(types):
        if q_num is None:
            q_num = 40  # sensible default
        shares = _normalize_ratios(ratio_nums)
        counts = [int(round(s * q_num)) for s in shares]
        # adjust rounding to match exactly q_num
        _fix_sum(counts, q_num)
        by_type = {}
        for t, c in zip(types, counts):
            by_type[t] = by_type.get(t, 0) + c
        return _complete_distribution(by_type=by_type, blueprint=blueprint, total_questions=q_num, total_marks=m_total)

    # Parse "marks distribution: 1:0.2, 2:0.3, 5:0.5"
    kv = re.findall(r"(\d)\s*[:=]\s*(0?\.\d+|1(?:\.0)?)", text)
    if kv:
        marks = [int(k) for k, _ in kv]
        ratios = [float(v) for _, v in kv]
        shares = _normalize_ratios(ratios)
        if q_num is None:
            q_num = 40
        counts = [int(round(s * q_num)) for s in shares]
        _fix_sum(counts, q_num)
        by_marks = {marks[i]: counts[i] for i in range(len(marks))}
        return _complete_distribution(by_marks=by_marks, blueprint=blueprint, total_questions=q_num, total_marks=m_total)

    # Fallback: if we only know total questions, mirror blueprint proportions by section counts
    if q_num:
        by_type = _mirror_blueprint_types(q_num, blueprint)
        return _complete_distribution(by_type=by_type, blueprint=blueprint, total_questions=q_num, total_marks=m_total)

    # Last resort
    q_num = 40
    by_type = {"MCQ": 10, "VSA": 10, "SA": 10, "LA": 10}
    return _complete_distribution(by_type=by_type, blueprint=blueprint, total_questions=q_num, total_marks=m_total)


# ----------------------------- internals -----------------------------

def _split_even(total: int, buckets: int) -> List[int]:
    base = total // buckets
    rem = total - base * buckets
    out = [base] * buckets
    for i in range(rem):
        out[i] += 1
    return out


def _normalize_ratios(nums: List[float]) -> List[float]:
    s = sum(nums)
    if s <= 0:
        return [1.0 / len(nums)] * len(nums)
    return [x / s for x in nums]


def _fix_sum(counts: List[int], target: int) -> None:
    delta = target - sum(counts)
    i = 0
    while delta != 0 and counts:
        counts[i % len(counts)] += 1 if delta > 0 else -1
        delta = target - sum(counts)
        i += 1


def _mirror_blueprint_types(total_q: int, blueprint: Dict[str, Any]) -> Dict[str, int]:
    secs = blueprint.get("sections") or []
    if not secs:
        return {"MCQ": total_q}  # degenerate
    # weight by section counts
    weights = {}
    for s in secs:
        typ = (s.get("type") or "").upper()
        weights[typ] = weights.get(typ, 0) + int(s.get("count", 0))
    names = list(weights.keys())
    nums = [weights[n] for n in names]
    shares = _normalize_ratios(nums)
    counts = [int(round(sh * total_q)) for sh in shares]
    _fix_sum(counts, total_q)
    return {names[i]: counts[i] for i in range(len(names))}


def _complete_distribution(
    *,
    by_marks: Optional[Dict[int, int]] = None,
    by_type: Optional[Dict[str, int]] = None,
    blueprint: Dict[str, Any],
    total_questions: int,
    total_marks: Optional[int],
) -> Dict[str, Any]:
    by_marks = dict(by_marks or {})
    by_type = dict(by_type or {})

    # Fill the other view using blueprint’s marks mapping
    if not by_marks and by_type:
        for t, c in by_type.items():
            m = _marks_for_type(t, blueprint)
            by_marks[m] = by_marks.get(m, 0) + c

    if not by_type and by_marks:
        # invert with best-effort type mapping
        for m, c in by_marks.items():
            # pick a type that has marks_each == m
            chosen = None
            for s in blueprint.get("sections", []):
                if int(s.get("marks_each", -1)) == int(m):
                    chosen = (s.get("type") or "").upper()
                    break
            if not chosen:
                # fallback to default map
                chosen = next((k for k, v in DEFAULT_MARKS_FOR_TYPE.items() if v == int(m)), "SA")
            by_type[chosen] = by_type.get(chosen, 0) + c

    # Compute total marks if requested
    calc_marks = None
    try:
        calc_marks = sum(int(m) * int(n) for m, n in by_marks.items())
    except Exception:
        calc_marks = None

    return {
        "by_marks": {int(k): int(v) for k, v in sorted(by_marks.items())},
        "by_type": {str(k): int(v) for k, v in by_type.items()},
        "total_questions": int(total_questions),
        "total_marks": int(calc_marks) if calc_marks is not None else (int(total_marks) if total_marks else None),
    }
