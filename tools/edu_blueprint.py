# tools/edu_blueprint.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from indexing.settings import load as load_settings
from indexing.retriever import retrieve as rag_retrieve
from config_home import project_rag_dir


# ----------------------------- data model -----------------------------

@dataclass
class SectionRule:
    id: str                # "A" | "B" | ...
    type: str              # "MCQ" | "VSA" | "SA" | "LA" | "CASE_STUDY" | etc.
    count: int
    marks_each: int
    internal_choice: int = 0

@dataclass
class Blueprint:
    board: str             # "CBSE"
    klass: int             # 10
    subject: str           # "Mathematics"
    session: str = ""      # e.g., "2024-25" (optional)
    max_marks: Optional[int] = None
    total_questions: Optional[int] = None
    sections: List[SectionRule] = None
    constraints: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "board": self.board,
            "class": self.klass,
            "subject": self.subject,
            "session": self.session,
            "max_marks": self.max_marks,
            "total_questions": self.total_questions,
            "sections": [asdict(s) for s in (self.sections or [])],
            "constraints": self.constraints or {},
        }


# ----------------------------- helpers -----------------------------

_SECTION_LINE_RX = re.compile(
    r"""(?ix)
    section \s+ (?P<sec>[A-Z]) .*?
    (?:consists|contains|has)\s+ (?P<count>\d{1,3}) \s+ (?:questions?|items?) \s+ (?:of|worth) \s+
    (?P<marks>\d{1,2}) \s+ mark
    """,
)

_CASE_STUDY_RX = re.compile(
    r"""(?i)\b(case\s*study|case-study)\b""",
)

_TOK_RX = re.compile(r"\w+")


def _guess_type(section_id: str, line: str, marks_each: int) -> str:
    s = line.lower()
    if "mcq" in s or "objective" in s or "multiple choice" in s or marks_each == 1:
        return "MCQ"
    if _CASE_STUDY_RX.search(s):
        return "CASE_STUDY"
    # heuristic map by marks
    if marks_each == 2:
        return "VSA"       # very short answer
    if marks_each == 3:
        return "SA"        # short answer
    if marks_each >= 5:
        return "LA"        # long answer
    return "SECTION"


def _normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _extract_rules_from_pages(pages: List[str]) -> Tuple[List[SectionRule], Optional[int], Optional[int]]:
    """
    Parse "General Instructions" style language into SectionRule entries.
    Returns (sections, total_questions, max_marks).
    """
    sections: List[SectionRule] = []
    total_q: Optional[int] = None
    total_m: Optional[int] = None

    blob = _normalize_text("\n".join(pages))
    # Try obvious "Maximum Marks: 80" / "Max Marks – 80"
    mm = re.search(r"(?i)\b(max(?:imum)?\s*marks?)\s*[:\-–]\s*(\d{1,3})", blob)
    if mm:
        try:
            total_m = int(mm.group(2))
        except Exception:
            pass

    # Try "This question paper contains 38 questions"
    tq = re.search(r"(?i)\bcontains?\s+(\d{1,3})\s+questions?\b", blob)
    if tq:
        try:
            total_q = int(tq.group(1))
        except Exception:
            pass

    # Section lines
    for line in blob.split("\n"):
        m = _SECTION_LINE_RX.search(line)
        if not m:
            continue
        sec = (m.group("sec") or "").strip()
        count = int(m.group("count"))
        marks_each = int(m.group("marks"))

        typ = _guess_type(sec, line, marks_each)

        # optional internal choice detection
        ic = 0
        ic_m = re.search(r"(?i)internal\s+choice\s+in\s+(\d+)\s+questions?", line)
        if ic_m:
            try:
                ic = int(ic_m.group(1))
            except Exception:
                ic = 0

        sections.append(SectionRule(id=sec, type=typ, count=count, marks_each=marks_each, internal_choice=ic))

    # Deduplicate sections by id (keep max count if duplicates found across pages)
    by_id: Dict[str, SectionRule] = {}
    for s in sections:
        if (prev := by_id.get(s.id)) is None or s.count > prev.count:
            by_id[s.id] = s

    return list(by_id.values()), total_q, total_m


def _score_heading(ch: dict) -> float:
    md = ch.get("metadata") or {}
    typ = (md.get("chunk_type") or "").lower()
    sc = ch.get("score") or 0.0
    # prefer headings and "general instructions" sections
    if "heading" in typ:
        sc *= 1.15
    txt = (ch.get("document") or "").lower()
    if "general instruction" in txt or "section a" in txt:
        sc *= 1.10
    return sc


# ----------------------------- public API -----------------------------

def build_blueprint(
    project_root: str,
    subject: str,
    klass: int,
    board: str = "CBSE",
    rag_path: Optional[str] = None,
    filename_hint: Optional[str] = None,
    session: str = "",
) -> Blueprint:
    """
    Build a blueprint by retrieving 'General Instructions' and section descriptions
    from indexed SQPs/syllabi for the given {board, class, subject}.

    If you know the file name (e.g., 'MathsStandard-SQP.pdf') pass filename_hint
    to bias retrieval via retriever's filename boost.
    """
    cfg = load_settings(None)
    rag_dir = rag_path or cfg.get("chroma_dir") or project_rag_dir(project_root)

    # Seed query — filename hint (if provided) improves pull
    q = f"{subject} class {klass} sample question paper general instructions section A B C D E marks"
    if filename_hint:
        q = f"{filename_hint} {q}"

    res = rag_retrieve(project_root=project_root, rag_path=rag_dir, query=q, k=60, where=None, min_score=(cfg.get("router") or {}).get("threshold"))

    chunks = res.get("chunks") or []
    # Take top ~20 highest signals for parsing
    top = sorted(chunks, key=_score_heading, reverse=True)[:20]
    pages = [c.get("document") or "" for c in top if (c.get("document") or "").strip()]

    sections, total_q, total_m = _extract_rules_from_pages(pages)

    # Fallback if we couldn't parse — create a minimal skeleton
    if not sections:
        logger.warning("edu_blueprint: no sections parsed; falling back to generic pattern")
        sections = [
            SectionRule(id="A", type="MCQ", count=20, marks_each=1),
            SectionRule(id="B", type="VSA", count=5, marks_each=2),
            SectionRule(id="C", type="SA", count=6, marks_each=3),
            SectionRule(id="D", type="LA", count=4, marks_each=5),
            SectionRule(id="E", type="CASE_STUDY", count=3, marks_each=4),
        ]
        total_q = sum(s.count for s in sections)
        total_m = 80

    bp = Blueprint(
        board=board,
        klass=int(klass),
        subject=str(subject),
        session=session or "",
        max_marks=total_m,
        total_questions=total_q,
        sections=sections,
        constraints={
            "diagrams_allowed": True,
            "calculator_allowed": False,
        },
    )
    return bp


def blueprint_path(project_root: str, board: str, klass: int, subject: str) -> Path:
    root = Path(project_root)
    out_dir = root / "data" / "edu" / "blueprints"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{board}-{klass}-{subject}".replace(" ", "_")
    return out_dir / f"{safe}.json"


def save_blueprint(project_root: str, bp: Blueprint) -> Path:
    p = blueprint_path(project_root, bp.board, bp.klass, bp.subject)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bp.to_dict(), indent=2), encoding="utf-8")
    logger.info("edu_blueprint: saved blueprint -> {}", p)
    return p


def load_blueprint(project_root: str, board: str, klass: int, subject: str) -> Optional[Blueprint]:
    p = blueprint_path(project_root, board, int(klass), subject)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        sections = [SectionRule(**s) for s in data.get("sections", [])]
        return Blueprint(
            board=data.get("board", board),
            klass=int(data.get("class", klass)),
            subject=data.get("subject", subject),
            session=data.get("session", ""),
            max_marks=data.get("max_marks"),
            total_questions=data.get("total_questions"),
            sections=sections,
            constraints=data.get("constraints") or {},
        )
    except Exception as e:
        logger.error("edu_blueprint: failed to load blueprint {}: {}", p, e)
        return None
