# tools/edu_generator.py
from __future__ import annotations

import json
import random
from typing import Dict, Any, List, Optional
from loguru import logger

from indexing.settings import load as load_settings
from indexing.retriever import retrieve as rag_retrieve
from config_home import project_rag_dir


# ----------------------------- exemplars -----------------------------

def collect_exemplars(
    project_root: str,
    subject: str,
    klass: int,
    *,
    rag_path: Optional[str] = None,
    per_type: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pull a few representative chunks per type from the corpus to steer style.
    """
    cfg = load_settings(None)
    rag_dir = rag_path or cfg.get("chroma_dir") or project_rag_dir(project_root)

    exemplars: Dict[str, List[Dict[str, Any]]] = {}

    # Simple set of steering queries by type
    queries = {
        "MCQ": f"{subject} class {klass} multiple choice questions",
        "VSA": f"{subject} class {klass} very short answer questions",
        "SA":  f"{subject} class {klass} short answer questions",
        "LA":  f"{subject} class {klass} long answer questions",
        "CASE_STUDY": f"{subject} class {klass} case study questions",
    }

    for typ, q in queries.items():
        try:
            res = rag_retrieve(project_root=project_root, rag_path=rag_dir, query=q, k=12)
        except Exception as e:
            logger.warning("collect_exemplars: retrieve failed for {}: {}", typ, e)
            continue
        chunks = res.get("chunks") or []
        # prefer 'questions' chunk type if present
        chunks.sort(key=lambda c: (("questions" in (c.get("metadata", {}).get("chunk_type") or "").lower()), c.get("score") or 0.0), reverse=True)
        exemplars[typ] = chunks[:per_type]
    return exemplars


# ----------------------------- generation pack -----------------------------

def build_generation_pack(
    *,
    blueprint: Dict[str, Any],
    distribution: Dict[str, Any],
    subject: str,
    klass: int,
    board: str = "CBSE",
    include_solutions: bool = True,
    seed: Optional[int] = None,
    exemplars: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Compose a model-ready prompt pack for generating a paper (and optional answer key).
    We do NOT call the model here; the registry/tool or agent can decide how to use the pack.
    """
    if seed is not None:
        random.seed(int(seed))

    # Human-readable constraints summary
    secs = blueprint.get("sections") or []
    sec_lines = [
        f"Section {s.get('id')}: {s.get('count')} × {s.get('marks_each')} marks  [{s.get('type')}]"
        + (f", internal choice in {s.get('internal_choice')}" if s.get("internal_choice") else "")
        for s in secs
    ]

    dist_lines = []
    by_marks = distribution.get("by_marks") or {}
    by_type = distribution.get("by_type") or {}
    if by_marks:
        dist_lines.append("By marks: " + ", ".join(f"{m}→{n}" for m, n in by_marks.items()))
    if by_type:
        dist_lines.append("By type: " + ", ".join(f"{t}→{n}" for t, n in by_type.items()))

    # Build context block from exemplars (short!)
    context_blocks = []
    if exemplars:
        for typ, chunks in exemplars.items():
            if not chunks:
                continue
            lines = []
            for c in chunks[:3]:
                md = c.get("metadata") or {}
                tag = md.get("file_name") or md.get("file_path") or "?"
                page = md.get("page_number")
                header = f"[{tag}{' p.' + str(page) if page else ''} · {typ}]"
                body = (c.get("document") or "").strip()
                body = body[:400]  # keep tiny
                lines.append(f"{header}\n{body}")
            context_blocks.append("\n\n".join(lines))
    context = "\n\n---\n\n".join(context_blocks)

    sys = (
        "You are an exam paper author. Generate NEW, original questions that match the requested blueprint and distribution.\n"
        "Follow CBSE Class formatting. Keep language clear and syllabus-appropriate. Avoid copying from exemplars.\n"
        "When solutions are requested, show clear steps and final answers. For MCQs, provide the correct option and a one-line justification."
    )

    user = (
        f"BOARD: {board}\n"
        f"CLASS: {klass}\n"
        f"SUBJECT: {subject}\n\n"
        f"BLUEPRINT SECTIONS:\n" + "\n".join(f"- {line}" for line in sec_lines) + "\n\n"
        f"REQUESTED DISTRIBUTION:\n" + "\n".join(f"- {line}" for line in dist_lines) + "\n\n"
        f"OUTPUT FORMAT:\n"
        f"1) A neatly numbered QUESTION PAPER body.\n"
        f"2) A separate ANSWER KEY section {'(REQUIRED)' if include_solutions else '(OPTIONAL: SKIP IF NOT REQUESTED)'}.\n"
        f"3) Maintain internal choice style if counts imply it; clearly label subparts for case studies.\n"
        f"4) Ensure total questions and marks match the distribution.\n"
    )

    if context:
        user += "\nCONTEXT EXAMPLES (for style only, do NOT copy):\n" + context

    # Hints to encourage novelty
    user += (
        "\n\nNOVELTY CONSTRAINTS:\n"
        "- Vary numbers/contexts from any examples.\n"
        "- Avoid verbatim reuse; paraphrase concepts.\n"
        "- Keep within Class syllabus; avoid out-of-scope topics.\n"
    )

    pack = {
        "system": sys,
        "user": user,
        "blueprint": blueprint,
        "distribution": distribution,
        "include_solutions": include_solutions,
        "seed": seed,
    }
    return pack
