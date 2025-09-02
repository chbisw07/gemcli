import re
from typing import Optional
from loguru import logger


def _detect_intent(user_prompt: str) -> str:
    p = (user_prompt or "").lower()
    intent = "general"
    matched = None

    if any(k in p for k in ["refactor", "restructure", "clean up", "simplify"]):
        intent, matched = "refactor", "refactor|restructure|clean up|simplify"
    elif any(k in p for k in ["optimize", "performance", "speed up", "latency", "throughput"]):
        intent, matched = "optimize", "optimize|performance|speed up|latency|throughput"
    elif any(k in p for k in ["compare", "vs", "versus", "advantages", "disadvantages", "tradeoff"]):
        intent, matched = "compare", "compare|vs|versus|advantages|disadvantages|tradeoff"
    elif any(k in p for k in ["root cause", "bug", "issue", "failure", "why failing", "diagnose"]):
        intent, matched = "rca", "root cause|bug|issue|failure|why failing|diagnose"
    elif any(k in p for k in ["design", "architecture", "propose", "plan", "roadmap"]):
        intent, matched = "design", "design|architecture|propose|plan|roadmap"

    logger.debug(
        "intents._detect_intent → intent='{}' matched='{}' prompt='{}...'",
        intent, matched, (p[:160] + ("…" if len(p) > 160 else "")),
    )
    return intent


_SYNTH_TEMPLATES = {
    "rca": """You are a precise code analyst. Produce a concise Root Cause Analysis grounded in the given execution results.
MANDATORY: cite file and line ranges with short code excerpts (3–8 lines).
Sections:
1) Root cause(s)
2) Supporting evidence (file:line + excerpt)
3) Risks/unknowns
4) Recommended fix steps (do NOT write or apply code)
RESULTS JSON:
{results}
""",
    "refactor": """You are a senior refactoring guide. Propose a concrete refactoring plan for the codebase based on the execution results.
MANDATORY: cite file and line ranges with short code excerpts (3–8 lines).
Focus on:
- Specific pain points (long functions, mixed responsibilities, tight coupling)
- A stepwise refactor plan (what functions to split/rename/move; where)
- Safety strategy (tests, incremental rollout)
- Risks/unknowns
RESULTS JSON:
{results}
""",
    "optimize": """You are a performance engineer. Suggest actionable optimizations grounded in the execution results.
MANDATORY: cite file and line ranges with short code excerpts (3–8 lines).
Cover:
- Likely bottlenecks (I/O, batching, re-embedding, duplicate work)
- Concrete improvements (batch sizes, caching, skip-logic, data structures)
- Trade-offs and risks
RESULTS JSON:
{results}
""",
    "compare": """You are a technical reviewer. Compare the two approaches referenced by the user prompt, grounded in the execution results.
MANDATORY: cite file and line ranges with short code excerpts (3–8 lines).
Structure:
- Comparison criteria (correctness, complexity, perf, maintainability)
- Side-by-side analysis with evidence
- Recommendation and conditions
RESULTS JSON:
{results}
""",
    "design": """You are a system designer. Propose an architecture or plan grounded in the execution results.
MANDATORY: cite file and line ranges with short code excerpts (3–8 lines).
Include:
- Current-state summary
- Proposed design (components, responsibilities)
- Migration plan
- Risks/unknowns
RESULTS JSON:
{results}
""",
    "general": """You are a precise code analyst. Produce a concise, actionable answer grounded in the execution results.
MANDATORY: cite file and line ranges with short code excerpts (3–8 lines).
Include:
- Direct answer to the user’s question
- Supporting evidence
- Risks/unknowns (if any)
- Next actions
RESULTS JSON:
{results}
""",
}


def _build_synth_prompt(user_prompt: str, results_json: str, extra_context: Optional[str] = None) -> str:
    intent = _detect_intent(user_prompt)
    tmpl = _SYNTH_TEMPLATES.get(intent, _SYNTH_TEMPLATES["general"])
    header = f"USER PROMPT:\n{user_prompt.strip()}\n"
    if extra_context and extra_context.strip():
        header += f"\nADDITIONAL CONTEXT:\n{extra_context.strip()}\n"
    out = header + "\n" + tmpl.format(results=results_json)
    logger.debug(
        "intents._build_synth_prompt → intent='{}' results_len={} extra_context={}",
        intent, len(results_json or ""), bool(extra_context and extra_context.strip())
    )
    return out
