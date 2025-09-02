# tools/direct_parser.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from loguru import logger


def try_direct_actions(query: str, root: Path) -> Optional[dict]:
    """
    Attempt to directly route the query to a known tool with parsed arguments.
    Returns a dict like:
      {"tool": "replace_in_file", "arguments": {...}} or None if not matched.
    """
    q = (query or "").strip()
    logger.debug("direct_parser.try_direct_actions: q='{}...'", q[:120].replace("\n", " "))

    # Pattern: Replace 'X' with 'Y' in file.txt (dry run)
    match = re.search(r"replace ['\"](.+?)['\"] with ['\"](.+?)['\"] in (.+?)( \(dry run\))?$", q.lower())
    if match:
        find = match.group(1)
        repl = match.group(2)
        path = match.group(3).strip()
        dry_run = bool(match.group(4))
        res = {"tool": "replace_in_file", "arguments": {"path": path, "find": find, "replace": repl, "dry_run": dry_run}}
        logger.info("direct_parser: matched replace_in_file → {}", res)
        return res

    # Pattern: Create a temp helper file under path with content XYZ
    match = re.search(r"create (?:a )?temp .*? file under (.+?)(?: with content (.+))?$", q.lower())
    if match:
        path = match.group(1).strip()
        content = match.group(2).strip() if match.group(2) else "Hello, world!"
        res = {"tool": "write_file", "arguments": {"path": path, "content": content, "overwrite": True}}
        logger.info("direct_parser: matched write_file → {}", res)
        return res

    # Pattern: Format python files under some directory
    match = re.search(r"format python files(?: under (.+?))?(?: \(dry run\))?$", q.lower())
    if match:
        subdir = match.group(1).strip() if match.group(1) else ""
        dry_run = "dry run" in q.lower()
        res = {"tool": "format_python_files", "arguments": {"subdir": subdir, "dry_run": dry_run, "line_length": 88}}
        logger.info("direct_parser: matched format_python_files → {}", res)
        return res

    logger.debug("direct_parser: no direct match")
    return None
