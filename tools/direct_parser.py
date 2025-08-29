# gemcli/tools/direct_parser.py

import re
from pathlib import Path
from typing import Optional


def try_direct_actions(query: str, root: Path) -> Optional[dict]:
    """
    Attempt to directly route the query to a known tool with parsed arguments.
    Returns a dict like:
      {"tool": "replace_in_file", "arguments": {...}} or None if not matched.
    """
    query = query.strip().lower()

    # Pattern: Replace 'X' with 'Y' in file.txt (dry run)
    match = re.search(r"replace ['\"](.+?)['\"] with ['\"](.+?)['\"] in (.+?)( \(dry run\))?$", query)
    if match:
        find = match.group(1)
        repl = match.group(2)
        path = match.group(3).strip()
        dry_run = bool(match.group(4))
        return {
            "tool": "replace_in_file",
            "arguments": {
                "path": path,
                "find": find,
                "replace": repl,
                "dry_run": dry_run,
            }
        }

    # Pattern: Create a temp helper file under path with content XYZ
    match = re.search(r"create (?:a )?temp .*? file under (.+?)(?: with content (.+))?$", query)
    if match:
        path = match.group(1).strip()
        content = match.group(2).strip() if match.group(2) else "Hello, world!"
        return {
            "tool": "write_file",
            "arguments": {
                "path": path,
                "content": content,
                "overwrite": True,
            }
        }

    # Pattern: Format python files under some directory
    match = re.search(r"format python files(?: under (.+?))?(?: \(dry run\))?$", query)
    if match:
        subdir = match.group(1).strip() if match.group(1) else ""
        dry_run = "dry run" in query
        return {
            "tool": "format_python_files",
            "arguments": {
                "subdir": subdir,
                "dry_run": dry_run,
                "line_length": 88
            }
        }

    return None
