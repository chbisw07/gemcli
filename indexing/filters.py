import os
from typing import Iterable, Set

def should_index(path: str, cfg) -> bool:
    p = path.replace("\\", "/")
    _, ext = os.path.splitext(p)
    if ext and ext.lower() not in {e.lower() for e in cfg["supported_file_types"]}:
        return False
    name = os.path.basename(p)
    if name in set(cfg["ignore_files"]):
        return False
    if any(part in set(cfg["ignore_dir"]) for part in p.split("/")):
        return False
    if ext.lower() in set(cfg["ignore_ext"]):
        return False
    return True

def iter_files(root: str, cfg) -> Iterable[str]:
    for dirpath, dirs, files in os.walk(root):
        # prune ignored dirs
        dirs[:] = [d for d in dirs if d not in set(cfg["ignore_dir"])]
        for f in files:
            fp = os.path.join(dirpath, f)
            if should_index(fp, cfg):
                yield fp
