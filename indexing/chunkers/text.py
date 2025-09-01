import os, hashlib
def _sha(s: str): import hashlib; return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()

def chunk(file_path: str, cfg: dict):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    maxc = int(cfg["chunking"]["text"]["max_chars"])
    ov = int(cfg["chunking"]["text"]["overlap"])
    out, i = [], 0
    rel = os.path.relpath(file_path)
    while i < len(t):
        seg = t[i:i+maxc]
        out.append({
            "id": _sha(rel + f":{i}"),
            "document": seg,
            "metadata": {"file_path": rel, "symbol_kind": "text", "qualified_name": f"{rel}:chunk@{i}"}
        })
        i += max(1, maxc - ov)
    return out
