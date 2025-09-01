from typing import Dict, List

CANON = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "tel", "telephone"],
    "name": ["name", "full_name", "fullname"],
}

def heuristic_map(columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for c in columns:
        lc = c.lower().strip()
        for canon, aliases in CANON.items():
            if lc == canon or lc in aliases:
                mapping[c] = canon
                break
    return mapping
