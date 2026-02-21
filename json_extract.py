from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


def _escape_newlines_in_json_strings(text: str) -> str:
    """
    Repair helper: replace literal newlines inside JSON strings with escaped sequences.

    Some local backends occasionally emit invalid JSON by placing raw '\n'/'\r' characters
    inside quoted strings. This pass makes such JSON parseable without changing meaning.
    """
    if not text:
        return ""

    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            out.append(ch)
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        out.append(ch)

    return "".join(out)


def extract_first_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Best-effort extraction of the first valid JSON object from model output.

    Returns (obj, json_text). If parsing fails, obj=None and json_text is a best-effort slice.
    """
    s = (text or "").strip()
    if not s:
        return None, ""

    dec = json.JSONDecoder()
    start = s.find("{")
    if start == -1:
        return None, s
    try:
        obj, end = dec.raw_decode(s[start:])
        json_text = s[start : start + end]
        return (obj, json_text) if isinstance(obj, dict) else (None, json_text)
    except Exception:
        end = s.rfind("}")
        if end != -1 and end > start:
            slice_text = s[start : end + 1]
            repaired = _escape_newlines_in_json_strings(slice_text)
            try:
                obj2 = json.loads(repaired)
                if isinstance(obj2, dict):
                    return obj2, repaired
            except Exception:
                pass
            return None, slice_text
        return None, s[start:]
