from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_cases.json"


@dataclass(frozen=True)
class SyntheticCase:
    id: str
    title: str
    description: str
    text: str


def load_cases() -> List[SyntheticCase]:
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    out: List[SyntheticCase] = []
    for row in raw:
        out.append(
            SyntheticCase(
                id=str(row.get("id", "")).strip(),
                title=str(row.get("title", "")).strip(),
                description=str(row.get("description", "")).strip(),
                text=str(row.get("text", "")).strip(),
            )
        )
    return [c for c in out if c.id and c.title and c.text]


def get_case(case_id: str) -> SyntheticCase | None:
    target = (case_id or "").strip()
    if not target:
        return None
    for item in load_cases():
        if item.id == target:
            return item
    return None
