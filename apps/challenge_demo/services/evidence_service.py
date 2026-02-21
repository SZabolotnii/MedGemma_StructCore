from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "config" / "evidence_claims.json"


def load_evidence_rows() -> List[Dict[str, str]]:
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    rows: List[Dict[str, str]] = []
    for item in data:
        rows.append(
            {
                "Claim ID": str(item.get("claim_id", "")).strip(),
                "Claim": str(item.get("claim", "")).strip(),
                "Metric": str(item.get("metric", "")).strip(),
                "Status": str(item.get("status", "")).strip(),
                "Artifact": str(item.get("artifact", "")).strip(),
            }
        )
    return rows
