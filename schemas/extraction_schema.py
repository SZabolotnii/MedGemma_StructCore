"""Pydantic models for structured extraction results.

This module defines a strict KVT4 contract used by tests and lightweight
validation utilities for canonical keyword compliance.
"""

from __future__ import annotations

import re
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator


VALID_CLUSTERS = {
    "DEMOGRAPHICS",
    "VITALS",
    "LABS",
    "PROBLEMS",
    "SYMPTOMS",
    "MEDICATIONS",
    "PROCEDURES",
    "UTILIZATION",
    "DISPOSITION",
}

NUMERIC_ONLY_CLUSTERS = {"VITALS", "LABS"}
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_DATE_RE = re.compile(r"^\d{8}$")
_ALLOWED_TIMESTAMPS = {"Past", "Admission", "Discharge", "Unknown"}


CANONICAL_KEYWORDS: Dict[str, set[str]] = {
    "VITALS": {
        "Heart Rate",
        "Systolic BP",
        "Diastolic BP",
        "Respiratory Rate",
        "Temperature",
        "SpO2",
        "Weight",
    },
    "LABS": {
        "Hemoglobin",
        "Hematocrit",
        "WBC",
        "Platelet",
        "Sodium",
        "Potassium",
        "Creatinine",
        "BUN",
        "Glucose",
        "Bicarbonate",
    },
    "DEMOGRAPHICS": {"Age", "Sex"},
}


class MedicalFact(BaseModel):
    cluster: str
    keyword: str
    value: str
    timestamp: str

    @field_validator("cluster")
    @classmethod
    def validate_cluster(cls, value: str) -> str:
        if value not in VALID_CLUSTERS:
            raise ValueError(f"Invalid cluster: {value}")
        return value

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if value in _ALLOWED_TIMESTAMPS or _DATE_RE.match(value):
            return value
        raise ValueError(f"Invalid timestamp: {value}")

    @model_validator(mode="after")
    def validate_value_contract(self) -> "MedicalFact":
        if self.cluster in NUMERIC_ONLY_CLUSTERS and not _NUM_RE.match(self.value):
            raise ValueError(
                f"{self.cluster} values must be numeric-only, got: {self.value}"
            )
        return self


class ExtractionResult(BaseModel):
    facts: List[MedicalFact] = Field(default_factory=list)

    def to_pipe_delimited(self) -> List[str]:
        return [
            f"{fact.cluster}|{fact.keyword}|{fact.value}|{fact.timestamp}"
            for fact in self.facts
        ]

    @classmethod
    def from_pipe_delimited(cls, lines: List[str]) -> "ExtractionResult":
        facts: List[MedicalFact] = []
        for raw in lines:
            line = str(raw).strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4:
                raise ValueError(f"Expected 4 parts in line: {line}")
            try:
                facts.append(
                    MedicalFact(
                        cluster=parts[0],
                        keyword=parts[1],
                        value=parts[2],
                        timestamp=parts[3],
                    )
                )
            except ValueError:
                # Keep parsing valid lines; invalid facts are ignored.
                continue
        return cls(facts=facts)

    def filter_by_cluster(self, cluster: str) -> "ExtractionResult":
        return ExtractionResult(facts=[fact for fact in self.facts if fact.cluster == cluster])


def validate_canonical_keywords(result: ExtractionResult) -> Dict[str, object]:
    canonical_used: Dict[str, List[str]] = {}
    non_canonical: Dict[str, List[str]] = {}
    canonical_count = 0
    total = len(result.facts)

    for fact in result.facts:
        canonical_for_cluster = CANONICAL_KEYWORDS.get(fact.cluster)

        # For open-vocabulary clusters (e.g. PROBLEMS/SYMPTOMS), do not penalize.
        if not canonical_for_cluster:
            canonical_count += 1
            continue

        if fact.keyword in canonical_for_cluster:
            canonical_count += 1
            canonical_used.setdefault(fact.cluster, []).append(fact.keyword)
        else:
            non_canonical.setdefault(fact.cluster, []).append(fact.keyword)

    for key in list(canonical_used.keys()):
        canonical_used[key] = sorted(set(canonical_used[key]))
    for key in list(non_canonical.keys()):
        non_canonical[key] = sorted(set(non_canonical[key]))

    compliance = (canonical_count / total) if total else 1.0
    return {
        "canonical_used": canonical_used,
        "non_canonical": non_canonical,
        "canonical_compliance": compliance,
    }


__all__ = [
    "MedicalFact",
    "ExtractionResult",
    "validate_canonical_keywords",
    "CANONICAL_KEYWORDS",
]

