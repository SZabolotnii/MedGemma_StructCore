"""
Prompt package.

The two-stage structured pipeline uses prompt templates defined in:
- `prompts/optimized_prompt.py`

This `__init__` intentionally re-exports only lightweight normalization helpers.
"""

from .synonyms_mapping import (
    DIAGNOSIS_SYNONYMS,
    SYMPTOM_SYNONYMS,
    MEDICATION_SYNONYMS,
    normalize_diagnosis,
    normalize_symptom,
    terms_match
)

__all__ = [
    "DIAGNOSIS_SYNONYMS",
    "SYMPTOM_SYNONYMS",
    "MEDICATION_SYNONYMS",
    "normalize_diagnosis",
    "normalize_symptom",
    "terms_match",
]
