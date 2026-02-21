"""Pydantic schemas for structured medical fact extraction."""

from .extraction_schema import MedicalFact, ExtractionResult, validate_canonical_keywords

__all__ = ["MedicalFact", "ExtractionResult", "validate_canonical_keywords"]
