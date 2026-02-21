"""
DSPy-free utilities for KVT4 parsing and normalization.

This module contains all parsing, normalization, and validation logic
that does NOT depend on DSPy. It can be used in production pipelines
without importing the DSPy framework.

Extracted from dspy_integration.py as part of Phase 0 decomposition.
"""

import ast
import json
import os
import re
from typing import List, Optional

# =============================================================================
# REGEX PATTERNS
# =============================================================================

_MEDGEMMA_INTERNAL_TOKEN_RE = re.compile(r"<unused\d+>")
_MEDGEMMA_THOUGHT_LINE_RE = re.compile(r"^\s*(<unused\d+>\w*\s*)?thought\b.*$", re.IGNORECASE)
_DSPY_QUOTED_FACT_RE = re.compile(r"«([^»]+)»")
_PARTIAL_JSON_FACT_RE = re.compile(
    r"""\{\s*["']cluster["']\s*:\s*["'](?P<cluster>[^"']+)["']\s*,\s*"""
    r"""["']keyword["']\s*:\s*["'](?P<keyword>[^"']+)["']\s*,\s*"""
    r"""["']value["']\s*:\s*(?P<value>"[^"]*"|'[^']*'|-?\d+(?:\.\d+)?|true|false|null)\s*,\s*"""
    r"""["']timestamp["']\s*:\s*["'](?P<timestamp>[^"']+)["']\s*\}""",
    re.IGNORECASE,
)
_PARTIAL_GROUPED_CLUSTER_BLOCK_RE = re.compile(
    r'"(?P<cluster>DEMOGRAPHICS|VITALS|LABS|PROBLEMS|SYMPTOMS|MEDICATIONS|PROCEDURES|UTILIZATION|DISPOSITION)"\s*:\s*\[',
    re.IGNORECASE,
)
_PARTIAL_GROUPED_ITEM_RE = re.compile(
    r"""\{\s*["']K["']\s*:\s*["'](?P<k>[^"']+)["']\s*,\s*"""
    r"""["']V["']\s*:\s*(?P<v>"[^"]*"|-?\d+(?:\.\d+)?|true|false)\s*,\s*"""
    r"""["']T["']\s*:\s*["'](?P<t>[^"']+)["']\s*\}""",
    re.IGNORECASE,
)


# =============================================================================
# CANONICAL KEYWORDS (MVP)
# =============================================================================

CANONICAL_VITALS = [
    "Heart Rate",
    "Systolic BP",
    "Diastolic BP",
    "Respiratory Rate",
    "Temperature",
    "SpO2",
    "Weight",
]

CANONICAL_LABS = [
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
]

CANONICAL_DEMOGRAPHICS = [
    "Age",
    "Sex",
]

STRICT_KEYWORDS_READMISSION: dict[str, set[str]] = {
    "DEMOGRAPHICS": set(CANONICAL_DEMOGRAPHICS),
    "VITALS": set(CANONICAL_VITALS),
    "LABS": set(CANONICAL_LABS),
    "MEDICATIONS": {
        "Medication Count",
        "New Medications Count",
        "Polypharmacy",
        "Anticoagulation",
        "Insulin Therapy",
        "Opioid Therapy",
        "Diuretic Therapy",
    },
    "PROCEDURES": {
        "Any Procedure",
        "Surgery",
        "Dialysis",
        "Mechanical Ventilation",
    },
    "UTILIZATION": {
        "Prior Admissions 12mo",
        "ED Visits 6mo",
        "Days Since Last Admission",
        "Current Length of Stay",
    },
    "DISPOSITION": {
        "Discharge Disposition",
        "Mental Status",
    },
}
READMISSION_CLUSTERS = {
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


# =============================================================================
# OUTPUT PARSING HELPERS
# =============================================================================

def strip_medgemma_internal_tokens(text: str) -> str:
    """Remove MedGemma internal tokens and thinking blocks from text.
    
    IMPORTANT: Only strips the internal token itself (e.g., "<unused95>").
    Does NOT consume adjacent alphanumerics, because models sometimes emit tokens
    immediately followed by a fact prefix (e.g., "<unused95>DEMOGRAPHICS|..."),
    and we must not delete "DEMOGRAPHICS".
    """
    if not text:
        return ""

    # Remove internal tokens
    cleaned = _MEDGEMMA_INTERNAL_TOKEN_RE.sub("", text)

    # Drop explicit thought lines and thinking blocks
    lines = []
    in_thinking_block = False
    for line in cleaned.splitlines():
        line_lower = line.lower().strip()

        # Check if entering thinking mode
        if any(marker in line_lower for marker in ['thought', 'the user wants', 'here\'s my plan', 'input:', 'output:', 'constraints:']):
            in_thinking_block = True
            continue

        # Check if exiting thinking mode (actual fact payload starts).
        # Support both KVT4 lines and JSON-like payload fragments.
        if in_thinking_block:
            looks_like_kvt4 = ('|' in line and line.count('|') >= 3)
            looks_like_json_payload = (
                '{"K"' in line
                or '"facts"' in line
                or line.lstrip().startswith("{")
                or line.strip().startswith("```json")
            )
            if looks_like_kvt4 or looks_like_json_payload:
                in_thinking_block = False

        # Skip if in thinking block
        if in_thinking_block:
            continue

        # Skip thought lines
        if _MEDGEMMA_THOUGHT_LINE_RE.match(line):
            continue

        lines.append(line)

    return "\n".join(lines)


def _looks_like_kvt_fact(line: str) -> bool:
    """Validate if a line looks like a valid KVT4 fact.
    
    Expects either:
    - 4-part: CLUSTER|Keyword|Value|Timestamp  (preferred)
    - 3-part: Keyword|Value|Timestamp         (legacy)
    """
    if not line:
        return False
    s = line.strip()
    if len(s) < 5 or len(s) > 400:
        return False
    
    pipe_count = s.count("|")
    if pipe_count not in (2, 3):
        return False
    parts = [p.strip() for p in s.split("|")]
    if len(parts) not in (3, 4):
        return False

    # Keep parser permissive by default (unit tests expect 3-part legacy facts too).
    allow_kvt3 = str(os.getenv("ALLOW_KVT3", "1")).strip() == "1"
    if len(parts) == 3 and not allow_kvt3:
        return False
    parts_lower = [p.lower() for p in parts]

    # Filter common headers / schema lines.
    if parts_lower == ["k", "v", "t"]:
        return False
    if parts_lower == ["category", "keyword", "value", "timestamp"]:
        return False
    if (
        len(parts_lower) == 4
        and parts_lower[0].startswith("category")
        and parts_lower[1] == "keyword"
        and parts_lower[2] == "value"
        and parts_lower[3].startswith("timestamp")
    ):
        return False
    if "format" in parts_lower[0] and parts_lower[1:3] == ["keyword", "value"]:
        return False
    if parts_lower[0].startswith(("format", "output format")) and "timestamp" in parts_lower[-1]:
        return False

    # Filter instruction lines
    if any(marker in parts_lower[1] for marker in ['any diagnosis', 'any symptom', 'any procedure', 'value:']):
        return False
    if '(' in parts[1] and ')' in parts[1]:
        return False

    # Length heuristics to avoid capturing prose with incidental pipes.
    if len(parts[0]) > 80 or len(parts[1]) > 80 or len(parts[2]) > 200:
        return False
    if len(parts) == 4 and len(parts[3]) > 40:
        return False

    # Word-count heuristics: KVT lines are short phrases, not full sentences.
    w0 = len(parts[0].split())
    w1 = len(parts[1].split())
    w2 = len(parts[2].split())
    if w0 > 8 or w1 > 8 or w2 > 14:
        return False
    if len(parts) == 4 and len(parts[3].split()) > 4:
        return False
    return all(parts)


def _normalize_kvt_fact(line: str) -> str:
    """Normalize a KVT fact line by stripping whitespace and quotes."""
    parts = [p.strip().strip("«»\"'") for p in line.strip().split("|")]
    return "|".join(parts)


def _map_category_to_cluster(category: str) -> str:
    """Map category aliases to canonical cluster names."""
    c = (category or "").strip().lower()
    if not c:
        return ""
    mapping = {
        "vitals": "VITALS",
        "vital": "VITALS",
        "labs": "LABS",
        "lab": "LABS",
        "demographics": "DEMOGRAPHICS",
        "demo": "DEMOGRAPHICS",
        "conditions": "PROBLEMS",
        "condition": "PROBLEMS",
        "problems": "PROBLEMS",
        "problem": "PROBLEMS",
        "symptoms": "SYMPTOMS",
        "symptom": "SYMPTOMS",
        "medications": "MEDICATIONS",
        "medication": "MEDICATIONS",
        "procedures": "PROCEDURES",
        "procedure": "PROCEDURES",
        "utilization": "UTILIZATION",
        "disposition": "DISPOSITION",
    }
    return mapping.get(c, category.strip())


def _infer_cluster_from_keyword(keyword: str) -> str:
    """Infer cluster from keyword using canonical lists."""
    k = (keyword or "").strip()
    if not k:
        return ""
    if k in CANONICAL_VITALS:
        return "VITALS"
    if k in CANONICAL_LABS:
        return "LABS"
    if k in CANONICAL_DEMOGRAPHICS:
        return "DEMOGRAPHICS"
    # Minimal readmission-fixed keywords
    if k in {"Prior Admissions 12mo", "ED Visits 6mo", "Days Since Last Admission", "Current Length of Stay"}:
        return "UTILIZATION"
    if k in {"Discharge Disposition", "Mental Status"}:
        return "DISPOSITION"
    if k in {"Any Procedure", "Surgery", "Dialysis", "Mechanical Ventilation"}:
        return "PROCEDURES"
    if k in {"Medication Count", "New Medications Count", "Polypharmacy", "Anticoagulation", "Insulin Therapy", "Opioid Therapy", "Diuretic Therapy"}:
        return "MEDICATIONS"
    return ""


def _kvt4_from_fact_dict(d: dict) -> Optional[str]:
    """Convert structured fact dict into CLUSTER|Keyword|Value|Timestamp."""
    if not isinstance(d, dict):
        return None

    def _first_present(*keys: str):
        for key in keys:
            if key in d and d[key] is not None:
                return d[key]
        return None

    # Accept multiple key spellings
    cluster = _first_present("cluster", "Cluster", "CLUSTER", "C", "category", "Category")
    keyword = _first_present("keyword", "Keyword", "KEYWORD", "K")
    value = _first_present("value", "Value", "VALUE", "V")
    timestamp = _first_present("timestamp", "Timestamp", "TIMESTAMP", "T")

    keyword_s = str(keyword).strip() if keyword is not None else ""
    value_s = str(value).strip() if value is not None else ""
    timestamp_s = str(timestamp).strip() if timestamp is not None else ""

    cluster_s = str(cluster).strip() if cluster is not None else ""
    cluster_s = _map_category_to_cluster(cluster_s)

    if not cluster_s:
        cluster_s = _infer_cluster_from_keyword(keyword_s)
    if not cluster_s:
        cluster_s = "UNKNOWN"

    if not keyword_s or not value_s:
        return None
    if not timestamp_s:
        timestamp_s = "Unknown"

    return f"{cluster_s}|{keyword_s}|{value_s}|{timestamp_s}"


def _fact_dict_has_explicit_cluster(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    for key in ("cluster", "Cluster", "CLUSTER", "C", "category", "Category"):
        v = d.get(key)
        if v is not None and str(v).strip():
            return True
    return False


def _unwrap_model_output(text: str) -> str:
    """Strip common model output wrappers that obscure the JSON/KVT payload.

    Removes (in order):
    1) Standalone START / END boundary markers (whole line, case-insensitive)
    2) Markdown code fences (``` or ```json) — both closed and unclosed
    """
    if not text:
        return text

    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper() in {"START", "END", "```", "```JSON", "```json"}:
            continue
        if stripped.startswith("```"):
            continue
        lines.append(line)
    result = "\n".join(lines).strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result, re.IGNORECASE)
    if fence_match:
        result = fence_match.group(1).strip()
    return result


def extract_kvt_fact_lines(text: str) -> List[str]:
    """
    Extract candidate K|V|T / Category|K|V|T lines from arbitrary model output.

    Handles common formats:
    - Plain pipe-delimited lines
    - DSPy-rendered lists like: [1] «Vitals|Temperature|37.2°C|20240110»
    - JSON objects/lists containing "facts"
    - Python literal lists of strings
    """
    if not text:
        return []

    cleaned = strip_medgemma_internal_tokens(text).strip()
    cleaned = _unwrap_model_output(cleaned)
    if not cleaned:
        return []

    candidates: List[str] = []
    structured_extracted = False

    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    def add_fact(s: str) -> None:
        s2 = _normalize_kvt_fact(s)
        # Recover 5-part drift lines like:
        #   "CLUSTER|VITALS|Heart Rate|67|Admission"
        # by dropping the leading CLUSTER/CLUSTERS token.
        #
        # This recovery must happen here (not only in line-by-line parsing) because
        # these strings can also appear inside JSON payloads.
        if s2.count("|") == 4:
            parts5 = [p.strip() for p in s2.split("|")]
            if len(parts5) == 5 and parts5[0].strip().upper() in {"CLUSTER", "CLUSTERS"}:
                s2 = "|".join(parts5[1:])
        # Recover 4-part drift lines where the model outputs a literal CLUSTER/CLUSTERS
        # token as the cluster field and omits the actual cluster, e.g.:
        #   "CLUSTERS|Heart Rate|67|Admission"
        # Infer the intended cluster from (Keyword, Value) when unambiguous.
        if s2.count("|") == 3:
            parts4 = [p.strip() for p in s2.split("|")]
            if len(parts4) == 4 and parts4[0].strip().upper() in {"CLUSTER", "CLUSTERS"}:
                kw = parts4[1].strip()
                val = parts4[2].strip()
                ts = parts4[3].strip()
                # Filter prompt-template headers like: "CLUSTER|Keyword|Value|Timestamp"
                if (
                    kw.casefold() in {"keyword", "k"}
                    and val.casefold() in {"value", "v"}
                    and ts.casefold() in {"timestamp", "t", "unknown", "admission", "discharge", "past"}
                ):
                    return
                inferred = _infer_cluster_from_keyword(kw)
                if not inferred:
                    v_l = val.strip().lower()
                    if v_l in {"acute", "chronic", "exist", "not exist"}:
                        inferred = "PROBLEMS"
                    elif v_l in {"yes", "no", "severe"}:
                        inferred = "SYMPTOMS"
                if inferred:
                    s2 = f"{inferred}|{kw}|{val}|{ts}"
        if _looks_like_kvt_fact(s2):
            candidates.append(s2)

    def _map_category_to_cluster(category: str) -> str:
        c = (category or "").strip().lower()
        if not c:
            return ""
        mapping = {
            "vitals": "VITALS",
            "vital": "VITALS",
            "labs": "LABS",
            "lab": "LABS",
            "demographics": "DEMOGRAPHICS",
            "demo": "DEMOGRAPHICS",
            "conditions": "PROBLEMS",
            "condition": "PROBLEMS",
            "problems": "PROBLEMS",
            "problem": "PROBLEMS",
            "symptoms": "SYMPTOMS",
            "symptom": "SYMPTOMS",
            "medications": "MEDICATIONS",
            "medication": "MEDICATIONS",
            "procedures": "PROCEDURES",
            "procedure": "PROCEDURES",
            "utilization": "UTILIZATION",
            "disposition": "DISPOSITION",
        }
        return mapping.get(c, category.strip())

    def _infer_cluster_from_keyword(keyword: str) -> str:
        k = (keyword or "").strip()
        if not k:
            return ""
        if k in CANONICAL_VITALS:
            return "VITALS"
        if k in CANONICAL_LABS:
            return "LABS"
        if k in CANONICAL_DEMOGRAPHICS:
            return "DEMOGRAPHICS"
        # Minimal readmission-fixed keywords (ontology v1).
        if k in {"Prior Admissions 12mo", "ED Visits 6mo", "Days Since Last Admission", "Current Length of Stay"}:
            return "UTILIZATION"
        if k in {"Discharge Disposition", "Mental Status"}:
            return "DISPOSITION"
        if k in {"Any Procedure", "Surgery", "Dialysis", "Mechanical Ventilation"}:
            return "PROCEDURES"
        if k in {"Medication Count", "New Medications Count", "Polypharmacy", "Anticoagulation", "Insulin Therapy", "Opioid Therapy", "Diuretic Therapy"}:
            return "MEDICATIONS"
        return ""

    def _kvt4_from_fact_dict(d: dict) -> Optional[str]:
        """Convert common structured fact dicts into CLUSTER|Keyword|Value|Timestamp."""
        if not isinstance(d, dict):
            return None

        def _first_present(*keys: str):
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return None

        # Accept multiple key spellings (legacy + short keys).
        cluster = _first_present("cluster", "Cluster", "CLUSTER", "C", "category", "Category")
        keyword = _first_present("keyword", "Keyword", "KEYWORD", "K")
        value = _first_present("value", "Value", "VALUE", "V")
        timestamp = _first_present("timestamp", "Timestamp", "TIMESTAMP", "T")

        keyword_s = str(keyword).strip() if keyword is not None else ""
        value_s = str(value).strip() if value is not None else ""
        timestamp_s = str(timestamp).strip() if timestamp is not None else ""
        # Drop prompt-template placeholders that are not real facts.
        if keyword_s.casefold() in {"keyword", "k"} and value_s.casefold() in {"value", "v"}:
            if timestamp_s.casefold() in {"timestamp", "t", "unknown", "admission", "discharge", "past"}:
                return None

        cluster_s = str(cluster).strip() if cluster is not None else ""
        # If we got a "category" like "vitals/labs", map it into prompt-style clusters.
        cluster_s = _map_category_to_cluster(cluster_s)

        if not cluster_s:
            cluster_s = _infer_cluster_from_keyword(keyword_s)
        if not cluster_s:
            cluster_s = "UNKNOWN"

        if not keyword_s or not value_s:
            return None
        if not timestamp_s:
            timestamp_s = "Unknown"

        return f"{cluster_s}|{keyword_s}|{value_s}|{timestamp_s}"

    def _kvt4_lines_from_grouped_obj(obj: dict) -> List[str]:
        """Convert grouped JSON object into KVT4 lines.

        Supported layout:
        {
          "LABS":[{"K":"Creatinine","V":1.2,"T":"Discharge"}],
          "PROBLEMS":[{"K":"Hypertension","V":"chronic","T":"Past"}]
        }
        """
        if not isinstance(obj, dict):
            return []

        out_lines: List[str] = []
        for raw_cluster, raw_entries in obj.items():
            cluster_norm = _map_category_to_cluster(str(raw_cluster).strip())
            cluster_upper = cluster_norm.upper()
            if cluster_upper not in READMISSION_CLUSTERS:
                continue

            entries: List[dict] = []
            if isinstance(raw_entries, list):
                entries = [it for it in raw_entries if isinstance(it, dict)]
            elif isinstance(raw_entries, dict):
                entries = [raw_entries]
            else:
                continue

            for ent in entries:
                keyword = ent["K"] if "K" in ent else ent.get("keyword", ent.get("Keyword"))
                value = ent["V"] if "V" in ent else ent.get("value", ent.get("Value"))
                timestamp = ent["T"] if "T" in ent else ent.get("timestamp", ent.get("Timestamp"))
                fact_obj = {
                    "cluster": cluster_upper,
                    "keyword": keyword,
                    "value": value,
                    "timestamp": timestamp,
                }
                ln = _kvt4_from_fact_dict(fact_obj)
                if ln:
                    out_lines.append(ln)
        return out_lines

    # 1) JSON / Python list attempts (whole string + best-effort substrings)
    json_like = cleaned
    substrings: List[str] = [json_like]
    first_obj = json_like.find("{")
    last_obj = json_like.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        substrings.append(json_like[first_obj : last_obj + 1])
    first_arr = json_like.find("[")
    last_arr = json_like.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        substrings.append(json_like[first_arr : last_arr + 1])

    cleaned_strip = cleaned.strip()
    for s in list(dict.fromkeys(substrings)):
        s_strip = s.strip()
        if not s_strip:
            continue
        is_derived_array_substring = (
            s_strip.startswith("[") and s_strip.endswith("]") and s_strip != cleaned_strip
        )
        try:
            before = len(candidates)
            obj = json.loads(s_strip)
            if isinstance(obj, dict):
                facts = obj.get("facts")
                if isinstance(facts, list):
                    for it in facts:
                        if isinstance(it, str):
                            add_fact(it)
                        elif isinstance(it, dict):
                            ln = _kvt4_from_fact_dict(it)
                            if ln:
                                add_fact(ln)
                else:
                    grouped_lines = _kvt4_lines_from_grouped_obj(obj)
                    if grouped_lines:
                        for ln in grouped_lines:
                            add_fact(ln)
                        continue
                    # Sometimes the whole object is a single fact dict.
                    ln = _kvt4_from_fact_dict(obj)
                    if ln:
                        add_fact(ln)
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, str):
                        add_fact(it)
                    elif isinstance(it, dict):
                        # Avoid duplicate UNKNOWN facts when a grouped JSON object is
                        # also parsed via its inner array substring (cluster context lost).
                        if is_derived_array_substring and not _fact_dict_has_explicit_cluster(it):
                            continue
                        ln = _kvt4_from_fact_dict(it)
                        if ln:
                            add_fact(ln)
            if len(candidates) > before:
                structured_extracted = True
        except Exception:
            pass

        try:
            before = len(candidates)
            obj = ast.literal_eval(s_strip)
            if isinstance(obj, dict):
                facts = obj.get("facts") if isinstance(obj.get("facts"), list) else None
                if facts is not None:
                    for it in facts:
                        if isinstance(it, str):
                            add_fact(it)
                        elif isinstance(it, dict):
                            ln = _kvt4_from_fact_dict(it)
                            if ln:
                                add_fact(ln)
                else:
                    grouped_lines = _kvt4_lines_from_grouped_obj(obj)
                    if grouped_lines:
                        for ln in grouped_lines:
                            add_fact(ln)
                        continue
                    ln = _kvt4_from_fact_dict(obj)
                    if ln:
                        add_fact(ln)
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, str):
                        add_fact(it)
                    elif isinstance(it, dict):
                        if is_derived_array_substring and not _fact_dict_has_explicit_cluster(it):
                            continue
                        ln = _kvt4_from_fact_dict(it)
                        if ln:
                            add_fact(ln)
            if len(candidates) > before:
                structured_extracted = True
        except Exception:
            pass

    # If we already extracted structured facts, do not run heuristic recovery
    # branches below (they may introduce noisy duplicates on valid JSON payloads).
    if structured_extracted and candidates:
        return _dedupe_preserve_order(candidates)

    # 1b) Partial/truncated JSON recovery:
    # If the model output is cut mid-stream, json.loads fails even when many
    # complete fact objects were already emitted. Recover those complete objects.
    for m in _PARTIAL_JSON_FACT_RE.finditer(cleaned):
        c = str(m.group("cluster") or "").strip()
        k = str(m.group("keyword") or "").strip()
        v_tok = str(m.group("value") or "").strip()
        t = str(m.group("timestamp") or "").strip()
        if v_tok.startswith('"') and v_tok.endswith('"') and len(v_tok) >= 2:
            v = v_tok[1:-1]
        elif v_tok.startswith("'") and v_tok.endswith("'") and len(v_tok) >= 2:
            v = v_tok[1:-1]
        else:
            v = v_tok
        if c and k and v and t:
            add_fact(f"{c}|{k}|{v}|{t}")

    # 1c) Partial/truncated grouped JSON recovery:
    # Recover complete {"K","V","T"} entries within each cluster block even when
    # root JSON is truncated and json.loads fails.
    cluster_hits = list(_PARTIAL_GROUPED_CLUSTER_BLOCK_RE.finditer(cleaned))
    if cluster_hits:
        for idx, hit in enumerate(cluster_hits):
            cluster = str(hit.group("cluster") or "").strip().upper()
            block_start = hit.end()
            block_end = cluster_hits[idx + 1].start() if idx + 1 < len(cluster_hits) else len(cleaned)
            block = cleaned[block_start:block_end]
            for item in _PARTIAL_GROUPED_ITEM_RE.finditer(block):
                k = str(item.group("k") or "").strip()
                t = str(item.group("t") or "").strip()
                v_tok = str(item.group("v") or "").strip()
                if not k or not t:
                    continue
                if v_tok.startswith('"') and v_tok.endswith('"') and len(v_tok) >= 2:
                    v = v_tok[1:-1]
                elif v_tok.casefold() in {"true", "false"}:
                    v = v_tok.casefold()
                else:
                    v = v_tok
                if v:
                    add_fact(f"{cluster}|{k}|{v}|{t}")

    # 2) Extract between DSPy quotes «...»
    for m in _DSPY_QUOTED_FACT_RE.finditer(cleaned):
        inner = m.group(1).strip()
        if "|" in inner:
            add_fact(inner)

    # 2b) Narrative markdown recovery.
    # Some small models emit facts as multi-line markdown blocks:
    #   **CLUSTER:** DEMOGRAPHICS
    #   **Keyword:** Sex
    #   **Value:** male
    #   **Timestamp:** Admission
    # Recover these into KVT4 lines.
    _narrative_kv_re = re.compile(
        r"\*{0,2}(cluster|keyword|value|timestamp)\s*:?\s*\*{0,2}\s*(.+)",
        re.IGNORECASE,
    )
    cur: dict = {}
    for line in cleaned.splitlines():
        m = _narrative_kv_re.match(line.strip())
        if not m:
            continue
        field = m.group(1).strip().lower()
        val = m.group(2).strip().strip("*").strip()
        if field == "cluster":
            if cur.get("cluster") and cur.get("keyword") and cur.get("value"):
                ts = cur.get("timestamp", "Unknown")
                add_fact(f"{cur['cluster']}|{cur['keyword']}|{cur['value']}|{ts}")
            cur = {"cluster": val}
        elif field == "keyword":
            # Flush previous fact within the same cluster before starting a new keyword
            if cur.get("cluster") and cur.get("keyword") and cur.get("value"):
                ts = cur.get("timestamp", "Unknown")
                add_fact(f"{cur['cluster']}|{cur['keyword']}|{cur['value']}|{ts}")
            cluster_keep = cur.get("cluster", "")
            cur = {"cluster": cluster_keep, "keyword": val}
        elif field in ("value", "timestamp"):
            cur[field] = val
    # flush last accumulated fact
    if cur.get("cluster") and cur.get("keyword") and cur.get("value"):
        ts = cur.get("timestamp", "Unknown")
        add_fact(f"{cur['cluster']}|{cur['keyword']}|{cur['value']}|{ts}")

    # 2c) Cluster-heading + inline JSON item recovery.
    # Some models emit planning text like:
    #   * **VITALS:**
    #     ... -> {"K":"Heart Rate","V":54,"T":"Admission"}
    # Recover such entries by tracking the current cluster heading.
    heading_re = re.compile(r"\*{0,2}\s*([A-Z][A-Z ]{2,})\s*:\s*\*{0,2}\s*$")
    cluster_inline_re = re.compile(
        r"\b(DEMOGRAPHICS|VITALS|LABS|PROBLEMS|SYMPTOMS|MEDICATIONS|PROCEDURES|UTILIZATION|DISPOSITION)\b",
        re.IGNORECASE,
    )
    item_re = re.compile(
        r'\{\s*"K"\s*:\s*"(?P<k>[^"]+)"\s*,\s*"V"\s*:\s*(?P<v>"[^"]*"|-?\d+(?:\.\d+)?|true|false)\s*,\s*"T"\s*:\s*"(?P<t>Past|Admission|Discharge|Unknown)"\s*\}',
        re.IGNORECASE,
    )
    cur_cluster = ""
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Avoid cluster-bleed on compact one-line JSON objects:
        # grouped payloads should be handled by structured parsing above.
        if line.startswith("{") or line.startswith("["):
            continue

        # Accept headings like "**VITALS:**", "VITALS:", "*   **VITALS:**"
        norm = re.sub(r"^[*•\-\s]+", "", line)
        norm = norm.strip("* ").strip()
        hm = heading_re.match(norm)
        if hm:
            c_raw = hm.group(1).strip().upper()
            c_norm = _map_category_to_cluster(c_raw)
            c_up = str(c_norm).strip().upper()
            if c_up in READMISSION_CLUSTERS:
                cur_cluster = c_up
            continue

        # Inline headings like:
        # "- **VITALS:** ... -> {\"K\":\"Heart Rate\",...}"
        cm = cluster_inline_re.search(norm)
        if cm:
            c_raw = cm.group(1).strip().upper()
            c_norm = _map_category_to_cluster(c_raw)
            c_up = str(c_norm).strip().upper()
            if c_up in READMISSION_CLUSTERS:
                cur_cluster = c_up

        if not cur_cluster:
            continue
        for m in item_re.finditer(line):
            k = str(m.group("k") or "").strip()
            v_tok = str(m.group("v") or "").strip()
            t = str(m.group("t") or "").strip()
            if not k or not t:
                continue
            if v_tok.startswith('"') and v_tok.endswith('"') and len(v_tok) >= 2:
                v = v_tok[1:-1]
            else:
                v = v_tok.casefold() if v_tok.casefold() in {"true", "false"} else v_tok
            if v:
                add_fact(f"{cur_cluster}|{k}|{v}|{t}")

    # 3) Line-by-line heuristics (bullets / numbering / quoted JSON fragments)
    for line in cleaned.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\[\d+\]\s*", "", s)
        s = re.sub(r"^[-*•]\s*", "", s)
        s = s.strip().strip("«»\"'")
        s = s.rstrip(",")
        if "|" in s:
            # Recover 5-part "CLUSTER|<cluster>|<keyword>|<value>|<timestamp>" drift by
            # dropping the leading CLUSTER/CLUSTERS token.
            #
            # This format appears in some Stage2 outputs and should be treated as
            # equivalent to the canonical 4-part KVT4 line.
            if s.count("|") == 4:
                parts5 = [p.strip() for p in s.split("|")]
                if len(parts5) == 5 and parts5[0].strip().upper() in {"CLUSTER", "CLUSTERS"}:
                    recovered = "|".join(parts5[1:])
                    add_fact(recovered)
                    continue

            # Recover 2-part drift: "CLUSTER|Keyword=Value".
            # Map to KVT4 with Unknown timestamp; downstream sanitizer can canonicalize.
            if s.count("|") == 1:
                left, right = [x.strip() for x in s.split("|", 1)]
                if "=" in right:
                    key, value = [x.strip() for x in right.split("=", 1)]
                    if left and key and value:
                        add_fact(f"{left}|{key}|{value}|Unknown")
                        continue
            add_fact(s)

    # De-duplicate while preserving order
    return _dedupe_preserve_order(candidates)

def normalize_readmission_kvt4_lines(lines: List[str]) -> tuple[List[str], dict]:
    """Normalize KVT4 lines into canonical READMISSION_MVP form.

    Goals:
    - Boost strict-format usability by deterministic canonicalization
    - Reduce drift (Blood Pressure -> SBP/DBP, Oxygen Saturation -> SpO2, etc.)
    - Enforce numeric-only values for VITALS/LABS (+ known numeric fields)
    - Enforce at most one line per (CLUSTER, Keyword) via timestamp-priority dedupe

    Returns: (normalized_lines, stats)
    """

    def _parse_line(line: str) -> Optional[tuple[str, str, str, str]]:
        if not isinstance(line, str):
            return None
        s = line.strip()
        # Strip leading markdown bullets
        s = re.sub(r"^[-*\u2022]\s+", "", s).strip()
        if not s:
            return None

        pipe_count = s.count("|")
        if pipe_count == 3:
            parts = [p.strip() for p in s.split("|")]
            if len(parts) != 4:
                return None
            c, k, v, t = parts
            if not c or not k or not v:
                return None
            return c, k, v, t or "Unknown"

        if pipe_count == 2:
            # 3-part:
            # - Keyword|Value|Timestamp (legacy, missing cluster)
            # - CLUSTER|Keyword|Value   (drift, missing timestamp)
            parts = [p.strip() for p in s.split("|")]
            if len(parts) != 3:
                return None
            a, b, c = parts
            if not a or not b or not c:
                return None

            cluster_guess = a.strip().strip("*<>").strip().upper()
            if cluster_guess in READMISSION_CLUSTERS:
                return cluster_guess, b, c, "Unknown"

            k, v, t = a, b, c
            cluster = _infer_cluster_from_keyword(k)
            if not cluster:
                return None
            return cluster, k, v, t or "Unknown"

        return None

    def _normalize_timestamp(t: str) -> str:
        tt = (t or "").strip()
        if not tt:
            return "Unknown"
        if tt in {"Admission", "Discharge", "Past", "Unknown"}:
            return tt
        if tt.casefold() == "adm":
            return "Admission"
        if tt.casefold() == "dc":
            return "Discharge"
        return "Unknown"

    def _fill_unknown_timestamp(cluster: str, keyword: str, value: str) -> str:
        """Best-effort timestamp fill for strict-eval stability.

        Policy is ontology-driven (not note-section heuristics):
        - DEMOGRAPHICS/VITALS/LABS/SYMPTOMS/MEDICATIONS/PROCEDURES: Admission
        - DISPOSITION: Discharge
        - UTILIZATION: Past
        - PROBLEMS: Past if chronic, Discharge if acute, else Past
        """
        c = (cluster or "").strip().upper()
        v = (value or "").strip().lower()

        if c == "DISPOSITION":
            return "Discharge"
        if c == "UTILIZATION":
            return "Past"
        if c == "PROBLEMS":
            if v == "acute":
                return "Discharge"
            if v == "chronic":
                return "Past"
            # Default: history-like framing
            return "Past"
        if c in {"DEMOGRAPHICS", "VITALS", "LABS", "SYMPTOMS", "MEDICATIONS", "PROCEDURES"}:
            return "Admission"
        return "Admission"

    def _first_number(value: str) -> Optional[str]:
        m = re.search(r"-?\d+(?:\.\d+)?", value or "")
        return m.group(0) if m else None

    # Keyword aliases (strict clusters).
    vital_alias = {
        "HR": "Heart Rate",
        "Pulse": "Heart Rate",
        "Temp": "Temperature",
        "O2 Sat": "SpO2",
        "Oxygen Saturation": "SpO2",
        "SpO2": "SpO2",
        "Resp": "Respiratory Rate",
        "RR": "Respiratory Rate",
        "Blood Pressure": "Blood Pressure",  # special-case splitter
        "BP": "Blood Pressure",
        "Systolic": "Systolic BP",
        "Diastolic": "Diastolic BP",
        "SBP": "Systolic BP",
        "DBP": "Diastolic BP",
    }
    lab_alias = {
        "Hgb": "Hemoglobin",
        "Hct": "Hematocrit",
        "Plt": "Platelet",
        "Platelets": "Platelet",
        "Na": "Sodium",
        "K": "Potassium",
        "Cr": "Creatinine",
        "HCO3": "Bicarbonate",
        "Bicarb": "Bicarbonate",
        "WBC": "WBC",
        "BUN": "BUN",
    }
    sex_alias = {"m": "male", "male": "male", "f": "female", "female": "female"}

    # Dedupe priority can be configured per mode.
    # For full readmission feature set we generally care about discharge/most-recent.
    ts_priority = [s.strip() for s in os.getenv("MEDGEMMA_TIMESTAMP_PRIORITY", "Discharge,Admission,Past,Unknown").split(",") if s.strip()]
    ts_rank = {t: i for i, t in enumerate(ts_priority)}

    stats = {
        "input_lines": len(lines or []),
        "parsed_kvt4": 0,
        "dropped_placeholders": 0,
        "dropped_noncanonical": 0,
        "dropped_by_allowed_clusters": 0,
        "expanded_bp": 0,
        "dedup_dropped": 0,
        "output_lines": 0,
        "canonical_keyword_rate_strict": None,
        "numeric_only_rate_vitals_labs": None,
        "duplicates_after_dedup": 0,
    }

    allowed_clusters_env = os.getenv("MEDGEMMA_ALLOWED_CLUSTERS", "").strip()
    allowed_clusters = None
    if allowed_clusters_env:
        allowed_clusters = {c.strip().upper() for c in allowed_clusters_env.split(",") if c.strip()}

    # First pass: normalize + expand BP
    normalized_candidates: List[tuple[str, str, str, str]] = []
    fill_unknown = os.getenv("MEDGEMMA_TIMESTAMP_FILL_UNKNOWN", "1").strip().lower() in {"1", "true", "yes"}
    for line in lines or []:
        parsed = _parse_line(line)
        if not parsed:
            continue
        c, k, v, t = parsed
        stats["parsed_kvt4"] += 1

        c_up = str(c).strip().strip("*<>").strip().upper()
        if allowed_clusters is not None and c_up not in allowed_clusters:
            stats["dropped_by_allowed_clusters"] += 1
            continue
        t_norm = _normalize_timestamp(t)
        k_norm = k.strip()
        v_norm = v.strip()

        # Drop obvious placeholders
        if v_norm in {"___", "__", "_", "N/A", "NA", "null", "None"}:
            stats["dropped_placeholders"] += 1
            continue

        # Cluster-specific normalization
        if c_up == "DEMOGRAPHICS":
            if k_norm == "Sex":
                vv = sex_alias.get(v_norm.strip().lower())
                if not vv:
                    stats["dropped_noncanonical"] += 1
                    continue
                v_norm = vv
            elif k_norm == "Age":
                num = _first_number(v_norm)
                if not num:
                    stats["dropped_noncanonical"] += 1
                    continue
                v_norm = num

        elif c_up == "VITALS":
            k_norm = vital_alias.get(k_norm, k_norm)
            if k_norm == "Blood Pressure":
                # Expand 120/80 -> SBP + DBP
                m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", v_norm)
                if not m:
                    stats["dropped_noncanonical"] += 1
                    continue
                sbp, dbp = m.group(1), m.group(2)
                normalized_candidates.append(("VITALS", "Systolic BP", sbp, t_norm))
                normalized_candidates.append(("VITALS", "Diastolic BP", dbp, t_norm))
                stats["expanded_bp"] += 1
                continue

            # Enforce numeric-only for vitals
            num = _first_number(v_norm)
            if not num:
                stats["dropped_noncanonical"] += 1
                continue
            v_norm = num

        elif c_up == "LABS":
            k_norm = lab_alias.get(k_norm, k_norm)
            num = _first_number(v_norm)
            if not num:
                stats["dropped_noncanonical"] += 1
                continue
            v_norm = num

        elif c_up == "UTILIZATION":
            num = _first_number(v_norm)
            if not num:
                stats["dropped_noncanonical"] += 1
                continue
            v_norm = num

        elif c_up == "MEDICATIONS":
            if k_norm in {"Medication Count", "New Medications Count"}:
                num = _first_number(v_norm)
                if not num:
                    stats["dropped_noncanonical"] += 1
                    continue
                v_norm = num
            elif k_norm in {"Polypharmacy", "Anticoagulation", "Insulin Therapy", "Opioid Therapy", "Diuretic Therapy"}:
                vv = v_norm.strip().lower()
                if vv in {"yes", "y", "true", "1"}:
                    v_norm = "yes"
                elif vv in {"no", "n", "false", "0"}:
                    v_norm = "no"
                else:
                    stats["dropped_noncanonical"] += 1
                    continue

        elif c_up == "PROCEDURES":
            vv = v_norm.strip().lower()
            if k_norm == "Dialysis":
                vv = vv.replace("canceled", "cancelled")
                if vv in {"decided", "started", "done", "cancelled", "no"}:
                    v_norm = vv
                elif vv in {"yes", "y", "true", "1", "performed", "present", "active", "positive"}:
                    # Conservative mapping: dialysis is present but state unknown.
                    v_norm = "decided"
                elif vv in {"n", "false", "0", "absent", "negative", "none"}:
                    v_norm = "no"
                else:
                    stats["dropped_noncanonical"] += 1
                    continue
            elif k_norm == "Mechanical Ventilation":
                if vv == "no":
                    v_norm = "no"
                else:
                    num = _first_number(v_norm)
                    v_norm = num if num else "yes"
            elif k_norm in {"Any Procedure", "Surgery"}:
                if vv in {"yes", "y", "true", "1", "performed", "done", "started", "present"}:
                    v_norm = "yes"
                elif vv in {"no", "n", "false", "0", "absent", "not performed", "cancelled"}:
                    v_norm = "no"
                else:
                    stats["dropped_noncanonical"] += 1
                    continue
            else:
                # Unknown procedure keyword: keep only if value looks boolean.
                if vv in {"yes", "y", "true", "1", "performed", "done", "started", "present"}:
                    v_norm = "yes"
                elif vv in {"no", "n", "false", "0", "absent", "not performed", "cancelled"}:
                    v_norm = "no"
                else:
                    stats["dropped_noncanonical"] += 1
                    continue

        elif c_up == "DISPOSITION":
            if k_norm == "Discharge Disposition":
                vv = v_norm.strip().lower()
                # Normalize into the prompt enums.
                if "home with" in vv or "home w" in vv or "services" in vv:
                    v_norm = "Home with Services"
                elif vv == "home" or vv.startswith("home "):
                    v_norm = "Home"
                elif "snf" in vv or "skilled nursing" in vv:
                    v_norm = "SNF"
                elif "rehab" in vv:
                    v_norm = "Rehab"
                elif "ltac" in vv:
                    v_norm = "LTAC"
                elif "hospice" in vv:
                    v_norm = "Hospice"
                elif "ama" in vv or "against medical advice" in vv:
                    v_norm = "AMA"
                else:
                    stats["dropped_noncanonical"] += 1
                    continue
            elif k_norm == "Mental Status":
                vv = v_norm.strip().lower()
                if "confus" in vv:
                    v_norm = "confused"
                elif "letharg" in vv:
                    v_norm = "lethargic"
                elif "alert" in vv:
                    v_norm = "alert"
                elif "orient" in vv:
                    v_norm = "oriented"
                else:
                    stats["dropped_noncanonical"] += 1
                    continue

        elif c_up == "PROBLEMS":
            vv = re.sub(r"\s+", " ", v_norm.strip().lower())
            if vv in {"chronic", "acute", "exist", "not exist"}:
                v_norm = vv
            elif vv in {"past", "history", "historical", "pmh", "chronic condition", "chronic disease"}:
                v_norm = "chronic"
            elif vv in {"discharge", "discharged", "active", "current"}:
                v_norm = "acute"
            elif vv in {"present", "yes", "true", "1", "positive", "confirmed", "exists"}:
                v_norm = "exist"
            elif vv in {"no", "none", "false", "0", "absent", "negative", "not present", "ruled out"}:
                v_norm = "not exist"
            else:
                stats["dropped_noncanonical"] += 1
                continue

        elif c_up == "SYMPTOMS":
            vv = re.sub(r"\s+", " ", v_norm.strip().lower())
            if vv in {"yes", "no", "severe"}:
                v_norm = vv
            elif vv in {"present", "positive", "true", "1", "y", "symptomatic"}:
                v_norm = "yes"
            elif vv in {"none", "absent", "negative", "false", "0", "n", "denied", "denies"}:
                v_norm = "no"
            elif "severe" in vv or vv in {"marked", "significant"}:
                v_norm = "severe"
            else:
                stats["dropped_noncanonical"] += 1
                continue

        # Drop non-canonical keywords for strict clusters (objective ones).
        if c_up in STRICT_KEYWORDS_READMISSION:
            if k_norm not in STRICT_KEYWORDS_READMISSION[c_up]:
                stats["dropped_noncanonical"] += 1
                continue

        if fill_unknown and t_norm == "Unknown":
            t_norm = _fill_unknown_timestamp(c_up, k_norm, v_norm)

        normalized_candidates.append((c_up, k_norm, v_norm, t_norm))

    # Second pass: dedupe by (CLUSTER, Keyword) using timestamp priority.
    best: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    for c, k, v, t in normalized_candidates:
        key = (c, k)
        cur = best.get(key)
        if cur is None:
            best[key] = (c, k, v, t)
            continue
        _, _, _, t_prev = cur
        r_new = ts_rank.get(t, 999)
        r_prev = ts_rank.get(t_prev, 999)
        if r_new < r_prev:
            best[key] = (c, k, v, t)
        else:
            stats["dedup_dropped"] += 1

    out_lines = [f"{c}|{k}|{v}|{t}" for (c, k), (c, k, v, t) in best.items()]
    out_lines.sort(key=lambda s: (s.split("|")[0], s.split("|")[1]))

    # Metrics: canonical + numeric-only compliance for VITALS/LABS.
    strict_total = 0
    strict_ok = 0
    vitlab_total = 0
    vitlab_numeric = 0
    key_counts: dict[tuple[str, str], int] = {}
    for ln in out_lines:
        parsed = _parse_line(ln)
        if not parsed:
            continue
        c, k, v, _t = parsed
        key_counts[(c, k)] = key_counts.get((c, k), 0) + 1
        if c in STRICT_KEYWORDS_READMISSION:
            strict_total += 1
            if k in STRICT_KEYWORDS_READMISSION[c]:
                strict_ok += 1
        if c in {"VITALS", "LABS"}:
            vitlab_total += 1
            if re.fullmatch(r"-?\d+(?:\.\d+)?", v.strip()):
                vitlab_numeric += 1

    stats["duplicates_after_dedup"] = sum(1 for cnt in key_counts.values() if cnt > 1)
    stats["output_lines"] = len(out_lines)
    stats["canonical_keyword_rate_strict"] = (strict_ok / strict_total) if strict_total else 1.0
    stats["numeric_only_rate_vitals_labs"] = (vitlab_numeric / vitlab_total) if vitlab_total else 1.0

    return out_lines, stats


def _normalize_mode(mode: Optional[str]) -> str:
    """Normalize mode string to canonical format."""
    if not mode:
        return "READMISSION_DISCHARGE"
    mode = mode.upper().replace("-", "_")
    if mode in {"CCDE", "CCDE_ADMISSION"}:
        return "CCDE_ADMISSION"
    elif mode in {"TABULAR", "READMISSION_TABULAR", "MVP_TABULAR", "TOON_TABULAR"}:
        return "READMISSION_TABULAR"
    elif mode in {"STRUCTURED", "READMISSION_STRUCTURED", "PYDANTIC", "STRUCTURED_OUTPUT"}:
        return "READMISSION_STRUCTURED"
    else:
        return "READMISSION_DISCHARGE"


# =============================================================================
# CUSTOM DSPy ADAPTER FOR MEDGEMMA (Local Transformers)
# =============================================================================
