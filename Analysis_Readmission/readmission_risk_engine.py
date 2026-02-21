#!/usr/bin/env python3
"""Rule-based 30-day readmission risk classification engine.

Reference implementation of the algorithm described in ALGORITHM_DESIGN.md.

Input: TOON lines (CLUSTER|Keyword|Value|Timestamp)
Output: Risk classification + days-to-readmission prediction

Usage:
    # From TOON string
    engine = ReadmissionRiskEngine()
    result = engine.score_from_toon(toon_text)
    print(result)

    # From TOON file
    result = engine.score_from_file("path/to/extraction.txt")

    # From JSONL training data
    results = engine.score_from_jsonl("dspy_fine_tuning/data/trainset_full.jsonl")
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParsedFact:
    cluster: str
    keyword: str
    value: Union[float, str]
    timestamp: str
    is_numeric: bool
    plausibility_ok: bool = True


@dataclass
class ClusterScore:
    cluster: str
    score: int
    max_score: int
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class InteractionResult:
    pattern_id: str
    pattern_name: str
    bonus: int
    description: str


@dataclass
class SurvivalCurve:
    """P(readmit by day t) for several horizons."""
    horizons: Dict[int, float]  # {7: 0.05, 14: 0.12, 21: 0.18, 30: 0.23}


@dataclass
class RiskResult:
    # Scores
    composite_score: int
    cluster_scores: Dict[str, ClusterScore]
    interaction_bonus: int
    interactions_triggered: List[InteractionResult]

    # Risk classification
    probability: float
    risk_category: str  # Low / Medium / High / Critical
    risk_color: str

    # Days prediction
    estimated_days: float
    days_bucket: str  # "0-7 days" / "8-14 days" / "15-30 days"
    survival_curve: SurvivalCurve

    # Explainability
    risk_factors: List[str]
    protective_factors: List[str]
    missing_clusters: List[str]
    data_completeness: float
    confidence: str  # high / medium / low

    # Raw data
    n_facts_parsed: int
    n_facts_dropped: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CLUSTERS = {
    "DEMOGRAPHICS", "VITALS", "LABS", "PROBLEMS", "SYMPTOMS",
    "MEDICATIONS", "PROCEDURES", "UTILIZATION", "DISPOSITION",
}

NUMERIC_CLUSTERS = {"VITALS", "LABS", "UTILIZATION"}

OBJECTIVE_CLUSTERS = {"DEMOGRAPHICS", "VITALS", "LABS", "UTILIZATION", "DISPOSITION"}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReadmissionRiskEngine:
    """Main entry point for readmission risk scoring."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"
        self._config_dir = config_dir
        self._scoring_rules = self._load_json("scoring_rules.json")
        self._problem_groups = self._load_json("snomed_problem_groups.json")["groups"]
        self._symptom_groups = self._load_json("symptom_urgency_groups.json")["groups"]

        # Build lookup indexes
        self._problem_synonym_index = self._build_synonym_index(self._problem_groups)
        self._symptom_synonym_index = self._build_synonym_index(self._symptom_groups)

        # Calibration parameters
        cal = self._scoring_rules["_meta"]["calibration"]
        self._alpha = cal["alpha"]
        self._beta = cal["beta"]

        # Days prediction parameters
        days_cfg = self._scoring_rules["DAYS_PREDICTION"]["models"]
        reg = days_cfg["regression"]["parameters"]
        self._d_max = reg["D_max"]
        self._gamma = reg["gamma"]

        surv = days_cfg["survival"]["parameters"]
        self._k_base = surv["k_base"]

    # -- Loading helpers ----------------------------------------------------

    def _load_json(self, filename: str) -> Dict[str, Any]:
        p = self._config_dir / filename
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _build_synonym_index(groups: List[Dict]) -> Dict[str, str]:
        """Map lowercase synonym → group id."""
        idx: Dict[str, str] = {}
        for g in groups:
            gid = g["id"]
            for syn in g.get("synonyms", []):
                key = syn.strip().lower()
                if key not in idx:
                    idx[key] = gid
        return idx

    def _match_to_group(
        self,
        keyword: str,
        synonym_index: Dict[str, str],
        groups: List[Dict],
    ) -> Optional[Dict]:
        """Smart matching: exact > word-boundary substring > raw substring.

        Avoids false matches like 'tia' in 'essential' by preferring
        word-boundary matches and longer synonyms.
        """
        kw_lower = keyword.strip().lower()

        # 1) Exact match (full keyword == synonym)
        gid = synonym_index.get(kw_lower)
        if gid:
            return self._group_by_id(groups, gid)

        # Tokenize keyword into words for word-boundary matching
        kw_words = set(re.split(r"[\s,;/\-()]+", kw_lower))

        # 2) Word-boundary match: synonym is a whole word within the keyword
        #    OR keyword starts/ends with the synonym as a distinct token
        best_wb_match: Optional[str] = None
        best_wb_len = 0

        # 3) Raw substring match (fallback, requires min 4 chars to avoid noise)
        best_sub_match: Optional[str] = None
        best_sub_len = 0

        for syn, gid in synonym_index.items():
            if syn not in kw_lower:
                continue

            # Check if it's a word-boundary match
            is_word_match = (
                syn in kw_words  # exact word token
                or kw_lower.startswith(syn + " ")
                or kw_lower.endswith(" " + syn)
                or (" " + syn + " ") in kw_lower
            )

            if is_word_match and len(syn) > best_wb_len:
                best_wb_match = gid
                best_wb_len = len(syn)
            elif not is_word_match and len(syn) >= 4 and len(syn) > best_sub_len:
                # Only use raw substring for synonyms >= 4 chars
                best_sub_match = gid
                best_sub_len = len(syn)

        # Prefer word-boundary matches over raw substring
        chosen = best_wb_match or best_sub_match
        if chosen:
            return self._group_by_id(groups, chosen)

        return None

    # -- Layer 1: Parser & Normalizer ----------------------------------------

    @staticmethod
    def _try_parse_float(value: str) -> Optional[float]:
        """Best-effort numeric parse.

        Stage2 should emit numeric-only values for numeric fields, but in practice
        we sometimes see light decoration like '3 days'. For scoring purposes we
        accept the first numeric token, but we avoid parsing ratios like '120/80'.
        """
        s = (value or "").strip()
        if not s:
            return None
        # Avoid BP-style ratios and similar formats.
        if "/" in s:
            return None
        # Fast path: pure float
        try:
            return float(s)
        except Exception:
            pass
        # Fallback: extract first numeric token
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _split_semantic_items(value: str, *, limit: int = 20) -> List[str]:
        """Split a semicolon/comma/newline separated list into normalized items."""
        raw = (value or "").strip()
        if not raw:
            return []
        parts: List[str] = []
        for seg in re.split(r"[;\n]+", raw):
            seg = seg.strip()
            if not seg:
                continue
            for item in seg.split(","):
                it = " ".join(item.strip().split())
                if not it:
                    continue
                parts.append(it.strip(" -"))
                if len(parts) >= limit:
                    break
            if len(parts) >= limit:
                break
        # Dedup while preserving order.
        out: List[str] = []
        seen: set[str] = set()
        for it in parts:
            k = it.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    @staticmethod
    def _strip_prefix(keyword: str, prefixes: List[str]) -> str:
        k = (keyword or "").strip()
        k_cf = k.casefold()
        for p in prefixes:
            p_cf = p.casefold()
            if k_cf.startswith(p_cf):
                k = k[len(p) :].strip()
                k_cf = k.casefold()
        return k

    @staticmethod
    def _normalize_discharge_disposition(value: str) -> str:
        """Normalize common discharge disposition variants to the scoring allowlist."""
        v = (value or "").strip()
        v_cf = v.casefold()
        if not v:
            return v
        # Canonical allowlist (scoring_rules.json): Home, Home with Services, Rehab, SNF, LTAC, Hospice, AMA
        if v_cf in {"home with service", "home w service", "home with svc", "home w/ service"}:
            return "Home with Services"
        if v_cf in {"home with services", "home w services", "home w/ services", "home health", "home health care"}:
            return "Home with Services"
        if v_cf in {"hospice residence", "hospice care"}:
            return "Hospice"
        return v

    @staticmethod
    def _normalize_mental_status(value: str) -> str:
        v = (value or "").strip()
        v_cf = v.casefold()
        if not v:
            return v
        if "alert" in v_cf and "orient" in v_cf:
            return "alert"
        if v_cf in {"a&o", "ao", "a/ox3", "a/ox4"}:
            return "alert"
        return v

    def parse_toon(self, toon_text: str) -> Tuple[Dict[str, List[ParsedFact]], int, int]:
        """Parse TOON text into structured facts.

        Returns (facts_by_cluster, n_parsed, n_dropped).
        """
        facts: Dict[str, List[ParsedFact]] = {}
        n_parsed = 0
        n_dropped = 0
        seen_objective: set = set()

        for raw_line in toon_text.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) != 4:
                n_dropped += 1
                continue

            cluster, keyword, value, timestamp = (p.strip() for p in parts)

            if cluster not in VALID_CLUSTERS:
                n_dropped += 1
                continue

            # Strip common semantic prefixes embedded in the keyword.
            if cluster == "PROBLEMS":
                keyword = self._strip_prefix(keyword, ["PMH:", "PMH/Comorbidities:", "Discharge Dx:", "Working Dx:", "Complication:", "Complications:"])
            elif cluster == "SYMPTOMS":
                keyword = self._strip_prefix(keyword, ["ADM:", "DC:"])

            # Expand common Stage2 aggregate semantic lines into per-item facts.
            # This makes the scorer robust to model drift like:
            #   PROBLEMS|Discharge Dx|CHF; COPD|Discharge
            # instead of emitting one line per diagnosis.
            if cluster == "PROBLEMS":
                kw_cf = keyword.strip().casefold()
                acute_keys = {"discharge dx", "working dx", "complication", "complications"}
                chronic_keys = {"pmh/comorbidities", "pmh", "comorbidities", "past medical history"}
                items = self._split_semantic_items(value)
                if kw_cf in acute_keys and items:
                    for it in items:
                        fact = ParsedFact(
                            cluster="PROBLEMS",
                            keyword=it,
                            value="acute",
                            timestamp="Discharge",
                            is_numeric=False,
                            plausibility_ok=True,
                        )
                        facts.setdefault("PROBLEMS", []).append(fact)
                        n_parsed += 1
                    continue
                if kw_cf in chronic_keys and items:
                    for it in items:
                        fact = ParsedFact(
                            cluster="PROBLEMS",
                            keyword=it,
                            value="chronic",
                            timestamp="Past",
                            is_numeric=False,
                            plausibility_ok=True,
                        )
                        facts.setdefault("PROBLEMS", []).append(fact)
                        n_parsed += 1
                    continue

            # Numeric parsing:
            # - Strictly numeric clusters MUST parse (else drop).
            # - Non-numeric clusters may still have numeric keywords (e.g. MEDICATIONS Medication Count,
            #   PROCEDURES Mechanical Ventilation days). Those should parse so scoring rules apply.
            is_numeric = False
            parsed_value: Union[float, str] = value

            kw_rules = self._scoring_rules.get(cluster, {}).get("keywords", {}).get(keyword, {})
            kw_type = kw_rules.get("type") if isinstance(kw_rules, dict) else None

            if cluster in NUMERIC_CLUSTERS:
                v = self._try_parse_float(value)
                if v is None:
                    n_dropped += 1
                    continue
                parsed_value = v
                is_numeric = True
            elif kw_type == "range":
                v = self._try_parse_float(value)
                if v is None:
                    n_dropped += 1
                    continue
                parsed_value = v
                is_numeric = True
            elif kw_type == "mixed":
                # Mixed: numeric is optional; keep as string if parsing fails.
                v = self._try_parse_float(value)
                if v is not None:
                    parsed_value = v
                    is_numeric = True

            # Plausibility check
            plausibility_ok = True
            if is_numeric:
                plausibility_ok = self._check_plausibility(cluster, keyword, parsed_value)

            # Dedup for objective clusters
            if cluster in OBJECTIVE_CLUSTERS:
                key = (cluster, keyword)
                if key in seen_objective:
                    # Keep the one with better timestamp
                    n_dropped += 1
                    continue
                seen_objective.add(key)

            fact = ParsedFact(
                cluster=cluster,
                keyword=keyword,
                value=parsed_value,
                timestamp=timestamp,
                is_numeric=is_numeric,
                plausibility_ok=plausibility_ok,
            )
            facts.setdefault(cluster, []).append(fact)
            n_parsed += 1

        return facts, n_parsed, n_dropped

    def _check_plausibility(self, cluster: str, keyword: str, value: float) -> bool:
        cluster_rules = self._scoring_rules.get(cluster, {}).get("keywords", {})
        kw_rules = cluster_rules.get(keyword, {})
        plaus = kw_rules.get("plausibility")
        if plaus:
            return plaus["min"] <= value <= plaus["max"]
        return True

    # -- Layer 2: Concept Mapper --------------------------------------------

    def map_problem_to_group(self, keyword: str) -> Optional[Dict]:
        """Map a PROBLEMS keyword to a SNOMED concept group."""
        return self._match_to_group(keyword, self._problem_synonym_index, self._problem_groups)

    def map_symptom_to_group(self, keyword: str) -> Optional[Dict]:
        """Map a SYMPTOMS keyword to an urgency group."""
        return self._match_to_group(keyword, self._symptom_synonym_index, self._symptom_groups)

    @staticmethod
    def _group_by_id(groups: List[Dict], gid: str) -> Optional[Dict]:
        for g in groups:
            if g["id"] == gid:
                return g
        return None

    # -- Layer 3: Cluster Scorers -------------------------------------------

    def _score_range_keyword(self, rules: Dict, value: float) -> Tuple[int, str]:
        """Score a numeric value using range rules. Returns (score, label)."""
        for r in rules.get("ranges", []):
            if r["min"] <= value <= r["max"]:
                return r["score"], r.get("label", "")
        return 0, ""

    def score_demographics(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["DEMOGRAPHICS"]["keywords"]
        score = 0
        factors: List[str] = []

        age_found = False
        for f in facts:
            if f.keyword == "Age" and f.is_numeric:
                age_found = True
                pts, label = self._score_range_keyword(rules["Age"], f.value)
                score += pts
                if pts > 0:
                    factors.append(f"Age {int(f.value)} ({label}, +{pts})")
            elif f.keyword == "Sex":
                val = str(f.value).lower()
                pts = rules["Sex"]["values"].get(val, 0)
                score += pts
                if pts > 0:
                    factors.append(f"Sex={val} (+{pts})")

        if not age_found:
            default = rules["Age"].get("missing_score", 2)
            score += default
            factors.append(f"Age missing (default +{default})")

        return ClusterScore("DEMOGRAPHICS", score, 10, factors)

    def score_vitals(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["VITALS"]["keywords"]
        score = 0
        factors: List[str] = []

        for f in facts:
            if not f.is_numeric or not f.plausibility_ok:
                continue
            kw_rules = rules.get(f.keyword)
            if not kw_rules or kw_rules.get("type") == "no_direct_score":
                continue
            pts, label = self._score_range_keyword(kw_rules, f.value)
            score += pts
            if pts > 0:
                factors.append(f"{f.keyword}={f.value} ({label}, +{pts})")

        return ClusterScore("VITALS", score, 25, factors)

    def score_labs(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["LABS"]["keywords"]
        score = 0
        factors: List[str] = []

        for f in facts:
            if not f.is_numeric or not f.plausibility_ok:
                continue
            kw_rules = rules.get(f.keyword)
            if not kw_rules:
                continue
            pts, label = self._score_range_keyword(kw_rules, f.value)
            score += pts
            if pts > 0:
                factors.append(f"{f.keyword}={f.value} ({label}, +{pts})")

        return ClusterScore("LABS", score, 30, factors)

    def score_problems(self, facts: List[ParsedFact]) -> ClusterScore:
        score = 0
        factors: List[str] = []
        active_groups: Dict[str, int] = {}  # group_id -> max weight

        include_values = {"chronic", "acute", "exist"}

        for f in facts:
            val = str(f.value).lower().strip()
            if val not in include_values:
                continue

            group = self.map_problem_to_group(f.keyword)
            if group:
                gid = group["id"]
                w = group["risk_weight"]
                if gid not in active_groups or w > active_groups[gid]:
                    active_groups[gid] = w
                    factors.append(f"{f.keyword} → {group['name']} (weight {w})")

        base_score = sum(active_groups.values())

        # Multimorbidity bonus
        n_groups = len(active_groups)
        mm_bonus = 0
        if n_groups > 3:
            mm_bonus = min(n_groups - 3, 5)
            factors.append(f"Multimorbidity: {n_groups} groups (+{mm_bonus})")

        score = min(base_score + mm_bonus, 40)
        return ClusterScore("PROBLEMS", score, 40, factors)

    def score_symptoms(self, facts: List[ParsedFact]) -> ClusterScore:
        sev_mult = {"severe": 1.5, "yes": 1.0, "no": 0.0}
        score = 0.0
        factors: List[str] = []
        active_groups: Dict[str, float] = {}
        active_count = 0

        for f in facts:
            val = str(f.value).lower().strip()
            mult = sev_mult.get(val, 0.0)
            if mult == 0.0:
                continue

            active_count += 1
            group = self.map_symptom_to_group(f.keyword)
            if group:
                gid = group["id"]
                w = group["risk_weight"] * mult
                if gid not in active_groups or w > active_groups[gid]:
                    active_groups[gid] = w
                    factors.append(f"{f.keyword}={val} → {group['name']} (+{w:.1f})")

        base_score = sum(active_groups.values())

        # Active symptom count bonus
        bonus = 0
        if active_count > 3:
            bonus = 2
            factors.append(f"Active symptoms: {active_count} (>3, +2)")

        score = min(int(round(base_score + bonus)), 15)
        return ClusterScore("SYMPTOMS", score, 15, factors)

    def score_medications(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["MEDICATIONS"]["keywords"]
        score = 0
        factors: List[str] = []
        med_count_val: Optional[float] = None

        for f in facts:
            kw_rules = rules.get(f.keyword)
            if not kw_rules:
                continue

            if kw_rules["type"] == "range" and f.is_numeric:
                pts, label = self._score_range_keyword(kw_rules, f.value)
                score += pts
                if f.keyword == "Medication Count":
                    med_count_val = f.value
                if pts > 0:
                    factors.append(f"{f.keyword}={f.value} ({label}, +{pts})")

            elif kw_rules["type"] == "categorical":
                val = str(f.value).lower().strip()
                pts = kw_rules["values"].get(val, 0)
                score += pts
                if pts > 0:
                    factors.append(f"{f.keyword}={val} (+{pts})")

        # Derived polypharmacy: if med_count >= 5 and Polypharmacy not already scored
        polypharmacy_scored = any("Polypharmacy" in f for f in factors)
        if med_count_val is not None and med_count_val >= 5 and not polypharmacy_scored:
            score += 3
            factors.append(f"Derived Polypharmacy (Med Count={int(med_count_val)} >=5, +3)")

        return ClusterScore("MEDICATIONS", min(score, 15), 15, factors)

    def score_procedures(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["PROCEDURES"]["keywords"]
        score = 0
        factors: List[str] = []
        specific_scored = False

        for f in facts:
            kw_rules = rules.get(f.keyword)
            if not kw_rules:
                continue

            if f.keyword == "Mechanical Ventilation":
                # Mixed type: numeric > 0 or categorical
                if f.is_numeric and f.value > 0:
                    score += kw_rules["score_if_any_positive"]
                    factors.append(f"Mechanical Ventilation={f.value} days (+{kw_rules['score_if_any_positive']})")
                    specific_scored = True
                elif str(f.value).lower().strip() != "no":
                    score += kw_rules["score_if_any_positive"]
                    factors.append(f"Mechanical Ventilation={f.value} (+{kw_rules['score_if_any_positive']})")
                    specific_scored = True

            elif f.keyword == "Dialysis":
                val = str(f.value).lower().strip()
                pts = kw_rules["values"].get(val, 0)
                score += pts
                if pts > 0:
                    factors.append(f"Dialysis={val} (+{pts})")
                    specific_scored = True

            elif f.keyword == "Surgery":
                val = str(f.value).lower().strip()
                pts = kw_rules["values"].get(val, 0)
                score += pts
                if pts > 0:
                    factors.append(f"Surgery={val} (+{pts})")
                    specific_scored = True

            elif f.keyword == "Any Procedure":
                # Only score if no specific procedure was scored
                pass  # handled below

        # Fallback: Any Procedure
        if not specific_scored:
            for f in facts:
                if f.keyword == "Any Procedure":
                    val = str(f.value).lower().strip()
                    pts = rules["Any Procedure"]["values"].get(val, 0)
                    score += pts
                    if pts > 0:
                        factors.append(f"Any Procedure={val} (generic fallback, +{pts})")
                    break

        return ClusterScore("PROCEDURES", min(score, 15), 15, factors)

    def score_utilization(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["UTILIZATION"]["keywords"]
        score = 0
        factors: List[str] = []

        for f in facts:
            if not f.is_numeric:
                continue
            kw_rules = rules.get(f.keyword)
            if not kw_rules:
                continue
            pts, label = self._score_range_keyword(kw_rules, f.value)
            score += pts
            if pts > 0:
                factors.append(f"{f.keyword}={f.value} ({label}, +{pts})")

        return ClusterScore("UTILIZATION", min(score, 20), 20, factors)

    def score_disposition(self, facts: List[ParsedFact]) -> ClusterScore:
        rules = self._scoring_rules["DISPOSITION"]["keywords"]
        score = 0
        factors: List[str] = []

        for f in facts:
            kw_rules = rules.get(f.keyword)
            if not kw_rules:
                continue
            val = str(f.value).strip()
            if f.keyword == "Discharge Disposition":
                val = self._normalize_discharge_disposition(val)
            elif f.keyword == "Mental Status":
                val = self._normalize_mental_status(val)
            # Try exact match first, then case-insensitive
            pts = kw_rules["values"].get(val, kw_rules["values"].get(val.lower(), 0))
            score += pts
            if pts > 0:
                factors.append(f"{f.keyword}={val} (+{pts})")

        return ClusterScore("DISPOSITION", min(score, 15), 15, factors)

    # -- Layer 4: Pattern Detector ------------------------------------------

    def detect_interactions(
        self,
        facts: Dict[str, List[ParsedFact]],
        cluster_scores: Dict[str, ClusterScore],
    ) -> List[InteractionResult]:
        """Detect cross-cluster clinical patterns."""
        results: List[InteractionResult] = []

        # Helper: get numeric value for a cluster/keyword
        def get_val(cluster: str, keyword: str) -> Optional[float]:
            for f in facts.get(cluster, []):
                if f.keyword == keyword and f.is_numeric:
                    return f.value
            return None

        def get_str(cluster: str, keyword: str) -> Optional[str]:
            for f in facts.get(cluster, []):
                if f.keyword == keyword:
                    return str(f.value).lower().strip()
            return None

        def has_symptom_group(group_id: str) -> bool:
            for f in facts.get("SYMPTOMS", []):
                val = str(f.value).lower().strip()
                if val in ("yes", "severe"):
                    g = self.map_symptom_to_group(f.keyword)
                    if g and g["id"] == group_id:
                        return True
            return False

        def has_problem_group(group_id: str) -> bool:
            for f in facts.get("PROBLEMS", []):
                val = str(f.value).lower().strip()
                if val in ("chronic", "acute", "exist"):
                    g = self.map_problem_to_group(f.keyword)
                    if g and g["id"] == group_id:
                        return True
            return False

        # --- Sepsis Pattern ---
        hr = get_val("VITALS", "Heart Rate")
        sbp = get_val("VITALS", "Systolic BP")
        rr = get_val("VITALS", "Respiratory Rate")
        wbc = get_val("LABS", "WBC")
        temp = get_val("VITALS", "Temperature")

        if hr is not None and hr > 100:
            has_hemodynamic = (sbp is not None and sbp < 100) or (rr is not None and rr > 22)
            has_infection = (
                (wbc is not None and (wbc > 12 or wbc < 4))
                or (temp is not None and temp > 100.4)
            )
            if has_hemodynamic and has_infection:
                results.append(InteractionResult(
                    "sepsis_pattern", "Sepsis / SIRS Pattern", 10,
                    f"HR={hr}, SBP={sbp}, RR={rr}, WBC={wbc}, Temp={temp}",
                ))

        # --- AKI Pattern ---
        cr = get_val("LABS", "Creatinine")
        bun = get_val("LABS", "BUN")
        k = get_val("LABS", "Potassium")
        na = get_val("LABS", "Sodium")
        bicarb = get_val("LABS", "Bicarbonate")

        if cr is not None and cr > 1.5 and bun is not None and bun > 30:
            has_electrolyte = (
                (k is not None and k > 5.0)
                or (na is not None and na < 135)
                or (bicarb is not None and bicarb < 22)
            )
            if has_electrolyte:
                results.append(InteractionResult(
                    "aki_pattern", "Acute Kidney Injury Pattern", 8,
                    f"Cr={cr}, BUN={bun}, K={k}, Na={na}, Bicarb={bicarb}",
                ))

        # --- Decompensated HF ---
        if has_problem_group("heart_failure"):
            has_decomp_sign = (
                has_symptom_group("edema_fluid")
                or has_symptom_group("respiratory_distress")
                or (bun is not None and bun > 40)
            )
            if has_decomp_sign:
                results.append(InteractionResult(
                    "decompensated_hf", "Decompensated Heart Failure", 8,
                    "Heart failure + fluid overload/dyspnea/elevated BUN",
                ))

        # --- Frailty Syndrome ---
        age = get_val("DEMOGRAPHICS", "Age")
        hgb = get_val("LABS", "Hemoglobin")
        mental = get_str("DISPOSITION", "Mental Status")
        disp = get_str("DISPOSITION", "Discharge Disposition")
        n_problem_groups = len(set(
            self.map_problem_to_group(f.keyword)["id"]
            for f in facts.get("PROBLEMS", [])
            if str(f.value).lower().strip() in ("chronic", "acute", "exist")
            and self.map_problem_to_group(f.keyword) is not None
        ))

        if age is not None and age > 75:
            frailty_count = 0
            if n_problem_groups >= 3:
                frailty_count += 1
            if hgb is not None and hgb < 10:
                frailty_count += 1
            if mental in ("confused", "lethargic"):
                frailty_count += 1
            if disp in ("snf", "ltac", "rehab"):
                frailty_count += 1
            if frailty_count >= 2:
                results.append(InteractionResult(
                    "frailty_syndrome", "Frailty Syndrome", 6,
                    f"Age={age}, problems={n_problem_groups}, Hgb={hgb}, mental={mental}, disp={disp}",
                ))

        # --- Unstable Discharge ---
        if disp == "ama":
            results.append(InteractionResult(
                "unstable_discharge", "Unstable Discharge (AMA)", 5,
                "Discharge Against Medical Advice",
            ))
        elif mental in ("confused", "lethargic") and disp in ("home", None):
            results.append(InteractionResult(
                "unstable_discharge", "Unstable Discharge (altered + Home)", 5,
                f"Mental={mental}, Disposition={disp}",
            ))

        # --- Respiratory Failure ---
        spo2 = get_val("VITALS", "SpO2")
        if spo2 is not None and spo2 < 92:
            has_resp = (rr is not None and rr > 24) or has_symptom_group("respiratory_distress")
            if has_resp:
                results.append(InteractionResult(
                    "respiratory_failure", "Respiratory Failure Pattern", 6,
                    f"SpO2={spo2}, RR={rr}",
                ))

        # --- Metabolic Crisis ---
        glucose = get_val("LABS", "Glucose")
        if glucose is not None and glucose > 300:
            has_metabolic = (
                (bicarb is not None and bicarb < 18)
                or (k is not None and k > 5.5)
            )
            if has_metabolic:
                results.append(InteractionResult(
                    "metabolic_crisis", "Metabolic Crisis (DKA/HHS)", 6,
                    f"Glucose={glucose}, Bicarb={bicarb}, K={k}",
                ))

        # --- Bleeding Risk ---
        plt = get_val("LABS", "Platelet")
        anticoag = get_str("MEDICATIONS", "Anticoagulation")
        if hgb is not None and hgb < 8:
            has_bleed_risk = (
                (plt is not None and plt < 100)
                or anticoag == "yes"
            )
            if has_bleed_risk:
                results.append(InteractionResult(
                    "bleeding_risk", "Active Bleeding Risk", 6,
                    f"Hgb={hgb}, Plt={plt}, Anticoag={anticoag}",
                ))

        return results

    # -- Layer 5: Risk Aggregator -------------------------------------------

    def _logistic(self, score: int) -> float:
        """Convert composite score to probability via logistic function."""
        z = self._alpha + self._beta * score
        return 1.0 / (1.0 + math.exp(-z))

    def _classify_risk(self, score: int) -> Tuple[str, str]:
        """Return (category, color) for a given composite score."""
        for cat in self._scoring_rules["_meta"]["risk_categories"]:
            if cat["score_min"] <= score <= cat["score_max"]:
                return cat["name"], cat["color"]
        return "Critical", "red"

    # -- Layer 6: Days Predictor --------------------------------------------

    def _predict_days(self, score: int) -> float:
        """Estimate days to readmission (point estimate)."""
        return max(1.0, self._d_max * math.exp(-self._gamma * score))

    def _predict_bucket(self, estimated_days: float) -> str:
        if estimated_days <= 7:
            return "0-7 days"
        elif estimated_days <= 14:
            return "8-14 days"
        else:
            return "15-30 days"

    def _predict_survival(self, score: int, p_30d: float) -> SurvivalCurve:
        """Compute P(readmit by day t) for several horizons."""
        k = self._k_base + 0.02 * (score - 30)
        k = max(0.5, k)  # floor to avoid degenerate cases

        horizons: Dict[int, float] = {}
        denom = 1.0 - math.exp(-k)
        if abs(denom) < 1e-9:
            denom = 1e-9

        for t in [7, 14, 21, 30]:
            f_t = (1.0 - math.exp(-(t / 30.0) * k)) / denom
            p_t = p_30d * f_t
            horizons[t] = round(min(max(p_t, 0.0), 1.0), 4)

        return SurvivalCurve(horizons=horizons)

    # -- Main Scoring Pipeline -----------------------------------------------

    def score(self, facts: Dict[str, List[ParsedFact]], n_parsed: int = 0, n_dropped: int = 0) -> RiskResult:
        """Run full scoring pipeline on parsed facts."""

        # Layer 3: Cluster scores
        cluster_scores: Dict[str, ClusterScore] = {}
        cluster_scores["DEMOGRAPHICS"] = self.score_demographics(facts.get("DEMOGRAPHICS", []))
        cluster_scores["VITALS"] = self.score_vitals(facts.get("VITALS", []))
        cluster_scores["LABS"] = self.score_labs(facts.get("LABS", []))
        cluster_scores["PROBLEMS"] = self.score_problems(facts.get("PROBLEMS", []))
        cluster_scores["SYMPTOMS"] = self.score_symptoms(facts.get("SYMPTOMS", []))
        cluster_scores["MEDICATIONS"] = self.score_medications(facts.get("MEDICATIONS", []))
        cluster_scores["PROCEDURES"] = self.score_procedures(facts.get("PROCEDURES", []))
        cluster_scores["UTILIZATION"] = self.score_utilization(facts.get("UTILIZATION", []))
        cluster_scores["DISPOSITION"] = self.score_disposition(facts.get("DISPOSITION", []))

        # Layer 4: Interaction detection
        interactions = self.detect_interactions(facts, cluster_scores)
        interaction_bonus = sum(i.bonus for i in interactions)

        # Layer 5: Aggregate
        composite = sum(cs.score for cs in cluster_scores.values()) + interaction_bonus
        probability = self._logistic(composite)
        category, color = self._classify_risk(composite)

        # Layer 6: Days prediction
        est_days = self._predict_days(composite)
        bucket = self._predict_bucket(est_days)
        survival = self._predict_survival(composite, probability)

        # Explainability
        risk_factors: List[str] = []
        protective_factors: List[str] = []
        for cs in cluster_scores.values():
            risk_factors.extend(cs.contributing_factors)

        # Identify protective factors (normal values in important clusters)
        for cluster in ["VITALS", "LABS"]:
            cs = cluster_scores[cluster]
            if cs.score == 0 and facts.get(cluster):
                protective_factors.append(f"Normal {cluster.lower()} at discharge")
        if cluster_scores["DISPOSITION"].score == 0 and facts.get("DISPOSITION"):
            protective_factors.append("Stable disposition (Home, alert)")

        for i in interactions:
            risk_factors.append(f"[PATTERN] {i.pattern_name} (+{i.bonus})")

        # Missing data
        missing_clusters = [c for c in VALID_CLUSTERS if c not in facts or not facts[c]]
        completeness = 1.0 - len(missing_clusters) / len(VALID_CLUSTERS)

        if completeness >= 0.7:
            confidence = "high"
        elif completeness >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return RiskResult(
            composite_score=composite,
            cluster_scores=cluster_scores,
            interaction_bonus=interaction_bonus,
            interactions_triggered=interactions,
            probability=round(probability, 4),
            risk_category=category,
            risk_color=color,
            estimated_days=round(est_days, 1),
            days_bucket=bucket,
            survival_curve=survival,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            missing_clusters=sorted(missing_clusters),
            data_completeness=round(completeness, 2),
            confidence=confidence,
            n_facts_parsed=n_parsed,
            n_facts_dropped=n_dropped,
        )

    # -- Convenience Methods ------------------------------------------------

    def score_from_toon(self, toon_text: str) -> RiskResult:
        """Score from raw TOON text."""
        facts, n_parsed, n_dropped = self.parse_toon(toon_text)
        return self.score(facts, n_parsed, n_dropped)

    def score_from_file(self, path: Union[str, Path]) -> RiskResult:
        """Score from a TOON text file."""
        text = Path(path).read_text(encoding="utf-8")
        return self.score_from_toon(text)

    def score_from_jsonl(self, path: Union[str, Path], limit: int = 0) -> List[Tuple[str, RiskResult]]:
        """Score all entries in a JSONL file (trainset_full format).

        Returns list of (hadm_id, RiskResult).
        """
        results: List[Tuple[str, RiskResult]] = []
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                obj = json.loads(line)
                hadm_id = str(obj.get("hadm_id", f"row_{i}"))
                completion = obj.get("completion", "")
                if completion:
                    result = self.score_from_toon(completion)
                    results.append((hadm_id, result))
        return results


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_result(result: RiskResult, hadm_id: str = "") -> str:
    """Format RiskResult as human-readable report."""
    lines: List[str] = []
    header = f"=== Readmission Risk Report"
    if hadm_id:
        header += f" (hadm_id: {hadm_id})"
    header += " ==="
    lines.append(header)
    lines.append("")

    # Summary
    lines.append(f"RISK: {result.risk_category} ({result.risk_color})")
    lines.append(f"Probability of 30-day readmission: {result.probability:.1%}")
    lines.append(f"Composite score: {result.composite_score}")
    lines.append(f"Confidence: {result.confidence} (data completeness: {result.data_completeness:.0%})")
    lines.append("")

    # Days prediction
    lines.append("--- Days-to-Readmission Prediction ---")
    lines.append(f"Point estimate: ~{result.estimated_days:.0f} days")
    lines.append(f"Bucket: {result.days_bucket}")
    lines.append("Survival curve:")
    for t, p in sorted(result.survival_curve.horizons.items()):
        lines.append(f"  P(readmit by day {t:2d}): {p:.1%}")
    lines.append("")

    # Cluster breakdown
    lines.append("--- Cluster Scores ---")
    for cluster in ["DEMOGRAPHICS", "VITALS", "LABS", "PROBLEMS", "SYMPTOMS",
                     "MEDICATIONS", "PROCEDURES", "UTILIZATION", "DISPOSITION"]:
        cs = result.cluster_scores.get(cluster)
        if cs:
            lines.append(f"  {cluster}: {cs.score}/{cs.max_score}")
    lines.append(f"  INTERACTIONS: +{result.interaction_bonus}")
    lines.append(f"  TOTAL: {result.composite_score}")
    lines.append("")

    # Risk factors
    if result.risk_factors:
        lines.append("--- Risk Factors ---")
        for rf in result.risk_factors:
            lines.append(f"  - {rf}")
        lines.append("")

    # Protective factors
    if result.protective_factors:
        lines.append("--- Protective Factors ---")
        for pf in result.protective_factors:
            lines.append(f"  + {pf}")
        lines.append("")

    # Triggered patterns
    if result.interactions_triggered:
        lines.append("--- Clinical Patterns Detected ---")
        for ix in result.interactions_triggered:
            lines.append(f"  [{ix.pattern_id}] {ix.pattern_name}: +{ix.bonus} pts")
            lines.append(f"    Evidence: {ix.description}")
        lines.append("")

    # Missing data
    if result.missing_clusters:
        lines.append(f"--- Missing Data ({len(result.missing_clusters)} clusters) ---")
        for mc in result.missing_clusters:
            lines.append(f"  ? {mc}")
        lines.append("")

    lines.append(f"Facts parsed: {result.n_facts_parsed}, dropped: {result.n_facts_dropped}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Rule-based 30-day readmission risk engine")
    sub = ap.add_subparsers(dest="cmd")

    # Score a single TOON file
    p_file = sub.add_parser("file", help="Score a single TOON file")
    p_file.add_argument("path", help="Path to TOON text file")

    # Score from JSONL
    p_jsonl = sub.add_parser("jsonl", help="Score all entries in a JSONL file")
    p_jsonl.add_argument("path", help="Path to JSONL file")
    p_jsonl.add_argument("--limit", type=int, default=0, help="Limit number of entries")
    p_jsonl.add_argument("--summary", action="store_true", help="Show summary statistics only")

    # Score from inline TOON text
    p_inline = sub.add_parser("inline", help="Score inline TOON text (pipe to stdin)")

    args = ap.parse_args()
    engine = ReadmissionRiskEngine()

    if args.cmd == "file":
        result = engine.score_from_file(args.path)
        print(format_result(result))

    elif args.cmd == "jsonl":
        results = engine.score_from_jsonl(args.path, limit=args.limit)

        if args.summary:
            scores = [r.composite_score for _, r in results]
            probs = [r.probability for _, r in results]
            categories = {}
            for _, r in results:
                categories[r.risk_category] = categories.get(r.risk_category, 0) + 1

            print(f"=== Summary ({len(results)} patients) ===")
            print(f"Score: mean={sum(scores)/len(scores):.1f}, "
                  f"min={min(scores)}, max={max(scores)}, "
                  f"median={sorted(scores)[len(scores)//2]}")
            print(f"P(readmit): mean={sum(probs)/len(probs):.1%}")
            print("Risk categories:")
            for cat in ["Low", "Medium", "High", "Critical"]:
                n = categories.get(cat, 0)
                pct = n / len(results) * 100 if results else 0
                print(f"  {cat}: {n} ({pct:.0f}%)")

            days = [r.estimated_days for _, r in results]
            print(f"Days estimate: mean={sum(days)/len(days):.1f}, "
                  f"min={min(days):.1f}, max={max(days):.1f}")
        else:
            for hadm_id, result in results:
                print(format_result(result, hadm_id))
                print("\n" + "=" * 60 + "\n")

    elif args.cmd == "inline":
        import sys
        toon_text = sys.stdin.read()
        result = engine.score_from_toon(toon_text)
        print(format_result(result))

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
