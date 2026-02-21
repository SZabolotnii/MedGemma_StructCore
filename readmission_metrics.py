"""
DSPy-free metrics for READMISSION KVT4 extraction evaluation.

This is extracted from `scripts/compare_baseline_vs_two_stage_single_doc.py` so that
other runners (e.g., two-stage structured pipeline) can import metrics without
pulling heavy inference dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# Keep semantics aligned with `scripts/compare_baseline_vs_two_stage_single_doc.py`.
NUMERIC_CLUSTERS = {"VITALS", "LABS", "UTILIZATION"}
SEMANTIC_CLUSTERS = {"PROBLEMS", "SYMPTOMS"}

ALL_CLUSTERS = [
    "DEMOGRAPHICS",
    "VITALS",
    "LABS",
    "PROBLEMS",
    "SYMPTOMS",
    "MEDICATIONS",
    "PROCEDURES",
    "UTILIZATION",
    "DISPOSITION",
]


def _keyword_norm(k: str) -> str:
    return " ".join((k or "").strip().split()).casefold()


def _semantic_value_present(v: str) -> bool | None:
    """
    Map PROBLEMS/SYMPTOMS values into a presence boolean for cluster-agnostic matching.
    - PROBLEMS: acute/chronic/exist -> present, not exist -> absent
    - SYMPTOMS: yes/severe -> present, no -> absent
    """
    s = (v or "").strip().casefold()
    if not s:
        return None
    if s in {"acute", "chronic", "exist"}:
        return True
    if s in {"not exist"}:
        return False
    if s in {"yes", "severe"}:
        return True
    if s in {"no"}:
        return False
    return None


def _parse_kvt4(line: str) -> Tuple[str, str, str, str] | None:
    parts = [p.strip() for p in (line or "").split("|")]
    if len(parts) != 4:
        return None
    c, k, v, t = parts
    if not (c and k and v and t):
        return None
    return c.strip().upper(), k, v, t


def _lines_to_map(lines: List[str]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for ln in lines or []:
        parsed = _parse_kvt4(ln)
        if not parsed:
            continue
        c, k, v, t = parsed
        out[(c, k)] = (v, t)
    return out


def _parse_float(s: str) -> float | None:
    try:
        return float(str(s).strip())
    except Exception:
        return None


def _values_match(cluster: str, pred_v: str, gt_v: str) -> bool:
    pv = str(pred_v).strip()
    gv = str(gt_v).strip()
    if (cluster or "").strip().upper() in NUMERIC_CLUSTERS:
        pf = _parse_float(pv)
        gf = _parse_float(gv)
        if pf is None or gf is None:
            return False
        tol = max(0.01, 0.10 * abs(gf))
        return abs(pf - gf) <= tol
    return pv.lower() == gv.lower()


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def compute_metrics(
    pred_lines: List[str],
    gt_lines: List[str],
    *,
    require_timestamp_match: bool,
    semantic_keyword_only_match: bool = False,
) -> Tuple[Metrics, Dict[str, Any]]:
    pred = _lines_to_map(pred_lines)
    gt = _lines_to_map(gt_lines)

    tp = fp = fn = 0
    matched: set[Tuple[str, str]] = set()
    details: Dict[str, Any] = {"tp": [], "fp": [], "fn": []}

    gt_semantic_by_kw: Dict[str, List[Tuple[Tuple[str, str], Tuple[str, str]]]] = {}
    if semantic_keyword_only_match:
        for (gc, gk), (gv, gt_t) in gt.items():
            if gc in SEMANTIC_CLUSTERS:
                kw = _keyword_norm(gk)
                gt_semantic_by_kw.setdefault(kw, []).append(((gc, gk), (gv, gt_t)))

    for (c, k), (pv, pt) in pred.items():
        if semantic_keyword_only_match and c in SEMANTIC_CLUSTERS:
            kw = _keyword_norm(k)
            candidates = gt_semantic_by_kw.get(kw) or []
            if not candidates:
                fp += 1
                details["fp"].append({"cluster": c, "keyword": k, "pred": {"V": pv, "T": pt}, "reason": "not_in_gt"})
                continue

            pred_present = _semantic_value_present(pv)
            best_match = None
            best_reason = None
            for (gc, gk), (gv, gt_t) in candidates:
                if (gc, gk) in matched:
                    continue
                if require_timestamp_match and (pt != gt_t):
                    best_reason = "timestamp_mismatch"
                    continue
                gt_present = _semantic_value_present(gv)
                if pred_present is None or gt_present is None:
                    best_reason = "value_mismatch"
                    continue
                if pred_present != gt_present:
                    best_reason = "value_mismatch"
                    continue
                best_match = ((gc, gk), (gv, gt_t))
                break

            if best_match is None:
                fp += 1
                details["fp"].append(
                    {
                        "cluster": c,
                        "keyword": k,
                        "pred": {"V": pv, "T": pt},
                        "reason": best_reason or "value_mismatch",
                        "note": "semantic_keyword_only_match_enabled",
                    }
                )
                continue

            (gc, gk), (gv, gt_t) = best_match
            tp += 1
            matched.add((gc, gk))
            details["tp"].append(
                {
                    "cluster": gc,
                    "keyword": gk,
                    "pred": {"C": c, "K": k, "V": pv, "T": pt},
                    "gt": {"C": gc, "K": gk, "V": gv, "T": gt_t},
                    "note": "semantic_keyword_only_match_enabled",
                }
            )
            continue

        gt_item = gt.get((c, k))
        if gt_item is None:
            fp += 1
            details["fp"].append({"cluster": c, "keyword": k, "pred": {"V": pv, "T": pt}, "reason": "not_in_gt"})
            continue

        gv, gt_t = gt_item
        if require_timestamp_match and (pt != gt_t):
            fp += 1
            details["fp"].append(
                {"cluster": c, "keyword": k, "pred": {"V": pv, "T": pt}, "gt": {"V": gv, "T": gt_t}, "reason": "timestamp_mismatch"}
            )
            continue
        if not _values_match(c, pv, gv):
            fp += 1
            details["fp"].append(
                {"cluster": c, "keyword": k, "pred": {"V": pv, "T": pt}, "gt": {"V": gv, "T": gt_t}, "reason": "value_mismatch"}
            )
            continue

        tp += 1
        matched.add((c, k))
        details["tp"].append({"cluster": c, "keyword": k, "pred": {"V": pv, "T": pt}, "gt": {"V": gv, "T": gt_t}})

    for (c, k), (gv, gt_t) in gt.items():
        if (c, k) in matched:
            continue
        fn += 1
        details["fn"].append({"cluster": c, "keyword": k, "gt": {"V": gv, "T": gt_t}})

    return Metrics(tp=tp, fp=fp, fn=fn), details


def compute_per_cluster_counts(details: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in ALL_CLUSTERS}
    for bucket in ("tp", "fp", "fn"):
        for item in details.get(bucket, []) or []:
            c = str(item.get("cluster") or "").strip().upper()
            if not c:
                continue
            if c not in counts:
                counts[c] = {"tp": 0, "fp": 0, "fn": 0}
            counts[c][bucket] += 1
    return counts


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def _safe_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return (tp / denom) if denom else 0.0


@dataclass(frozen=True)
class DownstreamMetricConfig:
    cluster_weights: Dict[str, float]
    lambda_fn: float
    lambda_fp: float
    critical_fn_clusters: List[str]
    critical_fp_clusters: List[str]
    gates: Dict[str, Dict[str, float]]


DEFAULT_DOWNSTREAM_CONFIG = DownstreamMetricConfig(
    cluster_weights={
        "VITALS": 0.20,
        "LABS": 0.15,
        "PROBLEMS": 0.15,
        "SYMPTOMS": 0.05,
        "MEDICATIONS": 0.05,
        "PROCEDURES": 0.05,
        "UTILIZATION": 0.20,
        "DISPOSITION": 0.15,
        "DEMOGRAPHICS": 0.00,
    },
    lambda_fn=0.02,
    lambda_fp=0.01,
    critical_fn_clusters=["DISPOSITION", "UTILIZATION", "PROBLEMS"],
    critical_fp_clusters=["VITALS", "LABS"],
    gates={"VITALS": {"min_recall": 0.85}},
)


def compute_downstream_score(
    details: Dict[str, Any],
    *,
    cfg: DownstreamMetricConfig,
) -> Tuple[float, Dict[str, Any]]:
    per_cluster = compute_per_cluster_counts(details)

    gate_report: Dict[str, Any] = {"passed": True, "failed": []}
    for cluster, rules in (cfg.gates or {}).items():
        c = str(cluster).strip().upper()
        if c not in per_cluster:
            continue
        tp = per_cluster[c]["tp"]
        fn = per_cluster[c]["fn"]
        recall = _safe_recall(tp, fn)
        min_recall = float(rules.get("min_recall", 0.0))
        if recall < min_recall:
            gate_report["passed"] = False
            gate_report["failed"].append({"cluster": c, "recall": recall, "min_recall": min_recall})

    if not gate_report["passed"]:
        return float("-inf"), {"gate": gate_report, "per_cluster": per_cluster}

    weighted_f1_sum = 0.0
    f1_by_cluster: Dict[str, float] = {}
    for cluster, w in (cfg.cluster_weights or {}).items():
        c = str(cluster).strip().upper()
        if c not in per_cluster:
            continue
        tp = per_cluster[c]["tp"]
        fp = per_cluster[c]["fp"]
        fn = per_cluster[c]["fn"]
        f1 = _safe_f1(tp, fp, fn)
        f1_by_cluster[c] = f1
        weighted_f1_sum += float(w) * f1

    critical_fn = sum(per_cluster.get(c, {}).get("fn", 0) for c in (x.upper() for x in cfg.critical_fn_clusters))
    critical_fp = sum(per_cluster.get(c, {}).get("fp", 0) for c in (x.upper() for x in cfg.critical_fp_clusters))
    penalty = cfg.lambda_fn * critical_fn + cfg.lambda_fp * critical_fp

    score = weighted_f1_sum - penalty
    report = {
        "gate": gate_report,
        "weighted_f1_sum": weighted_f1_sum,
        "critical_fn": critical_fn,
        "critical_fp": critical_fp,
        "penalty": penalty,
        "score": score,
        "f1_by_cluster": f1_by_cluster,
        "per_cluster": per_cluster,
        "config": {
            "cluster_weights": cfg.cluster_weights,
            "lambda_fn": cfg.lambda_fn,
            "lambda_fp": cfg.lambda_fp,
            "critical_fn_clusters": cfg.critical_fn_clusters,
            "critical_fp_clusters": cfg.critical_fp_clusters,
            "gates": cfg.gates,
        },
    }
    return score, report
