# ============================================================
# METRICS — BC5CDR Clinical NER System
# ============================================================
# BC5CDR specific:
#   - Seqeval strict entity-level F1 (Chemical + Disease)
#   - Per-entity F1 breakdown
#   - PSI drift monitoring
#   - Cost-sensitive evaluation (pharma/clinical context)
# ============================================================

import logging
import numpy as np
import pandas as pd
from typing import List, Dict

from src.config import (ID2LABEL, ENTITY_TYPES, HIGH_RISK_ENTITIES,
                        PSI_MODERATE, PSI_HIGH,
                        FN_COST_CHEMICAL, FN_COST_DISEASE, FP_COST_REVIEW)

logger = logging.getLogger(__name__)


# ============================================================
# SEQEVAL — STRICT ENTITY-LEVEL F1
# ============================================================

def compute_seqeval_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    verbose: bool = True
) -> Dict:
    """
    Computes strict entity-level F1 using seqeval.
    Industry standard for NER — BC5CDR papers use this exact metric.

    Returns dict with overall + per-entity precision, recall, F1.
    """
    try:
        from seqeval.metrics import (precision_score, recall_score,
                                     f1_score, classification_report)
        from seqeval.scheme import IOB2

        precision = precision_score(y_true, y_pred, mode="strict", scheme=IOB2)
        recall    = recall_score(y_true, y_pred, mode="strict", scheme=IOB2)
        f1        = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)
        report    = classification_report(
            y_true, y_pred, mode="strict", scheme=IOB2, output_dict=True
        )

        metrics = {
            "overall_precision": round(precision, 4),
            "overall_recall":    round(recall,    4),
            "overall_f1":        round(f1,        4),
            "per_entity":        {}
        }

        for entity, vals in report.items():
            if isinstance(vals, dict) and entity not in (
                "macro avg", "micro avg", "weighted avg"
            ):
                metrics["per_entity"][entity] = {
                    "precision": round(vals.get("precision", 0), 4),
                    "recall":    round(vals.get("recall",    0), 4),
                    "f1":        round(vals.get("f1-score",  0), 4),
                    "support":   int(vals.get("support",     0)),
                }

        if verbose:
            logger.info("=" * 55)
            logger.info("ENTITY-LEVEL EVALUATION (Strict Seqeval — BC5CDR)")
            logger.info("  Overall P=%.4f  R=%.4f  F1=%.4f",
                        precision, recall, f1)
            for ent, v in metrics["per_entity"].items():
                logger.info("  %-12s P=%.4f R=%.4f F1=%.4f support=%d",
                            ent, v["precision"], v["recall"],
                            v["f1"], v["support"])
            logger.info("=" * 55)

        return metrics

    except ImportError:
        logger.warning(
            "seqeval not installed — falling back to token-level metrics.\n"
            "  ➜ Install with:  pip install seqeval"
        )
        return _fallback_token_metrics(y_true, y_pred)


def _fallback_token_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> Dict:
    """Token-level F1 excluding O-class (used if seqeval unavailable)."""
    from sklearn.metrics import precision_recall_fscore_support

    flat_true = [l for sent in y_true for l in sent if l != "O"]
    flat_pred = [l for sent, ptrue in zip(y_pred, y_true)
                   for l, tl in zip(sent, ptrue) if tl != "O"]

    if not flat_true:
        return {"overall_f1": 0.0, "overall_precision": 0.0,
                "overall_recall": 0.0, "per_entity": {}}

    p, r, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average="micro", zero_division=0
    )
    return {
        "overall_precision": round(float(p),  4),
        "overall_recall":    round(float(r),  4),
        "overall_f1":        round(float(f1), 4),
        "per_entity":        {}
    }


# ============================================================
# TOKEN ACCURACY (informational only)
# ============================================================

def token_accuracy(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> float:
    """
    Token-level accuracy (biased by O-class — informational only).
    NEVER use as primary metric for NER.
    """
    correct = total = 0
    for true_sent, pred_sent in zip(y_true, y_pred):
        for t, p in zip(true_sent, pred_sent):
            correct += int(t == p)
            total   += 1
    return round(correct / max(total, 1), 4)


# ============================================================
# COST-SENSITIVE EVALUATION (BC5CDR pharma context)
# ============================================================

def cost_sensitive_ner_evaluation(
    y_true_sents: List[List[str]],
    y_pred_sents: List[List[str]],
) -> Dict:
    """
    Business cost model for BC5CDR:

    Missed Chemical (drug): ₹500 — wrong drug in ADE report = regulatory risk
    Missed Disease:         ₹300 — wrong diagnosis code = billing/insurance error
    False Positive:         ₹50  — unnecessary manual pharmacist review

    Based on pharma company clinical NLP workflows.
    """
    fn_chemical = fn_disease = fp_count = 0

    for true_sent, pred_sent in zip(y_true_sents, y_pred_sents):
        for true_label, pred_label in zip(true_sent, pred_sent):
            if true_label.startswith("B-") and pred_label == "O":
                # False negative — missed entity
                etype = true_label[2:]
                if etype == "Chemical":
                    fn_chemical += 1
                elif etype == "Disease":
                    fn_disease += 1
            elif pred_label.startswith("B-") and true_label == "O":
                # False positive
                fp_count += 1

    fn_chem_cost  = fn_chemical * FN_COST_CHEMICAL
    fn_dis_cost   = fn_disease  * FN_COST_DISEASE
    fp_cost       = fp_count    * FP_COST_REVIEW
    total_cost    = fn_chem_cost + fn_dis_cost + fp_cost

    result = {
        "missed_chemical_entities":  fn_chemical,
        "missed_disease_entities":   fn_disease,
        "false_positive_entities":   fp_count,
        "fn_chemical_cost_inr":      round(fn_chem_cost, 2),
        "fn_disease_cost_inr":       round(fn_dis_cost,  2),
        "fp_review_cost_inr":        round(fp_cost,      2),
        "total_estimated_cost_inr":  round(total_cost,   2),
    }

    logger.info(
        "Cost eval | FN_Chemical=%d FN_Disease=%d FP=%d | "
        "Chem_loss=₹%.0f Dis_loss=₹%.0f FP_cost=₹%.0f Total=₹%.0f",
        fn_chemical, fn_disease, fp_count,
        fn_chem_cost, fn_dis_cost, fp_cost, total_cost
    )

    return result


# ============================================================
# PSI — POPULATION STABILITY INDEX
# ============================================================

def psi_label_distribution(
    ref_labels: List[str],
    new_labels: List[str]
) -> float:
    """
    PSI on entity label distributions.
    Detects shift in Chemical vs Disease frequency.
    Same edge-based implementation as credit risk project.
    """
    from collections import Counter

    ref_counts = Counter(ref_labels)
    new_counts = Counter(new_labels)
    all_labels = sorted(set(list(ref_counts.keys()) + list(new_counts.keys())))

    ref_total = sum(ref_counts.values()) + 1e-9
    new_total = sum(new_counts.values()) + 1e-9
    psi_value = 0.0

    for label in all_labels:
        exp_pct = ref_counts.get(label, 0) / ref_total
        act_pct = new_counts.get(label, 0) / new_total
        exp_pct = max(exp_pct, 1e-6)
        act_pct = max(act_pct, 1e-6)
        psi_value += (exp_pct - act_pct) * np.log(exp_pct / act_pct)

    return float(psi_value)


# ============================================================
# PER-ENTITY F1 VARIANCE
# ============================================================

def per_entity_f1_variance(per_entity_metrics: Dict) -> float:
    """F1 variance across entity types — used in Champion-Challenger gate 3."""
    f1_scores = [v["f1"] for v in per_entity_metrics.values() if v["support"] > 0]
    if len(f1_scores) < 2:
        return 0.0
    return float(np.var(f1_scores))


# ============================================================
# ENTITY METRICS DATAFRAME
# ============================================================

def entity_metrics_to_dataframe(metrics: Dict) -> pd.DataFrame:
    """Converts per-entity metrics dict to DataFrame."""
    per_entity = metrics.get("per_entity", {})
    if not per_entity:
        return pd.DataFrame()

    rows = []
    for entity, vals in per_entity.items():
        rows.append({
            "Entity":    entity,
            "Precision": vals.get("precision", 0),
            "Recall":    vals.get("recall",    0),
            "F1":        vals.get("f1",        0),
            "Support":   vals.get("support",   0),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
