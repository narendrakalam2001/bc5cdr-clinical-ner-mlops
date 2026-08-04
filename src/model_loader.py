# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — BC5CDR Clinical NER System
# ============================================================
#
# CHALLENGER MODEL SYSTEM (mirrors credit risk project exactly):
#   Real clinical AI teams never blindly replace a model in production.
#   Pharma companies / biomedical NLP teams follow formal champion vs challenger:
#
#   Champion  = current production model (latest_model.json)
#   Challenger = newly trained model
#
#   Promotion logic:
#     Challenger promoted ONLY if it beats Champion on ALL 3 gates:
#       1. F1 improvement  >= MIN_F1_IMPROVEMENT   (default 0.5%)
#       2. Overall F1      >= MIN_F1_THRESHOLD      (default 0.70)
#       3. Per-entity gap  <= MAX_ENTITY_GAP        (default 0.15)
#
#   If challenger loses → champion stays, challenger archived.
#   Full comparison saved to ner_models/challenger_log.json
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import (MODEL_DIR, MIN_F1_IMPROVEMENT,
                         MIN_F1_THRESHOLD, MAX_ENTITY_GAP)

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads ner_models/latest_model.json → loads model + returns info.

    For BERT models  → returns model path (load separately with HuggingFace)
    For CRF models   → returns joblib-loaded dict with crf + feature_extractor

    Returns:
        (model_info_dict, threshold)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            f"Run scripts/train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_type = registry.get("model_type", "crf")
    model_path = registry.get("model_path", "")
    threshold  = float(registry.get("confidence_threshold", 0.80))

    logger.info("Champion model: %s | type=%s | threshold=%.4f",
                registry.get("model_name"), model_type, threshold)

    if model_type == "bert":
        # Return path — caller loads with HuggingFace
        return registry, threshold

    elif model_type == "crf":
        full_path = os.path.join(MODEL_DIR, os.path.basename(model_path)) \
            if not os.path.isabs(model_path) else model_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"CRF model file not found: {full_path}")

        model_dict = joblib.load(full_path)
        logger.info("CRF champion loaded: %s", full_path)
        return model_dict, threshold

    else:
        return registry, threshold


def load_bert_champion():
    """
    Loads BioBERT champion model from saved path.
    Returns (model, tokenizer) or (None, None) if not found.
    """
    try:
        from transformers import (AutoTokenizer,
                                   AutoModelForTokenClassification)
        from src.config  import ID2LABEL, LABEL2ID, NUM_LABELS

        registry_path = os.path.join(MODEL_DIR, "latest_model.json")
        if not os.path.exists(registry_path):
            return None, None

        with open(registry_path) as f:
            registry = json.load(f)

        if registry.get("model_type") != "bert":
            return None, None

        model_path = registry.get("model_path", "")

                if not os.path.exists(model_path):
                    # Local folder missing — this happens on Render/fresh clones
                    # because ner_model_BioBERT_v1/ (400MB+) is NOT pushed to git.
                    # Instead we host it on HuggingFace Hub and download it here,
                    # once, at server startup.
                    hf_repo = os.getenv("HF_MODEL_REPO", "")
                    if hf_repo:
                        logger.info("Local model not found — downloading from HF Hub: %s", hf_repo)
                        try:
                            from huggingface_hub import snapshot_download
                            downloaded_path = snapshot_download(repo_id=hf_repo)
                            model_path = downloaded_path
                            logger.info("Downloaded champion model to: %s", model_path)
                        except Exception as e:
                            logger.error("Failed to download model from HF Hub (%s): %s", hf_repo, e)
                            return None, None
                    else:
                        logger.warning("BERT model path not found: %s", model_path)
                        logger.warning("Set HF_MODEL_REPO env var to auto-download from HuggingFace Hub")
                        return None, None

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels = NUM_LABELS,
            id2label   = ID2LABEL,
            label2id   = LABEL2ID,
        )

        logger.info("BERT champion loaded: %s", model_path)
        return model, tokenizer

    except Exception as e:
        logger.error("Failed to load BERT champion: %s", e)
        return None, None


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    """
    Reads current champion model card to get its metrics.
    Returns empty dict if no champion exists yet.

    Exact same pattern as credit risk project's _load_champion_metrics().
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # FIX: read model_card_path directly from registry — no brittle string parsing
    card_path = registry.get("model_card_path", "")

    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})

    return {
        "model_name":      card.get("model_details", {}).get("name", "unknown"),
        "f1":              float(metrics.get("test_f1",       0)),
        "precision":       float(metrics.get("test_precision",0)),
        "recall":          float(metrics.get("test_recall",   0)),
        "entity_variance": float(metrics.get("entity_f1_variance", 0)),
        "threshold":       float(card.get("thresholds", {}).get("confidence_threshold", 0.80)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_name:      str,
    challenger_f1:        float,
    challenger_precision: float,
    challenger_recall:    float,
    challenger_gap:       float,    # per-entity F1 variance
    challenger_path:      str,
    challenger_threshold: float,
    challenger_card_path: str = "",
    model_type:           str = "bert",
) -> dict:
    """
    Compares challenger vs current champion.
    Exact same 3-gate logic as credit risk project.

    Args:
        challenger_name      : model name e.g. 'BioBERT'
        challenger_f1        : challenger test F1
        challenger_precision : challenger precision
        challenger_recall    : challenger recall
        challenger_gap       : per-entity F1 variance (generalization gap proxy)
        challenger_path      : path to saved model
        challenger_threshold : confidence threshold
        challenger_card_path : path to model card JSON
        model_type           : 'bert' or 'crf'

    Returns:
        result dict with decision: 'PROMOTED' or 'REJECTED'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    # ── If no champion → auto-promote ────────────────────────
    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_name, challenger_path, challenger_threshold,
                         challenger_card_path, model_type)
        result = {
            "decision":              "PROMOTED",
            "reason":                "No existing champion — first model auto-promoted",
            "challenger_name":       challenger_name,
            "challenger_f1":         round(challenger_f1,        4),
            "challenger_precision":  round(challenger_precision,  4),
            "challenger_recall":     round(challenger_recall,     4),
            "champion_name":         None,
            "champion_f1":           None,
            "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_f1   = champion.get("f1",   0.0)
    champion_name = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-20s  F1=%.4f", champion_name,   champion_f1)
    logger.info("  Challenger: %-20s  F1=%.4f", challenger_name, challenger_f1)
    logger.info("=" * 55)

    # ── Gate 1: F1 improvement ───────────────────────────────
    gate1_f1_improvement = (challenger_f1 - champion_f1) >= MIN_F1_IMPROVEMENT

    # ── Gate 2: Minimum absolute F1 ──────────────────────────
    gate2_min_f1 = challenger_f1 >= MIN_F1_THRESHOLD

    # ── Gate 3: Per-entity variance (generalization gap) ─────
    gate3_gap = challenger_gap <= MAX_ENTITY_GAP

    gates_passed = gate1_f1_improvement and gate2_min_f1 and gate3_gap

    if gates_passed:
        decision = "PROMOTED"
        reason   = (
            f"Challenger beats champion: "
            f"F1 {champion_f1:.4f} → {challenger_f1:.4f} "
            f"(+{challenger_f1 - champion_f1:.4f})"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_name, challenger_path, challenger_threshold,
                         challenger_card_path, model_type)

    else:
        decision = "REJECTED"
        failed   = []
        if not gate1_f1_improvement:
            failed.append(
                f"F1 improvement {challenger_f1 - champion_f1:+.4f} < {MIN_F1_IMPROVEMENT}"
            )
        if not gate2_min_f1:
            failed.append(f"F1 {challenger_f1:.4f} < min threshold {MIN_F1_THRESHOLD}")
        if not gate3_gap:
            failed.append(f"entity variance {challenger_gap:.4f} > {MAX_ENTITY_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":              decision,
        "reason":                reason,
        "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":       challenger_name,
        "challenger_f1":         round(challenger_f1,        4),
        "challenger_precision":  round(challenger_precision,  4),
        "challenger_recall":     round(challenger_recall,     4),
        "challenger_gap":        round(challenger_gap,        4),
        "champion_name":         champion_name,
        "champion_f1":           round(champion_f1,           4),
        "gates": {
            "f1_improvement_passed": gate1_f1_improvement,
            "min_f1_passed":         gate2_min_f1,
            "gap_passed":            gate3_gap,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(
    model_name:      str,
    model_path:      str,
    threshold:       float,
    model_card_path: str = "",
    model_type:      str = "bert"
) -> None:
    """
    Updates latest_model.json.
    FIX: stores model_card_path directly — no brittle filename parsing.
    Mirrors credit risk _update_registry().
    """
    registry = {
        "model_name":          model_name,
        "model_path":          model_path,
        "model_type":          model_type,
        "confidence_threshold":round(threshold, 4),
        "model_card_path":     model_card_path,
        "updated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s (%s)", model_name, model_type)


def _save_challenger_log(result: dict) -> None:
    """Appends challenger comparison result to challenger_log.json."""
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved → %s", CHALLENGER_LOG)
