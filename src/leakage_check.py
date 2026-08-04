# ============================================================
# LEAKAGE CHECK — BC5CDR Clinical NER System
# ============================================================
# NER-specific leakage checks:
#   1. Sentence-level overlap (same sentence in train + test)
#   2. Entity-level memorization (exact entity strings)
#   3. Label imbalance severity check
#   4. Token vocabulary overlap (train vs test OOV rate)
# ============================================================

import logging
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# MAIN LEAKAGE DETECTION
# ============================================================

def detect_leakage(
    train_sents: List[List[Tuple[str, str]]],
    test_sents:  List[List[Tuple[str, str]]],
    overlap_threshold:    float = 0.01,    # > 1% sentence overlap → warn
    entity_mem_threshold: float = 0.80,   # > 80% test entities seen in train → warn
) -> List[str]:
    """
    Runs NER-specific leakage heuristics.

    Args:
        train_sents        : list of training sentences
        test_sents         : list of test sentences
        overlap_threshold  : sentence-level overlap fraction to warn at
        entity_mem_threshold: entity memorization fraction to warn at

    Returns:
        List of warning strings. Empty = no leakage detected.
    """
    warnings = []

    # ── 1. Sentence-level overlap ─────────────────────────────
    warnings.extend(_check_sentence_overlap(train_sents, test_sents, overlap_threshold))

    # ── 2. Entity memorization ───────────────────────────────
    warnings.extend(_check_entity_memorization(train_sents, test_sents, entity_mem_threshold))

    # ── 3. Label leakage (target in features) ────────────────
    warnings.extend(_check_label_in_features(train_sents))

    # ── 4. Token OOV analysis ─────────────────────────────────
    _report_oov_rate(train_sents, test_sents)

    # ── 5. Label imbalance ───────────────────────────────────
    _check_label_imbalance(train_sents)

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no obvious leakage detected")

    return warnings


# ============================================================
# CHECK 1 — SENTENCE-LEVEL OVERLAP
# ============================================================

def _check_sentence_overlap(
    train_sents: List[List[Tuple[str, str]]],
    test_sents:  List[List[Tuple[str, str]]],
    threshold:   float
) -> List[str]:
    """
    Checks if any exact sentences appear in both train and test.
    Exact duplicate sentences → model has seen test labels during training.
    """
    warnings = []

    def sent_to_str(sent):
        return " ".join(token for token, _ in sent)

    train_strs = set(sent_to_str(s) for s in train_sents)
    test_strs  = [sent_to_str(s) for s in test_sents]

    overlaps = [s for s in test_strs if s in train_strs]
    overlap_rate = len(overlaps) / max(len(test_strs), 1)

    logger.info("Sentence overlap: %d / %d test sentences (%.2f%%)",
                len(overlaps), len(test_strs), overlap_rate * 100)

    if overlap_rate > threshold:
        warnings.append(
            f"[LEAKAGE] {len(overlaps)} test sentences ({overlap_rate:.1%}) "
            f"appear verbatim in training set → possible sentence-level leakage"
        )

    return warnings


# ============================================================
# CHECK 2 — ENTITY MEMORIZATION
# ============================================================

def _check_entity_memorization(
    train_sents: List[List[Tuple[str, str]]],
    test_sents:  List[List[Tuple[str, str]]],
    threshold:   float
) -> List[str]:
    """
    Checks if model may memorize entity strings.
    High overlap in named entities between train and test is expected
    in closed-domain medical NER — we warn if it's near-complete (> threshold).
    """
    warnings = []

    train_entities = _extract_entity_strings(train_sents)
    test_entities  = _extract_entity_strings(test_sents)

    if not test_entities:
        return warnings

    overlap = len(train_entities & test_entities) / len(test_entities)

    logger.info("Entity string overlap (train ∩ test / test): %.2f%%", overlap * 100)

    if overlap > threshold:
        warnings.append(
            f"[INFO] {overlap:.1%} of test entity strings appear in training data. "
            f"This is normal for closed-domain medical NER (drug names are finite). "
            f"Ensure evaluation uses held-out sentences, not just held-out tokens."
        )
        # Note: this is informational for NER — entity string overlap is expected,
        # unlike tabular data where feature = target overlap is true leakage

    return warnings


# ============================================================
# CHECK 3 — LABEL IN FEATURES
# ============================================================

def _check_label_in_features(
    train_sents: List[List[Tuple[str, str]]]
) -> List[str]:
    """
    In NER, the feature IS the token string. Leakage would occur if
    gold labels were somehow encoded in the token strings themselves.
    Checks for tokens that look like BIO labels.
    """
    warnings = []
    suspicious = []

    for sent in train_sents:
        for token, label in sent:
            # Token that looks like a BIO label is suspicious
            if token.startswith("B-") or token.startswith("I-"):
                suspicious.append(token)

    if suspicious:
        warnings.append(
            f"[LEAKAGE] {len(suspicious)} tokens look like BIO labels "
            f"(e.g. {suspicious[:3]}). Check data parsing — "
            f"columns may be swapped (label column read as token)."
        )

    return warnings


# ============================================================
# CHECK 4 — OOV RATE
# ============================================================

def _report_oov_rate(
    train_sents: List[List[Tuple[str, str]]],
    test_sents:  List[List[Tuple[str, str]]]
) -> None:
    """Reports token OOV rate (informational — no warning threshold)."""
    train_vocab = set(token.lower() for s in train_sents for token, _ in s)
    test_tokens = [token.lower() for s in test_sents for token, _ in s]

    if not test_tokens:
        return

    oov_tokens = [t for t in test_tokens if t not in train_vocab]
    oov_rate   = len(oov_tokens) / len(test_tokens)

    logger.info(
        "Token OOV rate: %d / %d test tokens (%.2f%%) not seen in training vocab",
        len(oov_tokens), len(test_tokens), oov_rate * 100
    )

    if oov_rate > 0.30:
        logger.warning(
            "High OOV rate (%.1f%%) — BioBERT subword tokenization "
            "will handle unknown tokens, but classical models may struggle",
            oov_rate * 100
        )


# ============================================================
# CHECK 5 — LABEL IMBALANCE
# ============================================================

def _check_label_imbalance(
    train_sents: List[List[Tuple[str, str]]]
) -> None:
    """Reports class imbalance (heavy O-label dominance is expected in NER)."""
    label_counts: Counter = Counter()
    for sent in train_sents:
        for _, label in sent:
            label_counts[label] += 1

    total      = sum(label_counts.values())
    o_fraction = label_counts.get("O", 0) / max(total, 1)

    logger.info("Label distribution — O: %.1f%% | entity tokens: %.1f%%",
                o_fraction * 100, (1 - o_fraction) * 100)

    if o_fraction > 0.95:
        logger.warning(
            "Extreme label imbalance: %.1f%% O-tokens. "
            "Use class weights or entity-level F1 for evaluation (not accuracy).",
            o_fraction * 100
        )

    # Top entity types
    entity_counts = {k: v for k, v in label_counts.items() if k.startswith("B-")}
    if entity_counts:
        logger.info("Entity counts: %s", dict(sorted(entity_counts.items(),
                                                      key=lambda x: -x[1])[:10]))


# ============================================================
# HELPERS
# ============================================================

def _extract_entity_strings(
    sentences: List[List[Tuple[str, str]]]
) -> set:
    """Extract all entity text spans from sentences."""
    entities   = set()
    for sent in sentences:
        current_tokens  = []
        current_entity  = None
        for token, label in sent:
            if label.startswith("B-"):
                if current_tokens and current_entity:
                    entities.add((" ".join(current_tokens), current_entity))
                current_tokens = [token]
                current_entity = label[2:]
            elif label.startswith("I-") and current_entity:
                current_tokens.append(token)
            else:
                if current_tokens and current_entity:
                    entities.add((" ".join(current_tokens), current_entity))
                current_tokens = []
                current_entity = None
        if current_tokens and current_entity:
            entities.add((" ".join(current_tokens), current_entity))

    return {text for text, _ in entities}
