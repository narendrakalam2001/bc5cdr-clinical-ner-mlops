# ============================================================
# PREDICTION SERVICE — BC5CDR Clinical NER System
# Entities: Chemical (Drug) + Disease
# ============================================================

import re, logging
import numpy as np
from typing import List, Dict, Optional

from src.preprocessing import decode_predictions
from src.config        import (CONFIDENCE_THRESHOLD, HIGH_RISK_ENTITIES,
                                LOW_CONFIDENCE_ALERT)

logger = logging.getLogger(__name__)


def clinical_tokenize(text: str) -> List[str]:
    """
    Tokenizes biomedical text preserving chemical names and disease terms.
    Handles hyphenated compounds (e.g., 'Cyclosporin-induced'),
    abbreviations (e.g., 'TNF-alpha'), and dosage patterns.
    """
    token_pattern = re.compile(
        r"\d+\.?\d*\s*(?:mg|mcg|ml|g|kg|IU|mEq|units?|mmol|mL|L)\b"
        r"|[A-Za-z0-9]+-[A-Za-z0-9]+"   # hyphenated compounds
        r"|[A-Za-z0-9]+"
        r"|[^\s\w]",
        re.IGNORECASE
    )
    tokens = [m.group().strip() for m in token_pattern.finditer(text)
              if m.group().strip()]
    return tokens if tokens else text.split()


def predict_entities(
    text: str,
    model_state: Dict,
    threshold: float = CONFIDENCE_THRESHOLD,
    return_conf: bool = True,
) -> Dict:
    """Full entity extraction for one biomedical text."""
    tokens = clinical_tokenize(text)
    if not tokens:
        return {"input_text": text, "tokens": [], "entities": [],
                "requires_review": False, "alert": None}

    model_type = model_state.get("model_type", "none")

    if model_type == "bert":
        pred_labels, confidences = _bert_single_predict(
            tokens, model_state["model"], model_state["tokenizer"])
    elif model_type == "crf":
        pred_labels, confidences = _crf_single_predict(
            tokens, model_state["crf"], model_state["extractor"])
    else:
        pred_labels = ["O"] * len(tokens)
        confidences = [0.0]  * len(tokens)

    entities = decode_predictions(tokens, _labels_to_ids(pred_labels), confidences)

    if threshold > 0:
        entities = [e for e in entities if e["confidence"] >= threshold]

    entities = _post_process_entities(entities)
    requires_review, alert = _flag_for_review(entities, confidences)

    if not return_conf:
        for e in entities:
            e.pop("confidence", None)

    return {
        "input_text":      text,
        "tokens":          tokens,
        "token_labels":    pred_labels,
        "entities":        entities,
        "entity_count":    len(entities),
        "entity_types":    list(set(e["entity_type"] for e in entities)),
        "requires_review": requires_review,
        "alert":           alert,
    }


def predict_batch_entities(
    texts: List[str],
    model_state: Dict,
    threshold: float = CONFIDENCE_THRESHOLD,
    return_conf: bool = True,
) -> List[Dict]:
    results = []
    for text in texts:
        try:
            result = predict_entities(text, model_state, threshold, return_conf)
        except Exception as e:
            result = {"input_text": text, "entities": [],
                      "error": str(e), "requires_review": True}
        results.append(result)
    return results


def _bert_single_predict(tokens, model, tokenizer):
    try:
        import torch
        from src.config import MAX_SEQ_LENGTH, ID2LABEL

        model.eval()
        encoding = tokenizer(
            [tokens], is_split_into_words=True,
            truncation=True, padding="max_length",
            max_length=MAX_SEQ_LENGTH, return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs   = model(**{k: v for k, v in encoding.items()
                                 if k != "token_type_ids" or
                                 hasattr(model.config, "type_vocab_size")})
            probs     = torch.softmax(outputs.logits, dim=-1)
            pred_ids  = torch.argmax(probs, dim=-1)[0]
            max_probs = probs.max(dim=-1).values[0]

        word_ids = encoding.word_ids(batch_index=0)
        pred_labels, confidences = [], []
        prev_word = None
        for k, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word:
                prev_word = word_id
                continue
            if word_id >= len(tokens):
                break
            pred_labels.append(ID2LABEL.get(int(pred_ids[k]), "O"))
            confidences.append(float(max_probs[k]))
            prev_word = word_id

        while len(pred_labels) < len(tokens):
            pred_labels.append("O")
            confidences.append(0.0)

        return pred_labels[:len(tokens)], confidences[:len(tokens)]
    except Exception as e:
        logger.error("BERT inference error: %s", e)
        return ["O"] * len(tokens), [0.0] * len(tokens)


def _crf_single_predict(tokens, crf_model, feature_extractor):
    try:
        features    = [feature_extractor.transform_sentence(tokens)]
        pred_labels = crf_model.predict(features)[0]
        try:
            marginals   = crf_model.predict_marginals(features)[0]
            confidences = [max(m.values()) for m in marginals]
        except Exception:
            confidences = [1.0] * len(pred_labels)
        return pred_labels, confidences
    except Exception as e:
        logger.error("CRF inference error: %s", e)
        return ["O"] * len(tokens), [0.0] * len(tokens)


def _labels_to_ids(labels: List[str]) -> List[int]:
    from src.config import LABEL2ID
    return [LABEL2ID.get(l, 0) for l in labels]


def _post_process_entities(entities: List[Dict]) -> List[Dict]:
    """Post-process BC5CDR entities."""
    processed = []
    for ent in entities:
        ent = ent.copy()
        if ent["entity_type"] == "Chemical":
            ent["is_high_risk"]  = True
            ent["review_reason"] = "Chemical/Drug entity — verify in ADE context"
        elif ent["entity_type"] == "Disease":
            ent["is_high_risk"]  = True
            ent["review_reason"] = "Disease entity — verify diagnosis coding"
        processed.append(ent)
    return processed


def _flag_for_review(entities: List[Dict], confidences: List[float]):
    """Flag for human review."""
    avg_conf = float(np.mean(confidences)) if confidences else 1.0

    if any(e["entity_type"] == "Chemical" for e in entities) and \
       any(e["entity_type"] == "Disease"  for e in entities):
        alert = ("⚠️ Chemical-Disease pair detected — "
                 "potential adverse drug reaction. Pharmacovigilance review.")
        return True, alert

    if avg_conf < LOW_CONFIDENCE_ALERT:
        alert = (f"⚠️ Low confidence ({avg_conf:.2f}) — "
                 "verify entity extraction manually")
        return True, alert

    return False, None
