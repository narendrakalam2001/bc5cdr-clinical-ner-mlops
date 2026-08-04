# ============================================================
# PREPROCESSING — BC5CDR Clinical NER System
# ============================================================
# Handles:
#   - Token-level feature engineering for classical NER baselines
#   - BERT tokenizer alignment (subword → word-level BIO labels)
#   - WindowFeatureExtractor (mirrors Clipper from credit risk)
#   - Dataset builder for HuggingFace Trainer
# ============================================================

import re
import logging
import numpy as np

from typing import List, Tuple, Dict, Optional, Any

from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.preprocessing   import LabelEncoder

from src.config import (LABEL2ID, ID2LABEL, MAX_SEQ_LENGTH,
                        WINDOW_SIZE, BIO_LABELS, RANDOM_STATE)

logger = logging.getLogger(__name__)


# ============================================================
# WINDOW FEATURE EXTRACTOR
# (mirrors Clipper from credit risk — custom sklearn transformer)
# ============================================================

class WindowFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts token-level features using a context window.
    Used for classical CRF / tree-based NER baselines.

    Features per token:
      - token text (lowercased)
      - is_upper, is_title, is_digit, has_hyphen, has_digit
      - prefix2, prefix3, suffix2, suffix3
      - Context window: prev_N and next_N token features

    get_feature_names_out() implemented (mirrors Clipper pattern).
    """

    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window

    def fit(self, X, y=None):
        # Stateless transformer — fit is no-op but needed for sklearn API
        self.n_features_in_ = len(self._single_token_features("sample"))
        return self

    def transform(self, X):
        """
        X: list of sentences, each = list of tokens (strings)
        Returns: list of feature dicts (one per token, all sentences flattened)
        """
        all_features = []
        for sentence in X:
            sent_features = self._extract_sentence_features(sentence)
            all_features.extend(sent_features)
        return all_features

    def transform_sentence(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Extract features for a single sentence (used at inference)."""
        return self._extract_sentence_features(tokens)

    def _extract_sentence_features(self, tokens: List[str]) -> List[Dict[str, Any]]:
        features = []
        for i, token in enumerate(tokens):
            feat = self._single_token_features(token, prefix="tok")

            # ── Context window ─────────────────────────────
            for offset in range(1, self.window + 1):
                # Previous tokens
                prev_idx = i - offset
                if prev_idx >= 0:
                    prev_feat = self._single_token_features(tokens[prev_idx], prefix=f"prev{offset}")
                    feat.update(prev_feat)
                else:
                    feat[f"prev{offset}_BOS"] = True

                # Next tokens
                next_idx = i + offset
                if next_idx < len(tokens):
                    next_feat = self._single_token_features(tokens[next_idx], prefix=f"next{offset}")
                    feat.update(next_feat)
                else:
                    feat[f"next{offset}_EOS"] = True

            features.append(feat)
        return features

    def _single_token_features(self, token: str, prefix: str = "tok") -> Dict[str, Any]:
        """Extract orthographic features for a single token."""
        return {
            f"{prefix}_lower":     token.lower(),
            f"{prefix}_is_upper":  token.isupper(),
            f"{prefix}_is_title":  token.istitle(),
            f"{prefix}_is_digit":  token.isdigit(),
            f"{prefix}_has_digit": any(c.isdigit() for c in token),
            f"{prefix}_has_hyphen":"-" in token,
            f"{prefix}_len":       len(token),
            f"{prefix}_prefix2":   token[:2].lower(),
            f"{prefix}_prefix3":   token[:3].lower(),
            f"{prefix}_suffix2":   token[-2:].lower(),
            f"{prefix}_suffix3":   token[-3:].lower(),
        }

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Mirrors Clipper.get_feature_names_out() for sklearn compatibility."""
        single = list(self._single_token_features("sample", "tok").keys())
        all_names = list(single)
        for offset in range(1, self.window + 1):
            all_names += [f"prev{offset}_" + k.split("_", 1)[1]
                          for k in single] + [f"prev{offset}_BOS"]
            all_names += [f"next{offset}_" + k.split("_", 1)[1]
                          for k in single] + [f"next{offset}_EOS"]
        return np.array(all_names, dtype=object)


# ============================================================
# BERT TOKENIZATION + LABEL ALIGNMENT
# ============================================================

def tokenize_and_align_labels(
    tokenizer,
    sentences: List[List[Tuple[str, str]]],
    max_length: int = MAX_SEQ_LENGTH,
    label_all_subwords: bool = False
) -> Dict[str, List]:
    """
    Tokenizes sentences with HuggingFace tokenizer and aligns
    BIO labels to subword tokens.

    Strategy:
      - First subword of each word → original label
      - Subsequent subwords → -100 (ignored in loss) if label_all_subwords=False
        OR → same label if label_all_subwords=True

    Args:
        tokenizer         : HuggingFace tokenizer (BioBERT / BERT)
        sentences         : List of [(token, label), ...] sentences
        max_length        : max sequence length for padding/truncation
        label_all_subwords: if True, all subwords get the entity label

    Returns:
        dict with: input_ids, attention_mask, token_type_ids, labels
    """
    all_input_ids      = []
    all_attention_mask = []
    all_token_type_ids = []
    all_labels         = []

    for sent in sentences:
        tokens = [t for t, _ in sent]
        labels = [l for _, l in sent]

        # ── Tokenize word by word (word_ids tracks alignment) ─
        encoding = tokenizer(
            tokens,
            is_split_into_words = True,
            truncation          = True,
            padding             = "max_length",
            max_length          = max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        word_ids      = encoding.word_ids()
        aligned_labels = []
        prev_word_id  = None

        for word_id in word_ids:
            if word_id is None:
                # [CLS] or [SEP] or [PAD]
                aligned_labels.append(-100)

            elif word_id != prev_word_id:
                # First subword of a word → assign original label
                aligned_labels.append(LABEL2ID.get(labels[word_id], 0))

            else:
                # Continuation subword
                if label_all_subwords:
                    aligned_labels.append(LABEL2ID.get(labels[word_id], 0))
                else:
                    aligned_labels.append(-100)

            prev_word_id = word_id

        all_input_ids.append(encoding["input_ids"])
        all_attention_mask.append(encoding["attention_mask"])
        all_token_type_ids.append(encoding.get("token_type_ids",
                                               [0] * max_length))
        all_labels.append(aligned_labels)

    return {
        "input_ids":       all_input_ids,
        "attention_mask":  all_attention_mask,
        "token_type_ids":  all_token_type_ids,
        "labels":          all_labels,
    }


# ============================================================
# PYTORCH DATASET CLASS
# ============================================================

def build_ner_dataset(encoded: Dict[str, List]):
    """
    Wraps encoded dict into a PyTorch Dataset for HuggingFace Trainer.
    Import torch here to avoid hard dependency at module level.
    """
    try:
        import torch
        from torch.utils.data import Dataset

        class NERDataset(Dataset):
            def __init__(self, enc):
                self.enc = enc

            def __len__(self):
                return len(self.enc["input_ids"])

            def __getitem__(self, idx):
                return {
                    "input_ids":       torch.tensor(self.enc["input_ids"][idx],       dtype=torch.long),
                    "attention_mask":  torch.tensor(self.enc["attention_mask"][idx],  dtype=torch.long),
                    "token_type_ids":  torch.tensor(self.enc["token_type_ids"][idx],  dtype=torch.long),
                    "labels":          torch.tensor(self.enc["labels"][idx],          dtype=torch.long),
                }

        return NERDataset(encoded)

    except ImportError:
        logger.warning("PyTorch not installed — returning raw encoded dict")
        return encoded


# ============================================================
# DECODE PREDICTIONS → ENTITY SPANS
# ============================================================

def decode_predictions(
    tokens:     List[str],
    pred_ids:   List[int],
    confidence: Optional[List[float]] = None
) -> List[Dict]:
    """
    Converts per-token BIO predictions → entity spans.

    Returns list of:
        {entity_type, start_token, end_token, tokens, text, confidence}

    Example:
        tokens   = ["Patient", "takes", "Aspirin", "500", "mg"]
        pred_ids = [0, 0, B-Drug, B-Strength, I-Strength]
        →
        [
          {entity_type: "Drug",     tokens: ["Aspirin"],    text: "Aspirin",  ...},
          {entity_type: "Strength", tokens: ["500", "mg"],  text: "500 mg",   ...}
        ]
    """
    entities = []
    current  = None

    for i, pred_id in enumerate(pred_ids):
        if i >= len(tokens):
            break

        label = ID2LABEL.get(pred_id, "O")
        conf  = confidence[i] if confidence else 1.0

        if label.startswith("B-"):
            # Save previous entity
            if current:
                entities.append(current)
            entity_type = label[2:]
            current = {
                "entity_type": entity_type,
                "start_token": i,
                "end_token":   i,
                "tokens":      [tokens[i]],
                "confidence":  [conf],
            }

        elif label.startswith("I-") and current:
            entity_type = label[2:]
            # Continue only if matches current entity
            if current["entity_type"] == entity_type:
                current["end_token"] = i
                current["tokens"].append(tokens[i])
                current["confidence"].append(conf)
            else:
                # Mismatched I- tag → treat as new B-
                entities.append(current)
                current = {
                    "entity_type": entity_type,
                    "start_token": i,
                    "end_token":   i,
                    "tokens":      [tokens[i]],
                    "confidence":  [conf],
                }
        else:
            # O label → save current entity
            if current:
                entities.append(current)
                current = None

    # Flush last entity
    if current:
        entities.append(current)

    # ── Post-process: add text + avg confidence ───────────────
    for ent in entities:
        ent["text"]       = " ".join(ent["tokens"])
        ent["confidence"] = float(np.mean(ent["confidence"]))

    return entities


# ============================================================
# PSI COMPUTATION (mirrors credit risk psi() function)
# ============================================================

def compute_token_distribution_psi(
    ref_labels: List[str],
    new_labels: List[str],
    buckets:    int = len(BIO_LABELS)
) -> float:
    """
    Computes PSI on label distributions between reference and new data.
    Exact same edge-based approach as credit risk project.

    PSI < 0.10  → stable
    PSI 0.10-0.20 → moderate shift
    PSI > 0.20  → major shift, retrain
    """
    from collections import Counter

    ref_counts = Counter(ref_labels)
    new_counts = Counter(new_labels)

    all_labels = sorted(set(list(ref_counts.keys()) + list(new_counts.keys())))

    ref_total  = sum(ref_counts.values()) + 1e-9
    new_total  = sum(new_counts.values()) + 1e-9

    psi_value  = 0.0
    for label in all_labels:
        exp_pct = ref_counts.get(label, 0) / ref_total
        act_pct = new_counts.get(label, 0) / new_total

        exp_pct = max(exp_pct, 1e-6)
        act_pct = max(act_pct, 1e-6)

        psi_value += (exp_pct - act_pct) * np.log(exp_pct / act_pct)

    return float(psi_value)
