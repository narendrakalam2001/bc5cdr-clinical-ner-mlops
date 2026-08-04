# ============================================================
# PYTEST UNIT TESTS — BC5CDR Clinical NER System
# Run: pytest tests/test_pipeline_core.py -v
# ============================================================

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (WindowFeatureExtractor, decode_predictions,
                                compute_token_distribution_psi)
from src.data_loader   import (validate_sentences, sentences_to_dataframe,
                                compute_dataset_stats, _normalize_label,
                                _split_sentences)
from src.leakage_check import detect_leakage
from src.metrics       import (psi_label_distribution, cost_sensitive_ner_evaluation,
                                token_accuracy, per_entity_f1_variance)
from src.config        import (BIO_LABELS, LABEL2ID, ID2LABEL, ENTITY_TYPES,
                                MIN_F1_IMPROVEMENT, MIN_F1_THRESHOLD,
                                MAX_ENTITY_GAP, PSI_MODERATE, PSI_HIGH,
                                CONFIDENCE_THRESHOLD, NUM_LABELS)

# ── BC5CDR fixtures ───────────────────────────────────────────
@pytest.fixture
def sample_sentences():
    """BC5CDR style BIO sentences — Chemical + Disease."""
    return [
        [("Cyclosporin", "B-Chemical"), ("-", "O"),
         ("induced", "O"), ("nephrotoxicity", "B-Disease"), (".", "O")],
        [("Cisplatin", "B-Chemical"), ("causes", "O"),
         ("ototoxicity", "B-Disease"), ("and", "O"),
         ("hepatotoxicity", "B-Disease"), (".", "O")],
        [("Methotrexate", "B-Chemical"), ("and", "O"),
         ("Azathioprine", "B-Chemical"), ("in", "O"),
         ("rheumatoid", "B-Disease"), ("arthritis", "I-Disease"), (".", "O")],
    ]

@pytest.fixture
def sample_tokens():
    return [
        ["Cyclosporin", "-", "induced", "nephrotoxicity"],
        ["Cisplatin", "causes", "ototoxicity", "and", "hepatotoxicity"],
    ]

@pytest.fixture
def gold_labels():
    return [
        ["B-Chemical", "O", "O", "B-Disease"],
        ["B-Chemical", "O", "B-Disease", "O", "B-Disease"],
    ]

@pytest.fixture
def pred_labels():
    return [
        ["B-Chemical", "O", "O", "O"],
        ["B-Chemical", "O", "O", "O", "B-Disease"],
    ]


# ══ WINDOW FEATURE EXTRACTOR TESTS (6) ══════════════════════════
class TestWindowFeatureExtractor:

    def test_transform_returns_list_of_dicts(self, sample_tokens):
        fe = WindowFeatureExtractor(window=2)
        fe.fit(None)
        result = fe.transform(sample_tokens)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_transform_sentence_correct_length(self, sample_tokens):
        fe = WindowFeatureExtractor(window=2)
        fe.fit(None)
        tokens   = sample_tokens[0]
        features = fe.transform_sentence(tokens)
        assert len(features) == len(tokens)

    def test_required_feature_keys_present(self):
        fe = WindowFeatureExtractor(window=1)
        fe.fit(None)
        feats = fe.transform_sentence(["Cyclosporin"])
        assert "tok_lower"   in feats[0]
        assert "tok_is_upper" in feats[0]
        assert "tok_is_digit" in feats[0]
        assert "tok_prefix3"  in feats[0]
        assert "tok_suffix3"  in feats[0]

    def test_context_window_adds_prev_next(self):
        fe = WindowFeatureExtractor(window=1)
        fe.fit(None)
        feats = fe.transform_sentence(["take", "Cisplatin", "daily"])
        assert any("prev1" in k for k in feats[1].keys())
        assert any("next1" in k for k in feats[1].keys())

    def test_bos_eos_markers(self):
        fe = WindowFeatureExtractor(window=1)
        fe.fit(None)
        feats = fe.transform_sentence(["Cisplatin", "nephrotoxicity"])
        assert "prev1_BOS" in feats[0]
        assert "next1_EOS" in feats[1]

    def test_get_feature_names_returns_array(self):
        fe = WindowFeatureExtractor(window=2)
        fe.fit(None)
        names = fe.get_feature_names_out()
        assert isinstance(names, np.ndarray)
        assert len(names) > 0


# ══ DATA LOADER TESTS (5) ════════════════════════════════════════
class TestDataLoader:

    def test_validate_passes_valid_data(self, sample_sentences):
        validate_sentences(sample_sentences, split_name="test")

    def test_validate_raises_on_empty(self):
        with pytest.raises(ValueError, match="Empty"):
            validate_sentences([], split_name="test")

    def test_normalize_valid_bc5cdr_labels(self):
        assert _normalize_label("B-Chemical") == "B-Chemical"
        assert _normalize_label("I-Disease")  == "I-Disease"
        assert _normalize_label("O")           == "O"

    def test_normalize_unknown_to_o(self):
        assert _normalize_label("UNKNOWN") == "O"
        assert _normalize_label("OTHER")   == "O"

    def test_split_sentences_correct_sizes(self, sample_sentences):
        big_set = sample_sentences * 20   # 60 sentences
        train, test = _split_sentences(big_set, test_ratio=0.20)
        assert len(train) + len(test) == len(big_set)
        assert abs(len(test) / len(big_set) - 0.20) < 0.05


# ══ LEAKAGE DETECTION TESTS (4) ══════════════════════════════════
class TestLeakageDetection:

    def test_no_leakage_disjoint_splits(self, sample_sentences):
        train = sample_sentences[:2]
        test  = sample_sentences[2:]
        warnings = detect_leakage(train, test)
        assert not any("[LEAKAGE]" in w and "verbatim" in w for w in warnings)

    def test_overlap_detected(self, sample_sentences):
        warnings = detect_leakage(sample_sentences, sample_sentences)
        assert any("verbatim" in w for w in warnings)

    def test_empty_test_no_crash(self, sample_sentences):
        warnings = detect_leakage(sample_sentences, [])
        assert isinstance(warnings, list)

    def test_bio_labels_as_tokens_flagged(self):
        bad_sents = [[("B-Chemical", "O"), ("I-Disease", "O")]]
        warnings  = detect_leakage(bad_sents, [])
        assert any("B-" in w or "I-" in w for w in warnings)


# ══ METRICS TESTS (6) ════════════════════════════════════════════
class TestMetrics:

    def test_token_accuracy_perfect(self, gold_labels):
        acc = token_accuracy(gold_labels, gold_labels)
        assert acc == 1.0

    def test_token_accuracy_range(self, gold_labels, pred_labels):
        acc = token_accuracy(gold_labels, pred_labels)
        assert 0.0 <= acc <= 1.0

    def test_psi_identical_distributions_near_zero(self):
        labels = ["O"] * 50 + ["B-Chemical"] * 30 + ["B-Disease"] * 20
        psi    = psi_label_distribution(labels, labels)
        assert psi < 0.01

    def test_psi_different_distributions_exceeds_moderate(self):
        ref = ["O"] * 90 + ["B-Chemical"] * 10
        new = ["O"] * 10 + ["B-Chemical"] * 90
        psi = psi_label_distribution(ref, new)
        assert psi > PSI_MODERATE

    def test_cost_eval_returns_required_keys(self, gold_labels, pred_labels):
        result = cost_sensitive_ner_evaluation(gold_labels, pred_labels)
        for key in ["missed_chemical_entities", "missed_disease_entities",
                    "false_positive_entities", "total_estimated_cost_inr"]:
            assert key in result

    def test_per_entity_variance_returns_float(self):
        per_entity = {
            "Chemical": {"f1": 0.88, "support": 200},
            "Disease":  {"f1": 0.85, "support": 300},
        }
        variance = per_entity_f1_variance(per_entity)
        assert isinstance(variance, float)
        assert variance >= 0.0


# ══ PREPROCESSING TESTS (5) ══════════════════════════════════════
class TestPreprocessing:

    def test_decode_chemical_entity(self):
        tokens   = ["Cyclosporin", "-", "induced", "nephrotoxicity"]
        pred_ids = [LABEL2ID["B-Chemical"], LABEL2ID["O"],
                    LABEL2ID["O"], LABEL2ID["B-Disease"]]
        entities = decode_predictions(tokens, pred_ids)
        types    = [e["entity_type"] for e in entities]
        assert "Chemical" in types
        assert "Disease"  in types

    def test_decode_bi_span_merges_correctly(self):
        tokens   = ["rheumatoid", "arthritis"]
        pred_ids = [LABEL2ID["B-Disease"], LABEL2ID["I-Disease"]]
        entities = decode_predictions(tokens, pred_ids)
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "Disease"
        assert entities[0]["text"]        == "rheumatoid arthritis"

    def test_decode_empty_input_returns_empty(self):
        assert decode_predictions([], []) == []

    def test_decode_all_o_returns_no_entities(self):
        tokens   = ["The", "patient", "recovered"]
        pred_ids = [LABEL2ID["O"]] * 3
        entities = decode_predictions(tokens, pred_ids)
        assert entities == []

    def test_psi_identical_labels_near_zero(self):
        labels = ["O", "B-Chemical", "B-Disease", "I-Disease", "O"]
        psi    = compute_token_distribution_psi(labels, labels)
        assert psi < 0.01


# ══ CONFIG TESTS (4) ════════════════════════════════════════════
class TestConfig:

    def test_bc5cdr_entity_types(self):
        """BC5CDR has exactly 2 entity types: Chemical + Disease."""
        assert "Chemical" in ENTITY_TYPES
        assert "Disease"  in ENTITY_TYPES
        assert len(ENTITY_TYPES) == 2

    def test_bio_labels_contain_all_entity_types(self):
        for ent in ENTITY_TYPES:
            assert f"B-{ent}" in BIO_LABELS, f"Missing B-{ent}"
            assert f"I-{ent}" in BIO_LABELS, f"Missing I-{ent}"

    def test_label2id_and_id2label_are_inverses(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_num_labels_equals_bio_labels_count(self):
        assert NUM_LABELS == len(BIO_LABELS)
        assert NUM_LABELS == 5   # O + B-Chemical + I-Chemical + B-Disease + I-Disease
