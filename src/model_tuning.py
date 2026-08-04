# ============================================================
# MODEL TUNING — BC5CDR Clinical NER System
# ============================================================
# Dataset : BC5CDR (tner/bc5cdr) — 15,000+ PubMed sentences
# Entities: Chemical + Disease
# Benchmark: BioBERT F1 ~0.87 on BC5CDR test set
#
# Models:
#   1. BioBERT (primary — SOTA biomedical NER)
#      dmis-lab/biobert-base-cased-v1.2 — pretrained on PubMed + PMC
#   2. CRF baseline (sklearn-crfsuite — fast, interpretable)
#      WindowFeatureExtractor(window=2) orthographic features
#   3. Majority class baseline (sanity check — all O)
#
# Architecture mirrors credit risk project pattern:
#   BioBERT → sequence model (BERT variants, scaled features)
#   CRF     → feature-based model (unscaled, interpretable)
#
# HuggingFace Trainer used for BioBERT (best practice).
# sklearn-crfsuite used for CRF (fast, no GPU needed).
# ============================================================

import logging
import numpy as np

from typing import Dict, List, Tuple, Optional, Any
from src.config import (RANDOM_STATE, LEARNING_RATE, NUM_EPOCHS,
                        WARMUP_RATIO, WEIGHT_DECAY, GRAD_CLIP,
                        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
                        MAX_SEQ_LENGTH, NUM_LABELS,
                        LABEL2ID, ID2LABEL, PRETRAINED_MODEL_NAME,
                        FALLBACK_MODEL_NAME)

logger = logging.getLogger(__name__)


# ============================================================
# BERT-BASED NER MODEL (Primary)
# ============================================================

def build_bert_ner_model(model_name: str = PRETRAINED_MODEL_NAME):
    """
    Loads BioBERT (or BERT) with token classification head.

    BioBERT (dmis-lab/biobert-base-cased-v1.2):
      - Pre-trained on PubMed abstracts + PMC full texts
      - Superior to vanilla BERT for clinical/biomedical NER
      - Industry standard for BC5CDR benchmark — F1 ~0.87

    Returns:
        (model, tokenizer) tuple
    """
    try:
        from transformers import (AutoTokenizer,
                                   AutoModelForTokenClassification)

        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model: %s | num_labels=%d", model_name, NUM_LABELS)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels = NUM_LABELS,
            id2label   = ID2LABEL,
            label2id   = LABEL2ID,
            ignore_mismatched_sizes = True,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded | parameters=%.1fM", total_params / 1e6)

        return model, tokenizer

    except Exception as e:
        logger.warning("Failed to load %s: %s", model_name, e)
        if model_name != FALLBACK_MODEL_NAME:
            logger.info("Falling back to: %s", FALLBACK_MODEL_NAME)
            return build_bert_ner_model(FALLBACK_MODEL_NAME)
        raise


# ============================================================
# HUGGINGFACE TRAINING ARGUMENTS
# ============================================================

def get_training_args(output_dir: str = "ner_models/bert_run"):
    """
    Returns HuggingFace TrainingArguments.
    Compatible with transformers 4.x and 5.x.
    """
    try:
        from transformers import TrainingArguments
        import transformers

        # Parse version safely
        ver_parts = transformers.__version__.split(".")
        major = int(ver_parts[0])
        minor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
        t_version = (major, minor)

        # Build args dict — only include args that exist in this version
        import torch
        use_cuda = torch.cuda.is_available()

        common = dict(
            output_dir                  = output_dir,
            num_train_epochs            = NUM_EPOCHS,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            learning_rate               = LEARNING_RATE,
            weight_decay                = WEIGHT_DECAY,
            max_grad_norm               = GRAD_CLIP,
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_f1",
            greater_is_better           = True,
            logging_steps               = 50,
            save_total_limit            = 2,
            seed                        = RANDOM_STATE,
            report_to                   = "none",
            dataloader_num_workers      = 0,
            fp16                        = use_cuda,   # mixed precision — halves VRAM usage on GPU
            dataloader_pin_memory       = use_cuda,   # avoids the "pin_memory...no accelerator" warning on CPU
        )
        logger.info("Device: %s | fp16=%s | batch=%d x accum=%d (effective=%d)",
                     "cuda" if use_cuda else "cpu", use_cuda,
                     BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
                     BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

        # eval_strategy renamed in transformers >= 4.41
        if t_version >= (4, 41):
            common["eval_strategy"] = "epoch"
        else:
            common["evaluation_strategy"] = "epoch"

        # warmup_ratio removed in transformers >= 5.2 — use warmup_steps instead
        if t_version < (5, 2):
            common["warmup_ratio"] = WARMUP_RATIO
        else:
            # Convert ratio to steps (approximate for small datasets)
            common["warmup_steps"] = 10

        # logging_dir removed in transformers >= 5.2
        if t_version < (5, 2):
            common["logging_dir"] = "logs/bert_train"

        args = TrainingArguments(**common)
        logger.info("TrainingArguments created | transformers=%s", transformers.__version__)
        return args

    except ImportError:
        logger.warning("transformers not installed — cannot create TrainingArguments")
        return None
    except TypeError as e:
        # Fallback: remove unknown args one by one
        logger.warning("TrainingArguments error (%s) — using minimal config", e)
        try:
            from transformers import TrainingArguments
            args = TrainingArguments(
                output_dir          = output_dir,
                num_train_epochs    = NUM_EPOCHS,
                learning_rate       = LEARNING_RATE,
                per_device_train_batch_size = BATCH_SIZE,
                per_device_eval_batch_size  = BATCH_SIZE,
                save_strategy       = "epoch",
                eval_strategy       = "epoch",
                load_best_model_at_end = True,
                metric_for_best_model  = "eval_f1",
                report_to           = "none",
                seed                = RANDOM_STATE,
            )
            return args
        except Exception as e2:
            logger.warning("TrainingArguments fallback also failed: %s", e2)
            return None


# ============================================================
# COMPUTE METRICS (for HuggingFace Trainer)
# ============================================================

def make_compute_metrics(tokenizer=None):
    """
    Returns a compute_metrics function for HuggingFace Trainer.
    Uses seqeval strict entity-level F1.
    """
    from src.metrics import compute_seqeval_metrics

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions    = np.argmax(logits, axis=-1)

        # Decode predictions + labels (skip -100 padding tokens)
        true_labels = []
        true_preds  = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_labels_sent = []
            true_preds_sent  = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                true_labels_sent.append(ID2LABEL.get(int(label_id), "O"))
                true_preds_sent.append( ID2LABEL.get(int(pred_id),  "O"))
            true_labels.append(true_labels_sent)
            true_preds.append(true_preds_sent)

        metrics = compute_seqeval_metrics(true_labels, true_preds, verbose=False)

        return {
            "precision": metrics["overall_precision"],
            "recall":    metrics["overall_recall"],
            "f1":        metrics["overall_f1"],
        }

    return compute_metrics


# ============================================================
# CLASSICAL BASELINE — CRF MODEL
# ============================================================

def build_crf_model(
    algorithm:     str   = "lbfgs",
    c1:            float = 0.1,
    c2:            float = 0.1,
    max_iterations: int  = 100,
    all_possible_transitions: bool = True
):
    """
    sklearn-crfsuite CRF model.
    Extremely fast, interpretable, no GPU needed.
    Good baseline before BERT — used in clinical NLP circa 2015-2020.

    Returns:
        CRF model instance
    """
    try:
        import sklearn_crfsuite

        crf = sklearn_crfsuite.CRF(
            algorithm                 = algorithm,
            c1                        = c1,
            c2                        = c2,
            max_iterations            = max_iterations,
            all_possible_transitions  = all_possible_transitions,
        )

        logger.info("CRF model built | algorithm=%s c1=%.2f c2=%.2f max_iter=%d",
                    algorithm, c1, c2, max_iterations)
        return crf

    except ImportError:
        logger.warning(
            "sklearn-crfsuite not installed — CRF baseline unavailable.\n"
            "  ➜ Install with:  pip install sklearn-crfsuite"
        )
        return None


# ============================================================
# CRF TRAINING
# ============================================================

def train_crf_model(
    crf_model,
    train_features: List[List[Dict]],
    train_labels:   List[List[str]],
    val_features:   Optional[List[List[Dict]]] = None,
    val_labels:     Optional[List[List[str]]]  = None,
):
    """
    Trains CRF model on pre-extracted window features.

    Args:
        crf_model      : sklearn_crfsuite.CRF instance
        train_features : list of sentences, each = list of feature dicts
        train_labels   : list of sentences, each = list of BIO strings

    Returns:
        fitted CRF model
    """
    if crf_model is None:
        logger.warning("CRF model is None — skipping training")
        return None

    logger.info("Training CRF | train_sentences=%d", len(train_features))

    crf_model.fit(train_features, train_labels)

    if val_features and val_labels:
        val_preds = crf_model.predict(val_features)
        from src.metrics import compute_seqeval_metrics
        val_metrics = compute_seqeval_metrics(val_labels, val_preds, verbose=True)
        logger.info("CRF val F1: %.4f", val_metrics["overall_f1"])

    logger.info("CRF training done | classes=%d", len(crf_model.classes_))
    return crf_model


# ============================================================
# CRF HYPERPARAMETER GRID
# ============================================================

CRF_PARAM_GRID = {
    "c1":             [0.01, 0.05, 0.1, 0.2],
    "c2":             [0.01, 0.05, 0.1, 0.2],
    "max_iterations": [100, 200],
}


def tune_crf_model(
    train_features: List[List[Dict]],
    train_labels:   List[List[str]],
    val_features:   List[List[Dict]],
    val_labels:     List[List[str]],
    n_trials:       int = 8,
) -> Tuple[Any, Dict]:
    """
    Simple grid search over CRF hyperparameters.
    Returns (best_crf_model, best_params).
    """
    from src.metrics import compute_seqeval_metrics

    best_f1     = -1.0
    best_model  = None
    best_params = {}

    np.random.seed(RANDOM_STATE)
    c1_vals   = np.random.choice(CRF_PARAM_GRID["c1"],   n_trials)
    c2_vals   = np.random.choice(CRF_PARAM_GRID["c2"],   n_trials)
    iter_vals = np.random.choice(CRF_PARAM_GRID["max_iterations"], n_trials)

    for i, (c1, c2, max_iter) in enumerate(zip(c1_vals, c2_vals, iter_vals)):
        params = {"c1": float(c1), "c2": float(c2), "max_iterations": int(max_iter)}
        logger.info("CRF trial %d/%d | %s", i + 1, n_trials, params)

        crf = build_crf_model(**params)
        if crf is None:
            break

        crf.fit(train_features, train_labels)
        preds   = crf.predict(val_features)
        metrics = compute_seqeval_metrics(val_labels, preds, verbose=False)
        f1      = metrics["overall_f1"]

        logger.info("  Val F1=%.4f", f1)

        if f1 > best_f1:
            best_f1     = f1
            best_model  = crf
            best_params = params

    logger.info("Best CRF params: %s | F1=%.4f", best_params, best_f1)
    return best_model, best_params


# ============================================================
# MAJORITY CLASS BASELINE
# ============================================================

class MajorityClassNER:
    """
    Trivial baseline: predict O for all tokens.
    Establishes minimum performance bar.
    """

    def __init__(self):
        self.prediction = "O"

    def fit(self, X, y=None):
        return self

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        return [["O"] * len(sent) for sent in sentences]

    def __repr__(self):
        return "MajorityClassNER(prediction='O')"
