# ============================================================
# EVALUATION — BC5CDR Clinical NER System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib

from typing import List, Dict, Tuple, Any

from src.config import (MODEL_DIR, ID2LABEL, LABEL2ID, ENTITY_TYPES,
                        PRETRAINED_MODEL_NAME, MAX_SEQ_LENGTH)
from src.metrics import (compute_seqeval_metrics, cost_sensitive_ner_evaluation,
                          token_accuracy, entity_metrics_to_dataframe,
                          per_entity_f1_variance)

logger = logging.getLogger(__name__)


# ============================================================
# BERT INFERENCE
# ============================================================

def bert_predict(
    model, tokenizer,
    sentences:  List[List[str]],
    batch_size: int = 16,
    device:     str = None
) -> Tuple[List[List[str]], List[List[float]]]:
    """Runs BioBERT inference on list of token sentences."""
    if device is None:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    try:
        import torch
        model.eval()
        model.to(device)

        all_preds = []
        all_confs = []

        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i + batch_size]

            encoding = tokenizer(
                batch_sents,
                is_split_into_words    = True,
                truncation             = True,
                padding                = "max_length",
                max_length             = MAX_SEQ_LENGTH,
                return_attention_mask  = True,
                return_token_type_ids  = True,
                return_tensors         = "pt",
            )

            with torch.no_grad():
                outputs   = model(**{k: v.to(device) for k, v in encoding.items()
                                     if k != "token_type_ids" or
                                     hasattr(model.config, "type_vocab_size")})
                logits    = outputs.logits
                probs     = torch.softmax(logits, dim=-1)
                pred_ids  = torch.argmax(probs, dim=-1)
                max_probs = probs.max(dim=-1).values

            for j, sent in enumerate(batch_sents):
                word_ids    = encoding.word_ids(batch_index=j)
                token_preds = []
                token_confs = []
                prev_word   = None

                for k, word_id in enumerate(word_ids):
                    if word_id is None or word_id == prev_word:
                        prev_word = word_id
                        continue
                    token_preds.append(ID2LABEL.get(int(pred_ids[j][k]), "O"))
                    token_confs.append(float(max_probs[j][k]))
                    prev_word = word_id

                token_preds = token_preds[:len(sent)]
                token_confs = token_confs[:len(sent)]
                while len(token_preds) < len(sent):
                    token_preds.append("O")
                    token_confs.append(0.0)

                all_preds.append(token_preds)
                all_confs.append(token_confs)

            logger.info("Inference: %d / %d sentences",
                        min(i + batch_size, len(sentences)), len(sentences))

        return all_preds, all_confs

    except ImportError:
        logger.error("PyTorch not installed — cannot run BERT inference")
        return [["O"] * len(s) for s in sentences], [[0.0] * len(s) for s in sentences]


# ============================================================
# CRF INFERENCE
# ============================================================

def crf_predict(
    crf_model, feature_extractor,
    sentences: List[List[str]]
) -> Tuple[List[List[str]], List[List[float]]]:
    """Runs CRF inference on list of token sentences."""
    if crf_model is None:
        return [["O"] * len(s) for s in sentences], [[0.0] * len(s) for s in sentences]

    all_features = [feature_extractor.transform_sentence(sent) for sent in sentences]
    pred_labels  = crf_model.predict(all_features)

    try:
        marginals   = crf_model.predict_marginals(all_features)
        confidences = [
            [max(token_marginals.values()) for token_marginals in sent_marginals]
            for sent_marginals in marginals
        ]
    except Exception:
        confidences = [[1.0] * len(s) for s in pred_labels]

    return pred_labels, confidences


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    models:      Dict[str, Any],
    test_sents:  List,
    gold_labels: List[List[str]],
) -> pd.DataFrame:
    """Evaluates all trained models on test set."""
    results = []

    for name, info in models.items():
        pred_labels = info.get("predictions", [])
        if len(pred_labels) == 0:
            continue

        metrics = compute_seqeval_metrics(gold_labels, pred_labels, verbose=False)
        cost    = cost_sensitive_ner_evaluation(gold_labels, pred_labels)
        acc     = token_accuracy(gold_labels, pred_labels)

        results.append({
            "Model":              name,
            "F1":                 metrics["overall_f1"],
            "Precision":          metrics["overall_precision"],
            "Recall":             metrics["overall_recall"],
            "Token_Accuracy":     acc,
            "FN_Chemical":        cost["missed_chemical_entities"],
            "FN_Disease":         cost["missed_disease_entities"],
            "Total_Cost_INR":     cost["total_estimated_cost_inr"],
        })

    if not results:
        logger.warning("No model results to compare.")
        return pd.DataFrame(columns=["Model","F1","Precision","Recall",
                                      "Token_Accuracy","FN_Chemical",
                                      "FN_Disease","Total_Cost_INR"])

    df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    logger.info("Model comparison:\n%s", df.to_string())
    return df


# ============================================================
# SAVE MODELS
# ============================================================

def save_bert_model(model, tokenizer, model_name, model_dir=MODEL_DIR, version="v1"):
    save_path = os.path.join(model_dir, f"ner_model_{model_name}_{version}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("BERT model saved → %s", save_path)
    return save_path


def save_crf_model(crf_model, feature_extractor,
                   model_name="CRF", model_dir=MODEL_DIR, version="v1"):
    save_dict = {"crf": crf_model, "feature_extractor": feature_extractor,
                 "model_name": model_name, "version": version}
    path = os.path.join(model_dir, f"ner_model_{model_name}_{version}.joblib")
    joblib.dump(save_dict, path)
    logger.info("CRF model saved → %s", path)
    return path


def save_model_card(model_card, model_dir=MODEL_DIR, model_name="BioBERT"):
    path = os.path.join(model_dir, f"model_card_{model_name}_v1.json")
    with open(path, "w") as f:
        json.dump(model_card, f, indent=2)
    logger.info("Model card saved → %s", path)
    return path


# ============================================================
# BUILD MODEL CARD (Google Model Cards standard)
# ============================================================

def build_model_card(
    selected_name, train_size, val_size, test_size,
    metrics, cost_result, per_entity,
    threshold, model_params=None,
) -> Dict:
    """Google Model Cards standard JSON for BC5CDR NER."""
    return {
        "model_details": {
            "name":       f"BC5CDR Clinical NER — {selected_name}",
            "version":    "1.0",
            "type":       "Token Classification (Named Entity Recognition)",
            "framework":  "HuggingFace Transformers + PyTorch",
            "base_model": PRETRAINED_MODEL_NAME if "BERT" in selected_name.upper()
                          else "sklearn-crfsuite",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task":       "BC5CDR NER — Chemical and Disease extraction from PubMed abstracts",
            "benchmark":  "BC5CDR (BioCreative V CDR) — industry standard biomedical NER",
        },
        "intended_use": {
            "primary_use":  "Extract Chemical and Disease entities from biomedical literature",
            "use_cases":    ["Drug-disease relation extraction",
                             "Pharmacovigilance ADE detection",
                             "Clinical trial participant screening",
                             "Biomedical literature mining"],
            "out_of_scope": ["General-purpose NER", "Non-biomedical text",
                             "Real-time ICU monitoring"],
        },
        "dataset": {
            "source":          "BC5CDR — BioCreative V CDR Task (tner/bc5cdr on HuggingFace)",
            "domain":          "Biomedical / PubMed abstracts",
            "train_sentences": train_size,
            "val_sentences":   val_size,
            "test_sentences":  test_size,
            "entity_types":    ENTITY_TYPES,
            "benchmark_f1":    "~0.87 (BioBERT state-of-the-art on BC5CDR)",
        },
        "metrics": {
            "test_f1":              metrics.get("overall_f1",        0),
            "test_precision":       metrics.get("overall_precision", 0),
            "test_recall":          metrics.get("overall_recall",    0),
            "confidence_threshold": threshold,
            "per_entity_f1":        {k: v["f1"] for k, v in per_entity.items()},
        },
        "business_impact": {
            "missed_chemical_entities":  cost_result.get("missed_chemical_entities", 0),
            "missed_disease_entities":   cost_result.get("missed_disease_entities",  0),
            "total_estimated_cost_inr":  cost_result.get("total_estimated_cost_inr", 0),
            "note": "Cost based on pharma clinical NLP workflow estimates",
        },
        "pipeline_config": model_params or {},
        "thresholds":      {"confidence_threshold": threshold},
        "limitations": [
            "Trained on PubMed abstracts — may underperform on clinical EHR notes",
            "BC5CDR covers 2 entity types only (Chemical + Disease)",
            "Requires retraining for post-2024 drug approvals",
        ],
        "ethical_considerations": [
            "Model predictions must not replace physician judgment",
            "All Chemical entities in ADE context flagged for human review",
        ],
    }


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_ner_run(run_name, model_name, metrics, cost_result, model_params=None):
    """
    Logs a training run to MLflow.

    NOTE (Windows fix): we do NOT rely on MLflow's default tracking URI
    (bare "./mlruns"). MLflow builds that path itself via os.path.abspath()
    and does its own space-encoding, which is buggy on Windows when the
    user's home directory contains a space (e.g. "C:\\Users\\Narendra Kalam")
    — it ends up trying to touch a literal "...Narendra%20Kalam..." path and
    Windows throws "PermissionError: [WinError 5] Access is denied".

    Building the URI ourselves via pathlib's `.as_uri()` percent-encodes
    the path correctly, and MLflow decodes it back correctly on the read
    side — so this sidesteps the bug entirely, on any OS.
    """
    try:
        import mlflow
        from pathlib import Path

        mlruns_dir = Path(MODEL_DIR).resolve().parent / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        db_path = mlruns_dir / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

        mlflow.set_experiment("bc5cdr-clinical-ner")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name",  model_name)
            mlflow.log_param("base_model",  PRETRAINED_MODEL_NAME)
            mlflow.log_param("dataset",     "BC5CDR (tner/bc5cdr)")
            mlflow.log_param("max_seq_len", MAX_SEQ_LENGTH)
            if model_params:
                for k, v in model_params.items():
                    mlflow.log_param(k, v)
            mlflow.log_metric("test_f1",        metrics.get("overall_f1",        0))
            mlflow.log_metric("test_precision",  metrics.get("overall_precision", 0))
            mlflow.log_metric("test_recall",     metrics.get("overall_recall",    0))
            mlflow.log_metric("fn_chemical",     cost_result.get("missed_chemical_entities", 0))
            mlflow.log_metric("fn_disease",      cost_result.get("missed_disease_entities",  0))
            mlflow.log_metric("total_cost_inr",  cost_result.get("total_estimated_cost_inr", 0))
        logger.info("MLflow run logged: %s (tracking_uri=%s)", run_name, mlflow.get_tracking_uri())
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)