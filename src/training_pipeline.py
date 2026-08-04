# ============================================================
# TRAINING PIPELINE — BC5CDR Clinical NER System
# ============================================================
# Dataset : BC5CDR (tner/bc5cdr) — 15,000+ sentences
# Entities: Chemical + Disease
# Models  : BioBERT (primary) + CRF baseline
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config         import (RANDOM_STATE, MODEL_DIR, PRETRAINED_MODEL_NAME,
                                CONFIDENCE_THRESHOLD, ENTITY_TYPES, BIO_LABELS,
                                LABEL2ID, ID2LABEL, NUM_LABELS, MAX_SEQ_LENGTH,
                                BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE)
from src.data_loader    import (load_bc5cdr_dataset, validate_sentences,
                                compute_dataset_stats, sentences_to_dataframe)
from src.preprocessing  import (tokenize_and_align_labels, build_ner_dataset,
                                WindowFeatureExtractor, decode_predictions,
                                compute_token_distribution_psi)
from src.leakage_check  import detect_leakage
from src.model_tuning   import (build_bert_ner_model, get_training_args,
                                make_compute_metrics, build_crf_model,
                                train_crf_model, tune_crf_model, MajorityClassNER)
from src.metrics        import (compute_seqeval_metrics, cost_sensitive_ner_evaluation,
                                token_accuracy, entity_metrics_to_dataframe,
                                per_entity_f1_variance, psi_label_distribution)
from src.evaluation     import (bert_predict, crf_predict, evaluate_models,
                                save_bert_model, save_crf_model, save_model_card,
                                build_model_card, mlflow_log_ner_run)
from src.model_loader   import run_challenger_comparison

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


def run_training(data_dir: str = "data", skip_bert: bool = False):
    """
    Full BC5CDR Clinical NER training pipeline.

    Args:
        data_dir  : directory for local data files (fallback only)
        skip_bert : if True, run CRF only (fast, no GPU needed)
    """

    # ─────────────────────────────────────────────────────────
    # 1. LOAD + VALIDATE DATA
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 1: Loading BC5CDR dataset")

    splits = load_bc5cdr_dataset(data_dir=data_dir)

    train_sents = splits["train"]
    val_sents   = splits["val"]
    test_sents  = splits["test"]

    train_tokens_list = [[t for t, _ in s] for s in train_sents]
    val_tokens_list   = [[t for t, _ in s] for s in val_sents]
    test_tokens_list  = [[t for t, _ in s] for s in test_sents]

    train_gold = [[l for _, l in s] for s in train_sents]
    val_gold   = [[l for _, l in s] for s in val_sents]
    test_gold  = [[l for _, l in s] for s in test_sents]

    logger.info("Dataset | train=%d  val=%d  test=%d sentences",
                len(train_sents), len(val_sents), len(test_sents))

    entity_stats = compute_dataset_stats(train_sents)
    logger.info("Entity distribution:\n%s", entity_stats.to_string())

    # ─────────────────────────────────────────────────────────
    # 2. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 2: Leakage detection")

    leak_warnings = detect_leakage(train_sents, test_sents)
    if not leak_warnings:
        logger.info("No obvious leakage detected")

    # ─────────────────────────────────────────────────────────
    # 3. MAJORITY CLASS BASELINE
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 3: Majority class baseline")

    majority_clf   = MajorityClassNER()
    majority_preds = majority_clf.predict(test_tokens_list)
    majority_metrics = compute_seqeval_metrics(test_gold, majority_preds, verbose=False)
    logger.info("Majority baseline F1: %.4f (should be ~0.0)",
                majority_metrics["overall_f1"])

    all_model_results = {}

    # ─────────────────────────────────────────────────────────
    # 4. CRF BASELINE
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 4: Training CRF baseline")

    feature_extractor = WindowFeatureExtractor(window=2)
    feature_extractor.fit(None)

    train_features = [feature_extractor.transform_sentence(s) for s in train_tokens_list]
    val_features   = [feature_extractor.transform_sentence(s) for s in val_tokens_list]
    test_features  = [feature_extractor.transform_sentence(s) for s in test_tokens_list]

    crf_model, crf_params = tune_crf_model(
        train_features, train_gold,
        val_features,   val_gold,
        n_trials=8,
    )

    if crf_model:
        crf_preds, crf_confs = crf_predict(crf_model, feature_extractor, test_tokens_list)
        crf_metrics = compute_seqeval_metrics(test_gold, crf_preds, verbose=True)
        crf_acc     = token_accuracy(test_gold, crf_preds)

        all_model_results["CRF"] = {
            "predictions": crf_preds,
            "confidences": crf_confs,
            "metrics":     crf_metrics,
            "params":      crf_params,
        }
        logger.info("CRF | F1=%.4f  P=%.4f  R=%.4f  TokenAcc=%.4f",
                    crf_metrics["overall_f1"], crf_metrics["overall_precision"],
                    crf_metrics["overall_recall"], crf_acc)

    # ─────────────────────────────────────────────────────────
    # 5. BIOBERT TRAINING
    # ─────────────────────────────────────────────────────────
    if not skip_bert:
        logger.info("=" * 65)
        logger.info("STEP 5: Training BioBERT NER model on BC5CDR")

        try:
            # Check accelerate
            try:
                import accelerate
            except ImportError:
                raise RuntimeError(
                    "accelerate not installed.\n"
                    "  ➜ Fix: pip install \"accelerate>=1.1.0\""
                )

            from transformers import Trainer, EarlyStoppingCallback

            model, tokenizer = build_bert_ner_model(PRETRAINED_MODEL_NAME)
            training_args    = get_training_args(output_dir="ner_models/bert_run")
            compute_metrics  = make_compute_metrics(tokenizer)

            if training_args is None:
                raise RuntimeError("TrainingArguments could not be created.")

            logger.info("Tokenizing + aligning labels ...")
            train_encoded = tokenize_and_align_labels(tokenizer, train_sents)
            val_encoded   = tokenize_and_align_labels(tokenizer, val_sents)

            train_dataset = build_ner_dataset(train_encoded)
            val_dataset   = build_ner_dataset(val_encoded)

            trainer = Trainer(
                model           = model,
                args            = training_args,
                train_dataset   = train_dataset,
                eval_dataset    = val_dataset,
                compute_metrics = compute_metrics,
                callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
            )

            logger.info("Starting BioBERT training | epochs=%d lr=%.0e",
                        NUM_EPOCHS, LEARNING_RATE)
            trainer.train()

            bert_preds, bert_confs = bert_predict(model, tokenizer, test_tokens_list)
            bert_metrics = compute_seqeval_metrics(test_gold, bert_preds, verbose=True)
            bert_acc     = token_accuracy(test_gold, bert_preds)

            all_model_results["BioBERT"] = {
                "predictions": bert_preds,
                "confidences": bert_confs,
                "metrics":     bert_metrics,
                "model":       model,
                "tokenizer":   tokenizer,
            }
            logger.info("BioBERT | F1=%.4f  P=%.4f  R=%.4f  TokenAcc=%.4f",
                        bert_metrics["overall_f1"], bert_metrics["overall_precision"],
                        bert_metrics["overall_recall"], bert_acc)

        except Exception as e:
            logger.warning("BioBERT training failed: %s", e)
            if "accelerate" in str(e).lower():
                logger.warning("  ➜ Fix: pip install \"accelerate>=1.1.0\"")
            logger.info("Continuing with CRF model as champion")

    # ─────────────────────────────────────────────────────────
    # 6. MODEL COMPARISON
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 6: Model comparison")

    if not all_model_results:
        logger.warning(
            "No models trained. Install missing packages:\n"
            "  pip install seqeval sklearn-crfsuite\n"
            "Using MajorityClass as emergency fallback."
        )
        fallback_preds = [["O"] * len(s) for s in test_tokens_list]
        fallback_metrics = compute_seqeval_metrics(test_gold, fallback_preds, verbose=False)
        all_model_results["MajorityClass"] = {
            "predictions": fallback_preds,
            "confidences": [[0.0] * len(s) for s in fallback_preds],
            "metrics":     fallback_metrics,
            "params":      {},
        }

    summary_df = evaluate_models(all_model_results, test_sents, test_gold)
    print("\n" + "=" * 65)
    print("ALL MODELS SUMMARY")
    print("=" * 65)
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    # ─────────────────────────────────────────────────────────
    # 7. SELECT BEST MODEL
    # ─────────────────────────────────────────────────────────
    best_name    = summary_df.iloc[0]["Model"]
    best_info    = all_model_results[best_name]
    best_preds   = best_info["predictions"]
    best_metrics = best_info["metrics"]
    best_confs   = best_info.get("confidences", [[1.0]*len(s) for s in best_preds])

    logger.info("Best model: %s | F1=%.4f", best_name, best_metrics["overall_f1"])

    # ─────────────────────────────────────────────────────────
    # 8. BUSINESS COST EVALUATION
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 8: Business cost evaluation")

    cost_result = cost_sensitive_ner_evaluation(test_gold, best_preds)
    print("\nBUSINESS COST EVALUATION")
    for k, v in cost_result.items():
        print(f"  {k}: {v}")

    # ─────────────────────────────────────────────────────────
    # 9. PER-ENTITY ANALYSIS + PLOTS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 9: Per-entity analysis")

    entity_df = entity_metrics_to_dataframe(best_metrics)
    if not entity_df.empty:
        print("\nPER-ENTITY METRICS")
        print(entity_df.to_string())

        os.makedirs("docs/plots", exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {"Chemical": "#1976D2", "Disease": "#D32F2F"}
        bar_colors = [colors.get(e, "#90A4AE") for e in entity_df["Entity"]]
        ax.barh(entity_df["Entity"], entity_df["F1"], color=bar_colors)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"{best_name} — Per-Entity F1 (BC5CDR)")
        ax.axvline(x=0.80, color="orange", linestyle="--", label="Target F1=0.80")
        ax.set_xlim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig("docs/plots/per_entity_f1.png", dpi=120)
        plt.close()

        _plot_entity_confusion(test_gold, best_preds)

    # ─────────────────────────────────────────────────────────
    # 10. PSI DRIFT MONITORING
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 10: PSI drift monitoring")

    train_flat = [l for s in train_gold for l in s]
    test_flat  = [l for s in test_gold  for l in s]

    label_psi = psi_label_distribution(train_flat, test_flat)
    logger.info("Label PSI (train vs test): %.4f", label_psi)

    entity_psi = {}
    for ent in ENTITY_TYPES:
        ref_bin = [str(int(l == f"B-{ent}")) for l in train_flat]
        new_bin = [str(int(l == f"B-{ent}")) for l in test_flat]
        entity_psi[ent] = compute_token_distribution_psi(ref_bin, new_bin)

    psi_df = pd.DataFrame(
        list(entity_psi.items()), columns=["entity_type", "psi_score"]
    ).sort_values("psi_score", ascending=False)
    psi_df.to_csv(os.path.join(MODEL_DIR, "entity_drift_report.csv"), index=False)
    logger.info("Entity PSI:\n%s", psi_df.to_string())

    # ─────────────────────────────────────────────────────────
    # 11. SAVE MONITOR SCORES
    # ─────────────────────────────────────────────────────────
    flat_confs  = [c for sent_confs in best_confs  for c in sent_confs]
    flat_preds  = [l for s in best_preds for l in s]
    flat_true_f = [l for s in test_gold  for l in s]

    monitor_df = pd.DataFrame({
        "predicted_label": flat_preds,
        "true_label":      flat_true_f,
        "confidence":      flat_confs,
        "is_entity":       [int(l != "O") for l in flat_preds],
        "is_correct":      [int(p == t) for p, t in zip(flat_preds, flat_true_f)],
    })
    monitor_df.to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)

    # ─────────────────────────────────────────────────────────
    # 12. ENTITY VARIANCE
    # ─────────────────────────────────────────────────────────
    entity_variance = per_entity_f1_variance(best_metrics.get("per_entity", {}))
    logger.info("Per-entity F1 variance: %.4f", entity_variance)

    # ─────────────────────────────────────────────────────────
    # 13. BUILD + SAVE MODEL CARD
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 13: Building model card")

    model_card = build_model_card(
        selected_name = best_name,
        train_size    = len(train_sents),
        val_size      = len(val_sents),
        test_size     = len(test_sents),
        metrics       = best_metrics,
        cost_result   = cost_result,
        per_entity    = best_metrics.get("per_entity", {}),
        threshold     = CONFIDENCE_THRESHOLD,
        model_params  = best_info.get("params", {}),
    )
    model_card["metrics"]["entity_f1_variance"] = entity_variance
    card_path = save_model_card(model_card, MODEL_DIR, best_name)

    # ─────────────────────────────────────────────────────────
    # 14. SAVE MODEL
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 14: Saving model")

    if best_name == "BioBERT" and "model" in best_info:
        model_path = save_bert_model(
            best_info["model"], best_info["tokenizer"],
            model_name="BioBERT", model_dir=MODEL_DIR, version="v1"
        )
        model_type = "bert"
    else:
        model_path = save_crf_model(
            crf_model, feature_extractor,
            model_name="CRF", model_dir=MODEL_DIR, version="v1"
        )
        model_type = "crf"

    # ─────────────────────────────────────────────────────────
    # 15. CHAMPION-CHALLENGER
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 15: Champion vs Challenger comparison")

    challenger_result = run_challenger_comparison(
        challenger_name      = best_name,
        challenger_f1        = float(best_metrics["overall_f1"]),
        challenger_precision = float(best_metrics["overall_precision"]),
        challenger_recall    = float(best_metrics["overall_recall"]),
        challenger_gap       = float(entity_variance),
        challenger_path      = model_path,
        challenger_threshold = CONFIDENCE_THRESHOLD,
        challenger_card_path = card_path,
        model_type           = model_type,
    )

    print("\n" + "=" * 65)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"Reason: {challenger_result['reason']}")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────
    # 16. MLFLOW LOGGING
    # ─────────────────────────────────────────────────────────
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{best_name}"
    mlflow_log_ner_run(
        run_name     = run_name,
        model_name   = best_name,
        metrics      = best_metrics,
        cost_result  = cost_result,
        model_params = best_info.get("params", {}),
    )

    print("\n" + "=" * 65)
    print(f"TRAINING COMPLETE  |  Best model: {best_name}")
    print(f"F1={best_metrics['overall_f1']:.4f}  "
          f"P={best_metrics['overall_precision']:.4f}  "
          f"R={best_metrics['overall_recall']:.4f}")
    print(f"Challenger status : {challenger_result['decision']}")
    print("=" * 65)

    return best_name, best_preds, best_metrics


def _plot_entity_confusion(test_gold, best_preds):
    try:
        flat_true = [l for s in test_gold  for l in s]
        flat_pred = [l for s in best_preds for l in s]
        b_labels  = sorted(set(l for l in flat_true + flat_pred if l.startswith("B-")))
        if len(b_labels) < 2:
            return
        all_cm = sorted(set(
            [l for l in flat_true + flat_pred if l.startswith("B-")]
        ))
        cm = confusion_matrix(flat_true, flat_pred, labels=all_cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[l[2:] for l in all_cm],
                    yticklabels=[l[2:] for l in all_cm],
                    ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted Entity")
        ax.set_ylabel("True Entity")
        ax.set_title("Entity Confusion Matrix (BC5CDR)")
        plt.tight_layout()
        plt.savefig("docs/plots/entity_confusion_matrix.png", dpi=120)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot confusion matrix: %s", e)


if __name__ == "__main__":
    start = time.time()
    name, preds, metrics = run_training(data_dir="data")
    logger.info("Pipeline finished in %.1fs", time.time() - start)
