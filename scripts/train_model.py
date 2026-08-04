# ============================================================
# TRAIN MODEL — BC5CDR Clinical NER System
# ============================================================
# Usage:
#   python scripts/train_model.py                  # CRF + BioBERT
#   python scripts/train_model.py --skip-bert      # CRF only (fast)
#   python scripts/train_model.py --data-dir data  # custom data dir
# ============================================================

import sys, os, argparse, logging, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training_pipeline import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="BC5CDR Clinical NER — Training Script")
    parser.add_argument("--data-dir",  type=str,  default="data",
                        help="Local data directory (fallback if HuggingFace unavailable)")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Run CRF only — fast, no GPU needed")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("BC5CDR CLINICAL NER SYSTEM — Training Pipeline")
    logger.info("Dataset : BC5CDR (tner/bc5cdr) — 15,000+ PubMed sentences")
    logger.info("Entities: Chemical + Disease")
    logger.info("=" * 65)
    logger.info("Data dir  : %s", args.data_dir)
    logger.info("Skip BERT : %s", args.skip_bert)

    start = time.time()
    name, preds, metrics = run_training(
        data_dir  = args.data_dir,
        skip_bert = args.skip_bert,
    )

    logger.info("=" * 65)
    logger.info("Training complete in %.1fs", time.time() - start)
    logger.info("Best model : %s", name)
    logger.info("F1         : %.4f", metrics.get("overall_f1",        0))
    logger.info("Precision  : %.4f", metrics.get("overall_precision", 0))
    logger.info("Recall     : %.4f", metrics.get("overall_recall",    0))
    logger.info("BC5CDR benchmark (BioBERT SOTA): F1 ~0.87")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
