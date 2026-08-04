# ============================================================
# CONFIGURATION — BC5CDR Clinical NER System
# ============================================================
# Dataset : BC5CDR (BioCreative V CDR Task)
# Source  : HuggingFace — tner/bc5cdr
# Entities: Chemical (Drug) + Disease
# Size    : 15,000+ sentences from PubMed biomedical abstracts
# Benchmark: Industry standard — BioBERT achieves F1~0.87
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── BC5CDR Entity Types ───────────────────────────────────────
# BC5CDR has exactly 2 entity types:
#   Chemical → drugs, compounds, chemicals mentioned in biomedical text
#   Disease  → diseases, disorders, syndromes, symptoms
ENTITY_TYPES   = ["Chemical", "Disease"]
BIO_LABELS     = ["O"] + [f"B-{e}" for e in ENTITY_TYPES] + [f"I-{e}" for e in ENTITY_TYPES]
NUM_LABELS     = len(BIO_LABELS)   # = 5

# label → id and id → label mappings
LABEL2ID       = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL       = {idx: label for label, idx in LABEL2ID.items()}

# ── Model architecture ────────────────────────────────────────
PRETRAINED_MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"   # BioBERT — PubMed trained
FALLBACK_MODEL_NAME    = "bert-base-cased"                     # fallback if no internet
MAX_SEQ_LENGTH         = 128
BATCH_SIZE             = 4          # small — GeForce MX350 has only 2GB VRAM
GRADIENT_ACCUMULATION_STEPS = 4     # 4 x 4 = effective batch size 16 (same as before)
# ── Training hyperparameters ──────────────────────────────────
LEARNING_RATE     = 2e-5
NUM_EPOCHS        = 5
WARMUP_RATIO      = 0.1
WEIGHT_DECAY      = 0.01
GRAD_CLIP         = 1.0

# ── Feature engineering for baseline models ──────────────────
WINDOW_SIZE       = 2       # context window for CRF features

# ── Evaluation ───────────────────────────────────────────────
EVAL_METRIC       = "f1"    # seqeval strict entity-level F1

# ── Challenger promotion gates ────────────────────────────────
MIN_F1_IMPROVEMENT      = 0.005   # challenger must beat champion by >= 0.5%
MIN_F1_THRESHOLD        = 0.70    # challenger must have F1 >= 0.70
MAX_ENTITY_GAP          = 0.15    # per-entity F1 variance must be <= 0.15

# ── PSI drift thresholds ──────────────────────────────────────
PSI_MODERATE            = 0.10
PSI_HIGH                = 0.20

# ── Business thresholds ───────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.80    # below this → flag for human review
HIGH_RISK_ENTITIES      = ["Chemical", "Disease"]  # both critical in pharma
LOW_CONFIDENCE_ALERT    = 0.40    # avg confidence below this → alert

# ── Business cost (pharma/clinical context) ───────────────────
# Missed Chemical (drug) entity in ADE report = regulatory risk
# Missed Disease entity in diagnosis coding   = billing/insurance error
FN_COST_CHEMICAL        = 500.0   # ₹ per missed Chemical entity
FN_COST_DISEASE         = 300.0   # ₹ per missed Disease entity
FP_COST_REVIEW          = 50.0    # ₹ per false positive (manual review)

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR        = "ner_models"
METRICS_LOG      = "ner_models/metrics_log.csv"
LOG_DIR          = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs("data",    exist_ok=True)
