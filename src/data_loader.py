# ============================================================
# DATA LOADER — BC5CDR Clinical NER System
# ============================================================
# PRIMARY DATASET:
#   BC5CDR — BioCreative V Chemical-Disease Relation Task
#   Source : HuggingFace — tner/bc5cdr
#   Size   : 15,000+ sentences from PubMed abstracts
#   Entities: Chemical + Disease (BIO format, already done)
#   Splits : train / validation / test (pre-split by dataset)
#
# HOW TO USE (one-time setup):
#   pip install datasets
#   python scripts/train_model.py
#   (Dataset downloads automatically from HuggingFace)
#
# FALLBACK (if no internet):
#   Place CoNLL .txt files in data/ folder
#   Or any BIO-tagged CSV with Sentence/Word/Tag columns
#
# BC5CDR label format:
#   B-Chemical, I-Chemical — drug/compound mentions
#   B-Disease,  I-Disease  — disease/disorder mentions
#   O                      — outside any entity
# ============================================================

import os
import re
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from src.config import BIO_LABELS, LABEL2ID, RANDOM_STATE, ENTITY_TYPES

logger = logging.getLogger(__name__)

VALID_LABELS = set(BIO_LABELS)

# ── BC5CDR label normalisation ────────────────────────────────
# BC5CDR uses exact same names as our config — direct mapping
BC5CDR_LABEL_MAP = {
    "B-Chemical": "B-Chemical", "I-Chemical": "I-Chemical",
    "B-Disease":  "B-Disease",  "I-Disease":  "I-Disease",
    "O": "O",
    # Some versions use different casing
    "b-chemical": "B-Chemical", "i-chemical": "I-Chemical",
    "b-disease":  "B-Disease",  "i-disease":  "I-Disease",
    # NCBI-style variants
    "B-CHEMICAL": "B-Chemical", "I-CHEMICAL": "I-Chemical",
    "B-DISEASE":  "B-Disease",  "I-DISEASE":  "I-Disease",
}


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def load_bc5cdr_dataset(
    data_dir: str = "data"
) -> Dict[str, List[List[Tuple[str, str]]]]:
    """
    Loads BC5CDR dataset.

    Priority order:
        1. HuggingFace datasets (tner/bc5cdr) — AUTO DOWNLOAD
        2. CoNLL .txt files in data/ folder
        3. CSV files in data/ folder

    No manual download needed — runs automatically.

    Returns:
        {"train": [...], "val": [...], "test": [...]}
        Each split = list of sentences
        Each sentence = list of (token, BIO_label) tuples
    """
    # ── Priority 1: HuggingFace auto-download ─────────────────
    try:
        return _load_from_huggingface()
    except ImportError:
        logger.warning(
            "datasets library not installed.\n"
            "  ➜ Fix: pip install datasets\n"
            "  Falling back to local files in data/ folder."
        )
    except Exception as e:
        logger.warning(
            "HuggingFace load failed: %s\n"
            "  Falling back to local files in data/ folder.", e
        )

    # ── Priority 2: Local CoNLL files ─────────────────────────
    os.makedirs(data_dir, exist_ok=True)
    files_in_dir = os.listdir(data_dir)

    conll_candidates = [
        "train.txt", "train_data.txt", "bc5cdr_train.txt",
        "data.txt", "ner.txt", "bc5cdr.txt"
    ]
    conll_file = _find_file(data_dir, conll_candidates)
    if conll_file:
        logger.info("Loading CoNLL format: %s", conll_file)
        return _load_from_conll_splits(data_dir, conll_file)

    # ── Priority 3: CSV files ─────────────────────────────────
    csv_candidates = ["ner_dataset.csv", "bc5cdr.csv", "data.csv"]
    csv_file = _find_file(data_dir, csv_candidates)
    if csv_file:
        logger.info("Loading CSV format: %s", csv_file)
        return _load_from_csv(csv_file)

    # ── Nothing found ─────────────────────────────────────────
    raise FileNotFoundError(
        f"\n{'='*65}\n"
        f"❌  Could not load BC5CDR dataset.\n\n"
        f"SOLUTION (choose one):\n\n"
        f"  Option A — Auto download (recommended):\n"
        f"    pip install datasets\n"
        f"    python scripts/train_model.py\n\n"
        f"  Option B — Manual CoNLL files:\n"
        f"    Place train.txt / test.txt in data/ folder\n"
        f"    (BIO format: TOKEN LABEL per line, blank line = sentence boundary)\n"
        f"{'='*65}\n"
    )


# ============================================================
# LOADER 1 — HuggingFace (PRIMARY)
# ============================================================

def _load_from_huggingface() -> Dict[str, List[List[Tuple[str, str]]]]:
    """
    Downloads BC5CDR from HuggingFace automatically — via the Parquet export,
    NOT `datasets.load_dataset("tner/bc5cdr")`.

    Why not load_dataset(): `tner/bc5cdr` ships a legacy Python *loading
    script* (bc5cdr.py). As of `datasets` v4.0 (mid-2024), the library
    refuses to execute any dataset loading script at all — you get
    "RuntimeError: Dataset scripts are no longer supported, but found
    bc5cdr.py" no matter what flags you pass. Older `datasets` pins work
    around it but silently break again on the next `pip install --upgrade`.

    Instead, we pull the Parquet files that HuggingFace auto-generates for
    every dataset on the Hub (script-based or not) via the public
    datasets-server `/parquet` API, and read them with pandas. No script
    execution, no version pinning required, works on any `datasets`
    version (or none installed at all).
    """
    import io
    import requests
    import pandas as pd

    REPO = "tner/bc5cdr"
    id_to_label = _default_bc5cdr_id_map()

    logger.info("Downloading BC5CDR Parquet export from HuggingFace (%s)...", REPO)
    logger.info("This may take a few seconds on first run (cached after that).")

    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={REPO}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    parquet_files = resp.json().get("parquet_files", [])
    if not parquet_files:
        raise RuntimeError(f"No parquet files returned for {REPO}")

    split_map = {
        "train":      "train",
        "validation": "val",
        "test":       "test",
    }

    splits: Dict[str, List[List[Tuple[str, str]]]] = {}

    for hf_split, our_split in split_map.items():
        urls = [f["url"] for f in parquet_files if f.get("split") == hf_split]
        if not urls:
            logger.warning("Split '%s' not found in BC5CDR parquet export", hf_split)
            continue

        frames = []
        for url in urls:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            frames.append(pd.read_parquet(io.BytesIO(r.content)))
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        sentences = _convert_hf_dataframe(df, id_to_label)
        validate_sentences(sentences, split_name=our_split)
        splits[our_split] = sentences

        total_tokens  = sum(len(s) for s in sentences)
        entity_tokens = sum(1 for s in sentences for _, l in s if l != "O")
        logger.info(
            "Split %-8s | sentences=%5d | tokens=%6d | entity_tokens=%5d (%.1f%%)",
            our_split, len(sentences), total_tokens,
            entity_tokens, 100 * entity_tokens / max(total_tokens, 1)
        )

    # Handle missing val split
    if "val" not in splits and "train" in splits:
        logger.info("No validation split — splitting 10%% from train")
        splits["train"], splits["val"] = _split_sentences(splits["train"], 0.111)

    return splits


def _convert_hf_dataframe(
    df,
    id_to_label: dict,
) -> List[List[Tuple[str, str]]]:
    """
    Converts a BC5CDR parquet DataFrame → list of (token, label) sentences.

    Each row has:
        'tokens': ['Cyclosporin', '-', 'induced', 'nephrotoxicity', ...]
        'tags':   [1, 0, 0, 2, ...]   ← integer label IDs (see id_to_label)
    """
    logger.info("BC5CDR label map: %s", id_to_label)

    sentences = []
    skipped   = 0

    tok_col = "tokens" if "tokens" in df.columns else "words"
    tag_col = "tags" if "tags" in df.columns else "ner_tags"

    for tokens, tags in zip(df[tok_col], df[tag_col]):
        # parquet arrays come back as numpy arrays / lists
        tokens = list(tokens)
        tags   = list(tags)

        if not tokens or not tags or len(tokens) != len(tags):
            skipped += 1
            continue

        sent = []
        for token, tag_id in zip(tokens, tags):
            raw_label = id_to_label.get(int(tag_id), "O")
            label = _normalize_label(raw_label)
            sent.append((str(token), label))

        if sent:
            sentences.append(sent)

    if skipped:
        logger.warning("Skipped %d malformed examples", skipped)

    return sentences


def _default_bc5cdr_id_map() -> dict:
    """
    BC5CDR integer → label string mapping for tner/bc5cdr.

    Verified directly against the dataset card's label2id
    (https://huggingface.co/datasets/tner/bc5cdr/raw/main/dataset/label.json):
        O: 0, B-Chemical: 1, B-Disease: 2, I-Disease: 3, I-Chemical: 4

    NOTE: this order is NOT alphabetical / BIO-grouped — B-Disease (2) comes
    before I-Disease (3) comes before I-Chemical (4). A previous version of
    this map had the wrong order (I-Chemical/B-Disease swapped), which would
    have silently mislabeled entities. Double-check against the source if
    the upstream dataset schema ever changes.
    """
    return {
        0: "O",
        1: "B-Chemical",
        2: "B-Disease",
        3: "I-Disease",
        4: "I-Chemical",
    }

# ============================================================
# LOADER 2 — CoNLL .txt format
# ============================================================

def load_conll_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """
    Parses CoNLL BIO .txt file.
    Format: TOKEN LABEL (space/tab), blank line = sentence boundary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CoNLL file not found: {filepath}")

    sentences    = []
    current_sent = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
                continue
            if line.startswith("-DOCSTART-") or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            token = parts[0]
            label = _normalize_label(parts[-1])
            current_sent.append((token, label))

    if current_sent:
        sentences.append(current_sent)

    logger.info("Loaded %d sentences from CoNLL: %s", len(sentences), filepath)
    return sentences


def _load_from_conll_splits(
    data_dir: str, train_file: str
) -> Dict[str, List[List[Tuple[str, str]]]]:
    train = load_conll_file(train_file)
    validate_sentences(train, "train")

    test_file = _find_file(data_dir, ["test.txt", "test_data.txt"])
    test = load_conll_file(test_file) if test_file else None

    val_file = _find_file(data_dir, ["val.txt", "dev.txt", "validation.txt"])
    val = load_conll_file(val_file) if val_file else None

    if test is None:
        train, test = _split_sentences(train, 0.20)
    if val is None:
        train, val  = _split_sentences(train, 0.111)

    return {"train": train, "val": val, "test": test}


# ============================================================
# LOADER 3 — CSV format
# ============================================================

def _load_from_csv(filepath: str) -> Dict[str, List[List[Tuple[str, str]]]]:
    """Sentence#/Word/Tag CSV format."""
    df = pd.read_csv(filepath, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]

    sc = next((c for c in df.columns if "sentence" in c), None)
    wc = next((c for c in df.columns if "word" in c or "token" in c), None)
    tc = next((c for c in df.columns if "tag" in c or "label" in c), None)

    if not (sc and wc and tc):
        raise ValueError(f"CSV needs Sentence/Word/Tag columns. Found: {list(df.columns)}")

    df[sc] = df[sc].ffill()
    sentences = []
    for _, grp in df.groupby(sc, sort=False):
        sent = [(str(r[wc]).strip(), _normalize_label(str(r[tc]).strip()))
                for _, r in grp.iterrows()]
        if sent:
            sentences.append(sent)

    train, test = _split_sentences(sentences, 0.20)
    train, val  = _split_sentences(train,     0.111)
    return {"train": train, "val": val, "test": test}


# ============================================================
# VALIDATION
# ============================================================

def validate_sentences(
    sentences: List[List[Tuple[str, str]]],
    split_name: str = "data"
) -> None:
    """Validates: non-empty, valid BIO tags, BIO sequence integrity."""
    if not sentences:
        raise ValueError(f"[{split_name}] Empty — no sentences found")

    if len(sentences) < 10:
        logger.warning("[%s] Very small: only %d sentences", split_name, len(sentences))

    unknown  = set()
    bio_errs = 0

    for sent in sentences:
        prev = "O"
        for token, label in sent:
            if label not in VALID_LABELS:
                unknown.add(label)
            if label.startswith("I-"):
                etype = label[2:]
                if not (prev == f"B-{etype}" or prev == f"I-{etype}"):
                    bio_errs += 1
            prev = label

    if unknown:
        logger.warning("[%s] Unknown labels → mapped to O: %s", split_name, unknown)
    if bio_errs:
        logger.warning("[%s] %d BIO violations", split_name, bio_errs)
    else:
        logger.info("[%s] ✅ BIO check passed | sentences=%d", split_name, len(sentences))


# ============================================================
# STATISTICS + DATAFRAME HELPERS
# ============================================================

def compute_dataset_stats(
    sentences: List[List[Tuple[str, str]]]
) -> pd.DataFrame:
    """Entity-level frequency counts."""
    counts: Dict[str, int] = {}
    for sent in sentences:
        for _, label in sent:
            if label.startswith("B-"):
                etype = label[2:]
                counts[etype] = counts.get(etype, 0) + 1
    return pd.DataFrame(
        list(counts.items()), columns=["entity_type", "count"]
    ).sort_values("count", ascending=False).reset_index(drop=True)


def sentences_to_dataframe(
    sentences: List[List[Tuple[str, str]]]
) -> pd.DataFrame:
    """Flat token-level DataFrame."""
    records = []
    for sid, sent in enumerate(sentences):
        for pos, (token, label) in enumerate(sent):
            records.append({
                "sentence_id": sid,
                "token_pos":   pos,
                "token":       token,
                "label":       label,
                "label_id":    LABEL2ID.get(label, LABEL2ID["O"])
            })
    return pd.DataFrame(records)


# ============================================================
# HELPERS
# ============================================================

def _normalize_label(label: str) -> str:
    """Normalise label string to valid BIO tag."""
    label = label.strip()
    if label in VALID_LABELS:
        return label
    # Try BC5CDR map first
    if label in BC5CDR_LABEL_MAP:
        mapped = BC5CDR_LABEL_MAP[label]
        if mapped in VALID_LABELS:
            return mapped
    # Case-insensitive fallback
    for v in VALID_LABELS:
        if v.lower() == label.lower():
            return v
    # Common O-variants
    if label.upper() in ("OUTSIDE", "OTHER", "O"):
        return "O"
    return "O"


def _find_file(directory: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            logger.info("Found data file: %s", path)
            return path
    return None


def _split_sentences(
    sentences: List, test_ratio: float = 0.20
) -> Tuple[List, List]:
    """Reproducible random split."""
    np.random.seed(RANDOM_STATE)
    idx = np.random.permutation(len(sentences))
    cut = int(len(sentences) * (1 - test_ratio))
    return ([sentences[i] for i in idx[:cut]],
            [sentences[i] for i in idx[cut:]])
