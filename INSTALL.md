# 🚀 Installation Guide — Medical NER System

## Tumhare System Pe Error Ka Reason

Pipeline run karne ke liye 2 packages missing the:
- `seqeval` — NER evaluation library (entity-level F1)
- `sklearn-crfsuite` — CRF baseline model

## Step 1: Required Packages Install Karo (ZAROOR)

```bash
pip install seqeval sklearn-crfsuite
```

## Step 2: BioBERT ke liye (Optional — GPU chahiye ya CPU pe slow hoga)

```bash
pip install transformers torch
```

**CPU pe BioBERT bahut slow hoga (2-3 hours).** 
Agar sirf CRF run karna chahte ho (fast, 30 seconds):

```bash
python scripts/train_model.py --skip-bert
```

## Step 3: Full Install (sab kuch)

```bash
pip install -r requirements.txt
```

## Step 4: Train Karo

```bash
# Fast mode (CRF only, no GPU needed) — RECOMMENDED first run
python scripts/train_model.py --skip-bert

# Full mode (CRF + BioBERT)
python scripts/train_model.py
```

## Expected Output (CRF only mode)

```
STEP 1: Loading Medical NER dataset ✅
STEP 2: Leakage detection ✅  
STEP 3: Majority class baseline ✅
STEP 4: Training CRF baseline ✅
STEP 6: Model comparison ✅
STEP 7: Best model: CRF
...
TRAINING COMPLETE | Best model: CRF
F1=0.7xxx  P=0.7xxx  R=0.7xxx
```

## Error → Solution Map

| Error | Fix |
|-------|-----|
| `seqeval not installed` | `pip install seqeval` |
| `sklearn-crfsuite not installed` | `pip install sklearn-crfsuite` |
| `No module named 'transformers'` | `pip install transformers torch` OR use `--skip-bert` |
| `KeyError: 'F1'` | Install seqeval + sklearn-crfsuite, then re-run |
