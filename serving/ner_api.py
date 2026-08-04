# ============================================================
# BC5CDR CLINICAL NER API — FastAPI Serving
# ============================================================

import os, json, time, logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional

from src.model_loader         import load_latest_model, load_bert_champion
from services.prediction_service import predict_entities, predict_batch_entities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "BC5CDR Clinical NER API",
    description = (
        "Extracts Chemical (Drug) and Disease entities from biomedical text. "
        "Trained on BC5CDR (BioCreative V CDR) — 15,000+ PubMed abstracts. "
        "Industry benchmark: BioBERT achieves F1~0.87 on this dataset."
    ),
    version     = "1.0.0",
)

model_state = {
    "model": None, "tokenizer": None,
    "crf": None, "extractor": None,
    "info": None, "threshold": 0.80, "model_type": "none",
}

try:
    registry, threshold = load_latest_model()
    model_state["info"]      = registry
    model_state["threshold"] = threshold
    model_type = registry.get("model_type", "crf") if isinstance(registry, dict) else "crf"

    if model_type == "bert":
        model, tokenizer = load_bert_champion()
        model_state["model"]      = model
        model_state["tokenizer"]  = tokenizer
        model_state["model_type"] = "bert"
    elif model_type == "crf":
        model_dict = registry if isinstance(registry, dict) and "crf" in registry else None
        if model_dict:
            model_state["crf"]        = model_dict.get("crf")
            model_state["extractor"]  = model_dict.get("feature_extractor")
            model_state["model_type"] = "crf"
except Exception as e:
    logger.warning("Model loading failed (run train_model.py first): %s", e)


class NERInput(BaseModel):
    text: str
    return_confidence: Optional[bool] = True

    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v.strip()


class NERBatchInput(BaseModel):
    texts: List[str]
    return_confidence: Optional[bool] = True

    @validator("texts")
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError("texts list must not be empty")
        if len(v) > 100:
            raise ValueError("Max 100 texts per batch")
        return [t.strip() for t in v if t.strip()]


@app.get("/")
def home():
    return {
        "message":  "BC5CDR Clinical NER API 🧬",
        "dataset":  "BC5CDR (BioCreative V CDR) — 15k+ PubMed sentences",
        "entities": ["Chemical (Drug)", "Disease"],
        "benchmark":"BioBERT F1 ~0.87 on BC5CDR test set",
        "docs":     "/docs",
        "health":   "/health",
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": model_state["model"] is not None or model_state["crf"] is not None,
        "model_type":   model_state["model_type"],
        "threshold":    model_state["threshold"],
        "dataset":      "BC5CDR",
    }


@app.get("/model_info")
def model_info():
    registry_path = "ner_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            reg = json.load(f)
        card_path = reg.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                return json.load(f)
        return reg
    return {"error": "Model registry not found — run train_model.py first"}


@app.post("/predict")
def predict(request: NERInput):
    """
    Extract Chemical and Disease entities from biomedical text.

    Example input:
        "Cyclosporin-induced nephrotoxicity in renal transplant patients."

    Example output:
        entities: [
            {entity_type: "Chemical", text: "Cyclosporin", confidence: 0.94},
            {entity_type: "Disease",  text: "nephrotoxicity", confidence: 0.91}
        ]
    """
    start = time.time()
    if model_state["model_type"] == "none":
        raise HTTPException(503, "Model not loaded. Run scripts/train_model.py first.")
    try:
        result = predict_entities(
            text=request.text, model_state=model_state,
            threshold=model_state["threshold"], return_conf=request.return_confidence,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    _log_prediction(request.text, result.get("entities", []),
                    time.time() - start, model_state["model_type"])
    result["latency_seconds"] = round(time.time() - start, 4)
    result["model_type"]      = model_state["model_type"]
    return result


@app.post("/predict_batch")
def predict_batch(request: NERBatchInput):
    start = time.time()
    if model_state["model_type"] == "none":
        raise HTTPException(503, "Model not loaded. Run scripts/train_model.py first.")
    try:
        results = predict_batch_entities(
            texts=request.texts, model_state=model_state,
            threshold=model_state["threshold"], return_conf=request.return_confidence,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"results": results, "total_texts": len(results),
            "latency_seconds": round(time.time() - start, 4)}


def _log_prediction(text, entities, latency, model_type):
    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp":       time.time(),
        "text_length":     len(text),
        "n_entities":      len(entities),
        "has_chemical":    int(any(e["entity_type"] == "Chemical" for e in entities)),
        "has_disease":     int(any(e["entity_type"] == "Disease"  for e in entities)),
        "avg_confidence":  round(float(sum(e.get("confidence", 1.0)
                           for e in entities) / max(len(entities), 1)), 4),
        "model_type":      model_type,
        "latency_s":       round(latency, 4),
    }
    log_df = pd.DataFrame([record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
