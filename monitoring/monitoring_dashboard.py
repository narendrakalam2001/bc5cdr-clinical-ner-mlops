# ============================================================
# MONITORING DASHBOARD — BC5CDR Clinical NER System
# ============================================================
# 5 Sections:
#   1. Real-Time Alerts
#   2. Champion vs Challenger History
#   3. KPIs + Per-Entity Charts
#   4. PSI Drift Monitoring
#   5. Recent Predictions Log
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(
    page_title = "BC5CDR Clinical NER Dashboard",
    page_icon  = "🧬",
    layout     = "wide"
)

st.title("🧬 BC5CDR Clinical NER — Monitoring Dashboard")
st.caption("Chemical & Disease Entity Extraction | BC5CDR Benchmark | BioBERT F1 ~0.87")

API_URL = os.getenv(
    "NER_API_URL", "http://localhost:8000"
) + "/predict"

PSI_MODERATE         = 0.10
PSI_HIGH             = 0.20
LOW_CONFIDENCE_ALERT = 0.40

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "ner_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "ner_models", "entity_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "ner_models", "challenger_log.json")
CARD_PATH_DIR   = os.path.join(BASE_DIR, "ner_models")

ENTITY_COLORS = {"Chemical": "#1976D2", "Disease": "#D32F2F"}

# ── SIDEBAR — Live Prediction ──────────────────────────────────────────
st.sidebar.header("🔮 Live Entity Extraction")

sample_texts = {
    "Chemical-Disease 1":
        "Cyclosporin-induced nephrotoxicity in renal transplant patients.",
    "Chemical-Disease 2":
        "Cisplatin causes ototoxicity and nephrotoxicity in cancer patients.",
    "Drug Interaction":
        "Methotrexate combined with NSAIDs increases risk of hepatotoxicity.",
    "ADE Report":
        "Patient developed severe nausea and vomiting after Amoxicillin administration.",
    "Custom": ""
}

selected = st.sidebar.selectbox("Sample Biomedical Text", list(sample_texts.keys()))
clinical_text = st.sidebar.text_area(
    "Biomedical Text", value=sample_texts[selected], height=100,
    placeholder="Enter biomedical text from PubMed abstract..."
)

if st.sidebar.button("🔍 Extract Entities"):
    if clinical_text.strip():
        with st.sidebar:
            with st.spinner("Calling API..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"text": clinical_text, "return_confidence": True},
                        timeout=90
                    )
                    if response.status_code == 200:
                        result   = response.json()
                        entities = result.get("entities", [])
                        st.success(f"✅ Found {len(entities)} entities")
                        for ent in entities:
                            color = ENTITY_COLORS.get(ent["entity_type"], "#666")
                            st.markdown(
                                f"<span style='background:{color};color:white;"
                                f"padding:2px 8px;border-radius:4px;font-size:0.85em'>"
                                f"{ent['entity_type']}</span> "
                                f"**{ent['text']}** "
                                f"<small>(conf: {ent.get('confidence',1.0):.2f})</small>",
                                unsafe_allow_html=True
                            )
                        if result.get("requires_review"):
                            st.warning(result.get("alert", "⚠️ Review required"))
                    else:
                        st.error(f"API error: HTTP {response.status_code}")
                except requests.exceptions.Timeout:
                    st.warning("Timeout. Render is waking up. Retry in 30s.")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ══ SECTION 1 — ALERTS ════════════════════════════════════════════════
st.markdown("---")
st.subheader("🚨 Section 1 — Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    if "avg_confidence" in df_log.columns:
        avg_conf = df_log["avg_confidence"].mean()
        if avg_conf < LOW_CONFIDENCE_ALERT:
            st.error(f"🔴 LOW CONFIDENCE: avg={avg_conf:.4f} — Model may need retraining.")
            alerts_found = True
    if "has_chemical" in df_log.columns and "has_disease" in df_log.columns:
        chem_disease_rate = (df_log["has_chemical"] & df_log["has_disease"]).mean()
        if chem_disease_rate > 0.30:
            st.warning(
                f"🟡 HIGH Chemical-Disease Co-occurrence: {chem_disease_rate:.1%} "
                f"of predictions — potential ADE signal. Pharmacovigilance review."
            )
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "psi_score" in df_psi.columns and not df_psi.empty:
        max_psi    = df_psi["psi_score"].max()
        top_entity = df_psi.iloc[0].get("entity_type", "unknown")
        if max_psi >= PSI_HIGH:
            st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_entity}'. Retrain immediately.")
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_entity}'. Monitor closely.")
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")

# ══ SECTION 2 — CHAMPION vs CHALLENGER ════════════════════════════════
st.markdown("---")
st.subheader("🏆 Section 2 — Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)

    if challenger_log:
        latest = challenger_log[-1]
        icon   = "✅" if latest["decision"] == "PROMOTED" else "❌"

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Latest Challenger", latest.get("challenger_name", "—"))
        with c2: st.metric("Challenger F1",     latest.get("challenger_f1",   "—"))
        with c3: st.metric("Champion F1",        latest.get("champion_f1",     "—") or "First Run")
        with c4: st.metric("Decision",           f"{icon} {latest['decision']}")
        st.caption(f"Reason: {latest.get('reason', '—')}")

        if len(challenger_log) > 1:
            hist_df = pd.DataFrame([{
                "Run":           i+1,
                "Challenger":    r.get("challenger_name", "—"),
                "Challenger F1": r.get("challenger_f1",   0),
                "Champion F1":   r.get("champion_f1",     0) or 0,
                "Decision":      r.get("decision",        "—"),
            } for i, r in enumerate(challenger_log)])
            st.dataframe(hist_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(hist_df["Run"], hist_df["Challenger F1"],
                    marker="o", label="Challenger F1", color="#1976D2")
            ax.plot(hist_df["Run"], hist_df["Champion F1"],
                    marker="s", linestyle="--", label="Champion F1", color="#4CAF50")
            ax.axhline(0.87, color="orange", linestyle=":", linewidth=1.5,
                       label="BC5CDR BioBERT benchmark (0.87)")
            ax.set_xlabel("Training Run"); ax.set_ylabel("F1 Score")
            ax.set_title("Champion vs Challenger F1 Over Time")
            ax.legend(); ax.set_ylim(0, 1)
            plt.tight_layout(); st.pyplot(fig); plt.close()
else:
    st.info("No challenger history yet. Run train_model.py to start.")

# ══ SECTION 3 — KPIs + CHARTS ══════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Section 3 — Model KPIs + Per-Entity Performance")

card_files = [f for f in os.listdir(CARD_PATH_DIR)
              if f.startswith("model_card_") and f.endswith(".json")] \
    if os.path.exists(CARD_PATH_DIR) else []

if card_files:
    with open(os.path.join(CARD_PATH_DIR, sorted(card_files)[-1])) as f:
        card = json.load(f)

    m = card.get("metrics", {})
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Overall F1",    f"{m.get('test_f1',        0):.4f}")
    with k2: st.metric("Precision",     f"{m.get('test_precision', 0):.4f}")
    with k3: st.metric("Recall",        f"{m.get('test_recall',    0):.4f}")
    with k4: st.metric("Conf Threshold",f"{m.get('confidence_threshold', 0.80):.2f}")

    per_entity_f1 = m.get("per_entity_f1", {})
    if per_entity_f1:
        ent_df = pd.DataFrame(
            list(per_entity_f1.items()), columns=["Entity", "F1"]
        ).sort_values("F1", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 3))
        colors  = [ENTITY_COLORS.get(e, "#90A4AE") for e in ent_df["Entity"]]
        ax.barh(ent_df["Entity"], ent_df["F1"], color=colors)
        ax.axvline(0.87, color="orange", linestyle="--", linewidth=1.5,
                   label="BC5CDR BioBERT benchmark (0.87)")
        ax.axvline(0.70, color="red", linestyle=":", linewidth=1.5,
                   label="Min threshold (0.70)")
        ax.set_xlabel("F1 Score"); ax.set_title("Per-Entity F1 Score")
        ax.set_xlim(0, 1); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    biz = card.get("business_impact", {})
    if biz:
        st.markdown("**Business Impact Estimate**")
        b1, b2, b3 = st.columns(3)
        with b1: st.metric("Missed Chemical Entities", biz.get("missed_chemical_entities", "—"))
        with b2: st.metric("Missed Disease Entities",  biz.get("missed_disease_entities",  "—"))
        with b3: st.metric("Total Cost (INR)",
                           f"₹{biz.get('total_estimated_cost_inr', 0):,.0f}")
else:
    st.info("Model card not found. Run train_model.py to generate metrics.")

# ══ SECTION 4 — PSI DRIFT ══════════════════════════════════════════════
st.markdown("---")
st.subheader("📉 Section 4 — Entity Distribution Drift (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if not df_psi.empty and "psi_score" in df_psi.columns:

        def psi_color(v):
            if v >= PSI_HIGH:     return "background-color: #FFCDD2"
            elif v >= PSI_MODERATE: return "background-color: #FFF9C4"
            return "background-color: #C8E6C9"

        st.dataframe(
            df_psi.style.applymap(psi_color, subset=["psi_score"]),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = ["#D32F2F" if v >= PSI_HIGH else "#F57C00"
                      if v >= PSI_MODERATE else "#388E3C"
                      for v in df_psi["psi_score"]]
        ax.bar(df_psi.get("entity_type", df_psi.index),
               df_psi["psi_score"], color=bar_colors)
        ax.axhline(PSI_HIGH,     color="#D32F2F", linestyle="--",
                   linewidth=1.5, label=f"Critical (>{PSI_HIGH})")
        ax.axhline(PSI_MODERATE, color="#F57C00", linestyle="--",
                   linewidth=1.5, label=f"Moderate (>{PSI_MODERATE})")
        ax.set_xlabel("Entity Type"); ax.set_ylabel("PSI Score")
        ax.set_title("Entity Drift (PSI) — Train vs Current")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("PSI < 0.10: No drift | 0.10–0.20: Monitor | > 0.20: Retrain")
else:
    st.info("entity_drift_report.csv not found. Run train_model.py first.")

# ══ SECTION 5 — RECENT PREDICTIONS ════════════════════════════════════
st.markdown("---")
st.subheader("📋 Section 5 — Recent Predictions Log")

try:
    _base_url = os.getenv("NER_API_URL", "https://bc5cdr-clinical-ner-mlops.onrender.com").rstrip("/")
    _resp = requests.get(f"{_base_url}/recent_predictions", timeout=15)
    _resp.raise_for_status()
    _data = _resp.json()
    _preds = _data.get("predictions", [])

    if _preds:
        st.caption(f"Total predictions logged this session: {_data.get('total_logged', len(_preds))}")
        df_log = pd.DataFrame(_preds)
        if "timestamp" in df_log.columns:
            df_log["timestamp"] = pd.to_datetime(
                df_log["timestamp"], unit="s"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("No predictions logged yet this session. Make predictions via sidebar.")
except Exception as e:
    st.info(f"Could not fetch recent predictions from API: {e}")