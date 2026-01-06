import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# =========================
#  UI — THEME + STYLES
# =========================

PRIMARY_BLUE = "#5DADE2"
LIGHT_GREY = "#F2F3F4"
BLACK = "#111111"
RED = "#E74C3C"

st.set_page_config(
    page_title="Customer Call Intent Detector",
    page_icon="📞",
    layout="centered"
)

st.markdown(
    f"""
    <style>
        body {{ background-color: {LIGHT_GREY}; }}
        .main {{ background-color: {LIGHT_GREY}; }}
        .title-text {{ color:{BLACK}; font-weight:700; font-size:32px; }}
        .sub-text {{ color:{BLACK}; opacity:0.75; }}
        .phone-icon {{ color:{RED}; font-size:36px; }}
        .confidence {{
            background:white; padding:12px; border-radius:10px;
            border:1px solid #DDD;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
#  LOAD MODELS
# =========================

MODEL_DIR = Path(__file__).resolve().parent.parent / "NLP_notebook"

classifier = joblib.load(MODEL_DIR / "intent_classifier.pkl")   # FULL PIPELINE
label_encoder = joblib.load(MODEL_DIR / "intent_label_encoder.pkl")


# =========================
#  PREDICT FUNCTION
# =========================
def predict_intent(text):
    probs = classifier.predict_proba([text])[0]
    top_idx = int(np.argmax(probs))
    intent = label_encoder.inverse_transform([top_idx])[0]
    confidence = float(probs[top_idx])
    return intent, confidence, probs


# =========================
#  HEADER UI
# =========================

st.markdown("<div class='phone-icon'>📞</div>", unsafe_allow_html=True)
st.markdown("<div class='title-text'>Customer Call Intent Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-text'>Paste a call transcript or summary. The model predicts the customer’s intent & confidence score.</div>",
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Customer Call Notes / Transcript",
    height=200,
    placeholder="Example: I want to reset my online banking password…"
)

run_button = st.button("Analyze Intent")


# =========================
#  RUN PREDICTION
# =========================

if run_button:

    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        intent, confidence, probs = predict_intent(user_input)

        if confidence >= 0.65:
            conf_label = "High confidence"
            conf_color = "#2ECC71"
        elif confidence >= 0.40:
            conf_label = "Moderate confidence"
            conf_color = "#F1C40F"
        else:
            conf_label = "Low confidence"
            conf_color = "#E74C3C"

        st.subheader("Predicted Intent")

        st.markdown(
            f"""
            <div class='confidence'>
            <b>Intent:</b> <span style='color:{PRIMARY_BLUE}'>{intent}</span><br>
            <b>Confidence:</b> <span style='color:{conf_color}'>{confidence:.2f}</span> — {conf_label}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Show probability breakdown"):
            prob_table = {
                label_encoder.classes_[i]: float(probs[i])
                for i in range(len(probs))
            }
            st.json(prob_table)

st.write("---")
st.caption("Built with small dataset v1. More data = higher accuracy 🚀")
