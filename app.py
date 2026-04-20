
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lime.lime_text import LimeTextExplainer
import time
import os

# ── Absolute paths ────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    lr    = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf    = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    return lr, rf, tfidf

lr_model, rf_model, tfidf = load_models()

# ── Prediction function ───────────────────────────────────
def predict(text, model):
    vec   = tfidf.transform([text])
    proba = model.predict_proba(vec)[0]
    label = "FAKE" if proba[1] > 0.5 else "REAL"
    return label, proba

def predict_proba_lime(texts):
    vecs = tfidf.transform(texts)
    return lr_model.predict_proba(vecs)

# ── LIME explanation ──────────────────────────────────────
def explain(text):
    explainer = LimeTextExplainer(class_names=["Real", "Fake"])
    exp = explainer.explain_instance(
        text,
        predict_proba_lime,
        num_features=10,
        num_samples=1000
    )
    return exp

# ── Plot LIME ─────────────────────────────────────────────
def plot_explanation(exp):
    words_weights = exp.as_list()
    words   = [w[0] for w in words_weights]
    weights = [w[1] for w in words_weights]
    colors  = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.barh(words, weights, color=colors)
    ax.axvline(x=0, color="white", linewidth=0.8)
    ax.set_title(
        "Why did the model decide this?",
        fontsize=13, fontweight="bold",
        color="white", pad=12
    )
    ax.set_xlabel(
        "Red = pushes toward FAKE  |  Green = pushes toward REAL",
        color="#aaaaaa", fontsize=9
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    fake_patch = mpatches.Patch(color="#e74c3c", label="→ FAKE")
    real_patch = mpatches.Patch(color="#2ecc71", label="→ REAL")
    ax.legend(handles=[fake_patch, real_patch],
              facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────
st.title("🔍 Fake Review Detector")
st.markdown("### Paste any hotel or product review below")
st.markdown("The model will tell you if it\'s **fake or real** — and *explain why* word by word.")
st.markdown("---")

review = st.text_area(
    "Review Text",
    placeholder="e.g. Amazing hotel! The staff was incredibly helpful...",
    height=160
)

model_choice = st.radio(
    "Choose Model:",
    ["Logistic Regression (Recommended)", "Random Forest"],
    horizontal=True
)

analyze_btn = st.button("🔍 Analyze Review", use_container_width=True)

if analyze_btn:
    if not review.strip():
        st.warning("⚠️ Please paste a review first!")
    elif len(review.split()) < 5:
        st.warning("⚠️ Review is too short. Please enter at least 5 words.")
    else:
        model = lr_model if "Logistic" in model_choice else rf_model

        with st.spinner("Analyzing..."):
            label, proba = predict(review, model)
            time.sleep(0.5)

        st.markdown("---")

        if label == "FAKE":
            st.error(f"⚠️ This review appears to be **FAKE**")
        else:
            st.success(f"✅ This review appears to be **REAL**")

        col1, col2 = st.columns(2)
        col1.metric("Real Confidence", f"{proba[0]:.1%}")
        col2.metric("Fake Confidence", f"{proba[1]:.1%}")

        st.progress(float(proba[1]), text="Fake probability")

        st.markdown("---")
        st.markdown("### 🧠 Word-level Explanation")
        st.caption("Which words pushed the model toward FAKE or REAL?")

        with st.spinner("Generating explanation..."):
            exp = explain(review)

        fig = plot_explanation(exp)
        st.pyplot(fig)

        st.markdown("#### Top influencing words:")
        for word, weight in exp.as_list():
            direction = "🔴 → FAKE" if weight > 0 else "🟢 → REAL"
            st.markdown(f"- **{word}**: `{weight:+.4f}` {direction}")

st.markdown("---")
st.caption("Built with Logistic Regression + TF-IDF + LIME Explainability")
st.caption("Dataset: Deceptive Opinion Spam Corpus | Myle Ott et al.")
