# 🔍 Fake Review Detector with Explainability

> Detects fake/deceptive product and hotel reviews using Machine Learning — and explains **why** word by word using LIME.

🌐 **Live Demo:** [huggingface.co/spaces/Kanjani25/fake-review-detector](https://huggingface.co/spaces/Kanjani25/fake-review-detector)

---

## 📌 Problem Statement

Fake reviews are a growing problem on platforms like Amazon, Flipkart, and TripAdvisor. They mislead consumers and damage trust. Most existing detectors only say *what* — this project also explains *why*.

---

## 🧠 What Makes This Different

Most ML projects classify and stop there. This project adds an **explainability layer using LIME** that highlights exactly which words pushed the model toward FAKE or REAL — making the model transparent and trustworthy.

---

## 📊 Dataset

- **Source:** Deceptive Opinion Spam Corpus — Myle Ott et al.
- **Size:** 1,600 reviews (800 fake + 800 real)
- **Domain:** Hotel reviews from TripAdvisor
- **Labels:** deceptive (fake) vs truthful (real)

---

## 🔬 Approach

### 1. Exploratory Data Analysis
Engineered 9 behavioral features to find linguistic patterns in fake reviews:

| Feature | Finding |
|---|---|
| First Person Words (I/me/my) | Fake reviews use significantly more |
| Review Length | Similar — fakers put in effort |
| Sentiment Polarity | Not a strong discriminator alone |
| Subjectivity Score | Fake reviews slightly more subjective |

> Key Insight: Simple sentiment is not enough to detect fake reviews — vocabulary patterns matter more.

### 2. Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression + TF-IDF | 88.75% | Best performer — used in production |
| Random Forest + TF-IDF | 83.12% | Solid but lower than LR |
| DistilBERT (SST-2) | 51.25% | Sentiment model — wrong task alignment |

> Key Insight: DistilBERT underperformed because SST-2 detects sentiment, not deception. TF-IDF captures vocabulary patterns specific to deceptive writing.

### 3. Explainability with LIME

- **Fake reviews:** Words like amazing, experience, hotel brand names signal deception
- **Real reviews:** Words like location, floor, rate, helpful signal authenticity

> Liars oversell. Truth-tellers report specifics.

---

## Tech Stack

- Python, scikit-learn, LIME, Streamlit, HuggingFace, TextBlob, Matplotlib

---

## 🚀 Run Locally

---

## 💡 What I'd Add With More Time

- Multilingual support for Hindi and Indian languages
- Browser extension for Amazon/Flipkart
- Active learning loop with user feedback
- Fine-tuned BERT on deceptive text corpus
- SHAP global explanations

---

## References

- Myle Ott et al. — Finding Deceptive Opinion Spam by Any Stretch of the Imagination (ACL 2011)
- Ribeiro et al. — Why Should I Trust You?: Explaining the Predictions of Any Classifier

---

## Author

Kanjani25 — BTech CSE (AIML Specialization)
