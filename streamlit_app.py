from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FINCORP Customer Message Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path(__file__).parent / "models"
SENTIMENT_MODEL_PATH = MODEL_DIR / "fincorp_sentiment_pipeline.pkl"
EMOTION_MODEL_PATH = MODEL_DIR / "fincorp_emotion_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "sentiment_emotion_metadata.json"

DEFAULT_THRESHOLD = 0.60

st.markdown(
    """
    <style>
    :root {
        --navy: #0e1b3b;
        --blue: #1c3569;
        --teal: #19b8ad;
        --orange: #ff6337;
        --purple: #7c4dcc;
        --soft: #f4f7fb;
    }
    .stApp {background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 45%);}
    .main .block-container {max-width: 1180px; padding-top: 1.6rem;}
    .hero {
        padding: 1.7rem 2rem; border-radius: 20px;
        background: linear-gradient(125deg, var(--navy), var(--blue));
        color: white; margin-bottom: 1.2rem;
        box-shadow: 0 12px 28px rgba(14,27,59,.14);
    }
    .hero h1 {margin: 0; font-size: 2.15rem; color: white;}
    .hero p {margin: .45rem 0 0; color: #dce6ff; font-size: 1.02rem;}
    .result-card {
        border: 1px solid #e3e9f2; border-radius: 16px; padding: 1.15rem 1.2rem;
        background: white; box-shadow: 0 7px 18px rgba(14,27,59,.07);
    }
    .result-label {font-size: .76rem; font-weight: 800; letter-spacing: .08em; color: #667085;}
    .result-value {font-size: 1.75rem; font-weight: 800; color: var(--navy); margin: .15rem 0;}
    .review-ok {border-left: 7px solid var(--teal);}
    .review-needed {border-left: 7px solid var(--orange);}
    .small-note {font-size: .86rem; color: #667085;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e3e9f2; padding: 1rem;
        border-radius: 14px;
    }
    .stButton > button {
        background: var(--orange); color: white; border: 0; border-radius: 10px;
        font-weight: 700; min-height: 2.8rem;
    }
    .stButton > button:hover {background: #e84d24; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)


def secret_value(*names):
    for name in names:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass
    return None


def show_access_gate():
    expected_code = secret_value("APP_ACCESS_CODE", "ACCESS_CODE")
    if not expected_code:
        st.error("The application access code has not been configured.")
        st.caption("Add APP_ACCESS_CODE to Streamlit Community Cloud secrets.")
        st.stop()

    if st.session_state.get("fincorp_authenticated"):
        return

    st.markdown(
        """
        <div class="hero">
          <h1>FINCORP Customer Message Intelligence</h1>
          <p>Secure classroom demonstration of interpretable sentiment and emotion classification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, centre, right = st.columns([1, 1.35, 1])
    with centre:
        with st.form("access_form"):
            st.subheader("Authorised access")
            entered_code = st.text_input("Access code", type="password")
            submitted = st.form_submit_button("Enter application", use_container_width=True)
        if submitted:
            if entered_code == expected_code:
                st.session_state["fincorp_authenticated"] = True
                st.rerun()
            else:
                st.error("The access code is incorrect.")
    st.stop()


@st.cache_resource
def load_deployment_files():
    required = [SENTIMENT_MODEL_PATH, EMOTION_MODEL_PATH, METADATA_PATH]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing deployment file(s): " + ", ".join(missing)
        )

    sentiment_pipeline = joblib.load(SENTIMENT_MODEL_PATH)
    emotion_pipeline = joblib.load(EMOTION_MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return sentiment_pipeline, emotion_pipeline, metadata


def probability_output(pipeline, text):
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.named_steps["classifier"].classes_
    best_position = int(np.argmax(probabilities))
    ranked = pd.DataFrame({
        "Class": [str(value).title() for value in classes],
        "Probability": probabilities,
    }).sort_values("Probability", ascending=False)
    return str(classes[best_position]), float(probabilities[best_position]), ranked


def analyse_message(text, sentiment_pipeline, emotion_pipeline, threshold):
    sentiment, sentiment_confidence, sentiment_ranking = probability_output(
        sentiment_pipeline, text
    )
    emotion, emotion_confidence, emotion_ranking = probability_output(
        emotion_pipeline, text
    )
    requires_review = min(sentiment_confidence, emotion_confidence) < threshold
    return {
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "sentiment_ranking": sentiment_ranking,
        "emotion": emotion,
        "emotion_confidence": emotion_confidence,
        "emotion_ranking": emotion_ranking,
        "requires_review": requires_review,
    }


show_access_gate()

st.markdown(
    """
    <div class="hero">
      <h1>FINCORP Customer Message Intelligence</h1>
      <p>One message. Two complementary views: overall sentiment and expressed emotion.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    sentiment_model, emotion_model, model_metadata = load_deployment_files()
except Exception as error:
    st.error("The model files could not be loaded.")
    st.code(str(error))
    st.info(
        "Copy the two pipeline files and metadata JSON produced by the Colab notebook "
        "into the repository's models folder, then redeploy."
    )
    st.stop()

threshold = float(model_metadata.get("confidence_threshold", DEFAULT_THRESHOLD))

analyse_tab, method_tab, model_tab = st.tabs(
    ["Analyse a message", "How it works", "Model information"]
)

with analyse_tab:
    st.subheader("Enter a customer message")
    sample_options = {
        "Write my own": "",
        "Payment failure": "The payment failed again and I am extremely upset with the service.",
        "Positive service": "The issue was resolved quickly and I am very happy with the support.",
        "Uncertain enquiry": "I submitted the request yesterday and would like to know the status.",
    }
    selected_sample = st.selectbox("Optional classroom example", sample_options.keys())
    starting_text = sample_options[selected_sample]
    customer_text = st.text_area(
        "Customer message",
        value=starting_text,
        height=145,
        placeholder="Type or paste one customer message here...",
        max_chars=5_000,
    )

    if st.button("Analyse message", type="primary", use_container_width=True):
        if not customer_text.strip():
            st.warning("Please enter a customer message before running the analysis.")
        else:
            result = analyse_message(
                customer_text.strip(), sentiment_model, emotion_model, threshold
            )

            first, second = st.columns(2)
            with first:
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="result-label">SENTIMENT</div>
                      <div class="result-value">{result['sentiment'].title()}</div>
                      <div>Confidence: <b>{result['sentiment_confidence']:.1%}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with second:
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="result-label">EMOTION</div>
                      <div class="result-value">{result['emotion'].title()}</div>
                      <div>Confidence: <b>{result['emotion_confidence']:.1%}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            review_class = "review-needed" if result["requires_review"] else "review-ok"
            review_title = (
                "Human review recommended"
                if result["requires_review"]
                else "Automated result allowed"
            )
            review_text = (
                "At least one confidence score is below the operating threshold. "
                "Use the output as decision support and review the message manually."
                if result["requires_review"]
                else "Both confidence scores meet the current operating threshold. "
                "Human oversight may still be appropriate for high-impact cases."
            )
            st.markdown(
                f"""
                <div class="result-card {review_class}" style="margin-top:1rem;">
                  <div class="result-label">REVIEW DECISION</div>
                  <div class="result-value" style="font-size:1.3rem;">{review_title}</div>
                  <div>{review_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View class probabilities"):
                probability_col_1, probability_col_2 = st.columns(2)
                with probability_col_1:
                    st.markdown("**Sentiment probabilities**")
                    sentiment_chart = result["sentiment_ranking"].set_index("Class")
                    st.bar_chart(sentiment_chart, horizontal=True)
                with probability_col_2:
                    st.markdown("**Emotion probabilities**")
                    emotion_chart = result["emotion_ranking"].set_index("Class")
                    st.bar_chart(emotion_chart, horizontal=True)

            st.caption(
                f"Operating threshold: {threshold:.0%}. Confidence is the model's highest "
                "estimated class probability; it is not a guarantee of correctness."
            )

with method_tab:
    st.subheader("Two separate classification pipelines")
    st.markdown(
        """
        The same raw message is passed independently through two pipelines:

        1. **TF-IDF** converts words and short phrases into weighted numerical features.
        2. The **sentiment classifier** predicts negative, neutral, or positive.
        3. The **emotion classifier** predicts anger, disgust, fear, happiness, other, sadness, or surprise.
        4. The lowest of the two confidence scores determines whether human review is recommended.

        Sentiment is not used to predict emotion, and emotion is not used to predict sentiment.
        """
    )
    st.info(
        "The models support service triage and aggregate monitoring. They do not infer a "
        "person's stable psychological state or replace human judgement."
    )

with model_tab:
    st.subheader("Deployment metadata")
    sentiment_meta = model_metadata.get("sentiment", {})
    emotion_meta = model_metadata.get("emotion", {})

    metric_col_1, metric_col_2 = st.columns(2)
    with metric_col_1:
        st.markdown("#### Sentiment")
        st.write("Selected model:", sentiment_meta.get("model_name", "Not recorded"))
        sentiment_metrics = sentiment_meta.get("test_metrics", {})
        if sentiment_metrics:
            st.metric("Test accuracy", f"{sentiment_metrics.get('accuracy', 0):.1%}")
            st.metric("Test macro-F1", f"{sentiment_metrics.get('macro_f1', 0):.1%}")
    with metric_col_2:
        st.markdown("#### Emotion")
        st.write("Selected model:", emotion_meta.get("model_name", "Not recorded"))
        emotion_metrics = emotion_meta.get("test_metrics", {})
        if emotion_metrics:
            st.metric("Test accuracy", f"{emotion_metrics.get('accuracy', 0):.1%}")
            st.metric("Test macro-F1", f"{emotion_metrics.get('macro_f1', 0):.1%}")

    st.markdown("#### Limitations")
    for limitation in model_metadata.get("limitations", []):
        st.markdown(f"- {limitation}")

st.divider()
st.caption(
    "Educational decision-support demonstration. Do not submit confidential, personally "
    "identifiable, or sensitive customer information. Predictions may be incorrect."
)

