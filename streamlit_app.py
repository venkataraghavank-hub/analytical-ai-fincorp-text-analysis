import hmac
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FINCORP Customer Message Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SENTIMENT_MODEL_PATH = MODEL_DIR / "fincorp_sentiment_pipeline.pkl"
EMOTION_MODEL_PATH = MODEL_DIR / "fincorp_emotion_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "sentiment_emotion_metadata.json"
DEFAULT_THRESHOLD = 0.60


st.markdown(
    """
    <style>
    :root{
        --navy:#082b61;--blue:#2670b8;--teal:#13a4aa;--ink:#17324d;
        --muted:#647f9d;--line:#dbe7f3;--soft:#f3f9ff;--warning:#fff5db
    }
    .stApp{background:#f3f9ff;color:var(--ink)}
    .block-container{max-width:1400px;padding:1.2rem 2.2rem 2rem}
    header[data-testid="stHeader"]{background:transparent}
    #MainMenu,footer{visibility:hidden}
    [data-testid="stSidebar"]{background:#082b61}
    [data-testid="stSidebar"] *{color:#eef7ff}
    [data-testid="stSidebar"] div[role="radiogroup"] label{
        background:rgba(255,255,255,.07);border-radius:10px;
        padding:.55rem .7rem;margin:.2rem 0
    }
    .hero{padding:.25rem .25rem 1rem}
    .eyebrow{font-size:.76rem;font-weight:800;letter-spacing:.15em;color:var(--blue)}
    .hero h1{font-size:clamp(2rem,3vw,2.75rem);line-height:1.05;color:var(--navy);
        letter-spacing:-.035em;margin:.4rem 0}
    .hero p{font-size:1.03rem;color:#587596;margin:0;max-width:950px}
    .notice{margin:.05rem .25rem 1rem;padding:.72rem .95rem;border-left:4px solid var(--blue);
        border-radius:8px;background:#eaf4ff;color:#365d82;font-size:.88rem}
    .access{text-align:center;padding:.4rem 0 .8rem}
    .access h1{font-size:2rem;color:var(--navy);margin:.3rem 0}
    .access p{color:#65809d}
    .access-note{text-align:center;color:#7890a8;font-size:.78rem;margin-top:.75rem}
    [data-testid="stVerticalBlockBorderWrapper"]{
        background:rgba(255,255,255,.98);border:1px solid #dce8f4!important;
        border-radius:18px!important;box-shadow:0 10px 30px rgba(18,59,112,.07)
    }
    .section-title{font-size:1.3rem;font-weight:800;color:#092f66}
    .section-copy{color:#65809d;margin:.15rem 0 .9rem}
    .result-card{padding:1rem 1.1rem;border-radius:14px;background:#eefafc;
        border:1px solid #d8f0f1;min-height:145px}
    .result-label{font-size:.75rem;font-weight:800;letter-spacing:.09em;
        color:#53809a;text-transform:uppercase}
    .decision{font-size:1.65rem;font-weight:850;color:#087f89;margin:.3rem 0}
    .small-note{font-size:.82rem;color:#6c849b}
    .review-ok{border-left:6px solid var(--teal)}
    .review-needed{border-left:6px solid #e79020;background:var(--warning)}
    .safety{padding:.8rem 1rem;border:1px solid #f1b942;border-radius:11px;
        color:#8a4d05;background:#fff5db;font-weight:650}
    .footer{border-top:1px solid var(--line);margin-top:1.5rem;padding-top:1rem;
        color:#6d849b;font-size:.82rem}
    div[data-testid="stMetric"]{background:white;border:1px solid #dce8f4;
        border-radius:14px;padding:.85rem}
    div.stButton>button[kind="primary"]{background:var(--teal);border:0;
        border-radius:10px;min-height:3rem;font-weight:750}
    div.stButton>button[kind="primary"]:hover{background:#0b9299;border:0}
    @media(max-width:800px){.block-container{padding:1rem}}
    </style>
    """,
    unsafe_allow_html=True,
)


def require_access():
    """Show the same classroom access experience used by the reference app."""
    if st.session_state.get("class_access_granted", False):
        return

    try:
        expected_code = st.secrets.get("STUDENT_ACCESS_CODE", "")
    except Exception:
        expected_code = ""
    expected_code = expected_code or os.getenv("STUDENT_ACCESS_CODE", "")

    _, gate, _ = st.columns([1, 1.15, 1])
    with gate:
        with st.container(border=True):
            st.markdown(
                """
                <div class="access">
                  <div class="eyebrow">AI APPLICATIONS LAB</div>
                  <h1>Student Lab Access</h1>
                  <p>Enter the class access code to open FINCORP Customer Message Intelligence.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            entered_code = st.text_input(
                "Class access code",
                type="password",
                placeholder="Enter the code provided in class",
            )
            if st.button("Enter Application", type="primary", use_container_width=True):
                if expected_code and hmac.compare_digest(entered_code, expected_code):
                    st.session_state["class_access_granted"] = True
                    st.rerun()
                st.error("Incorrect class code. Please try again.")
            if not expected_code:
                st.warning("Class access has not been configured by the application owner.")
            st.markdown(
                '<div class="access-note">Access is restricted to classroom participants.</div>',
                unsafe_allow_html=True,
            )
    st.stop()


@st.cache_resource
def load_assets():
    required_files = [SENTIMENT_MODEL_PATH, EMOTION_MODEL_PATH, METADATA_PATH]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing deployment file(s): " + ", ".join(missing_files))

    sentiment_pipeline = joblib.load(SENTIMENT_MODEL_PATH)
    emotion_pipeline = joblib.load(EMOTION_MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return sentiment_pipeline, emotion_pipeline, metadata


def render_header(title, subtitle):
    title_column, controls = st.columns([5, 2])
    with title_column:
        st.markdown(
            f"""
            <div class="hero">
              <div class="eyebrow">ANALYTICAL AI · TEXT ANALYTICS</div>
              <h1>{title}</h1><p>{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with controls:
        st.link_button(
            "← Back to AI Applications Lab",
            "https://aiapplicationslab.in",
            use_container_width=True,
        )
        if st.button("Exit Lab", use_container_width=True):
            st.session_state["class_access_granted"] = False
            st.rerun()

    st.markdown(
        """
        <div class="notice"><strong>Educational decision-support prototype:</strong>
        The two models classify the text entered by the user. Results support review and
        prioritisation; they do not establish a person's psychological state or replace
        human judgement.</div>
        """,
        unsafe_allow_html=True,
    )


def probability_output(pipeline, text):
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.named_steps["classifier"].classes_
    best_position = int(np.argmax(probabilities))
    ranking = pd.DataFrame(
        {
            "Class": [str(value).title() for value in classes],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)
    return str(classes[best_position]), float(probabilities[best_position]), ranking


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


def render_analysis_page(sentiment_pipeline, emotion_pipeline, metadata):
    render_header(
        "Customer Message Intelligence",
        "Analyse one message through two independent views: overall sentiment and expressed emotion.",
    )
    threshold = float(metadata.get("confidence_threshold", DEFAULT_THRESHOLD))
    examples = {
        "Write my own": "",
        "Payment failure": "The payment failed again and I am extremely upset with the service.",
        "Positive service": "The issue was resolved quickly and I am very happy with the support.",
        "Status enquiry": "I submitted the request yesterday and would like to know the status.",
    }

    input_column, guidance_column = st.columns([3, 2], gap="large")
    with input_column:
        with st.container(border=True):
            st.markdown('<div class="section-title">Message for analysis</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">Use a classroom example or enter one short customer message.</div>',
                unsafe_allow_html=True,
            )
            selected_example = st.selectbox("Optional classroom example", examples)
            customer_text = st.text_area(
                "Customer message",
                value=examples[selected_example],
                height=165,
                max_chars=5000,
                placeholder="Type or paste one customer message here...",
            )
            run_analysis = st.button(
                "Analyse Message", type="primary", use_container_width=True
            )
    with guidance_column:
        with st.container(border=True):
            st.markdown('<div class="section-title">What the application returns</div>', unsafe_allow_html=True)
            st.write("**Sentiment** — negative, neutral, or positive")
            st.write("**Emotion** — one of seven affective labels")
            st.write("**Confidence** — highest estimated class probability")
            st.write("**Review status** — based on the lower confidence score")
            st.caption(f"Current human-review threshold: {threshold:.0%}")

    if not run_analysis:
        return
    if not customer_text.strip():
        st.warning("Please enter a customer message before running the analysis.")
        return

    result = analyse_message(
        customer_text.strip(), sentiment_pipeline, emotion_pipeline, threshold
    )
    st.markdown("### Analysis result")
    sentiment_column, emotion_column = st.columns(2)
    with sentiment_column:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Sentiment</div>
              <div class="decision">{result['sentiment'].title()}</div>
              <div>Model confidence: <strong>{result['sentiment_confidence']:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with emotion_column:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Emotion</div>
              <div class="decision">{result['emotion'].title()}</div>
              <div>Model confidence: <strong>{result['emotion_confidence']:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result["requires_review"]:
        review_class = "review-needed"
        review_title = "Human review recommended"
        review_copy = (
            "At least one confidence score is below the operating threshold. "
            "Review the original message before acting on the classification."
        )
    else:
        review_class = "review-ok"
        review_title = "Automated result allowed"
        review_copy = (
            "Both confidence scores meet the operating threshold. Human oversight may "
            "still be appropriate for sensitive or high-impact cases."
        )

    st.markdown(
        f"""
        <div class="result-card {review_class}" style="margin-top:1rem;min-height:auto">
          <div class="result-label">Review decision</div>
          <div class="decision" style="font-size:1.35rem">{review_title}</div>
          <div>{review_copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View all class probabilities"):
        first, second = st.columns(2)
        with first:
            st.markdown("#### Sentiment probabilities")
            st.dataframe(
                result["sentiment_ranking"],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        format="percent", min_value=0, max_value=1
                    )
                },
            )
        with second:
            st.markdown("#### Emotion probabilities")
            st.dataframe(
                result["emotion_ranking"],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        format="percent", min_value=0, max_value=1
                    )
                },
            )

    st.markdown(
        '<div class="safety">Confidence is not a guarantee of correctness. Do not enter confidential, personally identifiable, or sensitive customer information.</div>',
        unsafe_allow_html=True,
    )


def render_method_page():
    render_header(
        "How the Application Works",
        "A deployment view of the CRISP-DM text-classification workflow used in the FINCORP notebook.",
    )
    with st.container(border=True):
        st.markdown('<div class="section-title">Two independent pipelines</div>', unsafe_allow_html=True)
        st.markdown(
            """
            1. The raw message is cleaned inside each fitted pipeline.
            2. TF-IDF converts words and short phrases into weighted numerical features.
            3. One classifier predicts **sentiment**; another independently predicts **emotion**.
            4. The lowest confidence score determines whether human review is recommended.

            Sentiment is not used as an input for emotion, and emotion is not used as an input for sentiment.
            """
        )
    with st.container(border=True):
        st.markdown('<div class="section-title">Appropriate use</div>', unsafe_allow_html=True)
        st.write("The outputs can support service triage, communication design and aggregate monitoring.")
        st.write("They should not be treated as causal explanations, clinical assessments or final decisions about customers.")


def render_model_page(metadata):
    render_header(
        "Model Information",
        "Selected algorithms, test performance and operating limitations recorded during deployment.",
    )
    sentiment_metadata = metadata.get("sentiment", {})
    emotion_metadata = metadata.get("emotion", {})
    sentiment_metrics = sentiment_metadata.get("test_metrics", {})
    emotion_metrics = emotion_metadata.get("test_metrics", {})

    first, second = st.columns(2)
    with first:
        with st.container(border=True):
            st.markdown('<div class="section-title">Sentiment model</div>', unsafe_allow_html=True)
            st.write("Selected model:", sentiment_metadata.get("model_name", "Not recorded"))
            m1, m2 = st.columns(2)
            m1.metric("Test accuracy", f"{sentiment_metrics.get('accuracy', 0):.1%}")
            m2.metric("Test macro-F1", f"{sentiment_metrics.get('macro_f1', 0):.1%}")
    with second:
        with st.container(border=True):
            st.markdown('<div class="section-title">Emotion model</div>', unsafe_allow_html=True)
            st.write("Selected model:", emotion_metadata.get("model_name", "Not recorded"))
            m1, m2 = st.columns(2)
            m1.metric("Test accuracy", f"{emotion_metrics.get('accuracy', 0):.1%}")
            m2.metric("Test macro-F1", f"{emotion_metrics.get('macro_f1', 0):.1%}")

    with st.container(border=True):
        st.markdown('<div class="section-title">Limitations</div>', unsafe_allow_html=True)
        limitations = metadata.get("limitations", [])
        if limitations:
            for limitation in limitations:
                st.markdown(f"- {limitation}")
        else:
            st.write("No limitations were recorded in the deployment metadata.")


require_access()

try:
    sentiment_model, emotion_model, model_metadata = load_assets()
except Exception as error:
    st.error("The FINCORP deployment files could not be loaded.")
    st.code(str(error))
    st.info(
        "Keep this file in the repository root and place the two fitted pipelines and "
        "metadata JSON inside the models folder."
    )
    st.stop()

with st.sidebar:
    st.markdown("## FINCORP Lab")
    st.caption("Sentiment and Emotion Analysis")
    page = st.radio(
        "Navigate",
        ["Analyse a Message", "How It Works", "Model Information"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Interpretable TF-IDF pipelines · No neural networks")

if page == "Analyse a Message":
    render_analysis_page(sentiment_model, emotion_model, model_metadata)
elif page == "How It Works":
    render_method_page()
else:
    render_model_page(model_metadata)

st.markdown(
    '<div class="footer">AI Applications Lab · Educational use only · Model outputs require responsible human interpretation.</div>',
    unsafe_allow_html=True,
)

