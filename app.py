# ============================================================
# ComplaintIQ — AI-Powered Complaint Analysis System
# app.py — Main Streamlit Application
# ============================================================

import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
import io
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from risk_engine import analyse_complaint

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title  = "ComplaintIQ",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
HF_MODELS = {
    "DistilBERT": {
        "sentiment": "Sabrin23sultana/complaintiq-distilbert-sentiment",
        "priority":  "Sabrin23sultana/complaintiq-distilbert-priority",
    },
    "DeBERTa-v3-small": {
        "sentiment": "Sabrin23sultana/complaintiq-deberta-sentiment",
        "priority":  "Sabrin23sultana/complaintiq-deberta-priority",
    }
}

MAX_TOKEN_LEN = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RISK_COLORS = {
    "High":   "#FF4B4B",
    "Medium": "#FFA500",
    "Low":    "#21C354"
}

RISK_BG = {
    "High":   "#fff0f0",
    "Medium": "#fff8ee",
    "Low":    "#f0fff4"
}

# ============================================================
# MODEL LOADING — CACHED
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(repo_id: str, use_float32: bool = False):
    """Load tokenizer and model from HF Hub. Cached after first load."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model     = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        num_labels              = 2,
        ignore_mismatched_sizes = True
    )
    if use_float32:
        model = model.float()
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_models(model_choice: str):
    """Return loaded tokenizer and model pair for selected model."""
    use_f32 = model_choice == "DeBERTa-v3-small"
    with st.spinner(f"Loading {model_choice} models..."):
        tok_sent, mdl_sent = load_model(
            HF_MODELS[model_choice]["sentiment"],
            use_float32=use_f32
        )
        tok_pri, mdl_pri = load_model(
            HF_MODELS[model_choice]["priority"],
            use_float32=use_f32
        )
    return tok_sent, mdl_sent, tok_pri, mdl_pri


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def predict(text: str,
            tokenizer,
            model,
            use_float32: bool = False):
    """
    Run inference on a single text.
    Returns predicted label and confidence score.
    """
    encoded = tokenizer(
        text,
        max_length     = MAX_TOKEN_LEN,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt"
    )
    input_ids      = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    token_type_ids = encoded.get("token_type_ids")

    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(DEVICE)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits.float()
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label  = int(np.argmax(probs))
    conf   = float(probs[label])
    return label, conf, probs


def analyse_text(text: str,
                 tok_sent, mdl_sent,
                 tok_pri,  mdl_pri,
                 use_float32: bool = False) -> dict:
    """Run full analysis pipeline on a single complaint text."""
    sent_label, sent_conf, sent_probs = predict(
        text, tok_sent, mdl_sent, use_float32)
    pri_label,  pri_conf,  pri_probs  = predict(
        text, tok_pri,  mdl_pri,  use_float32)

    result = analyse_complaint(
        sentiment_pred = sent_label,
        priority_pred  = pri_label,
        sentiment_conf = sent_conf,
        priority_conf  = pri_conf,
        complaint_text = text
    )
    result["sentiment_probs"] = sent_probs.tolist()
    result["priority_probs"]  = pri_probs.tolist()
    return result


# ============================================================
# UI COMPONENTS
# ============================================================
def render_risk_card(result: dict):
    """Render the main risk assessment card."""
    risk     = result["risk_level"]
    color    = RISK_COLORS[risk]
    bg_color = RISK_BG[risk]
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 6px solid {color};
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 16px;
    ">
        <h2 style="color: {color}; margin: 0 0 4px 0;">
            {result['risk_icon']} Risk Level: {risk}
        </h2>
        <p style="color: #1a1a1a; margin: 0; font-size: 15px;">
            Risk Score: <strong>{result['risk_score']}</strong> / 1.0
            &nbsp;|&nbsp;
            Escalation Required:
            <strong>{'Yes' if result['escalation_flag'] else 'No'}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_metrics(result: dict):
    """Render sentiment and priority prediction columns."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎭 Emotional Tone")
        tone      = result["emotional_tone"]
        color     = "#FF4B4B" if tone == "Negative" else "#21C354"
        sent_conf = result["confidence_summary"]["sentiment_confidence"]
        sent_lvl  = result["confidence_summary"]["sentiment_confidence_level"]
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #e0e0e0;
        ">
            <h3 style="color: {color}; margin: 0 0 8px 0;">{tone}</h3>
            <p style="margin: 0; color: #444444; font-size: 14px;">
                Confidence: <strong>{sent_conf:.1%}</strong>
                ({sent_lvl})
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### ⏰ Urgency Level")
        urgency  = result["urgency_label"]
        color    = "#FF4B4B" if urgency == "Urgent" else "#21C354"
        pri_conf = result["confidence_summary"]["priority_confidence"]
        pri_lvl  = result["confidence_summary"]["priority_confidence_level"]
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #e0e0e0;
        ">
            <h3 style="color: {color}; margin: 0 0 8px 0;">{urgency}</h3>
            <p style="margin: 0; color: #444444; font-size: 14px;">
                Confidence: <strong>{pri_conf:.1%}</strong>
                ({pri_lvl})
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_confidence_bars(result: dict):
    """Render confidence probability bars for both tasks."""
    st.markdown("#### 📊 Prediction Confidence")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment**")
        sent_probs = result["sentiment_probs"]
        st.progress(float(sent_probs[0]),
                    text=f"Neutral: {sent_probs[0]:.1%}")
        st.progress(float(sent_probs[1]),
                    text=f"Negative: {sent_probs[1]:.1%}")

    with col2:
        st.markdown("**Priority**")
        pri_probs = result["priority_probs"]
        st.progress(float(pri_probs[0]),
                    text=f"Not Urgent: {pri_probs[0]:.1%}")
        st.progress(float(pri_probs[1]),
                    text=f"Urgent: {pri_probs[1]:.1%}")


def render_recommendations(result: dict):
    """Render actionable recommendations."""
    st.markdown("#### 📋 Recommended Actions")
    risk  = result["risk_level"]
    color = RISK_COLORS[risk]
    for rec in result["recommendations"]:
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-left: 3px solid {color};
            font-size: 14px;
            color: #1a1a1a;
            line-height: 1.5;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
            display: block;
            width: 100%;
            box-sizing: border-box;
        ">
            <span style="color: #1a1a1a;">{rec}</span>
        </div>
        """, unsafe_allow_html=True)


def render_complaint_stats(result: dict, text: str):
    """Render complaint text statistics."""
    st.markdown("#### 📝 Complaint Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count",  result["word_count"])
    col2.metric("Char Count",  len(text))
    col3.metric("Risk Score",  f"{result['risk_score']:.3f}")


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Render sidebar with model selection and project info."""
    with st.sidebar:
        st.title("🔍 ComplaintIQ")
        st.markdown(
            "AI-powered complaint analysis and "
            "escalation decision support."
        )
        st.divider()

        st.markdown("### ⚙️ Model Settings")
        model_choice = st.selectbox(
            "Select Model",
            options = ["DeBERTa-v3-small", "DistilBERT"],
            index   = 0,
            help    = (
                "DeBERTa-v3-small: Best performance\n"
                "DistilBERT: Faster inference"
            )
        )

        st.divider()
        st.markdown("### 📊 Model Performance")
        perf_df = pd.DataFrame({
            "Model":    ["TF-IDF+LR", "DistilBERT", "DeBERTa"],
            "Sent. F1": [0.6424, 0.6626, 0.6897],
            "Pri. F1":  [0.8229, 0.7941, 0.8330],
        })
        st.dataframe(
            perf_df,
            hide_index          = True,
            use_container_width = True
        )

        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown("""
        **ComplaintIQ** analyses customer complaints
        using NLP to detect:
        - 🎭 Emotional tone
        - ⏰ Urgency level
        - 🔴 Risk level
        - 📋 Recommended actions

        Built with DistilBERT and DeBERTa-v3-small
        fine-tuned on financial complaint data.
        """)

        st.divider()
        st.markdown(
            "<p style='font-size:12px; color:#888;'>"
            "AT2 NLP Project | 2025</p>",
            unsafe_allow_html=True
        )

    return model_choice


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Sidebar
    model_choice = render_sidebar()
    use_float32  = model_choice == "DeBERTa-v3-small"

    # Header
    st.markdown("""
    <h1 style='margin-bottom: 4px;'>
        🔍 ComplaintIQ
    </h1>
    <p style='color: #888; font-size: 16px; margin-top: 0;'>
        AI-powered complaint analysis and escalation
        decision support system
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Complaint Analysis",
        "📂 Batch Analysis (CSV)",
        "📊 Model Comparison"
    ])

    # ── TAB 1: Single Complaint ───────────────────────────
    with tab1:
        st.markdown("### Enter Customer Complaint")
        st.markdown(
            "Paste or type a customer complaint below "
            "to receive an instant risk assessment."
        )

        complaint_text = st.text_area(
            label            = "Complaint Text",
            placeholder      = (
                "e.g. I have been trying to resolve an "
                "unauthorised charge on my account for "
                "three weeks with no response..."
            ),
            height           = 180,
            label_visibility = "collapsed"
        )

        col1, col2, col3 = st.columns([1, 1, 4])
        analyse_btn = col1.button(
            "🔍 Analyse",
            type             = "primary",
            use_container_width = True
        )
        clear_btn = col2.button(
            "🗑️ Clear",
            type             = "secondary",
            use_container_width = True
        )

        if clear_btn:
            st.rerun()

        if analyse_btn:
            if not complaint_text.strip():
                st.warning(
                    "Please enter a complaint text to analyse.")
            elif len(complaint_text.split()) < 3:
                st.warning(
                    "Complaint text is too short. "
                    "Please enter at least a few sentences."
                )
            else:
                with st.spinner(
                    f"Analysing with {model_choice}..."
                ):
                    tok_sent, mdl_sent, tok_pri, mdl_pri = \
                        get_models(model_choice)
                    start  = time.perf_counter()
                    result = analyse_text(
                        complaint_text,
                        tok_sent, mdl_sent,
                        tok_pri,  mdl_pri,
                        use_float32=use_float32
                    )
                    elapsed = time.perf_counter() - start

                st.success(
                    f"Analysis complete in {elapsed:.2f}s "
                    f"using {model_choice}"
                )
                st.divider()

                render_risk_card(result)
                st.markdown("")
                render_prediction_metrics(result)
                st.markdown("")
                render_confidence_bars(result)
                st.markdown("")
                render_recommendations(result)
                st.markdown("")
                render_complaint_stats(result, complaint_text)

                with st.expander(
                    "🔧 Raw Analysis Output (JSON)"
                ):
                    display_result = {
                        k: v for k, v in result.items()
                        if k not in [
                            "sentiment_probs",
                            "priority_probs"
                        ]
                    }
                    st.json(display_result)

    # ── TAB 2: Batch Analysis ─────────────────────────────
    with tab2:
        st.markdown("### Batch Complaint Analysis")
        st.markdown(
            "Upload a CSV file containing complaint text. "
            "The file must have a column named "
            "`Consumer_complaint`."
        )

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type             = ["csv"],
            label_visibility = "collapsed"
        )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if "Consumer_complaint" not in batch_df.columns:
                    st.error(
                        "CSV must contain a column named "
                        "`Consumer_complaint`."
                    )
                else:
                    st.success(
                        f"File loaded: {len(batch_df)} complaints"
                    )
                    st.dataframe(
                        batch_df[["Consumer_complaint"]].head(5),
                        use_container_width=True
                    )

                    max_rows = st.slider(
                        "Number of complaints to analyse",
                        min_value = 1,
                        max_value = min(50, len(batch_df)),
                        value     = min(10, len(batch_df)),
                        help      = "Limited to 50 for performance"
                    )

                    if st.button(
                        "🔍 Run Batch Analysis",
                        type="primary"
                    ):
                        tok_sent, mdl_sent, tok_pri, mdl_pri = \
                            get_models(model_choice)

                        batch_subset = batch_df.head(
                            max_rows).reset_index(drop=True)
                        results_list = []
                        progress_bar = st.progress(
                            0, text="Analysing complaints...")

                        for i, row in batch_subset.iterrows():
                            text   = str(
                                row["Consumer_complaint"])[:1000]
                            result = analyse_text(
                                text,
                                tok_sent, mdl_sent,
                                tok_pri,  mdl_pri,
                                use_float32=use_float32
                            )
                            results_list.append({
                                "Complaint (preview)":
                                    text[:100] + "...",
                                "Emotional Tone":
                                    result["emotional_tone"],
                                "Urgency":
                                    result["urgency_label"],
                                "Risk Level":
                                    result["risk_level"],
                                "Risk Score":
                                    result["risk_score"],
                                "Escalation":
                                    result["escalation_flag"],
                                "Sentiment Conf":
                                    result["confidence_summary"][
                                        "sentiment_confidence"],
                                "Priority Conf":
                                    result["confidence_summary"][
                                        "priority_confidence"],
                            })
                            progress_bar.progress(
                                (i + 1) / max_rows,
                                text=(
                                    f"Analysing "
                                    f"{i+1}/{max_rows}..."
                                )
                            )

                        progress_bar.empty()
                        results_df = pd.DataFrame(results_list)

                        st.markdown("### Batch Results")
                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )

                        st.markdown("### Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(
                            "High Risk",
                            int((results_df["Risk Level"]
                                 == "High").sum())
                        )
                        c2.metric(
                            "Medium Risk",
                            int((results_df["Risk Level"]
                                 == "Medium").sum())
                        )
                        c3.metric(
                            "Low Risk",
                            int((results_df["Risk Level"]
                                 == "Low").sum())
                        )
                        c4.metric(
                            "Escalations",
                            int(results_df["Escalation"].sum())
                        )

                        csv_buffer = io.StringIO()
                        results_df.to_csv(
                            csv_buffer, index=False)
                        st.download_button(
                            label     = "⬇️ Download Results CSV",
                            data      = csv_buffer.getvalue(),
                            file_name = "complaintiq_results.csv",
                            mime      = "text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── TAB 3: Model Comparison ───────────────────────────
    with tab3:
        st.markdown("### Model Performance Comparison")
        st.markdown(
            "Comprehensive comparison of all three models "
            "evaluated on the held-out test set (263 samples)."
        )

        # ── Full comparison table ─────────────────────────
        st.markdown("#### 📋 Test Set Results")
        comparison_df = pd.DataFrame({
            "Model": [
                "TF-IDF + LR",
                "DistilBERT",
                "DeBERTa-v3-small"
            ],
            "Sentiment Macro F1": [0.6424, 0.6626, 0.6897],
            "Priority Macro F1":  [0.8229, 0.7941, 0.8330],
            "Avg Inference (ms)": [0.007,  16.48,  27.22],
            "Disk Size (MB)":     [0.84,   512.49, 1098.80],
            "Interpretable":      ["✅ Yes", "❌ No", "❌ No"],
            "Deployment Cost":    ["Very Low", "Moderate", "High"]
        })
        st.dataframe(
            comparison_df,
            hide_index          = True,
            use_container_width = True
        )

        st.divider()

        # ── Macro F1 bar charts ───────────────────────────
        st.markdown("#### 📊 Macro F1 Comparison")
        models  = ["TF-IDF + LR", "DistilBERT", "DeBERTa-v3-small"]
        colors  = ["#4575b4", "#74add1", "#d73027"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sentiment Task**")
            sent_f1s = [0.6424, 0.6626, 0.6897]
            sent_chart_df = pd.DataFrame({
                "Model":     models,
                "Macro F1":  sent_f1s
            }).set_index("Model")
            st.bar_chart(
                sent_chart_df,
                use_container_width = True,
                color               = "#d73027"
            )
            for model, f1 in zip(models, sent_f1s):
                delta = f1 - sent_f1s[0]
                delta_str = (
                    f"+{delta:.4f}" if delta > 0
                    else f"{delta:.4f}"
                )
                color = "normal" if delta >= 0 else "inverse"
                st.metric(
                    label = model,
                    value = f"{f1:.4f}",
                    delta = delta_str if model != "TF-IDF + LR"
                            else "baseline"
                )

        with col2:
            st.markdown("**Priority Task**")
            pri_f1s = [0.8229, 0.7941, 0.8330]
            pri_chart_df = pd.DataFrame({
                "Model":    models,
                "Macro F1": pri_f1s
            }).set_index("Model")
            st.bar_chart(
                pri_chart_df,
                use_container_width = True,
                color               = "#4575b4"
            )
            for model, f1 in zip(models, pri_f1s):
                delta = f1 - pri_f1s[0]
                delta_str = (
                    f"+{delta:.4f}" if delta > 0
                    else f"{delta:.4f}"
                )
                st.metric(
                    label = model,
                    value = f"{f1:.4f}",
                    delta = delta_str if model != "TF-IDF + LR"
                            else "baseline"
                )

        st.divider()

        # ── Per-class F1 tables ───────────────────────────
        st.markdown("#### 🔍 Per-Class F1 Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sentiment Classification**")
            sent_df = pd.DataFrame({
                "Model": models,
                "Neutral F1":      [0.79, 0.84, 0.83],
                "Negative F1":     [0.49, 0.49, 0.55],
                "Negative Recall": [0.54, 0.46, 0.60],
            })
            st.dataframe(
                sent_df,
                hide_index          = True,
                use_container_width = True
            )
            st.caption(
                "Negative Recall is the most critical metric — "
                "it measures how many negative complaints "
                "are correctly identified."
            )

        with col2:
            st.markdown("**Priority Classification**")
            pri_df = pd.DataFrame({
                "Model": models,
                "Not Urgent F1":     [0.73, 0.68, 0.75],
                "Urgent F1":         [0.92, 0.91, 0.92],
                "Not Urgent Recall": [0.79, 0.70, 0.84],
            })
            st.dataframe(
                pri_df,
                hide_index          = True,
                use_container_width = True
            )
            st.caption(
                "Not Urgent Recall measures how well the model "
                "avoids unnecessary escalations of "
                "non-urgent complaints."
            )

        st.divider()

        # ── Speed and efficiency ──────────────────────────
        st.markdown("#### ⚡ Speed & Efficiency Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "TF-IDF + LR",
            "0.007 ms/sample",
            "4,124× faster than DeBERTa",
            delta_color = "normal"
        )
        col2.metric(
            "DistilBERT",
            "16.48 ms/sample",
            "1.65× faster than DeBERTa",
            delta_color = "normal"
        )
        col3.metric(
            "DeBERTa-v3-small",
            "27.22 ms/sample",
            "Best performance",
            delta_color = "off"
        )

        st.markdown("")
        speed_df = pd.DataFrame({
            "Model":          models,
            "Inference (ms)": [0.007, 16.48, 27.22],
            "Disk Size (MB)": [0.84, 512.49, 1098.80],
            "Parameters (M)": [0.05, 66.96, 141.90]
        }).set_index("Model")

        st.markdown("**Inference Time (ms per sample)**")
        st.bar_chart(
            speed_df[["Inference (ms)"]],
            use_container_width = True,
            color               = "#ff7f0e"
        )

        st.divider()

        # ── Trade-off summary ─────────────────────────────
        st.markdown("#### 🎯 Model Selection Guide")
        st.markdown(
            "Choose the right model based on your deployment needs:"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="
                background: #e8f4f8;
                border-radius: 10px;
                padding: 20px;
                border-top: 4px solid #4575b4;
                color: #1a1a1a;
                min-height: 220px;
            ">
                <h4 style="color: #4575b4; margin-top: 0;">
                    TF-IDF + LR
                </h4>
                <p style="color: #1a1a1a; font-size: 13px;">
                    ✅ Best for high-volume real-time systems<br>
                    ✅ Fully interpretable decisions<br>
                    ✅ Minimal infrastructure needed<br>
                    ✅ 0.84 MB deployment size<br>
                    ⚠️ Lower sentiment detection<br>
                    ⚠️ Cannot capture context
                </p>
                <p style="
                    color: #4575b4;
                    font-weight: bold;
                    font-size: 13px;
                ">
                    Use when: speed and interpretability
                    are priorities
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="
                background: #e8f0f8;
                border-radius: 10px;
                padding: 20px;
                border-top: 4px solid #74add1;
                color: #1a1a1a;
                min-height: 220px;
            ">
                <h4 style="color: #74add1; margin-top: 0;">
                    DistilBERT
                </h4>
                <p style="color: #1a1a1a; font-size: 13px;">
                    ✅ Good sentiment detection<br>
                    ✅ Faster than DeBERTa<br>
                    ✅ Smaller than DeBERTa<br>
                    ⚠️ Underperforms TF-IDF on priority<br>
                    ⚠️ Not interpretable<br>
                    ⚠️ 512 MB deployment size
                </p>
                <p style="
                    color: #74add1;
                    font-weight: bold;
                    font-size: 13px;
                ">
                    Use when: moderate performance
                    with lower cost than DeBERTa
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="
                background: #fdf0f0;
                border-radius: 10px;
                padding: 20px;
                border-top: 4px solid #d73027;
                color: #1a1a1a;
                min-height: 220px;
            ">
                <h4 style="color: #d73027; margin-top: 0;">
                    DeBERTa-v3-small ⭐
                </h4>
                <p style="color: #1a1a1a; font-size: 13px;">
                    ✅ Best overall performance<br>
                    ✅ Best minority class recall<br>
                    ✅ Best sentiment detection<br>
                    ✅ Used in ComplaintIQ<br>
                    ⚠️ Highest inference time<br>
                    ⚠️ 1.1 GB deployment size
                </p>
                <p style="
                    color: #d73027;
                    font-weight: bold;
                    font-size: 13px;
                ">
                    Use when: maximum accuracy
                    is the priority
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Error analysis summary ────────────────────────
        st.markdown("#### 🔎 Error Analysis Summary")
        st.markdown(
            "Key findings from error analysis on "
            "DeBERTa-v3-small (263 test samples):"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sentiment Task Errors**")
            err_sent_df = pd.DataFrame({
                "Error Type":        ["False Positive", "False Negative"],
                "Count":             [39, 27],
                "Mean Confidence":   ["71.7%", "75.2%"],
                "Primary Cause":     [
                    "Label ambiguity",
                    "Implicit emotion"
                ]
            })
            st.dataframe(
                err_sent_df,
                hide_index          = True,
                use_container_width = True
            )
            st.caption(
                "Highest error rate in Loans (34.8%) "
                "and short complaints (Q1: 37.9%)"
            )

        with col2:
            st.markdown("**Priority Task Errors**")
            err_pri_df = pd.DataFrame({
                "Error Type":        ["False Positive", "False Negative"],
                "Count":             [9, 23],
                "Mean Confidence":   ["88.6%", "87.5%"],
                "Primary Cause":     [
                    "Legal language",
                    "Long complaint truncation"
                ]
            })
            st.dataframe(
                err_pri_df,
                hide_index          = True,
                use_container_width = True
            )
            st.caption(
                "Highest error rate in Money Transfer (23.7%) "
                "and long complaints (Q4: 30.3%)"
            )

        st.divider()

        # ── Key findings ──────────────────────────────────
        st.markdown("#### 💡 Key Findings")
        st.success(
            "🏆 **DeBERTa-v3-small** achieves the best "
            "performance on both tasks (Sentiment F1: 0.6897, "
            "Priority F1: 0.8330) and is used as the default "
            "model in ComplaintIQ."
        )
        st.warning(
            "⚡ **TF-IDF + LR** is 4,124× faster than DeBERTa "
            "at only 0.84 MB — the best choice for high-volume "
            "real-time environments where speed and "
            "interpretability are prioritised over marginal "
            "performance gains."
        )
        st.error(
            "📉 **DistilBERT** underperforms TF-IDF on the "
            "priority task (0.7941 vs 0.8229 Macro F1), "
            "confirming that urgency signals in this dataset "
            "are partially keyword-driven, favouring sparse "
            "TF-IDF representations on small training sets."
        )
        st.info(
            "🔀 **Hybrid system:** ComplaintIQ combines neural "
            "predictions with a keyword safety override layer, "
            "addressing high-confidence false negatives "
            "identified in error analysis. This reflects "
            "production best practice for critical "
            "classification systems."
        )


if __name__ == "__main__":
    main()
