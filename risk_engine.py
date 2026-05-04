# ============================================================
# risk_engine.py
# Decision-support layer for ComplaintIQ.
# ============================================================

import numpy as np

RISK_MATRIX = {
    (0, 0): "Low",
    (0, 1): "Medium",
    (1, 0): "Medium",
    (1, 1): "High"
}

RISK_SCORE_BANDS = {
    "Low":    (0.0,  0.33),
    "Medium": (0.34, 0.66),
    "High":   (0.67, 1.0)
}

RISK_COLORS = {
    "Low":    "🟢",
    "Medium": "🟡",
    "High":   "🔴"
}

RECOMMENDATIONS = {
    "High": [
        "🚨 Escalate immediately to senior support agent or team lead.",
        "📞 Initiate direct contact with customer within 1 hour.",
        "📋 Log complaint as high-priority in the support ticketing system.",
        "🔍 Review account history for prior complaints or escalations.",
        "📝 Document all actions taken for compliance and audit trail.",
        "💬 Acknowledge complaint in writing with expected resolution timeline."
    ],
    "Medium": [
        "⚠️ Flag complaint for review within 4 hours.",
        "📋 Assign to available support agent with relevant product expertise.",
        "📞 Schedule callback or follow-up within the same business day.",
        "🔍 Review complaint details for potential escalation triggers.",
        "💬 Send acknowledgement to customer confirming receipt of complaint."
    ],
    "Low": [
        "✅ Log complaint in standard support queue.",
        "📋 Assign to next available support agent.",
        "💬 Send automated acknowledgement to customer.",
        "📝 Monitor for follow-up or escalation within 48 hours."
    ]
}

SIGNAL_RECOMMENDATIONS = {
    "negative_high_confidence": (
        "😤 High confidence negative sentiment detected — "
        "prioritise empathetic communication in response."
    ),
    "urgent_high_confidence": (
        "⏰ High confidence urgent priority — "
        "do not defer to standard queue processing."
    ),
    "negative_low_confidence": (
        "🔎 Sentiment signal is uncertain — "
        "manual review of complaint tone recommended."
    ),
    "urgent_low_confidence": (
        "🔎 Urgency signal is uncertain — "
        "review complaint text manually before triaging."
    )
}

CONFIDENCE_THRESHOLD_HIGH = 0.75
CONFIDENCE_THRESHOLD_LOW  = 0.55


def compute_risk_score(sentiment_pred, priority_pred,
                        sentiment_conf, priority_conf):
    risk_level = RISK_MATRIX[(sentiment_pred, priority_pred)]
    band_min, band_max = RISK_SCORE_BANDS[risk_level]
    band_mid   = (band_min + band_max) / 2
    band_width = band_max - band_min
    avg_conf   = (sentiment_conf + priority_conf) / 2
    delta      = (avg_conf - 0.5) * band_width
    score      = np.clip(band_mid + delta, band_min, band_max)
    return round(float(score), 4)


def get_signal_recommendations(sentiment_pred, priority_pred,
                                 sentiment_conf, priority_conf):
    extra = []
    if sentiment_pred == 1:
        if sentiment_conf >= CONFIDENCE_THRESHOLD_HIGH:
            extra.append(
                SIGNAL_RECOMMENDATIONS["negative_high_confidence"])
        elif sentiment_conf < CONFIDENCE_THRESHOLD_LOW:
            extra.append(
                SIGNAL_RECOMMENDATIONS["negative_low_confidence"])
    if priority_pred == 1:
        if priority_conf >= CONFIDENCE_THRESHOLD_HIGH:
            extra.append(
                SIGNAL_RECOMMENDATIONS["urgent_high_confidence"])
        elif priority_conf < CONFIDENCE_THRESHOLD_LOW:
            extra.append(
                SIGNAL_RECOMMENDATIONS["urgent_low_confidence"])
    return extra


def analyse_complaint(sentiment_pred, priority_pred,
                       sentiment_conf, priority_conf,
                       complaint_text=""):
    risk_level      = RISK_MATRIX[(sentiment_pred, priority_pred)]
    risk_score      = compute_risk_score(
        sentiment_pred, priority_pred,
        sentiment_conf, priority_conf
    )
    emotional_tone  = "Negative" if sentiment_pred == 1 else "Neutral"
    urgency_label   = "Urgent"   if priority_pred  == 1 else "Not Urgent"
    escalation_flag = risk_level == "High"
    base_recs       = RECOMMENDATIONS[risk_level].copy()
    signal_recs     = get_signal_recommendations(
        sentiment_pred, priority_pred,
        sentiment_conf, priority_conf
    )
    all_recs = signal_recs + base_recs

    def conf_label(c):
        if c >= CONFIDENCE_THRESHOLD_HIGH:
            return "High"
        elif c >= CONFIDENCE_THRESHOLD_LOW:
            return "Moderate"
        else:
            return "Low"

    confidence_summary = {
        "sentiment_confidence":       round(sentiment_conf, 4),
        "sentiment_confidence_level": conf_label(sentiment_conf),
        "priority_confidence":        round(priority_conf, 4),
        "priority_confidence_level":  conf_label(priority_conf)
    }
    word_count = len(complaint_text.split()) if complaint_text else 0

    return {
        "risk_level":         risk_level,
        "risk_score":         risk_score,
        "risk_icon":          RISK_COLORS[risk_level],
        "emotional_tone":     emotional_tone,
        "urgency_label":      urgency_label,
        "escalation_flag":    escalation_flag,
        "recommendations":    all_recs,
        "confidence_summary": confidence_summary,
        "word_count":         word_count
    }
