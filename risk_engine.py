# ============================================================
# risk_engine.py
# Decision-support layer for ComplaintIQ.
# Includes keyword-based safety override for high-risk cases.
# ============================================================

import numpy as np
import re

# ============================================================
# RISK MATRIX
# ============================================================
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

# ============================================================
# KEYWORD OVERRIDE RULES
# ============================================================
# High-risk keywords are grouped by category.
# If 2+ keywords detected across any categories
# → force Negative + Urgent (High Risk)
# If 1 keyword detected
# → force Urgent only (Medium Risk minimum)
#
# This hybrid approach addresses high-confidence false
# negatives identified in Phase 2.7 error analysis,
# where legal/threat language was misclassified as neutral.

KEYWORD_CATEGORIES = {
    "legal_threat": [
        "legal action", "lawsuit", "sue", "court",
        "attorney", "lawyer", "solicitor", "litigation",
        "file a complaint", "press charges", "criminal"
    ],
    "financial_harm": [
        "stolen", "fraud", "fraudulent", "scam",
        "unauthorized", "unauthorised", "stolen money",
        "money missing", "funds missing", "theft",
        "embezzlement", "identity theft"
    ],
    "urgency_signal": [
        "immediately", "urgent", "emergency",
        "right now", "as soon as possible", "asap",
        "cannot wait", "time sensitive", "deadline",
        "eviction", "foreclosure", "repossession"
    ],
    "severe_distress": [
        "cannot eat", "cannot pay rent", "homeless",
        "losing my home", "losing my house",
        "family will suffer", "medical emergency",
        "life savings", "entire savings", "ruined",
        "destroyed", "devastated"
    ],
    "escalation_intent": [
        "bbb", "better business bureau", "cfpb",
        "consumer financial protection",
        "federal trade commission", "ftc",
        "news channel", "media", "go public",
        "social media", "report to", "authorities",
        "police", "fbi"
    ]
}

# Flatten all keywords for quick lookup
ALL_KEYWORDS = {
    kw: category
    for category, keywords in KEYWORD_CATEGORIES.items()
    for kw in keywords
}


def check_keyword_override(text: str,
                            sentiment_pred: int,
                            priority_pred: int):
    """
    Keyword-based safety override layer.

    Scans complaint text for high-risk keywords grouped
    by category. Override logic:
      - 2+ keyword categories matched
        → force sentiment=1 (Negative) + priority=1 (Urgent)
        → results in High Risk
      - 1 keyword category matched
        → force priority=1 (Urgent) only
        → results in Medium Risk minimum
      - 0 keywords matched
        → no override, use model predictions as-is

    This addresses high-confidence false negatives identified
    in Phase 2.7 where legal/financial threat language was
    misclassified as neutral by DeBERTa-v3-small.

    Returns:
        sentiment_pred  : int (potentially overridden)
        priority_pred   : int (potentially overridden)
        override_note   : str or None
        matched_keywords: list of matched keyword strings
        matched_categories: list of matched category names
    """
    text_lower = text.lower()

    # Find all keyword matches
    matched = {}
    for kw, category in ALL_KEYWORDS.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            if category not in matched:
                matched[category] = []
            matched[category].append(kw)

    matched_categories = list(matched.keys())
    matched_keywords   = [
        kw for kws in matched.values() for kw in kws
    ]

    override_note = None

    if len(matched_categories) >= 2:
        # Multiple risk categories — force High Risk
        sentiment_pred = 1
        priority_pred  = 1
        override_note  = (
            f"⚠️ Keyword override (High Risk): "
            f"detected {len(matched_keywords)} high-risk signal(s) "
            f"across {len(matched_categories)} categories "
            f"({', '.join(matched_categories)}). "
            f"Manual review recommended."
        )
    elif len(matched_categories) == 1:
        # Single risk category — force Urgent minimum
        priority_pred = 1
        category_name = matched_categories[0]
        override_note = (
            f"⚠️ Keyword override (Medium Risk minimum): "
            f"detected {category_name} signal(s): "
            f"{', '.join(matched_keywords[:3])}. "
            f"Manual review recommended."
        )

    return (
        sentiment_pred,
        priority_pred,
        override_note,
        matched_keywords,
        matched_categories
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================
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


# ============================================================
# CORE FUNCTIONS
# ============================================================
def compute_risk_score(sentiment_pred, priority_pred,
                        sentiment_conf, priority_conf):
    """
    Compute continuous risk score 0.0-1.0.
    Base score from risk band midpoint, modulated
    by average confidence of both predictions.
    """
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
    """
    Returns additional signal-based recommendations
    based on prediction confidence levels.
    """
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
    """
    Main engine function.

    Pipeline:
      1. Apply keyword override safety layer
      2. Compute risk level from risk matrix
      3. Compute continuous risk score
      4. Generate recommendations
      5. Return structured analysis dictionary

    Args:
        sentiment_pred  : 0 = Neutral, 1 = Negative
        priority_pred   : 0 = Not Urgent, 1 = Urgent
        sentiment_conf  : model confidence for sentiment
        priority_conf   : model confidence for priority
        complaint_text  : raw complaint text

    Returns:
        dict with full analysis results
    """
    # Step 1 — Keyword override
    (sentiment_pred,
     priority_pred,
     override_note,
     matched_keywords,
     matched_categories) = check_keyword_override(
        complaint_text,
        sentiment_pred,
        priority_pred
    )

    # Step 2 — Risk level
    risk_level      = RISK_MATRIX[(sentiment_pred, priority_pred)]
    emotional_tone  = "Negative" if sentiment_pred == 1 else "Neutral"
    urgency_label   = "Urgent"   if priority_pred  == 1 else "Not Urgent"
    escalation_flag = risk_level == "High"

    # Step 3 — Risk score
    risk_score = compute_risk_score(
        sentiment_pred, priority_pred,
        sentiment_conf, priority_conf
    )

    # Step 4 — Recommendations
    base_recs   = RECOMMENDATIONS[risk_level].copy()
    signal_recs = get_signal_recommendations(
        sentiment_pred, priority_pred,
        sentiment_conf, priority_conf
    )
    # Add override note as first recommendation if triggered
    override_recs = [override_note] if override_note else []
    all_recs      = override_recs + signal_recs + base_recs

    # Step 5 — Confidence summary
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
        "risk_level":          risk_level,
        "risk_score":          risk_score,
        "risk_icon":           RISK_COLORS[risk_level],
        "emotional_tone":      emotional_tone,
        "urgency_label":       urgency_label,
        "escalation_flag":     escalation_flag,
        "recommendations":     all_recs,
        "confidence_summary":  confidence_summary,
        "word_count":          word_count,
        "keyword_override":    override_note is not None,
        "matched_keywords":    matched_keywords,
        "matched_categories":  matched_categories
    }
