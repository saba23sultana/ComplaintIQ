# ComplaintIQ — AI-Powered Complaint Analysis & Escalation Decision Support System

## Project Overview

ComplaintIQ is an end-to-end Natural Language Processing (NLP) application developed 
as part of AT2 Group Project (2025). The system analyses unstructured customer complaint 
text and provides structured insights to support escalation decision-making in 
high-volume customer service environments.

The application performs two classification tasks:
- **Sentiment Classification** — detecting emotional tone (Neutral / Negative)
- **Priority Classification** — identifying urgency level (Not Urgent / Urgent)

Results are combined with a rule-based risk engine to generate a risk level 
(Low / Medium / High), escalation flag, and actionable recommendations for 
support agents.

---

## Live Application

🔗 **[ComplaintIQ on Streamlit Cloud](https://complaintiq-aaffeme9ruquaxwbw3bpze.streamlit.app)**

---

## Key Features

- 🔍 **Single complaint analysis** — instant risk assessment with confidence scores
- 📂 **Batch analysis** — CSV upload for processing multiple complaints
- 🔴 **Risk indicator** — Low / Medium / High with continuous risk score
- 📋 **Recommendation engine** — actionable escalation steps per risk level
- ⚠️ **Keyword safety override** — hybrid rule-based layer for high-risk signals
- 📊 **Model comparison** — performance metrics across all three model tiers
- 🔧 **Explainability** — confidence scores and raw JSON output

---

## NLP Approaches

Three model tiers are implemented and compared:

| Model | Type | Sentiment F1 | Priority F1 | Inference |
|-------|------|-------------|-------------|-----------|
| TF-IDF + Logistic Regression | Traditional baseline | 0.6424 | 0.8229 | 0.007 ms |
| DistilBERT | Lightweight transformer | 0.6626 | 0.7941 | 16.48 ms |
| DeBERTa-v3-small | Advanced transformer | 0.6897 | 0.8330 | 27.22 ms |

**DeBERTa-v3-small** is used as the production model in the application based on 
best Macro F1 performance across both tasks.

---

## Project Structure
complaintiq/
├── app.py              # Main Streamlit application
├── risk_engine.py      # Risk scoring and recommendation engine
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

---

## Dataset

- **Source:** [Customer Complaints Sentiment and Priority Dataset](https://www.kaggle.com/datasets/xjoury/customer-complaints-sentiment-and-priority-dataset)
- **Records:** 1,750 financial complaints
- **Labels:** Sentiment (0=Neutral, 1=Negative), Priority (0=Not Urgent, 1=Urgent)
- **Product categories:** 7 financial product classes, 250 records each
- **License:** CC BY 4.0

---

## Model Architecture

### Traditional Baseline
- TF-IDF vectorizer (max 20,000 features, unigrams + bigrams)
- Logistic Regression with `class_weight='balanced'`
- Interpretable via coefficient analysis

### Transformer Models
- **DistilBERT** (`distilbert-base-uncased`) — 66.96M parameters
- **DeBERTa-v3-small** (`microsoft/deberta-v3-small`) — 141.90M parameters
- Fine-tuned with weighted cross-entropy loss
- AdamW optimizer, lr=2e-5, linear warmup (10%), weight decay=0.01
- Early stopping with patience=2 on validation Macro F1

### Risk Engine
- Rule-based risk matrix combining sentiment and priority predictions
- Continuous risk score (0.0–1.0) modulated by model confidence
- Keyword-based safety override for high-risk signals (legal threats,
  financial harm, severe distress, escalation intent)
- Three-tier recommendation system (Low / Medium / High)

---

## Installation & Local Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Sabrin23sultana/complaintiq.git
cd complaintiq
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
streamlit run app.py
```

**5. Open in browser**

https://complaintiq-aaffeme9ruquaxwbw3bpze.streamlit.app

---

## Fine-tuned Models

All fine-tuned models are hosted on Hugging Face Hub:

| Model | Task | Link |
|-------|------|------|
| DeBERTa-v3-small | Sentiment | [complaintiq-deberta-sentiment](https://huggingface.co/Sabrin23sultana/complaintiq-deberta-sentiment) |
| DeBERTa-v3-small | Priority | [complaintiq-deberta-priority](https://huggingface.co/Sabrin23sultana/complaintiq-deberta-priority) |
| DistilBERT | Sentiment | [complaintiq-distilbert-sentiment](https://huggingface.co/Sabrin23sultana/complaintiq-distilbert-sentiment) |
| DistilBERT | Priority | [complaintiq-distilbert-priority](https://huggingface.co/Sabrin23sultana/complaintiq-distilbert-priority) |

Models are loaded automatically at runtime via `st.cache_resource`.

---

## Reproducibility

All experiments are fully reproducible:

- **Random seed:** 42 (applied across numpy, torch, sklearn)
- **Data splits:** Stratified 70/15/15 train/val/test (seed=42)
- **Training configs:** Saved as JSON in `models/` directory
- **Dependencies:** Pinned in `requirements.txt`

---

## Evaluation Results

### Test Set Performance (263 samples)

**Sentiment Classification:**

| Model | Accuracy | Macro F1 | Neutral F1 | Negative F1 |
|-------|----------|----------|------------|-------------|
| TF-IDF + LR | 0.7072 | 0.6424 | 0.79 | 0.49 |
| DistilBERT | 0.7529 | 0.6626 | 0.84 | 0.49 |
| DeBERTa-v3-small | 0.7490 | **0.6897** | 0.83 | **0.55** |

**Priority Classification:**

| Model | Accuracy | Macro F1 | Not Urgent F1 | Urgent F1 |
|-------|----------|----------|---------------|-----------|
| TF-IDF + LR | 0.8745 | 0.8229 | 0.73 | 0.92 |
| DistilBERT | 0.8593 | 0.7941 | 0.68 | 0.91 |
| DeBERTa-v3-small | 0.8783 | **0.8330** | **0.75** | **0.92** |

---

## Technical Report

The full technical report covering all project phases is available in the 
project submission. It includes:

- Project objectives and scope
- Data sources and EDA
- NLP methods and justification
- Model training and evaluation
- Error analysis and interpretability
- Ethical considerations
- Recommendations and next steps

---

## Ethical Considerations

- **Data privacy:** Dataset uses pre-redacted PII (XXXX placeholders)
  from the CFPB Consumer Complaint Database
- **Model bias:** Labels were manually assigned — potential subjectivity
  in sentiment/priority boundaries acknowledged
- **Over-reliance risk:** System is designed as decision support,
  not autonomous decision-making. Human review is always recommended
- **Fairness:** No demographic information used in classification.
  Product category patterns may introduce institution-level bias
  as identified in error analysis

---

## Project Phases

| Phase | Description |
|-------|-------------|
| 1.0–1.4 | Setup, data loading, EDA |
| 1.5–1.6 | Preprocessing pipelines, feature engineering |
| 1.7–1.8 | Data splits, augmentation |
| 2.1–2.2 | Task definition, TF-IDF + LR baseline |
| 2.3 | DistilBERT and DeBERTa-v3-small fine-tuning |
| 2.4 | Risk engine and recommendation system |
| 2.5 | Speed and efficiency comparison |
| 2.6 | Hyperparameter tuning |
| 2.7 | Error analysis |
| 2.8 | Model interpretability (coefficients + LIME) |
| 2.9 | Streamlit application development and deployment |

---

## Dependencies
streamlit>=1.32.0
torch>=2.0.0
transformers>=4.38.0
numpy>=1.24.0
pandas>=2.0.0
huggingface-hub>=0.21.0
sentencepiece>=0.1.99
protobuf>=3.20.0
accelerate>=0.27.0


---

## Acknowledgements

- Dataset: [xjoury on Kaggle](https://www.kaggle.com/datasets/xjoury/customer-complaints-sentiment-and-priority-dataset)
- Original data: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- Pretrained models: [Hugging Face](https://huggingface.co)
- Framework: [Streamlit](https://streamlit.io)

---

## License

This project is developed for academic purposes as part of AT2 NLP Assignment 2025.
Dataset is licensed under CC BY 4.0.
