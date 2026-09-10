# PS01: Predictive Equipment Health — Explainable Risk Assessment System

An end-to-end pipeline that estimates equipment failure risk, categorizes
equipment into risk levels, explains *why* each prediction was made, and
represents the model's confidence — built for the "Predictive Equipment
Health" problem statement (AIDS / X'O CODE).

## Objective

Given historical equipment measurements + system-status labels, build a
system that:

1. Estimates the probability/risk of equipment failure.
2. Categorizes equipment into risk levels (Low / Medium / High / Critical).
3. Identifies which measurements drive each prediction.
4. Provides a human-readable explanation for every high-risk alert.
5. Helps prioritize which equipment needs inspection first.

## Project Structure

```
equipment-risk-assessment/
├── README.md
├── requirements.txt
├── data/                      # place raw dataset here (not committed)
│   └── .gitkeep
├── notebooks/
│   └── 01_eda.ipynb           # exploratory data analysis
├── src/
│   ├── data_preprocessing.py  # cleaning, encoding, train/test split
│   ├── eda.py                 # automated EDA report generator
│   ├── imbalance.py           # class-imbalance handling strategies
│   ├── model_training.py      # trains + calibrates the risk model
│   ├── threshold_selection.py # cost-sensitive threshold tuning
│   ├── explainability.py      # SHAP-based global + per-instance explanations
│   ├── risk_assessment.py     # probability -> risk tier + narrative reason
│   ├── uncertainty.py         # confidence / uncertainty estimation
│   └── main.py                # orchestrates the full pipeline end-to-end
├── models/                    # saved model artifacts
└── reports/                   # generated evaluation + explanation reports
```

## How Each Constraint Is Addressed

| Constraint | How it's handled |
|---|---|
| Class imbalance (normal vs failure) | `imbalance.py` — stratified splits + `class_weight='balanced'` and optional SMOTE oversampling, chosen based on measured imbalance ratio |
| False alarms ≠ missed failures | `threshold_selection.py` — threshold is tuned against a configurable cost matrix (cost of missed failure >> cost of false alarm), not a fixed 0.5 cutoff |
| Accuracy not primary metric | Evaluation uses **PR-AUC, Recall on failure class, F2-score, and cost-weighted confusion matrix** instead of raw accuracy |
| Interpretable explanations | `explainability.py` uses SHAP values to generate both global feature-importance and per-instance "top contributing factors" |
| Team-defined decision threshold | Threshold is a configurable parameter (`config['decision_threshold']`), derived transparently from the cost matrix, and documented in `reports/threshold_justification.md` |
| Represent uncertainty | `uncertainty.py` reports calibrated probability + confidence band via ensemble variance / Platt scaling, not just a binary yes/no |
| Don't remove unusual observations | `data_preprocessing.py` flags outliers but never drops rows automatically — they're kept and surfaced in the EDA report for manual review |
| Justify evaluation metrics | See `reports/metrics_justification.md` |

## Pipeline Overview

```
raw data ──▶ preprocessing ──▶ EDA ──▶ imbalance-aware training
   ──▶ probability calibration ──▶ cost-sensitive threshold selection
   ──▶ risk tiering ──▶ SHAP explanation ──▶ uncertainty scoring
   ──▶ prioritized inspection report
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Put your dataset at data/equipment_data.csv
# 2. Run the full pipeline
python src/main.py --data data/equipment_data.csv --target status_column_name
```

Outputs:
- `reports/risk_report.csv` — every equipment unit with risk score, tier,
  top contributing factors, and confidence.
- `reports/model_evaluation.md` — PR-AUC, recall, F2, cost-weighted metrics.
- `reports/global_feature_importance.png` — SHAP summary plot.

## Output Schema (`risk_report.csv`)

| column | description |
|---|---|
| equipment_id | identifier from source data |
| failure_probability | calibrated probability [0,1] |
| risk_tier | Low / Medium / High / Critical |
| confidence | model's certainty in this estimate (High/Medium/Low) |
| top_factors | ranked list of measurements driving this score |
| explanation | plain-language reason for the alert |
| recommended_action | inspect now / schedule inspection / monitor |

## Notes

This repo is a scaffold designed to work with any structured
tabular equipment-monitoring dataset — column names are auto-detected
in `data_preprocessing.py`; update `config.py` if your dataset uses
non-standard naming for the target/status column.
