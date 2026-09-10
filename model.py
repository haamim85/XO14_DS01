"""
PS01 - Predictive Equipment Health: Modeling & Explainability
----------------------------------------------------------------
Takes the cleaned train/validation data from preprocess.py and:
  1. Trains a Logistic Regression baseline and a Random Forest model,
     both with class-weighting to address the ~3.7% failure rate.
  2. Evaluates with PR-AUC / F1 / recall / precision (NOT accuracy).
  3. Picks an operating threshold using a cost-weighted rule
     (missed failure costs more than a false alarm).
  4. Buckets predictions into Low / Medium / High / Critical risk.
  5. Explains each prediction:
       - Random Forest -> SHAP if available, else permutation importance
       - Logistic Regression -> exact per-instance coefficient contributions
         (fully interpretable, no extra library needed)

Usage:
    python model.py
Requires train_clean.csv / validation_clean.csv from preprocess.py in OUT_DIR.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, classification_report,
)

# --- EDIT THIS TO MATCH YOUR GOOGLE DRIVE FOLDER ---
# Must be the SAME folder preprocess.py's OUT_DIR wrote train_clean.csv /
# validation_clean.csv into. Run `!ls "/content/drive/MyDrive/PS01-hackathon"`
# in Colab to confirm before running this script.
OUT_DIR = "/content/drive/MyDrive/PS01-hackathon"
ID_COL = "Equipment_Record_ID"
TARGET = "Machine failure"

# Business cost assumption: a MISSED failure (false negative) is far more
# expensive than a FALSE ALARM (false positive). Adjust this ratio to match
# real maintenance/downtime costs if you have them.
COST_FN = 10   # cost of missing a real failure
COST_FP = 1    # cost of an unnecessary inspection


def load_clean_data():
    train = pd.read_csv(f"{OUT_DIR}/train_clean.csv")
    val = pd.read_csv(f"{OUT_DIR}/validation_clean.csv")
    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]
    return X_train, y_train, X_val, y_val, feature_cols, val


def train_models(X_train, y_train):
    log_reg = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    )
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return {"logistic_regression": log_reg, "random_forest": rf}


def evaluate(model, X_val, y_val, name):
    proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, proba)
    roc_auc = roc_auc_score(y_val, proba)
    pred_default = (proba >= 0.5).astype(int)
    print(f"\n=== {name} (threshold=0.5) ===")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1:      {f1_score(y_val, pred_default):.4f}")
    print(f"Precision: {precision_score(y_val, pred_default):.4f}")
    print(f"Recall:    {recall_score(y_val, pred_default):.4f}")
    print(confusion_matrix(y_val, pred_default))
    return proba, pr_auc


def choose_threshold(y_val, proba):
    """Pick the threshold minimizing expected cost = FN*COST_FN + FP*COST_FP."""
    prec, rec, thresh = precision_recall_curve(y_val, proba)
    thresh = np.append(thresh, 1.0)  # align lengths
    best_t, best_cost = 0.5, np.inf
    n_pos = y_val.sum()
    n_neg = len(y_val) - n_pos
    for p, r, t in zip(prec, rec, thresh):
        tp = r * n_pos
        fn = n_pos - tp
        fp = (tp / p - tp) if p > 0 else n_neg
        cost = fn * COST_FN + fp * COST_FP
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t)


def risk_bucket(p):
    if p < 0.10:
        return "Low"
    elif p < 0.30:
        return "Medium"
    elif p < 0.60:
        return "High"
    else:
        return "Critical"


def explain_logistic(model, X, feature_cols, top_n=3):
    """Exact per-row contribution = coefficient * feature_value (scaled space)."""
    coefs = model.coef_[0]
    contrib = X[feature_cols].values * coefs  # elementwise
    reasons = []
    for row in contrib:
        idx = np.argsort(-np.abs(row))[:top_n]
        parts = [f"{feature_cols[i]} ({'+' if row[i]>0 else ''}{row[i]:.2f})" for i in idx]
        reasons.append("; ".join(parts))
    return reasons


def try_shap_explanations(rf_model, X_sample, feature_cols):
    """Optional: richer per-instance explanations if `shap` is installed
    (available by default on Colab after `!pip install shap`)."""
    try:
        import shap
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        # shap_values[1] = contributions toward the positive (failure) class
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        reasons = []
        for row in sv:
            idx = np.argsort(-np.abs(row))[:3]
            parts = [f"{feature_cols[i]} ({'+' if row[i]>0 else ''}{row[i]:.3f})" for i in idx]
            reasons.append("; ".join(parts))
        return reasons, True
    except ImportError:
        return None, False


def main():
    X_train, y_train, X_val, y_val, feature_cols, val_raw = load_clean_data()

    models = train_models(X_train, y_train)

    results = {}
    for name, model in models.items():
        proba, pr_auc = evaluate(model, X_val, y_val, name)
        results[name] = {"model": model, "proba": proba, "pr_auc": pr_auc}

    # pick the model with the higher PR-AUC (appropriate for imbalanced data)
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_model = results[best_name]["model"]
    best_proba = results[best_name]["proba"]
    print(f"\n>>> Best model by PR-AUC: {best_name}")

    # threshold selection (cost-weighted)
    threshold = choose_threshold(y_val, best_proba)
    print(f">>> Chosen decision threshold: {threshold:.3f} "
          f"(cost ratio FN:FP = {COST_FN}:{COST_FP})")

    final_pred = (best_proba >= threshold).astype(int)
    print("\n=== Final report at chosen threshold ===")
    print(classification_report(y_val, final_pred, digits=3))

    # risk buckets + explanations
    risk_levels = [risk_bucket(p) for p in best_proba]

    shap_reasons, shap_used = (None, False)
    if best_name == "random_forest":
        shap_reasons, shap_used = try_shap_explanations(best_model, X_val, feature_cols)

    if shap_used:
        reasons = shap_reasons
        explain_method = "SHAP (TreeExplainer)"
    else:
        # Fallback: explain using the logistic regression model regardless of
        # which model is "best", since it is exactly interpretable without
        # extra dependencies (works everywhere, incl. this offline sandbox).
        reasons = explain_logistic(models["logistic_regression"], X_val, feature_cols)
        explain_method = "Logistic regression coefficient contribution (fallback — install `shap` for tree-based explanations)"

    out = val_raw[[ID_COL, TARGET]].copy()
    out["predicted_probability"] = best_proba
    out["predicted_label"] = final_pred
    out["risk_level"] = risk_levels
    out["top_contributing_factors"] = reasons
    out.to_csv(f"{OUT_DIR}/validation_predictions.csv", index=False)

    joblib.dump(best_model, f"{OUT_DIR}/best_model_{best_name}.joblib")
    joblib.dump(models["logistic_regression"], f"{OUT_DIR}/logistic_regression.joblib")
    joblib.dump(models["random_forest"], f"{OUT_DIR}/random_forest.joblib")

    with open(f"{OUT_DIR}/model_report.md", "w") as f:
        f.write("# Model Report\n\n")
        f.write(f"- Best model (by PR-AUC on validation): **{best_name}**\n")
        f.write(f"- PR-AUC: {results[best_name]['pr_auc']:.4f}\n")
        f.write(f"- Chosen decision threshold: **{threshold:.3f}** "
                f"(minimizes expected cost, FN:FP = {COST_FN}:{COST_FP})\n")
        f.write(f"- Explanation method used: {explain_method}\n")
        f.write("- Risk buckets: Low (<0.10), Medium (0.10-0.30), "
                "High (0.30-0.60), Critical (>=0.60) — probability of failure\n")
        f.write("\n## Why PR-AUC / cost-weighted threshold instead of accuracy\n")
        f.write("With ~3.7% positive class, a model predicting 'no failure' for "
                "everything would already be ~96% accurate while catching zero "
                "real failures. PR-AUC and a cost-weighted threshold better "
                "reflect the real objective: catching failures while limiting "
                "false alarms.\n")
        f.write("\n## Files\n")
        f.write("- `validation_predictions.csv` — probability, risk level, "
                "and top contributing factors per equipment record\n")
        f.write(f"- `best_model_{best_name}.joblib` — the selected model\n")
        f.write("- `logistic_regression.joblib`, `random_forest.joblib` — both trained models\n")

    print("\nSaved: validation_predictions.csv, model_report.md, and model .joblib files")


if __name__ == "__main__":
    main()