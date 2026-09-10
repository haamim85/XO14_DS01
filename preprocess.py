"""
PS01 - Predictive Equipment Health: Preprocessing & Cleaning Pipeline
----------------------------------------------------------------------
Dataset: industrial equipment monitoring (AI4I-style predictive maintenance).
Fit everything on TRAIN only, then apply the same transforms to VALIDATION
(and later, to any held-out TEST set) to avoid data leakage.

Usage:
    python preprocess.py
Outputs (in /mnt/user-data/outputs/):
    train_clean.csv, validation_clean.csv, preprocessing_pipeline.joblib,
    preprocessing_report.md
"""

import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib

# --- EDIT THESE 3 PATHS TO MATCH YOUR GOOGLE DRIVE FOLDER ---
# Run `!ls "/content/drive/MyDrive/PS01-hackathon"` in Colab first to confirm
# these filenames/paths are exactly right before running this script.
RAW_TRAIN = "/content/drive/MyDrive/PS01-hackathon/development_train.csv"
RAW_VAL   = "/content/drive/MyDrive/PS01-hackathon/development_validation.csv"
OUT_DIR   = "/content/drive/MyDrive/PS01-hackathon"

NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CAT_COL = "Type"
ID_COL = "Equipment_Record_ID"
TARGET = "Machine failure"
TYPE_ORDER = {"L": 0, "M": 1, "H": 2}  # L=low, M=medium, H=high quality/durability tier


def load_data():
    train = pd.read_csv(RAW_TRAIN)
    val = pd.read_csv(RAW_VAL)
    return train, val


def report_missingness(train, val):
    print("\n=== Missing values (train) ===")
    print(train.isna().sum())
    print("\n=== Missing values (validation) ===")
    print(val.isna().sum())
    print("\n=== Rows with >=1 missing value (train) ===",
          (train[NUM_COLS].isna().sum(axis=1) > 0).sum(), "/", len(train))


def impute_missing(train, val):
    """
    Median imputation, computed PER Type group on TRAIN, then applied to both
    train and validation. Median is used (not mean) because it is robust to
    the skew/outliers already present in Rotational speed and Torque.
    Falls back to the global train median if a Type/column combo has no data.
    """
    group_medians = train.groupby(CAT_COL)[NUM_COLS].median()
    global_medians = train[NUM_COLS].median()

    def fill(df):
        df = df.copy()
        for col in NUM_COLS:
            # fill using the row's own Type median, else the global median
            missing_mask = df[col].isna()
            if missing_mask.any():
                fill_vals = df.loc[missing_mask, CAT_COL].map(group_medians[col])
                fill_vals = fill_vals.fillna(global_medians[col])
                df.loc[missing_mask, col] = fill_vals
        return df

    train_imputed = fill(train)
    val_imputed = fill(val)
    return train_imputed, val_imputed, group_medians, global_medians


def engineer_features(df):
    df = df.copy()
    # Physical / domain-driven features (kept alongside raw features, not replacing them)
    df["Temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Torque_x_ToolWear"] = df["Torque [Nm]"] * df["Tool wear [min]"]  # mechanical strain proxy
    df["Speed_x_ToolWear"] = df["Rotational speed [rpm]"] * df["Tool wear [min]"]
    df["Log_RotSpeed"] = np.log1p(df["Rotational speed [rpm]"])  # right-skewed -> log
    df["Type_ordinal"] = df[CAT_COL].map(TYPE_ORDER)
    # one-hot as an alternative encoding some models/tools prefer
    df = pd.get_dummies(df, columns=[CAT_COL], prefix="Type", drop_first=False)
    return df


def flag_outliers(df):
    """Flag (do not remove) statistically unusual rows, per the challenge
    constraint: unusual observations must not simply be dropped."""
    df = df.copy()
    for col in ["Rotational speed [rpm]", "Torque [Nm]"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag_col = col.split(" ")[0] + "_is_outlier"
        df[flag_col] = ((df[col] < lo) | (df[col] > hi)).astype(int)
    return df


def scale_features(train_df, val_df, feature_cols):
    scaler = RobustScaler()  # robust to the retained outliers
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    return train_scaled, val_scaled, scaler


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    train, val = load_data()
    report_missingness(train, val)

    # duplicate check
    dup_train = train.duplicated(subset=[c for c in train.columns if c != ID_COL]).sum()
    dup_ids = train[ID_COL].duplicated().sum()
    print(f"\nDuplicate rows (excluding ID): {dup_train} | Duplicate IDs: {dup_ids}")

    # 1) impute
    train_i, val_i, group_medians, global_medians = impute_missing(train, val)
    assert train_i[NUM_COLS].isna().sum().sum() == 0
    assert val_i[NUM_COLS].isna().sum().sum() == 0

    # 2) outlier flags (kept, not removed)
    train_f = flag_outliers(train_i)
    val_f = flag_outliers(val_i)

    # 3) feature engineering + encoding
    train_e = engineer_features(train_f)
    val_e = engineer_features(val_f)

    # align columns between train/val (in case a Type level is missing in one split)
    for c in train_e.columns:
        if c not in val_e.columns:
            val_e[c] = 0
    val_e = val_e[train_e.columns]

    # 4) scale numeric + engineered continuous features (fit on train only)
    scale_cols = NUM_COLS + ["Temp_diff_K", "Power_W", "Torque_x_ToolWear",
                              "Speed_x_ToolWear", "Log_RotSpeed"]
    train_s, val_s, scaler = scale_features(train_e, val_e, scale_cols)

    # save
    train_s.to_csv(f"{OUT_DIR}/train_clean.csv", index=False)
    val_s.to_csv(f"{OUT_DIR}/validation_clean.csv", index=False)
    joblib.dump(
        {
            "scaler": scaler,
            "scale_cols": scale_cols,
            "group_medians": group_medians,
            "global_medians": global_medians,
            "type_order": TYPE_ORDER,
        },
        f"{OUT_DIR}/preprocessing_pipeline.joblib",
    )

    # quick report
    with open(f"{OUT_DIR}/preprocessing_report.md", "w") as f:
        f.write("# Preprocessing Report\n\n")
        f.write(f"- Train rows: {len(train)}, Validation rows: {len(val)}\n")
        f.write(f"- Duplicate rows (train): {dup_train}, Duplicate IDs: {dup_ids}\n")
        f.write(f"- Class balance (train): {train[TARGET].mean():.4f} positive\n")
        f.write("- Missing values were imputed using per-`Type` medians (fit on train only).\n")
        f.write("- Outliers in Rotational speed / Torque were flagged, not removed "
                "(per challenge constraints); they reflect a real inverse "
                "torque-speed relationship, not data errors.\n")
        f.write("- Engineered features: Temp_diff_K, Power_W, Torque_x_ToolWear, "
                "Speed_x_ToolWear, Log_RotSpeed, Type_ordinal, Type one-hot.\n")
        f.write("- Continuous features scaled with RobustScaler (fit on train, applied to val).\n")
        f.write("- Class imbalance (~3.7% positive) intentionally NOT resampled here — "
                "handle via class_weight / SMOTE at the modeling stage, on train folds only.\n")

    print("\nSaved:")
    print(f"  {OUT_DIR}/train_clean.csv  shape={train_s.shape}")
    print(f"  {OUT_DIR}/validation_clean.csv  shape={val_s.shape}")
    print(f"  {OUT_DIR}/preprocessing_pipeline.joblib")
    print(f"  {OUT_DIR}/preprocessing_report.md")


if __name__ == "__main__":
    main()