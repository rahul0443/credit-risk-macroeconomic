#!/usr/bin/env python3
"""
Capstone: Credit Risk Modeling Under Macroeconomic Conditions
Stage 2 of 6 — XGBoost Hyperparameter Search

This is a one-time search script designed to run on Google Colab.
The best parameters found here are already embedded in 4_model_v3.py,
so this file does not need to be re-run unless you want to redo the search.

Pipeline order:
  1_exploratory_data_analysis.py   ← run locally
  2_model_hyperparameter_search.py ← THIS FILE (Colab only, one-time)
  3_model_lgbm_search.py           ← Colab only, one-time
  4_model_v3.py                    ← run locally (uses tuned params from 2 & 3)
  5_run_pipeline.py                ← orchestrates 1 + 4 end-to-end

Setup in Colab:
    1. Upload accepted_2007_to_2018Q4.csv, FEDFUNDS.csv, UNRATE.csv to your Google Drive
    2. Run this cell first to mount Drive and install dependencies:

        from google.colab import drive
        drive.mount('/content/drive')
        !pip install xgboost scikit-learn pandas numpy matplotlib -q

    3. Update the three FILE paths below to match your Drive folder
    4. Run the script — expect 30–60 min on Colab free tier

What this does:
- Loads the full LendingClub dataset
- Applies the same leakage-safe preprocessing as model_improvement.py
- Runs RandomizedSearchCV (50 combinations) on XGBoost Borrower-only
- Saves best params + full results to JSON
- Reports best test ROC-AUC vs the baseline 0.702
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Config
# Update FILE PATHS to match your Google Drive folder before running on Colab.
# ─────────────────────────────────────────────────────────────────────────────

ACCEPTED_FILE = "/content/drive/MyDrive/capstone/accepted_2007_to_2018Q4.csv"
FEDFUNDS_FILE = "/content/drive/MyDrive/capstone/FEDFUNDS.csv"
UNRATE_FILE   = "/content/drive/MyDrive/capstone/UNRATE.csv"

OUTDIR = "/content/drive/MyDrive/capstone/hyperparam_outputs"
FIGDIR = os.path.join(OUTDIR, "figures")

RANDOM_SEED = 42
MAX_TRAIN   = 500_000  # cap training rows to match the budget used in 4_model_v3.py
N_BOOT      = 100      # bootstrap iterations for final AUC confidence interval
N_ITER      = 50       # random hyperparameter combinations to evaluate
CV_FOLDS    = 3        # folds for cross-validation during search

# Target definition must match 4_model_v3.py to ensure results are comparable.
GOOD_STATUSES = {"Fully Paid"}
DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
}
EXTRA_HARD_DROP = ["zip_code", "member_id"]

# Search space covers the main regularisation and complexity axes of XGBoost.
# Kept deliberately broad for the first search pass; narrow in follow-up runs.
XGB_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 400, 500],
    "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth":         [3, 4, 5, 6],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda":        [0.1, 0.5, 1.0, 2.0, 5.0],
    "min_child_weight":  [1, 3, 5, 10],
    "gamma":             [0.0, 0.1, 0.5, 1.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_outdirs() -> None:
    """Create output directories if they do not already exist."""
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase all column names.

    Lowercasing is applied here (unlike in file 1) because this script
    needs consistent casing when probing for FRED date columns.

    Args:
        df: Input DataFrame.

    Returns:
        Copy of df with cleaned column names.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def safe_to_datetime(s: pd.Series) -> pd.Series:
    """Parse a Series to datetime, coercing unparseable values to NaT instead of raising."""
    return pd.to_datetime(s, errors="coerce")


def create_month_key(dt: pd.Series) -> pd.Series:
    """Truncate a datetime Series to month-start timestamps for merge keys.

    Args:
        dt: Series of datetime or date-like values.

    Returns:
        Series of Timestamps representing the first day of each month.
    """
    dt = safe_to_datetime(dt)
    return dt.dt.to_period("M").dt.to_timestamp()


def infer_issue_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Locate and parse the loan issue-date column.

    Probes a priority list of known column names, then falls back to
    any column containing 'issue' followed by '_d' or ending in 'd'.

    Args:
        df: Raw loan DataFrame.

    Returns:
        Tuple of (df_with_parsed_dates, column_name_found).

    Raises:
        ValueError: If no issue-date column can be identified.
    """
    df = df.copy()
    candidates = ["issue_d", "issue_date", "issued", "issue"]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        for c in df.columns:
            if "issue" in c and ("_d" in c or c.endswith("d")):
                found = c
                break
    if found is None:
        raise ValueError("Could not find an issue date column (expected issue_d).")
    df[found] = safe_to_datetime(df[found])
    return df, found


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target column `y_default` from `loan_status`.

    Rows with ambiguous statuses (e.g. 'Current') are dropped to avoid
    label noise — only resolved outcomes are kept.

    Args:
        df: DataFrame containing a 'loan_status' column.

    Returns:
        Filtered DataFrame with added 'y_default' column (0 = good, 1 = default).

    Raises:
        ValueError: If 'loan_status' is not present.
    """
    df = df.copy()
    if "loan_status" not in df.columns:
        raise ValueError("Expected 'loan_status' column to build target.")
    before = len(df)
    df = df[df["loan_status"].isin(GOOD_STATUSES.union(DEFAULT_STATUSES))].copy()
    after = len(df)
    df["y_default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    print(f"Target filter (GOOD+DEFAULT): {before:,} -> {after:,}")
    return df


def read_fred_csv(path: str, value_col: str) -> pd.DataFrame:
    """Load a FRED download CSV and return a clean (date, value) DataFrame.

    Args:
        path:      Path to the FRED CSV file.
        value_col: Desired output name for the value column (e.g. 'FEDFUNDS').

    Returns:
        Two-column DataFrame with columns ['date', value_col], sorted by date.

    Raises:
        ValueError: If no date column or value column can be identified.
    """
    df = pd.read_csv(path)
    df = normalize_colnames(df)
    date_col = None
    for c in ["date", "observation_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"{path}: Could not find date column.")
    if value_col.lower() not in df.columns:
        non_date_cols = [c for c in df.columns if c != date_col]
        if len(non_date_cols) != 1:
            raise ValueError(f"{path}: Could not find {value_col}.")
        series_col = non_date_cols[0]
    else:
        series_col = value_col.lower()
    out = pd.DataFrame({
        "date": safe_to_datetime(df[date_col]),
        value_col: pd.to_numeric(df[series_col], errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")
    return out


def engineer_macro(fed: pd.DataFrame, unr: pd.DataFrame) -> pd.DataFrame:
    """Merge FEDFUNDS and UNRATE and engineer lag/change/rolling features.

    Features are computed at monthly frequency. Forward-fill is applied
    twice — once after the outer join (to cover months present in only
    one series) and once after adding lag features (to cover NaNs at
    the start of the series).

    Args:
        fed: DataFrame with columns ['date', 'FEDFUNDS'].
        unr: DataFrame with columns ['date', 'UNRATE'].

    Returns:
        Monthly DataFrame with FEDFUNDS, UNRATE, and derived lag/change/roll columns.
    """
    fed = fed.copy()
    unr = unr.copy()
    fed["month"] = create_month_key(fed["date"])
    unr["month"] = create_month_key(unr["date"])
    fed = fed.drop(columns=["date"]).drop_duplicates(subset=["month"]).sort_values("month")
    unr = unr.drop(columns=["date"]).drop_duplicates(subset=["month"]).sort_values("month")
    macro = pd.merge(fed, unr, on="month", how="outer").sort_values("month").ffill()
    for col in ["FEDFUNDS", "UNRATE"]:
        macro[f"{col}_lag1"]  = macro[col].shift(1)
        macro[f"{col}_lag3"]  = macro[col].shift(3)
        macro[f"{col}_lag6"]  = macro[col].shift(6)
        macro[f"{col}_chg1"]  = macro[col] - macro[f"{col}_lag1"]
        macro[f"{col}_chg3"]  = macro[col] - macro[f"{col}_lag3"]
        macro[f"{col}_roll3"] = macro[col].rolling(3).mean()
        macro[f"{col}_roll6"] = macro[col].rolling(6).mean()
    return macro.ffill()


def find_leakage_columns(df: pd.DataFrame) -> List[str]:
    """Scan column names for keywords associated with post-origination data.

    These columns are populated only after a loan is active and must be
    excluded before training to prevent label leakage.

    Args:
        df: DataFrame whose column names will be scanned.

    Returns:
        Sorted list of column names matching at least one leakage keyword.
    """
    leak_keywords = [
        "pymnt", "payment", "last_pymnt", "next_pymnt", "recover", "recovery",
        "collection", "collections", "settlement", "debt_settlement",
        "hardship", "out_prncp", "total_pymnt", "total_rec_", "last_fico",
        "acc_now_delinq", "chargeoff", "chargedoff", "charge_off",
    ]
    return sorted({col for col in df.columns if any(kw in col.lower() for kw in leak_keywords)})


def hard_drop_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns to unconditionally drop using regex patterns and an explicit list.

    Pattern matching catches column name variants not covered by keyword scan;
    the explicit list handles known edge cases that evade both regex and keywords.

    Args:
        df: DataFrame to inspect.

    Returns:
        Sorted list of column names to drop.
    """
    patterns = [
        r"^total_rec_", r"^total_pymnt", r"^recover", r"^collection",
        r"^last_pymnt", r"^next_pymnt", r"^out_prncp", r"^hardship",
        r"^settlement", r"^debt_settlement", r"^last_fico",
    ]
    must_drop = [
        col for col in df.columns
        if any(re.search(p, col, re.IGNORECASE) for p in patterns)
    ]
    explicit = [
        "total_rec_late_fee", "funded_amnt_inv", "collection_recovery_fee",
        "delinq_2yrs", "acc_now_delinq", "delinq_amnt",
        "mths_since_recent_revol_delinq", "sec_app_fico_range_low",
        "sec_app_fico_range_high", "sec_app_inq_last_6mths",
        "sec_app_mths_since_last_major_derog",
    ]
    must_drop += [c for c in explicit if c in df.columns]
    return sorted(set(must_drop))


def drop_all_missing_cols(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    label: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove columns that are entirely missing in the training set.

    Applying this filter on train only (then mirroring to val/test) prevents
    features with zero training signal from entering the preprocessor.

    Args:
        X_train: Training feature DataFrame.
        X_val:   Validation feature DataFrame.
        X_test:  Test feature DataFrame.
        label:   Optional label for the print statement (e.g. 'Borrower-only').

    Returns:
        Tuple of (X_train, X_val, X_test) with all-missing columns removed.
    """
    all_missing = [c for c in X_train.columns if X_train[c].notna().sum() == 0]
    if all_missing:
        print(f"[{label}] Dropping {len(all_missing)} all-missing cols from train")
        X_train = X_train.drop(columns=all_missing, errors="ignore")
        X_val   = X_val.drop(columns=all_missing, errors="ignore")
        X_test  = X_test.drop(columns=all_missing, errors="ignore")
    return X_train, X_val, X_test


@dataclass
class SplitData:
    """Container for the three temporal data splits."""
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame


def temporal_split(df: pd.DataFrame, date_col: str,
                   train_q: float = 0.70, val_q: float = 0.85) -> SplitData:
    """Split a DataFrame into train/val/test along the time axis.

    Uses quantile positions on the sorted month list rather than fixed dates,
    so the split proportions hold regardless of the date range in the data.

    Args:
        df:       DataFrame containing the date column.
        date_col: Name of the column used for sorting and splitting.
        train_q:  Quantile cutoff for end of training period (default 0.70).
        val_q:    Quantile cutoff for end of validation period (default 0.85).

    Returns:
        SplitData with .train, .val, and .test DataFrames.
    """
    d = df.sort_values(date_col).copy()
    months = pd.Series(d[date_col].dropna().unique()).sort_values()
    train_max = months.iloc[int(np.floor(train_q * (len(months) - 1)))]
    val_max   = months.iloc[int(np.floor(val_q   * (len(months) - 1)))]
    return SplitData(
        train=d[d[date_col] <= train_max].copy(),
        val  =d[(d[date_col] > train_max) & (d[date_col] <= val_max)].copy(),
        test =d[d[date_col] > val_max].copy(),
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise all string/object columns before passing to the preprocessor.

    Must be called on X_train, X_val, and X_test before fitting ColumnTransformer,
    because the transformer operates on the original DataFrame in-place.
    Percentage strings (e.g. int_rate '13.5%') are converted to numeric here
    rather than inside the pipeline to keep the pipeline stateless.

    Args:
        df: Feature DataFrame with potentially messy string columns.

    Returns:
        Cleaned copy of df.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
            # Convert percentage strings (e.g. '13.5%') to numeric floats.
            sample = df[col].dropna().head(50)
            if sample.astype(str).str.contains("%", na=False).any():
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("%", "", regex=False), errors="coerce"
                )
    # Force-coerce numeric-dtype columns to float to catch hidden string values.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer that imputes and scales numeric columns and
    one-hot encodes low-cardinality categoricals.

    High-cardinality categoricals (>20 unique values) are dropped entirely
    rather than encoded, to avoid inflating the feature space.
    X must already be cleaned by clean_dataframe() before calling this.

    Args:
        X: Feature DataFrame (post-cleaning, pre-transform).

    Returns:
        Tuple of (fitted_preprocessor, numeric_col_names, categorical_col_names).
    """
    X = X.copy()

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    raw_cat = [c for c in X.columns if c not in numeric_cols]
    categorical_cols = [c for c in raw_cat if X[c].nunique() <= 20]
    dropped = [c for c in raw_cat if c not in categorical_cols]
    if dropped:
        print(f"  [preprocessor] Dropped {len(dropped)} high-cardinality categoricals: {dropped[:10]}")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=15)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe,     categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, numeric_cols, categorical_cols


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute a standard set of binary classification metrics.

    Args:
        y_true:    Ground-truth binary labels.
        y_prob:    Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dictionary with keys: roc_auc, pr_auc, brier, tn, fp, fn, tp,
        precision@threshold, recall@threshold, fpr@threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc":              float(roc_auc_score(y_true, y_prob)),
        "pr_auc":               float(average_precision_score(y_true, y_prob)),
        "brier":                float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        f"precision@{threshold}": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        f"recall@{threshold}":    float(tp / (tp + fn)) if (tp + fn) else 0.0,
        f"fpr@{threshold}":       float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def bootstrap_auc(y_true: np.ndarray, y_prob: np.ndarray,
                  n_boot: int = 100, seed: int = 42) -> Dict:
    """Estimate a 95% bootstrap confidence interval for ROC-AUC.

    Resamples with replacement; iterations that produce only one class
    are skipped to avoid undefined AUC.

    Args:
        y_true:  Ground-truth binary labels.
        y_prob:  Predicted probabilities for the positive class.
        n_boot:  Number of bootstrap resamples (default 100).
        seed:    Random seed for reproducibility.

    Returns:
        Dictionary with keys: mean, ci_low, ci_high, n_boot (actual iterations used).
    """
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return {
        "mean":    float(aucs.mean()),
        "ci_low":  float(np.quantile(aucs, 0.025)),
        "ci_high": float(np.quantile(aucs, 0.975)),
        "n_boot":  int(len(aucs)),
    }


def plot_roc(y_true: np.ndarray, y_prob_baseline: np.ndarray,
             y_prob_tuned: np.ndarray, out_path: str) -> None:
    """Save an ROC curve comparing the baseline and tuned XGBoost models.

    Args:
        y_true:          Ground-truth binary labels.
        y_prob_baseline: Predicted probabilities from the baseline model.
        y_prob_tuned:    Predicted probabilities from the tuned model.
        out_path:        Full file path for the saved figure.
    """
    plt.figure(figsize=(9, 7))
    for name, prob in [("Baseline XGB (0.702)", y_prob_baseline),
                       ("Tuned XGB",             y_prob_tuned)]:
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("ROC Curve — Baseline vs Tuned XGBoost (Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"ROC plot saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the XGBoost hyperparameter search end-to-end.

    Steps:
      1.  Load full loan dataset and parse issue dates.
      2.  Load macro series (merged for completeness; excluded from features).
      3.  Engineer borrower features (sub-grade encoding, log income, credit age).
      4.  Drop leakage and hard-coded post-origination columns.
      5.  Build borrower-only feature set (macro columns excluded).
      6.  Apply temporal train/val/test split.
      7.  Clean DataFrames and fit the preprocessor.
      8.  Fit and evaluate the baseline XGBoost model.
      9.  Run RandomizedSearchCV over XGB_PARAM_DIST.
      10. Evaluate the best model on the held-out test set.
      11. Save results JSON and top-10 combinations CSV.
      12. Save ROC comparison plot.
    """
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)
    ensure_outdirs()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("=== Load accepted loans ===")
    loans = pd.read_csv(ACCEPTED_FILE, low_memory=False)
    loans = normalize_colnames(loans)
    print(f"Raw shape: {loans.shape[0]:,} x {loans.shape[1]:,}")

    loans, issue_col = infer_issue_date(loans)
    loans["issue_month"] = create_month_key(loans[issue_col])
    loans = make_target(loans)
    loans = loans.dropna(subset=["issue_month", "y_default"]).copy()
    print(f"After filtering: {loans.shape[0]:,} rows")

    # ── 2. Macro ──────────────────────────────────────────────────────────────
    # Macro is loaded and merged so the dataset stays comparable with 4_model_v3.py,
    # but macro columns are excluded from the feature set for this borrower-only search.
    print("=== Load macro (excluded from search — macro hurts XGB) ===")
    fed   = read_fred_csv(FEDFUNDS_FILE, "FEDFUNDS")
    unr   = read_fred_csv(UNRATE_FILE,   "UNRATE")
    macro = engineer_macro(fed, unr)
    loans = loans.merge(macro, left_on="issue_month", right_on="month", how="left").drop(columns=["month"])

    # ── 3. Feature engineering ────────────────────────────────────────────────
    if "sub_grade" in loans.columns:
        grades = list("ABCDEFG")
        sub_grade_map = {f"{g}{n}": i * 5 + n for i, g in enumerate(grades) for n in range(1, 6)}
        loans["sub_grade_enc"] = loans["sub_grade"].map(sub_grade_map)
        loans = loans.drop(columns=["sub_grade"])

    if "annual_inc" in loans.columns:
        loans["annual_inc"] = np.log1p(pd.to_numeric(loans["annual_inc"], errors="coerce"))

    if "earliest_cr_line" in loans.columns:
        ecl = safe_to_datetime(loans["earliest_cr_line"])
        loans["credit_age_months"] = ((loans["issue_month"] - ecl).dt.days / 30.44).clip(lower=0)
        loans = loans.drop(columns=["earliest_cr_line"])

    # ── 4. Leakage drop ───────────────────────────────────────────────────────
    print("=== Leakage drop ===")
    drop_cols = sorted(set(find_leakage_columns(loans)).union(hard_drop_columns(loans)))
    protected = {"y_default", "issue_month", issue_col, "loan_status"}
    drop_cols = [c for c in drop_cols if c not in protected]
    loans = loans.drop(columns=drop_cols, errors="ignore")
    extra = [c for c in EXTRA_HARD_DROP if c in loans.columns]
    loans = loans.drop(columns=extra, errors="ignore")
    print(f"Shape after leakage drop: {loans.shape[0]:,} x {loans.shape[1]:,}")

    # ── 5. Borrower-only feature set ──────────────────────────────────────────
    macro_cols = [c for c in loans.columns if c.startswith("FEDFUNDS") or c.startswith("UNRATE")]
    non_feature = {"y_default", "loan_status", issue_col}
    X = loans.drop(columns=[c for c in non_feature if c in loans.columns] + macro_cols, errors="ignore")

    split_df = X.copy()
    split_df["y_default"]   = loans["y_default"].values
    split_df["issue_month"] = loans["issue_month"].values
    split_df = split_df.loc[:, ~split_df.columns.duplicated()]

    # ── 6. Temporal split ─────────────────────────────────────────────────────
    print("=== Temporal split ===")
    splits = temporal_split(split_df, date_col="issue_month")

    train_df = splits.train
    val_df   = splits.val
    test_df  = splits.test

    if len(train_df) > MAX_TRAIN:
        train_df = train_df.sample(MAX_TRAIN, random_state=RANDOM_SEED)
        print(f"Downsampled train to: {len(train_df):,}")

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Train default rate: {train_df['y_default'].mean():.4f}")

    y_train = train_df["y_default"].values
    y_val   = val_df["y_default"].values
    y_test  = test_df["y_default"].values

    Xb_train = train_df[X.columns]
    Xb_val   = val_df[X.columns]
    Xb_test  = test_df[X.columns]

    Xb_train, Xb_val, Xb_test = drop_all_missing_cols(Xb_train, Xb_val, Xb_test, label="Borrower-only")

    # ── 7. Clean & preprocess ─────────────────────────────────────────────────
    print("=== Clean & fit preprocessor ===")
    Xb_train = clean_dataframe(Xb_train)
    Xb_val   = clean_dataframe(Xb_val)
    Xb_test  = clean_dataframe(Xb_test)
    print("  Dataframes cleaned.")

    pre, _, _ = build_preprocessor(Xb_train)
    Xtr = pre.fit_transform(Xb_train)
    Xva = pre.transform(Xb_val)
    Xte = pre.transform(Xb_test)
    print(f"Preprocessed shape: {Xtr.shape}")

    # ── 8. Baseline XGBoost ───────────────────────────────────────────────────
    print("\n=== Baseline XGBoost (replicating model_improvement.py) ===")
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    # XGBoost ≥2.0 moved early_stopping_rounds to the constructor.
    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    es_constructor = {"early_stopping_rounds": 30} if xgb_version >= (2, 0) else {}
    es_fit         = {} if xgb_version >= (2, 0) else {"early_stopping_rounds": 30}

    pos = float(y_train.sum())
    neg = float((y_train == 0).sum())
    spw = neg / pos if pos > 0 else 1.0  # up-weight the minority class

    baseline = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        scale_pos_weight=spw,
        verbosity=0,
        **es_constructor,
    )
    baseline.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=50, **es_fit)
    baseline_prob    = baseline.predict_proba(Xte)[:, 1]
    baseline_metrics = eval_metrics(y_test, baseline_prob)
    print(f"Baseline Test ROC-AUC: {baseline_metrics['roc_auc']:.4f}")

    # ── 9. Randomized search ──────────────────────────────────────────────────
    print(f"\n=== Randomized Search ({N_ITER} combinations, {CV_FOLDS}-fold CV) ===")
    print("This will take 30–60 minutes on Colab free tier. Sit back!")

    search_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        scale_pos_weight=spw,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=search_model,
        param_distributions=XGB_PARAM_DIST,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        verbose=2,    # prints progress for each combination as it runs
        random_state=RANDOM_SEED,
        n_jobs=1,     # XGBoost uses n_jobs=-1 internally; outer parallelism would conflict
        refit=True,   # refit best config on the full training set after search
    )

    search.fit(Xtr, y_train)

    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {json.dumps(search.best_params_, indent=2)}")

    # ── 10. Evaluate best model ────────────────────────────────────────────────
    print("\n=== Evaluate best model on held-out test set ===")
    tuned_prob    = search.best_estimator_.predict_proba(Xte)[:, 1]
    tuned_metrics = eval_metrics(y_test, tuned_prob)

    print(f"Baseline Test ROC-AUC : {baseline_metrics['roc_auc']:.4f}")
    print(f"Tuned    Test ROC-AUC : {tuned_metrics['roc_auc']:.4f}")
    print(f"Improvement           : {tuned_metrics['roc_auc'] - baseline_metrics['roc_auc']:+.4f}")

    boot = bootstrap_auc(y_test, tuned_prob, n_boot=N_BOOT)
    print(f"Tuned AUC 95% CI: [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")

    # ── 11. Save results ───────────────────────────────────────────────────────
    results = {
        "baseline": {
            "params": {
                "n_estimators": 300, "learning_rate": 0.1,
                "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
            },
            "test_metrics": baseline_metrics,
        },
        "tuned": {
            "best_params":   search.best_params_,
            "best_cv_score": float(search.best_score_),
            "test_metrics":  tuned_metrics,
            "bootstrap_auc": boot,
        },
        "improvement": float(tuned_metrics["roc_auc"] - baseline_metrics["roc_auc"]),
        "cv_results_top10": (
            pd.DataFrame(search.cv_results_)
            .sort_values("rank_test_score")
            .head(10)[["rank_test_score", "mean_test_score", "std_test_score", "params"]]
            .to_dict(orient="records")
        ),
    }

    results_path = os.path.join(OUTDIR, "hyperparam_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {results_path}")

    # ── 12. ROC plot ───────────────────────────────────────────────────────────
    plot_roc(
        y_test,
        baseline_prob,
        tuned_prob,
        out_path=os.path.join(FIGDIR, "roc_baseline_vs_tuned.png"),
    )

    # Top-10 combinations exported as CSV for inclusion in the report.
    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(10)
    cv_df.to_csv(os.path.join(OUTDIR, "top10_hyperparams.csv"), index=False)
    print(f"Top-10 combinations saved to: {OUTDIR}/top10_hyperparams.csv")

    print("\n=== DONE ===")
    print(f"All outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
