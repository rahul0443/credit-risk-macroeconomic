#!/usr/bin/env python3
"""
Capstone: Credit Risk Modeling Under Macroeconomic Conditions
Stage 4 of 6 — Macro Integration, Regime Analysis & Stress Simulation

Extends model_v2.py with three analyses:

  Macro comparison — Do macroeconomic features actually improve prediction?
           Trains two LGBM models side-by-side:
             Model A: Borrower-only features (replicates model_v2.py best)
             Model B: Borrower + Macro features (FEDFUNDS + UNRATE lags/changes)
           Bootstrap test for statistical significance of macro lift.

  Regime analysis — Does performance hold across economic regimes?
           Evaluates both models year-by-year on the val+test period (2015-2018).
           Reports AUC, default rate, and macro lift per year to show
           whether macro features improve stability across economic conditions.

  Stress simulation — Portfolio stress test under adverse macro scenarios.
           Uses Model B (macro model) to simulate three stress scenarios:
             - Mild Recession:    UNRATE +2%,  FEDFUNDS -1%
             - Severe Recession:  UNRATE +4%,  FEDFUNDS -2%  (2008-like)
             - Rate Shock:        UNRATE +1%,  FEDFUNDS +3%  (2022-like)
           Reports mean P(default) shift and high-risk loan rate per scenario.

Pipeline order:
  1_exploratory_data_analysis.py   ← run locally
  2_model_hyperparameter_search.py ← Colab only, one-time
  3_model_lgbm_search.py           ← Colab only, one-time
  4_model_v3.py                    ← THIS FILE (run locally)
  5_run_pipeline.py                ← orchestrates 1 + 4 end-to-end

Progress tracker:
  model_improvement.py  — XGB: 0.702, LR: 0.693   (baseline)
  model_v2.py           — LGBM: 0.7103             (best single model)
  model_v3.py           — macro analysis + regime + stress (this file)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ACCEPTED_FILE = os.path.join(BASE_DIR, "..", "data", "accepted_2007_to_2018Q4.csv")
FEDFUNDS_FILE = os.path.join(BASE_DIR, "..", "data", "FEDFUNDS.csv")
UNRATE_FILE   = os.path.join(BASE_DIR, "..", "data", "UNRATE.csv")

OUTDIR = os.path.join(BASE_DIR, "..", "outputs", "model")
FIGDIR = os.path.join(OUTDIR, "figures")

RANDOM_SEED = 42
MAX_TRAIN   = 500_000  # cap training rows to keep runtime and memory manageable
N_BOOT      = 100      # bootstrap iterations for AUC confidence intervals

# Target definition must match files 2 and 3 to ensure consistent labeling.
GOOD_STATUSES    = {"Fully Paid"}
DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"}
EXTRA_HARD_DROP  = ["zip_code", "member_id"]

# LGBM params tuned from the Colab search in 3_model_lgbm_search.py.
LGBM_PARAMS = {
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "subsample":         0.9,
    "colsample_bytree":  0.9,
    "reg_lambda":        5.0,
    "reg_alpha":         0.0,
    "min_child_samples": 50,
    "n_jobs":            2,
    "random_state":      RANDOM_SEED,
    "metric":            "auc",
}

# Maps each issue year to a broad economic regime label.
# Used to annotate the year-by-year evaluation table and chart.
REGIME_MAP = {
    2007: "Pre-Crisis",
    2008: "Crisis",
    2009: "Crisis",
    2010: "Recovery",
    2011: "Recovery",
    2012: "Recovery",
    2013: "Recovery",
    2014: "Recovery",
    2015: "Expansion",
    2016: "Expansion",
    2017: "Expansion",
    2018: "Expansion",
}

# Stress scenarios apply a parallel shift to all UNRATE-related and
# FEDFUNDS-related macro columns in the test set. Borrower features are unchanged.
STRESS_SCENARIOS = {
    "Baseline":         {"delta_unrate": 0.0, "delta_fedfunds":  0.0},
    "Mild Recession":   {"delta_unrate": 2.0, "delta_fedfunds": -1.0},
    "Severe Recession": {"delta_unrate": 4.0, "delta_fedfunds": -2.0},  # 2008-like
    "Rate Shock":       {"delta_unrate": 1.0, "delta_fedfunds":  3.0},  # 2022-like
}

# Loans with calibrated P(default) above this threshold are flagged as high-risk.
DEFAULT_THRESHOLD = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_outdirs() -> None:
    """Create output directories if they do not already exist."""
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase all column names.

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
    return safe_to_datetime(dt).dt.to_period("M").dt.to_timestamp()


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
    for c in ["issue_d", "issue_date", "issued", "issue"]:
        if c in df.columns:
            df[c] = safe_to_datetime(df[c])
            return df, c
    for c in df.columns:
        if "issue" in c and ("_d" in c or c.endswith("d")):
            df[c] = safe_to_datetime(df[c])
            return df, c
    raise ValueError("Could not find an issue date column.")


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target column `y_default` from `loan_status`.

    Rows with ambiguous statuses (e.g. 'Current') are dropped to avoid
    label noise — only resolved outcomes are kept.

    Args:
        df: DataFrame containing a 'loan_status' column.

    Returns:
        Filtered DataFrame with added 'y_default' column (0 = good, 1 = default).
    """
    df = df.copy()
    before = len(df)
    df = df[df["loan_status"].isin(GOOD_STATUSES | DEFAULT_STATUSES)].copy()
    df["y_default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    print(f"Target filter: {before:,} -> {len(df):,}")
    return df


def read_fred_csv(path: str, value_col: str) -> pd.DataFrame:
    """Load a FRED download CSV and return a clean (date, value) DataFrame.

    Args:
        path:      Path to the FRED CSV file.
        value_col: Desired output name for the value column (e.g. 'FEDFUNDS').

    Returns:
        Two-column DataFrame with columns ['date', value_col], sorted by date.

    Raises:
        ValueError: If no date column can be identified.
    """
    df = normalize_colnames(pd.read_csv(path))
    date_col = next((c for c in ["date", "observation_date"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"{path}: no date column found.")
    non_date = [c for c in df.columns if c != date_col]
    series_col = value_col.lower() if value_col.lower() in df.columns else non_date[0]
    return pd.DataFrame({
        "date":    safe_to_datetime(df[date_col]),
        value_col: pd.to_numeric(df[series_col], errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")


def engineer_macro(fed: pd.DataFrame, unr: pd.DataFrame) -> pd.DataFrame:
    """Merge FEDFUNDS and UNRATE and engineer lag/change/rolling features.

    Args:
        fed: DataFrame with columns ['date', 'FEDFUNDS'].
        unr: DataFrame with columns ['date', 'UNRATE'].

    Returns:
        Monthly DataFrame with FEDFUNDS, UNRATE, and derived lag/change/roll columns.
    """
    for df in [fed, unr]:
        df["month"] = create_month_key(df["date"])
    fed = fed.drop(columns=["date"]).drop_duplicates("month").sort_values("month")
    unr = unr.drop(columns=["date"]).drop_duplicates("month").sort_values("month")
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
    keywords = [
        "pymnt", "payment", "last_pymnt", "next_pymnt", "recover", "recovery",
        "collection", "settlement", "debt_settlement", "hardship", "out_prncp",
        "total_pymnt", "total_rec_", "last_fico", "acc_now_delinq",
        "chargeoff", "chargedoff", "charge_off",
    ]
    return sorted({c for c in df.columns if any(k in c.lower() for k in keywords)})


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
    must = [c for c in df.columns if any(re.search(p, c, re.I) for p in patterns)]
    explicit = [
        "total_rec_late_fee", "funded_amnt_inv", "collection_recovery_fee",
        "delinq_2yrs", "acc_now_delinq", "delinq_amnt",
        "mths_since_recent_revol_delinq", "sec_app_fico_range_low",
        "sec_app_fico_range_high", "sec_app_inq_last_6mths",
        "sec_app_mths_since_last_major_derog",
    ]
    must += [c for c in explicit if c in df.columns]
    return sorted(set(must))


def drop_all_missing_cols(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    label: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove columns that are entirely missing in the training set.

    Applying this filter on train only (then mirroring to val/test) prevents
    features with zero training signal from entering the preprocessor.

    Args:
        X_train: Training feature DataFrame.
        X_val:   Validation feature DataFrame.
        X_test:  Test feature DataFrame.
        label:   Optional label for the print statement.

    Returns:
        Tuple of (X_train, X_val, X_test) with all-missing columns removed.
    """
    missing = [c for c in X_train.columns if X_train[c].notna().sum() == 0]
    if missing:
        print(f"  [{label}] Dropping {len(missing)} all-missing cols")
        X_train = X_train.drop(columns=missing, errors="ignore")
        X_val   = X_val.drop(columns=missing, errors="ignore")
        X_test  = X_test.drop(columns=missing, errors="ignore")
    return X_train, X_val, X_test


def target_encode_state(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, X_test: pd.DataFrame,
    col: str = "addr_state", smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply smoothed mean target encoding to a high-cardinality categorical column.

    Encoding is fit only on the training set and applied to val/test to prevent
    leakage. Smoothing shrinks rare-state estimates toward the global mean,
    reducing variance for states with few observations.

    The formula is: encoded = (sum(y) + smoothing * global_mean) / (count + smoothing)

    Args:
        X_train:   Training feature DataFrame.
        y_train:   Training target array.
        X_val:     Validation feature DataFrame.
        X_test:    Test feature DataFrame.
        col:       Column to encode (default 'addr_state').
        smoothing: Smoothing strength; higher = more shrinkage toward global mean.

    Returns:
        Tuple of (X_train, X_val, X_test) with the original column replaced by
        a new '{col}_target_enc' numeric column.
    """
    if col not in X_train.columns:
        return X_train, X_val, X_test
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    global_mean = y_train.mean()
    train_s = X_train[col].astype(str).str.strip().str.upper()
    temp     = pd.DataFrame({"state": train_s, "y": y_train})
    stats    = temp.groupby("state")["y"].agg(["sum", "count"])
    stats["encoded"] = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
    enc_map  = stats["encoded"].to_dict()
    new_col  = f"{col}_target_enc"
    for df_s, series in [
        (X_train, train_s),
        (X_val,   X_val[col].astype(str).str.strip().str.upper()),
        (X_test,  X_test[col].astype(str).str.strip().str.upper()),
    ]:
        df_s[new_col] = series.map(enc_map).fillna(global_mean)
    X_train = X_train.drop(columns=[col], errors="ignore")
    X_val   = X_val.drop(columns=[col],   errors="ignore")
    X_test  = X_test.drop(columns=[col],  errors="ignore")
    print(f"  Target encoded '{col}' ({len(enc_map)} states, global_mean={global_mean:.4f})")
    return X_train, X_val, X_test


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer that imputes and scales numeric columns and
    one-hot encodes low-cardinality categoricals.

    High-cardinality categoricals (>20 unique values) are dropped entirely
    rather than encoded, to avoid inflating the feature space.

    Args:
        X: Feature DataFrame (post-cleaning, pre-transform).

    Returns:
        Tuple of (preprocessor, numeric_col_names, categorical_col_names).
    """
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            sample = X[col].dropna().head(50)
            if sample.astype(str).str.contains("%", na=False).any():
                X[col] = pd.to_numeric(
                    X[col].astype(str).str.replace("%", "", regex=False), errors="coerce"
                )
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    raw_cat  = [c for c in X.columns if c not in num_cols]
    cat_cols = [c for c in raw_cat if X[c].nunique() <= 20]
    dropped  = [c for c in raw_cat if c not in cat_cols]
    if dropped:
        print(f"  Dropped {len(dropped)} high-cardinality: {dropped[:8]}")
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=True, max_categories=15))])
    pre = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop", sparse_threshold=0.3,
    )
    return pre, num_cols, cat_cols


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 threshold: float = 0.5) -> Dict:
    """Compute a standard set of binary classification metrics.

    Args:
        y_true:    Ground-truth binary labels.
        y_prob:    Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dictionary with keys: roc_auc, pr_auc, brier, tn, fp, fn, tp.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc":  float(average_precision_score(y_true, y_prob)),
        "brier":   float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibrate_probs(
    val_probs:  np.ndarray,
    y_val:      np.ndarray,
    test_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Apply Platt scaling to correct inflated raw probabilities.

    scale_pos_weight improves AUC ranking but inflates raw probability outputs.
    Platt scaling fits a logistic regression on the 1-D val-set probabilities,
    then applies it to re-scale test probabilities so that mean P(default)
    approximates the actual default rate — essential for credible portfolio-level
    stress simulation numbers. AUC is rank-based and is NOT changed by calibration.

    Args:
        val_probs:  Raw predicted probabilities on the validation set.
        y_val:      Ground-truth labels for the validation set.
        test_probs: Raw predicted probabilities on the test set.

    Returns:
        Tuple of (calibrated_val_probs, calibrated_test_probs, fitted_platt_scaler).
    """
    from sklearn.linear_model import LogisticRegression as _LR
    platt = _LR(solver="lbfgs", max_iter=1000, C=1.0)
    platt.fit(val_probs.reshape(-1, 1), y_val)
    val_cal  = platt.predict_proba(val_probs.reshape(-1,  1))[:, 1]
    test_cal = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    return val_cal, test_cal, platt


def bootstrap_auc_diff(
    y_true: np.ndarray, a: np.ndarray, b: np.ndarray,
    n_boot: int = 100, seed: int = 42,
) -> Dict:
    """Estimate a 95% bootstrap confidence interval for AUC(a) - AUC(b).

    Used to test whether the macro model (a) is statistically better than
    the borrower-only model (b). A CI that excludes zero indicates a
    significant difference.

    Args:
        y_true: Ground-truth binary labels.
        a:      Predicted probabilities from model A.
        b:      Predicted probabilities from model B.
        n_boot: Number of bootstrap resamples (default 100).
        seed:   Random seed for reproducibility.

    Returns:
        Dictionary with keys: mean, ci_low, ci_high, n_boot.
        Returns None values if no valid bootstrap iterations complete.
    """
    rng, n, diffs = np.random.default_rng(seed), len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb  = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, a[idx]) - roc_auc_score(yb, b[idx]))
    if not diffs:
        return {"mean": None, "ci_low": None, "ci_high": None}
    d = np.array(diffs)
    return {
        "mean":    float(d.mean()),
        "ci_low":  float(np.quantile(d, 0.025)),
        "ci_high": float(np.quantile(d, 0.975)),
        "n_boot":  len(diffs),
    }


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
    t_max = months.iloc[int(np.floor(train_q * (len(months) - 1)))]
    v_max = months.iloc[int(np.floor(val_q   * (len(months) - 1)))]
    return SplitData(
        train=d[d[date_col] <= t_max].copy(),
        val  =d[(d[date_col] > t_max) & (d[date_col] <= v_max)].copy(),
        test =d[d[date_col] > v_max].copy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model runner — LGBM only (fastest, best performer)
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val:   pd.DataFrame, y_val:   np.ndarray,
    X_test:  pd.DataFrame, y_test:  np.ndarray,
    label: str, pre_fitted=None,
) -> Tuple[object, object, np.ndarray, np.ndarray, Dict, Dict]:
    """Fit a LightGBM classifier and evaluate on val and test sets.

    scale_pos_weight is computed from the training set to up-weight the
    minority (default) class without requiring class resampling.

    Args:
        X_train:     Training feature DataFrame.
        y_train:     Training target array.
        X_val:       Validation feature DataFrame.
        y_val:       Validation target array.
        X_test:      Test feature DataFrame.
        y_test:      Test target array.
        label:       Display name for this model (used in print statements).
        pre_fitted:  Pre-fitted ColumnTransformer to reuse; if None, a new
                     one is built and fit on X_train.

    Returns:
        Tuple of (model, preprocessor, val_probs, test_probs, val_metrics, test_metrics).
    """
    import lightgbm as lgb

    if pre_fitted is None:
        pre, _, _ = build_preprocessor(X_train)
        Xtr = pre.fit_transform(X_train)
    else:
        pre = pre_fitted
        Xtr = pre.transform(X_train)

    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)

    pos = float(y_train.sum())
    neg = float((y_train == 0).sum())
    spw = neg / pos if pos > 0 else 1.0  # up-weight the minority class

    model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)
    model.fit(
        Xtr, y_train,
        eval_set=[(Xva, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    val_probs  = model.predict_proba(Xva)[:, 1]
    test_probs = model.predict_proba(Xte)[:, 1]

    print(f"  [{label}] Val  AUC: {roc_auc_score(y_val,  val_probs):.4f}")
    print(f"  [{label}] Test AUC: {roc_auc_score(y_test, test_probs):.4f}")

    return (model, pre,
            val_probs, test_probs,
            eval_metrics(y_val,  val_probs),
            eval_metrics(y_test, test_probs))


# ─────────────────────────────────────────────────────────────────────────────
# Regime / year-by-year evaluation
# ─────────────────────────────────────────────────────────────────────────────

def regime_evaluation(
    y_eval:     np.ndarray,
    probs_a:    np.ndarray,
    probs_b:    np.ndarray,
    eval_years: np.ndarray,
) -> pd.DataFrame:
    """Compute per-year AUC for both models across the val+test evaluation period.

    Years with fewer than 50 loans or only one target class are skipped to
    avoid undefined or unreliable AUC estimates.

    Args:
        y_eval:     Ground-truth labels for the combined val+test period.
        probs_a:    Predicted probabilities from Model A (borrower-only).
        probs_b:    Predicted probabilities from Model B (macro).
        eval_years: Issue year for each observation in the eval period.

    Returns:
        DataFrame with columns: Year, Regime, N_loans, Default_rate,
        AUC_borrower, AUC_macro, Macro_lift.
    """
    rows = []
    for yr in sorted(np.unique(eval_years)):
        mask = eval_years == yr
        yb   = y_eval[mask]
        if len(np.unique(yb)) < 2 or mask.sum() < 50:
            continue
        auc_a = roc_auc_score(yb, probs_a[mask])
        auc_b = roc_auc_score(yb, probs_b[mask])
        rows.append({
            "Year":         int(yr),
            "Regime":       REGIME_MAP.get(int(yr), "Unknown"),
            "N_loans":      int(mask.sum()),
            "Default_rate": float(yb.mean()),
            "AUC_borrower": float(auc_a),
            "AUC_macro":    float(auc_b),
            "Macro_lift":   float(auc_b - auc_a),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Stress simulation
# ─────────────────────────────────────────────────────────────────────────────

def apply_macro_stress(
    X_test_macro: pd.DataFrame,
    macro_cols:   List[str],
    delta_unrate:    float = 0.0,
    delta_fedfunds:  float = 0.0,
) -> pd.DataFrame:
    """Apply a parallel shift to all macro columns in the test set.

    All UNRATE-related columns are shifted by delta_unrate and all
    FEDFUNDS-related columns by delta_fedfunds. Borrower features are
    left untouched, isolating the macro signal for the stress test.

    Args:
        X_test_macro:   Test feature DataFrame including macro columns.
        macro_cols:     List of macro column names to shift.
        delta_unrate:   Additive shift applied to UNRATE-related columns.
        delta_fedfunds: Additive shift applied to FEDFUNDS-related columns.

    Returns:
        Stressed copy of X_test_macro.
    """
    X_stress = X_test_macro.copy()
    for col in macro_cols:
        if "UNRATE"    in col:
            X_stress[col] = X_stress[col] + delta_unrate
        elif "FEDFUNDS" in col:
            X_stress[col] = X_stress[col] + delta_fedfunds
    return X_stress


def stress_simulation(
    model_b,
    pre_b,
    platt_b,
    Xm_test:    pd.DataFrame,
    y_test:     np.ndarray,
    macro_cols: List[str],
) -> pd.DataFrame:
    """Run all STRESS_SCENARIOS and report calibrated portfolio-level metrics.

    For each scenario, macro columns in the test set are shifted, Model B
    re-predicts, Platt calibration is applied so P(default) is on a realistic
    scale, and portfolio metrics are computed. AUC uses raw (uncalibrated)
    probabilities since it is rank-based and unaffected by calibration.

    Args:
        model_b:    Fitted Model B (borrower+macro LGBMClassifier).
        pre_b:      Fitted preprocessor for Model B.
        platt_b:    Fitted Platt scaler from calibrate_probs().
        Xm_test:    Test feature DataFrame (borrower + macro columns, pre-transform).
        y_test:     Ground-truth labels for the test set.
        macro_cols: List of macro column names to shift per scenario.

    Returns:
        DataFrame with one row per scenario and columns: Scenario, delta_UNRATE,
        delta_FEDFUNDS, mean_P_default, high_risk_pct, AUC, delta_mean_P,
        delta_highrisk.
    """
    rows = []
    for scenario_name, deltas in STRESS_SCENARIOS.items():
        X_stress  = apply_macro_stress(
            Xm_test, macro_cols,
            delta_unrate=deltas["delta_unrate"],
            delta_fedfunds=deltas["delta_fedfunds"],
        )
        X_stress_t = pre_b.transform(X_stress)
        raw_prob   = model_b.predict_proba(X_stress_t)[:, 1]
        # Platt calibration fixes the probability scale without changing AUC.
        cal_prob   = platt_b.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
        rows.append({
            "Scenario":       scenario_name,
            "delta_UNRATE":   deltas["delta_unrate"],
            "delta_FEDFUNDS": deltas["delta_fedfunds"],
            "mean_P_default": float(cal_prob.mean()),
            "high_risk_pct":  float((cal_prob > DEFAULT_THRESHOLD).mean()),
            "AUC":            float(roc_auc_score(y_test, raw_prob)),
        })

    df = pd.DataFrame(rows)
    baseline_mean = df.loc[df["Scenario"] == "Baseline", "mean_P_default"].values[0]
    baseline_hr   = df.loc[df["Scenario"] == "Baseline", "high_risk_pct"].values[0]
    df["delta_mean_P"]   = df["mean_P_default"] - baseline_mean
    df["delta_highrisk"] = df["high_risk_pct"]  - baseline_hr
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_comparison(y_true, curves: Dict[str, np.ndarray],
                        title: str, fname: str) -> None:
    """Save an ROC curve comparing multiple models on the same axes.

    Args:
        y_true:  Ground-truth binary labels.
        curves:  Dict mapping display name → predicted probability array.
        title:   Chart title string.
        fname:   Output filename (written to FIGDIR).
    """
    plt.figure(figsize=(9, 7))
    for name, prob in curves.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname), dpi=160)
    plt.close()
    print(f"  Plot saved: {fname}")


def plot_regime_auc(regime_df: pd.DataFrame, fname: str) -> None:
    """Save a grouped bar chart of AUC by year with a secondary default-rate axis.

    Displaying both models side by side per year makes the macro lift (or lack
    thereof) visible per economic regime at a glance.

    Args:
        regime_df: DataFrame from regime_evaluation() with per-year metrics.
        fname:     Output filename (written to FIGDIR).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regime_df))
    w = 0.35
    ax1.bar(x - w/2, regime_df["AUC_borrower"], w,
            label="Borrower-only", color="steelblue", alpha=0.85)
    ax1.bar(x + w/2, regime_df["AUC_macro"],    w,
            label="Borrower+Macro", color="darkorange", alpha=0.85)
    ax1.set_ylim(0.65, 0.77)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{int(r['Year'])}\n({r['Regime']})" for _, r in regime_df.iterrows()],
        fontsize=9,
    )
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("Model AUC by Year / Economic Regime")
    ax1.legend(loc="upper right")
    ax1.axhline(0.70, color="red", linestyle="--", linewidth=0.8,
                label="0.70 reference")

    ax2 = ax1.twinx()
    ax2.plot(x, regime_df["Default_rate"] * 100, "k--o",
             markersize=5, label="Default rate %")
    ax2.set_ylabel("Default Rate (%)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname), dpi=160)
    plt.close()
    print(f"  Plot saved: {fname}")


def plot_stress(stress_df: pd.DataFrame, fname: str) -> None:
    """Save a two-panel bar chart of stress scenario results.

    Left panel shows mean calibrated P(default); right panel shows
    the percentage of loans classified as high-risk (P > DEFAULT_THRESHOLD).

    Args:
        stress_df: DataFrame from stress_simulation() with per-scenario metrics.
        fname:     Output filename (written to FIGDIR).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["steelblue", "gold", "tomato", "mediumpurple"]
    scenarios = stress_df["Scenario"].tolist()
    ax1.bar(scenarios, stress_df["mean_P_default"] * 100, color=colors, alpha=0.85)
    ax1.set_ylabel("Mean P(default) [%]")
    ax1.set_title("Mean Predicted Default Probability\nby Stress Scenario")
    ax1.set_ylim(0, stress_df["mean_P_default"].max() * 130)
    for i, (_, row) in enumerate(stress_df.iterrows()):
        ax1.text(i, row["mean_P_default"] * 100 + 0.2,
                 f"{row['mean_P_default']*100:.1f}%", ha="center", fontsize=9)
    ax2.bar(scenarios, stress_df["high_risk_pct"] * 100, color=colors, alpha=0.85)
    ax2.set_ylabel(f"High-Risk Loans [%]  (P > {DEFAULT_THRESHOLD})")
    ax2.set_title(f"High-Risk Loan Rate (P > {DEFAULT_THRESHOLD})\nby Stress Scenario")
    ax2.set_ylim(0, stress_df["high_risk_pct"].max() * 130)
    for i, (_, row) in enumerate(stress_df.iterrows()):
        ax2.text(i, row["high_risk_pct"] * 100 + 0.2,
                 f"{row['high_risk_pct']*100:.1f}%", ha="center", fontsize=9)
    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname), dpi=160)
    plt.close()
    print(f"  Plot saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full model v3 pipeline end-to-end.

    Steps:
      1.  Load raw loan CSV and parse issue dates.
      2.  Load and engineer FRED macro features; merge onto loan data.
      3.  Engineer borrower features (sub-grade encoding, log income, credit age).
      4.  Drop leakage and hard-coded post-origination columns.
      5.  Build two feature sets: borrower-only (A) and borrower+macro (B).
      6.  Apply temporal train/val/test split; downsample training set.
      7.  Target-encode addr_state on both feature sets.
      8.  Train Model A (borrower-only LGBM).
      9.  Train Model B (borrower+macro LGBM).
      10. Calibrate Model B probabilities with Platt scaling.
      11. Bootstrap significance test: does macro help?
      12. Year-by-year regime evaluation.
      13. Portfolio stress simulation.
      14. Save all results to JSON.
    """
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)
    ensure_outdirs()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("=== 1. Load accepted loans ===")
    loans = pd.read_csv(ACCEPTED_FILE, low_memory=False)
    loans = normalize_colnames(loans)
    print(f"  Raw shape: {loans.shape[0]:,} x {loans.shape[1]:,}")

    loans, issue_col = infer_issue_date(loans)
    loans["issue_month"] = create_month_key(loans[issue_col])
    loans = make_target(loans)
    loans = loans.dropna(subset=["issue_month", "y_default"]).copy()
    print(f"  After filter: {loans.shape[0]:,} rows")

    # ── 2. Macro ──────────────────────────────────────────────────────────────
    print("\n=== 2. Load macro ===")
    fed   = read_fred_csv(FEDFUNDS_FILE, "FEDFUNDS")
    unr   = read_fred_csv(UNRATE_FILE,   "UNRATE")
    macro = engineer_macro(fed, unr)
    loans = loans.merge(macro, left_on="issue_month",
                        right_on="month", how="left").drop(columns=["month"])
    macro_feature_cols = [c for c in loans.columns
                          if c.startswith("FEDFUNDS") or c.startswith("UNRATE")]
    print(f"  Macro features added: {len(macro_feature_cols)}")
    print(f"  Macro cols: {macro_feature_cols}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\n=== 3. Feature engineering ===")
    if "sub_grade" in loans.columns:
        grades = list("ABCDEFG")
        sg_map = {f"{g}{n}": i * 5 + n
                  for i, g in enumerate(grades) for n in range(1, 6)}
        loans["sub_grade_enc"] = loans["sub_grade"].map(sg_map)
        loans = loans.drop(columns=["sub_grade"])

    if "annual_inc" in loans.columns:
        loans["annual_inc"] = np.log1p(
            pd.to_numeric(loans["annual_inc"], errors="coerce"))

    if "earliest_cr_line" in loans.columns:
        ecl = safe_to_datetime(loans["earliest_cr_line"])
        loans["credit_age_months"] = (
            (loans["issue_month"] - ecl).dt.days / 30.44).clip(lower=0)
        loans = loans.drop(columns=["earliest_cr_line"])

    # ── 4. Leakage drop ───────────────────────────────────────────────────────
    print("\n=== 4. Leakage drop ===")
    drop_cols = sorted(
        set(find_leakage_columns(loans)) | set(hard_drop_columns(loans))
    )
    protected = {"y_default", "issue_month", issue_col, "loan_status"}
    drop_cols = [c for c in drop_cols if c not in protected]
    loans = loans.drop(columns=drop_cols + EXTRA_HARD_DROP, errors="ignore")
    print(f"  Shape after leakage drop: {loans.shape[0]:,} x {loans.shape[1]:,}")

    # ── 5. Build two feature sets ─────────────────────────────────────────────
    # Model A uses borrower features only; Model B adds macro columns.
    # Keeping identical rows ensures any AUC difference is due to features, not data.
    non_feature = {"y_default", "loan_status", issue_col}
    X_full     = loans.drop(
        columns=[c for c in non_feature if c in loans.columns], errors="ignore"
    )
    X_borrower = X_full.drop(columns=macro_feature_cols, errors="ignore")
    print(f"\n  Borrower-only features : {X_borrower.shape[1]}")
    print(f"  Borrower+Macro features: {X_full.shape[1]}")

    # ── 6. Temporal split ─────────────────────────────────────────────────────
    print("\n=== 5. Temporal split ===")

    def build_split_df(X_df: pd.DataFrame) -> pd.DataFrame:
        """Attach target and issue_month to a feature DataFrame for splitting."""
        out = X_df.copy()
        out["y_default"]   = loans["y_default"].values
        out["issue_month"] = loans["issue_month"].values
        return out.loc[:, ~out.columns.duplicated()]

    splits_b = temporal_split(build_split_df(X_borrower), "issue_month")
    splits_m = temporal_split(build_split_df(X_full),     "issue_month")

    # Downsample training set; apply the same index to both feature sets
    # so Model A and Model B train on exactly the same loans.
    train_b = splits_b.train
    if len(train_b) > MAX_TRAIN:
        idx = train_b.sample(MAX_TRAIN, random_state=RANDOM_SEED).index
        train_b        = splits_b.train.loc[idx]
        splits_m.train = splits_m.train.loc[idx]

    print(f"  Train: {len(train_b):,} | "
          f"Val: {len(splits_b.val):,} | "
          f"Test: {len(splits_b.test):,}")

    y_train = train_b["y_default"].values
    y_val   = splits_b.val["y_default"].values
    y_test  = splits_b.test["y_default"].values

    Xb_train = train_b[X_borrower.columns]
    Xb_val   = splits_b.val[X_borrower.columns]
    Xb_test  = splits_b.test[X_borrower.columns]

    Xm_train = splits_m.train[X_full.columns]
    Xm_val   = splits_m.val[X_full.columns]
    Xm_test  = splits_m.test[X_full.columns]

    Xb_train, Xb_val, Xb_test = drop_all_missing_cols(
        Xb_train, Xb_val, Xb_test, "Borrower-only")
    Xm_train, Xm_val, Xm_test = drop_all_missing_cols(
        Xm_train, Xm_val, Xm_test, "Borrower+Macro")

    # Combine val+test issue months for the regime analysis in step 12.
    val_months  = splits_b.val["issue_month"]
    test_months = splits_b.test["issue_month"]
    eval_months = pd.concat([val_months, test_months], ignore_index=True)
    eval_years  = eval_months.dt.year.values
    y_eval      = np.concatenate([y_val, y_test])

    # ── 7. Target encode addr_state ───────────────────────────────────────────
    print("\n=== 6. Target encode addr_state ===")
    Xb_train, Xb_val, Xb_test = target_encode_state(
        Xb_train, y_train, Xb_val, Xb_test)
    Xm_train, Xm_val, Xm_test = target_encode_state(
        Xm_train, y_train, Xm_val, Xm_test)

    # ── 8. Train Model A ──────────────────────────────────────────────────────
    print("\n=== Train Model A: LGBM Borrower-only ===")
    pre_a, _, _ = build_preprocessor(Xb_train)
    pre_a.fit(Xb_train)
    (model_a, pre_a,
     val_probs_a, test_probs_a,
     val_metrics_a, test_metrics_a) = train_lgbm(
        Xb_train, y_train, Xb_val, y_val, Xb_test, y_test,
        "Borrower-only", pre_fitted=pre_a,
    )

    # ── 9. Train Model B ──────────────────────────────────────────────────────
    print("\n=== Train Model B: LGBM Borrower+Macro ===")
    pre_b, _, _ = build_preprocessor(Xm_train)
    pre_b.fit(Xm_train)
    (model_b, pre_b,
     val_probs_b, test_probs_b,
     val_metrics_b, test_metrics_b) = train_lgbm(
        Xm_train, y_train, Xm_val, y_val, Xm_test, y_test,
        "Borrower+Macro", pre_fitted=pre_b,
    )

    # ── 10. Probability calibration ───────────────────────────────────────────
    # Fit on val set, apply to test and stress scenarios.
    print("\n=== Probability Calibration (Platt Scaling — Model B) ===")
    val_cal_b, test_cal_b, platt_b = calibrate_probs(
        val_probs_b, y_val, test_probs_b
    )
    print(f"  Before calibration — mean P(default) on test : "
          f"{test_probs_b.mean():.1%}")
    print(f"  After  calibration — mean P(default) on test : "
          f"{test_cal_b.mean():.1%}  "
          f"(actual default rate: {y_test.mean():.1%})")
    print(f"  AUC unchanged      — raw: {roc_auc_score(y_test, test_probs_b):.4f}  "
          f"calibrated: {roc_auc_score(y_test, test_cal_b):.4f}")

    # ── 11. Bootstrap significance test: does macro improve AUC? ─────────────
    lift = bootstrap_auc_diff(y_test, test_probs_b, test_probs_a, n_boot=N_BOOT)

    print("\n" + "="*55)
    print("=== Macro Comparison Results: Does Macro Help? ===")
    print("="*55)
    print(f"  Model A — Borrower-only  Test AUC: {test_metrics_a['roc_auc']:.4f}")
    print(f"  Model B — Borrower+Macro Test AUC: {test_metrics_b['roc_auc']:.4f}")
    macro_lift = test_metrics_b["roc_auc"] - test_metrics_a["roc_auc"]
    print(f"  Macro lift              : {macro_lift:+.4f}")
    print(f"  Bootstrap 95% CI        : [{lift['ci_low']:.4f}, {lift['ci_high']:.4f}]")
    if lift["ci_low"] > 0:
        print("  → Macro features provide a STATISTICALLY SIGNIFICANT improvement.")
    elif lift["ci_high"] < 0:
        print("  → Macro features HURT performance (statistically significant).")
    else:
        print("  → Macro lift is NOT statistically significant (CI crosses zero).")
        print("    Borrower attributes already capture most macro signal via")
        print("    loan grade, interest rate, and issue timing.")

    plot_roc_comparison(
        y_test,
        {"Borrower-only": test_probs_a, "Borrower+Macro": test_probs_b},
        title="Macro Comparison: Borrower-only vs Borrower+Macro (Test ROC)",
        fname="gap1_roc_macro_comparison.png",
    )

    # ── 12. Regime evaluation ─────────────────────────────────────────────────
    print("\n=== Year-by-Year Regime Evaluation ===")

    eval_probs_a = np.concatenate([val_probs_a, test_probs_a])
    eval_probs_b = np.concatenate([val_probs_b, test_probs_b])

    regime_df = regime_evaluation(y_eval, eval_probs_a, eval_probs_b, eval_years)

    print(f"\n  {'Year':<6} {'Regime':<16} {'N':>7} {'DefRate':>8} "
          f"{'AUC_A':>8} {'AUC_B':>8} {'MacroLift':>10}")
    print("  " + "-" * 70)
    for _, row in regime_df.iterrows():
        sig = " *" if abs(row["Macro_lift"]) > 0.002 else "  "
        print(f"  {int(row['Year']):<6} {row['Regime']:<16} "
              f"{int(row['N_loans']):>7,} "
              f"{row['Default_rate']:>7.1%}  "
              f"{row['AUC_borrower']:>8.4f} "
              f"{row['AUC_macro']:>8.4f} "
              f"{row['Macro_lift']:>+10.4f}{sig}")

    plot_regime_auc(regime_df, fname="gap2_regime_auc_by_year.png")

    # ── 13. Portfolio stress simulation ───────────────────────────────────────
    # Run on the test set (2017-2018), the most recent held-out period.
    print("\n=== Portfolio Stress Simulation (Model B — Macro) ===")
    print(f"  High-risk threshold: P(default) > {DEFAULT_THRESHOLD}")
    print(f"  Macro cols being shifted: {len(macro_feature_cols)}")

    stress_df = stress_simulation(model_b, pre_b, platt_b, Xm_test, y_test, macro_feature_cols)

    print(f"\n  {'Scenario':<22} {'ΔUNRATE':>8} {'ΔFEDFUNDS':>10} "
          f"{'Mean P(def)':>12} {'Δ Mean P':>9} {'High-risk%':>11} {'ΔAUC':>7}")
    print("  " + "-" * 85)
    base_auc = stress_df.loc[stress_df["Scenario"] == "Baseline", "AUC"].values[0]
    for _, row in stress_df.iterrows():
        print(f"  {row['Scenario']:<22} "
              f"{row['delta_UNRATE']:>+7.1f}% "
              f"{row['delta_FEDFUNDS']:>+9.1f}%  "
              f"{row['mean_P_default']:>11.1%}  "
              f"{row['delta_mean_P']:>+8.2%}  "
              f"{row['high_risk_pct']:>10.1%}  "
              f"{row['AUC'] - base_auc:>+6.4f}")

    plot_stress(stress_df, fname="gap3_stress_simulation.png")

    # ── 14. Save all results ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("=== FINAL SUMMARY ===")
    print("="*55)
    print(f"  model_v2.py best          : LGBM 0.7103")
    print(f"  Model A — Borrower-only   : {test_metrics_a['roc_auc']:.4f}")
    print(f"  Model B — Borrower+Macro  : {test_metrics_b['roc_auc']:.4f}  "
          f"({macro_lift:+.4f})")
    print(f"\n  Analyses completed:")
    print(f"    Macro comparison  ✓ — see gap1_roc_macro_comparison.png")
    print(f"    Regime evaluation ✓ — see gap2_regime_auc_by_year.png")
    print(f"    Stress simulation ✓ — see gap3_stress_simulation.png")

    results = {
        "gap1_macro_comparison": {
            "model_a_borrower_only": {
                "test_auc": test_metrics_a["roc_auc"],
                "val_auc":  val_metrics_a["roc_auc"],
                "test_metrics": test_metrics_a,
            },
            "model_b_borrower_macro": {
                "test_auc": test_metrics_b["roc_auc"],
                "val_auc":  val_metrics_b["roc_auc"],
                "test_metrics": test_metrics_b,
            },
            "macro_lift_test":  macro_lift,
            "bootstrap_95ci":   lift,
            "macro_cols_used":  macro_feature_cols,
        },
        "gap2_regime_evaluation": regime_df.to_dict(orient="records"),
        "gap3_stress_simulation": stress_df.to_dict(orient="records"),
    }

    out_path = os.path.join(OUTDIR, "metrics_v3.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {out_path}")
    print(f"  Figures saved to     : {FIGDIR}/")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
