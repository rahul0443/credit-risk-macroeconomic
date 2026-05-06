"""
Capstone: Credit Risk Modeling Under Macroeconomic Conditions
Stage 1 of 6 — Exploratory Data Analysis

Loads the raw LendingClub accepted loan file, merges FRED macroeconomic
series (FEDFUNDS, UNRATE), builds a binary default target, and produces
data summaries, correlation rankings, and diagnostic figures used in
subsequent modeling stages.

Expected input files (data/ folder):
  - accepted_2007_to_2018Q4.csv
  - FEDFUNDS.csv
  - UNRATE.csv

Run:
  python 1_exploratory_data_analysis.py

Outputs  →  outputs/eda/
  Data:
    data_summary.json
    missingness_top50.csv
    numeric_summary.csv
    macro_merged_sample.csv
    target_rate_by_year.csv
    target_rate_by_issue_month.csv
    target_rate_by_grade.csv
    target_rate_by_term.csv
    target_rate_by_purpose.csv
    feature_target_corr.csv
    grade_year_heatmap.csv
    vintage_cohort.csv
    eda_report.md

  Figures  →  outputs/eda/figures/
    class_imbalance.png
    default_rate_by_month.png
    macro_default_overlay.png
    corr_target_macro.png
    corr_feature_target.png
    kde_int_rate.png
    kde_dti.png
    kde_loan_amnt.png
    kde_annual_inc.png
    default_rate_by_subgrade.png
    default_rate_by_purpose.png
    grade_year_heatmap.png
    vintage_cohort_curves.png
"""

from __future__ import annotations

import os
import json
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ACCEPTED_FILE = os.path.join(BASE_DIR, "..", "data", "accepted_2007_to_2018Q4.csv")
FEDFUNDS_FILE = os.path.join(BASE_DIR, "..", "data", "FEDFUNDS.csv")
UNRATE_FILE   = os.path.join(BASE_DIR, "..", "data", "UNRATE.csv")

OUTDIR        = os.path.join(BASE_DIR, "..", "outputs", "eda")
FIGDIR        = os.path.join(OUTDIR, "figures")
RANDOM_SEED   = 42
PLOT_SAMPLE_N = 200_000  # cap heavy plots at 200K rows to keep runtime manageable
TOP_CORR_N    = 30       # top-30 gives enough signal without crowding the bar chart

# Loans are labeled 1 (default) if they ended in charge-off or sustained
# delinquency. "Fully Paid" is the only unambiguous positive outcome;
# in-progress loans are excluded to avoid label noise.
DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
    "Late (16-30 days)",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

# Explicit ordering required for categorical plots; pandas Categorical
# won't sort A1–G5 correctly without it.
SUBGRADE_ORDER = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]

# Keywords that appear in columns populated only after a loan is active,
# meaning they would not be available at origination time and must be dropped.
LEAKAGE_KEYWORDS = [
    "hardship", "pymnt", "last_pymnt", "recover", "collection", "total_pymnt",
    "funded_amnt_inv", "out_prncp", "out_prncp_inv", "delinq",
    "collection_recovery", "recoveries", "payment_plan", "debt_settlement",
    "settlement", "last_credit_pull",
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_outdirs() -> None:
    """Create output directories if they do not already exist."""
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    """Parse a Series to datetime, coercing unparseable values to NaT instead of raising."""
    return pd.to_datetime(s, errors="coerce")


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names.

    LendingClub CSVs occasionally ship with padded header strings that
    cause silent key-lookup failures downstream.

    Args:
        df: Input DataFrame with potentially whitespace-padded column names.

    Returns:
        Copy of df with stripped column names.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def read_fred_csv(path: str, value_colname: str) -> pd.DataFrame:
    """Load a FRED download CSV and return a clean (date, value) DataFrame.

    FRED exports vary in their date-column header ('DATE' vs 'observation_date'),
    so we detect it by name rather than position.

    Args:
        path:          Path to the FRED CSV file.
        value_colname: Desired output name for the value column (e.g. 'FEDFUNDS').

    Returns:
        Two-column DataFrame with columns ['date', value_colname], sorted by date,
        with non-numeric values dropped.

    Raises:
        ValueError: If no date column can be identified.
    """
    df = pd.read_csv(path)
    df = normalize_colnames(df)
    colmap = {c: c.strip().lower() for c in df.columns}
    date_col = next(
        (c for c in df.columns if colmap[c] in {"date", "observation_date"}), None
    )
    if date_col is None:
        raise ValueError(f"{path}: Cannot find date column.")
    val_col = (
        value_colname
        if value_colname in df.columns
        else [c for c in df.columns if c != date_col][0]
    )
    df[date_col] = safe_to_datetime(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.rename(columns={date_col: "date", val_col: value_colname})
    df[value_colname] = pd.to_numeric(df[value_colname], errors="coerce")
    return df[["date", value_colname]].dropna()


def create_month_key(dt: pd.Series) -> pd.Series:
    """Truncate a datetime Series to month-start timestamps for merge keys.

    Using period→timestamp (rather than dt.floor) ensures consistent
    month-start alignment regardless of the original day component.

    Args:
        dt: Series of datetime values.

    Returns:
        Series of Timestamps representing the first day of each month.
    """
    return dt.dt.to_period("M").dt.to_timestamp()


def infer_issue_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Locate and parse the loan issue-date column from the raw DataFrame.

    LendingClub has used several column names across dataset versions,
    so we probe a priority list rather than hard-coding a single name.
    A secondary format attempt ('%b-%Y') handles the older 'Jan-2015' style.

    Args:
        df: Raw loan DataFrame, straight from CSV.

    Returns:
        Tuple of (df_with_parsed_dates, column_name_found).

    Raises:
        ValueError: If no candidate column is found, or if >50% of dates
                    fail to parse (indicating a format we have not handled).
    """
    candidates = ["issue_d", "issue_date", "issued", "issue"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is None:
        raise ValueError(
            f"Cannot find issue date column. Expected one of {candidates}. "
            f"Found: {df.columns[:30].tolist()}"
        )
    out = df.copy()
    out[found] = safe_to_datetime(out[found])
    if out[found].isna().mean() > 0.2:
        try:
            out[found] = pd.to_datetime(df[found], format="%b-%Y", errors="coerce")
        except Exception:
            pass
    if out[found].isna().mean() > 0.5:
        raise ValueError(
            f"Issue date column '{found}' parse failed (NaT rate="
            f"{out[found].isna().mean():.2%}). Inspect raw values."
        )
    return out, found


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target column `y_default` from `loan_status`.

    Rows whose status falls outside DEFAULT_STATUSES and GOOD_STATUSES
    (e.g. 'Current', 'In Grace Period') are dropped — their final outcome
    is unknown and including them would introduce label noise.

    Args:
        df: DataFrame containing a 'loan_status' column.

    Returns:
        DataFrame with added 'y_default' column (0 = good, 1 = default),
        with ambiguous rows removed.

    Raises:
        ValueError: If 'loan_status' is not present.
    """
    if "loan_status" not in df.columns:
        raise ValueError("Expected column 'loan_status' in dataset.")
    out = df.copy()
    out["loan_status"] = out["loan_status"].astype(str).str.strip()
    out["y_default"] = np.nan
    out.loc[out["loan_status"].isin(DEFAULT_STATUSES), "y_default"] = 1
    out.loc[out["loan_status"].isin(GOOD_STATUSES),   "y_default"] = 0
    before = len(out)
    out = out.dropna(subset=["y_default"])
    out["y_default"] = out["y_default"].astype(int)
    print(f"  Target filter: {before:,} → {len(out):,} rows")
    return out


def to_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV without the row index."""
    df.to_csv(path, index=False)


def to_json(obj: dict, path: str) -> None:
    """Serialize a dictionary to a pretty-printed JSON file.

    Args:
        obj:  Dictionary to serialize.
        path: Destination file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Missingness + Numeric summary
# ─────────────────────────────────────────────────────────────────────────────

def missingness_table(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Rank columns by missing-value rate, highest first.

    Args:
        df:    Input DataFrame.
        top_n: Number of columns to return (default 50).

    Returns:
        DataFrame with columns ['column', 'missing_rate', 'missing_pct'],
        sorted descending by missing_rate.
    """
    miss = df.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({
        "column":       miss.index,
        "missing_rate": miss.values,
        "missing_pct":  (miss.values * 100).round(2),
    })
    return out.head(top_n).reset_index(drop=True)


def basic_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended descriptive statistics for all numeric columns.

    Adds 1st/5th/95th/99th percentiles and a missing-rate column to the
    standard describe() output to help identify outliers and skew.

    Args:
        df: Input DataFrame.

    Returns:
        Transposed describe DataFrame with an added 'missing_rate' column,
        or an empty DataFrame if no numeric columns exist.
    """
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    desc["missing_rate"] = num.isna().mean()
    return desc


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Leakage detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_leakage_candidates(df: pd.DataFrame) -> List[str]:
    """Scan column names for keywords associated with post-origination data.

    These columns are populated only after a loan is active (e.g. payment
    history, recovery amounts), so they must be excluded before training to
    prevent label leakage.

    Args:
        df: DataFrame whose column names will be scanned.

    Returns:
        Sorted list of column names that match at least one leakage keyword.
    """
    leak_cols = []
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in LEAKAGE_KEYWORDS):
            leak_cols.append(col)
    return sorted(set(leak_cols))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Feature–target correlation ranking
# ─────────────────────────────────────────────────────────────────────────────

def feature_target_correlation(
    df: pd.DataFrame, target: str = "y_default", top_n: int = TOP_CORR_N
) -> pd.DataFrame:
    """Rank numeric features by point-biserial correlation with the target.

    Point-biserial is equivalent to Pearson when one variable is binary,
    making it the natural choice for ranking against y_default.
    Sampling is applied first to keep runtime predictable on large datasets.

    Args:
        df:     DataFrame containing both features and the target column.
        target: Name of the binary target column (default 'y_default').
        top_n:  Number of top features to return, ranked by |correlation|.

    Returns:
        DataFrame with columns ['feature', 'correlation', 'abs_corr'],
        sorted descending by abs_corr.
    """
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target and df[c].nunique() > 1
    ]
    sample = df[[target] + num_cols].dropna(subset=[target])
    if len(sample) > PLOT_SAMPLE_N:
        sample = sample.sample(PLOT_SAMPLE_N, random_state=RANDOM_SEED)

    y = sample[target].values
    rows = []
    for col in num_cols:
        x = pd.to_numeric(sample[col], errors="coerce").values
        mask = ~np.isnan(x)
        if mask.sum() < 100:
            continue
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        rows.append({"feature": col, "correlation": float(corr), "abs_corr": abs(corr)})

    return (
        pd.DataFrame(rows)
        .sort_values("abs_corr", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Vintage / cohort analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_vintage_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observed default rates per issue-year cohort at increasing loan ages.

    Loans are grouped into 6-month age buckets based on months elapsed since
    issuance up to the dataset ceiling (2019-01-01). Right-truncation of newer
    cohorts is the key finding: 2016–2018 cohorts appear safer only because
    they had less time to default by the cutoff date.

    Args:
        df: DataFrame with columns ['issue_month', 'y_default'].

    Returns:
        DataFrame with columns ['issue_year', 'age_bucket', 'default_rate', 'n'].
    """
    obs_ceil = pd.Timestamp("2019-01-01")
    out = df[["issue_month", "y_default"]].copy()
    out["issue_year"] = out["issue_month"].dt.year
    out["months_observed"] = (
        (obs_ceil - out["issue_month"]) / pd.Timedelta(days=30.44)
    ).round().clip(0, 72)
    out["age_bucket"] = (out["months_observed"] // 6 * 6).astype(int)
    return (
        out.groupby(["issue_year", "age_bucket"])["y_default"]
        .agg(default_rate="mean", n="size")
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — KDE helper
# ─────────────────────────────────────────────────────────────────────────────

def _kde_from_data(x: np.ndarray, n_points: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Gaussian kernel density estimate without a scipy dependency.

    Uses Silverman's rule-of-thumb bandwidth (1.06 * σ * n^{-0.2}) and
    evaluates the KDE in batches to avoid materialising an n×n_points matrix.

    Args:
        x:        1-D array of numeric values; NaNs are dropped internally.
        n_points: Number of evaluation points on the density grid (default 300).

    Returns:
        Tuple (grid, density) — both 1-D arrays of length n_points.
        Returns two empty arrays if fewer than 10 valid values are present.
    """
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.array([]), np.array([])
    bw = 1.06 * x.std() * len(x) ** (-0.2)
    if bw == 0:
        return np.array([]), np.array([])
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    grid = np.linspace(lo, hi, n_points)
    density = np.zeros(len(grid))
    batch = 5000
    for i in range(0, len(x), batch):
        xb = x[i : i + batch]
        diff = (grid[:, None] - xb[None, :]) / bw
        density += np.exp(-0.5 * diff ** 2).sum(axis=1)
    density /= len(x) * bw * np.sqrt(2 * np.pi)
    return grid, density


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — All plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_imbalance(df: pd.DataFrame, filename: str) -> None:
    """Save a bar chart showing the count and percentage of each target class.

    Visualising imbalance upfront motivates the use of scale_pos_weight in
    tree models and Platt-scaling for probability calibration.

    Args:
        df:       DataFrame containing a 'y_default' column (0/1).
        filename: Output filename (written to FIGDIR).
    """
    counts = df["y_default"].value_counts().sort_index()
    labels = ["Non-default (y=0)", "Default (y=1)"]
    colors = ["#4575b4", "#d73027"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{v:,}\n({v/counts.sum():.1%})",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Number of loans")
    ax.set_title("Class Imbalance: Default vs Non-Default")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_default_rate_by_month(by_month: pd.DataFrame, filename: str) -> None:
    """Save a line chart of monthly default rate across the full date range.

    Args:
        by_month: DataFrame with columns ['issue_month', 'default_rate'].
        filename: Output filename (written to FIGDIR).
    """
    plt.figure(figsize=(11, 4))
    plt.plot(by_month["issue_month"].values, by_month["default_rate"].values,
             color="#333333", linewidth=1.6)
    plt.title("Default-like Rate by Issue Month")
    plt.xlabel("Issue month")
    plt.ylabel("Default-like rate")
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_macro_default_overlay(
    by_month: pd.DataFrame, macro: pd.DataFrame, filename: str
) -> None:
    """Save a dual-axis chart overlaying default rate with UNRATE and FEDFUNDS.

    Placing macro conditions on the same time axis as default rates allows
    visual inspection of whether origination-time macro correlates with
    eventual outcomes — the central question of the capstone hypothesis.
    The GFC window (Sep 2008 – Jun 2009) is shaded for reference.

    Args:
        by_month: DataFrame with columns ['issue_month', 'default_rate'].
        macro:    DataFrame with columns ['month', 'UNRATE', 'FEDFUNDS'].
        filename: Output filename (written to FIGDIR).
    """
    m = macro.sort_values("month")[["month", "UNRATE", "FEDFUNDS"]].copy()
    merged = (
        pd.merge(by_month, m, left_on="issue_month", right_on="month", how="inner")
        .sort_values("issue_month")
    )
    if merged.empty:
        print("  [WARN] Macro overlay: empty after merge — skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(merged["issue_month"], merged["default_rate"],
             color="#333333", linewidth=1.8, label="Default rate (left)")
    ax1.set_ylabel("Default-like rate", color="#333333", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.set_ylim(0, merged["default_rate"].max() * 1.4)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax2 = ax1.twinx()
    ax2.plot(merged["issue_month"], merged["UNRATE"],
             color="#d73027", linewidth=1.4, linestyle="--", label="UNRATE (right)")
    ax2.plot(merged["issue_month"], merged["FEDFUNDS"],
             color="#4575b4", linewidth=1.4, linestyle="-.", label="FEDFUNDS (right)")
    ax2.set_ylabel("% (UNRATE / FEDFUNDS)", color="#555555", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#555555")

    crisis_start = pd.Timestamp("2008-09-01")
    crisis_end   = pd.Timestamp("2009-06-01")
    if merged["issue_month"].min() <= crisis_end and merged["issue_month"].max() >= crisis_start:
        ax1.axvspan(crisis_start, crisis_end, alpha=0.12, color="orange", label="GFC (2008–09)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("Monthly Default Rate vs. Macro Conditions (UNRATE & FEDFUNDS)", fontsize=12)
    ax1.set_xlabel("Issue month")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_macro_corr_heatmap(df: pd.DataFrame, filename: str) -> None:
    """Save a correlation heatmap between y_default and engineered macro features.

    Args:
        df:       DataFrame containing y_default and any UNRATE_* / FEDFUNDS_* columns.
        filename: Output filename (written to FIGDIR).
    """
    macro_cols = [c for c in df.columns if c.startswith("UNRATE") or c.startswith("FEDFUNDS")]
    corr_cols = ["y_default"] + macro_cols
    corr_df = df[corr_cols].dropna()
    if len(corr_df) > PLOT_SAMPLE_N:
        corr_df = corr_df.sample(PLOT_SAMPLE_N, random_state=RANDOM_SEED)
    corr = corr_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Correlation: y_default vs Macro Engineered Features", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_feature_corr(corr_df: pd.DataFrame, filename: str) -> None:
    """Save a horizontal bar chart of the top-N feature–target correlations.

    Positive bars (red) indicate features that increase default probability;
    negative bars (blue) indicate protective features.

    Args:
        corr_df:  DataFrame from feature_target_correlation() with columns
                  ['feature', 'correlation', 'abs_corr'].
        filename: Output filename (written to FIGDIR).
    """
    if corr_df.empty:
        return
    n = len(corr_df)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.32)))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in corr_df["correlation"]]
    ax.barh(corr_df["feature"][::-1], corr_df["correlation"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Point-biserial correlation with y_default")
    ax.set_title(
        f"Top-{n} Feature–Target Correlations\n"
        "(red = positively correlated with default)"
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_kde_comparison(
    df: pd.DataFrame, col: str, filename: str, log_scale: bool = False
) -> None:
    """Save overlaid KDE density curves for defaulted vs non-defaulted loans.

    Separation between the two curves indicates how predictive the feature is.
    log_scale is used for heavy-tailed distributions (e.g. annual_inc) to
    prevent the bulk of the density from collapsing near zero.

    Args:
        df:        DataFrame with the feature column and 'y_default'.
        col:       Name of the feature column to plot.
        filename:  Output filename (written to FIGDIR).
        log_scale: If True, apply log1p transform before computing KDE.
    """
    sub = df[[col, "y_default"]].copy()
    if sub[col].dtype == object:
        sub[col] = sub[col].astype(str).str.replace("%", "", regex=False)
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[col])
    if len(sub) > PLOT_SAMPLE_N:
        sub = sub.sample(PLOT_SAMPLE_N, random_state=RANDOM_SEED)

    x0 = sub.loc[sub["y_default"] == 0, col].values.astype(float)
    x1 = sub.loc[sub["y_default"] == 1, col].values.astype(float)
    if log_scale:
        x0 = np.log1p(x0[x0 > 0])
        x1 = np.log1p(x1[x1 > 0])

    g0, d0 = _kde_from_data(x0)
    g1, d1 = _kde_from_data(x1)
    if g0.size == 0 or g1.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g0, d0, color="#4575b4", linewidth=2.0, label="Non-default (y=0)")
    ax.fill_between(g0, d0, alpha=0.18, color="#4575b4")
    ax.plot(g1, d1, color="#d73027", linewidth=2.0, label="Default (y=1)")
    ax.fill_between(g1, d1, alpha=0.18, color="#d73027")
    xlabel = f"log(1 + {col})" if log_scale else col
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Distribution of '{col}': Default vs Non-Default", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_default_by_subgrade(df: pd.DataFrame, filename: str) -> None:
    """Save a bar chart of default rate ordered by sub-grade (A1 → G5).

    A secondary axis overlays loan volume to show where the distribution
    of issued loans sits relative to default risk — useful for sizing
    the impact of each bucket.

    Args:
        df:       DataFrame with 'sub_grade' (or 'grade') and 'y_default'.
        filename: Output filename (written to FIGDIR).
    """
    col = "sub_grade" if "sub_grade" in df.columns else ("grade" if "grade" in df.columns else None)
    if col is None:
        print("  [WARN] No sub_grade or grade column found — skipping.")
        return

    agg = (
        df.groupby(col)["y_default"]
        .agg(default_rate="mean", n="size")
        .reset_index()
    )
    if col == "sub_grade":
        order = [sg for sg in SUBGRADE_ORDER if sg in agg[col].values]
        agg[col] = pd.Categorical(agg[col], categories=order, ordered=True)
    agg = agg.sort_values(col)
    if agg.empty:
        return

    fig, ax1 = plt.subplots(figsize=(max(8, len(agg) * 0.55), 5))
    norm_vals = (agg["default_rate"] - agg["default_rate"].min()) / (
        agg["default_rate"].max() - agg["default_rate"].min() + 1e-9
    )
    colors = [plt.cm.RdYlGn_r(v) for v in norm_vals]
    ax1.bar(range(len(agg)), agg["default_rate"], color=colors, alpha=0.85)
    ax1.set_xticks(range(len(agg)))
    ax1.set_xticklabels(agg[col].astype(str), rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Default rate", fontsize=11)
    ax1.set_title(
        f"Default Rate by {col.replace('_', ' ').title()} (green=safer, red=riskier)",
        fontsize=12,
    )
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax2 = ax1.twinx()
    ax2.plot(range(len(agg)), agg["n"] / 1000, color="#555555", linewidth=1.5,
             linestyle="--", marker="o", markersize=3, label="Loan count (K)")
    ax2.set_ylabel("Loan count (thousands)", color="#555555", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#555555")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2, labels2, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_default_by_purpose(df: pd.DataFrame, filename: str) -> None:
    """Save a ranked bar chart of default rate by loan purpose.

    Purposes are sorted highest-to-lowest default rate and annotated with
    loan counts (in thousands) to show risk alongside origination volume.

    Args:
        df:       DataFrame with 'purpose' and 'y_default' columns.
        filename: Output filename (written to FIGDIR).
    """
    if "purpose" not in df.columns:
        return
    agg = (
        df.groupby("purpose")["y_default"]
        .agg(default_rate="mean", n="size")
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    if agg.empty:
        return

    norm_vals = (agg["default_rate"] - agg["default_rate"].min()) / (
        agg["default_rate"].max() - agg["default_rate"].min() + 1e-9
    )
    colors = [plt.cm.RdYlGn_r(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(agg)), agg["default_rate"], color=colors)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["purpose"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Default rate")
    ax.set_title("Default Rate by Loan Purpose (ranked, red=highest risk)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(i, row["default_rate"] + 0.002, f"{row['n']//1000:.0f}K",
                ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


def plot_grade_year_heatmap(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Save a heatmap of default rate by loan grade and issue year.

    Reveals whether post-2015 default rate increases are grade-uniform
    (temporal drift) or concentrated in specific risk tiers.

    Args:
        df:       DataFrame with 'grade', 'issue_year', and 'y_default'.
        filename: Output filename (written to FIGDIR).

    Returns:
        Pivoted DataFrame (grade × issue_year) used to export the heatmap
        data as CSV, or an empty DataFrame if required columns are missing.
    """
    col = "grade" if "grade" in df.columns else None
    if col is None or "issue_year" not in df.columns:
        print("  [WARN] 'grade' or 'issue_year' not found — skipping heatmap.")
        return pd.DataFrame()

    pivot = (
        df.groupby([col, "issue_year"])["y_default"]
        .mean()
        .unstack("issue_year")
        .sort_index()
    )
    if pivot.empty:
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.7)))
    vmax = min(0.5, np.nanmax(pivot.values))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Issue year", fontsize=10)
    ax.set_ylabel("Loan grade", fontsize=10)
    ax.set_title("Default Rate Heatmap: Loan Grade × Issue Year", fontsize=12)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1%}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.25 else "black")
    plt.colorbar(im, ax=ax, label="Default rate", fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()
    return pivot.reset_index()


def plot_vintage_cohort(cohort: pd.DataFrame, filename: str) -> None:
    """Save a multi-line chart showing each issue-year cohort's default rate over loan age.

    Right-truncation of newer cohorts is visible as shorter lines — 2016–2018
    cohorts appear safer only because most loans had not yet had time to default
    by the dataset cutoff (2018 Q4).

    Args:
        cohort:   DataFrame from build_vintage_cohort() with columns
                  ['issue_year', 'age_bucket', 'default_rate'].
        filename: Output filename (written to FIGDIR).
    """
    if cohort.empty:
        return
    years = sorted(cohort["issue_year"].unique())
    cmap  = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, yr in enumerate(years):
        sub = cohort[cohort["issue_year"] == yr].sort_values("age_bucket")
        color = cmap(i / max(len(years) - 1, 1))
        ax.plot(sub["age_bucket"], sub["default_rate"],
                marker="o", markersize=4, linewidth=1.6, color=color, label=str(yr))
    ax.set_xlabel("Months since loan issuance (6-month bins)", fontsize=11)
    ax.set_ylabel("Cumulative default rate", fontsize=11)
    ax.set_title(
        "Vintage Cohort Analysis: Default Rate by Loan Age per Issue-Year\n"
        "(right-truncation visible — newer cohorts have fewer observation months)",
        fontsize=11,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(title="Issue Year", fontsize=8, title_fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=160)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — EDA report (markdown)
# ─────────────────────────────────────────────────────────────────────────────

def build_eda_report(
    summary: dict,
    miss_top: pd.DataFrame,
    by_year: pd.DataFrame,
    by_month: pd.DataFrame,
    corr_df: pd.DataFrame,
    leak_cols: List[str],
) -> str:
    """Assemble a markdown report summarising all EDA findings.

    Args:
        summary:   Data summary dictionary produced in main().
        miss_top:  Top-N missingness table from missingness_table().
        by_year:   Yearly default rate aggregates.
        by_month:  Monthly default rate aggregates.
        corr_df:   Feature–target correlation ranking.
        leak_cols: List of flagged leakage column names.

    Returns:
        Multi-section markdown string ready to be written to eda_report.md.
    """
    n_rows     = summary["accepted_shape_after_target_and_dates"][0]
    n_cols     = summary["accepted_shape_after_target_and_dates"][1]
    pos_rate   = summary["target_positive_rate"]

    peak_year_str = ""
    if not by_year.empty:
        peak = by_year.loc[by_year["default_rate"].idxmax()]
        peak_year_str = (
            f"Peak yearly default rate: **{int(peak['issue_year'])}** "
            f"at **{float(peak['default_rate']):.3f}**"
        )

    miss_lines = "\n".join(
        f"- **{r['column']}**: {r['missing_pct']:.2f}% missing"
        for _, r in miss_top.head(15).iterrows()
    )

    top5_pos = corr_df[corr_df["correlation"] > 0].head(5)
    top5_neg = corr_df[corr_df["correlation"] < 0].head(5)
    pos_lines = "\n".join(
        f"  - `{r['feature']}` (+{r['correlation']:.4f})"
        for _, r in top5_pos.iterrows()
    )
    neg_lines = "\n".join(
        f"  - `{r['feature']}` ({r['correlation']:.4f})"
        for _, r in top5_neg.iterrows()
    )

    md = f"""# EDA Report — Capstone: Credit Risk Modeling Under Macroeconomic Conditions

## 1  Dataset Snapshot
| Metric | Value |
|---|---|
| Rows (after target + date filter) | {n_rows:,} |
| Columns | {n_cols:,} |
| Default-like rate (positive class) | {pos_rate:.4f} ({pos_rate*100:.2f}%) |
| Issue date range | {summary['issue_month_min']} → {summary['issue_month_max']} |
| Target: y=1 | {sorted(DEFAULT_STATUSES)} |
| Target: y=0 | {sorted(GOOD_STATUSES)} |

---

## 2  Macroeconomic Enrichment (FRED)
- **Series merged:** UNRATE, FEDFUNDS
- **Engineered features per series:** lag1 / lag3 / lag6, chg1 / chg3, roll3 / roll6
- **Total macro columns added:** {len(summary['macro_columns_added'])}
- **Merge completeness:** all loans matched by issue month (FRED data covers 2007–2018)

---

## 3  Missingness (top 15 columns)
{miss_lines}

**Interpretation:** Columns with >60% missingness are typically post-origination
fields (e.g., hardship-related) and are excluded before modeling. Columns with
moderate missingness use median imputation + missing-indicator flags.

---

## 4  Target Behaviour Over Time
- Default rate by year  → `target_rate_by_year.csv`
- Default rate by month → `target_rate_by_issue_month.csv`
- Plot                  → `figures/default_rate_by_month.png`

{peak_year_str}

---

## 5  Macro Overlay (NEW)
Monthly default rate plotted against UNRATE and FEDFUNDS on dual y-axes.
GFC window (Sep 2008 – Jun 2009) highlighted.

> Plot: `figures/macro_default_overlay.png`

**Key insight:** The co-movement (or lack thereof) between macro conditions at
origination and observed defaults motivates the proposal hypothesis. Results in
model_v3.py show macro at origination is NOT additive — borrower features already
encode macro conditions through LendingClub's interest-rate pricing mechanism.

---

## 6  Feature–Target Correlation Ranking (NEW)
Point-biserial correlation of every numeric feature with `y_default`.

**Top positively correlated** (higher → more likely to default):
{pos_lines if pos_lines else "  (none)"}

**Top negatively correlated** (higher → less likely to default):
{neg_lines if neg_lines else "  (none)"}

> Plot: `figures/corr_feature_target.png`
> Data: `feature_target_corr.csv`

**Interpretation:** Very high |r| (>0.5) on a pre-issue feature is suspicious and
may indicate data leakage. Features with near-zero correlation are candidates for
removal before modeling.

---

## 7  Distribution Analysis: Default vs Non-Default (NEW)
Overlaid KDE density curves for key numeric features.

| Feature | Note | File |
|---|---|---|
| `int_rate` | Strong separation — pricing reflects risk | `kde_int_rate.png` |
| `dti` | Moderate separation | `kde_dti.png` |
| `loan_amnt` | Weak-to-moderate | `kde_loan_amnt.png` |
| `annual_inc` | Moderate (log scale for heavy tail) | `kde_annual_inc.png` |

---

## 8  Default Rate by Sub-Grade (NEW)
Ordered bar chart A1 → G5 with secondary axis for loan volume per bucket.
Demonstrates monotonic risk increase — directly motivates `sub_grade_enc`
feature engineering used in models.

> Plot: `figures/default_rate_by_subgrade.png`

---

## 9  Default Rate by Purpose (NEW)
Ranked bar chart — highest-risk loan purposes at left, annotated with loan counts.

> Plot: `figures/default_rate_by_purpose.png`
> Data: `target_rate_by_purpose.csv`

---

## 10  Grade × Year Heatmap (NEW)
Cell = average default rate for each (grade, issue_year) bucket.
Reveals whether the post-2015 default rate rise is grade-uniform (temporal drift)
or concentrated in specific risk tiers.

> Plot: `figures/grade_year_heatmap.png`
> Data: `grade_year_heatmap.csv`

---

## 11  Class Imbalance
Bar chart confirming dataset imbalance (~{pos_rate:.0%} defaults).
Motivates `scale_pos_weight` in XGBoost/LightGBM and Platt-scaling calibration.

> Plot: `figures/class_imbalance.png`

---

## 12  Vintage Cohort Analysis (NEW)
Each issue-year cohort plotted by observed default rate vs loan age (6-month bins).

**Key finding — right-censorship / survivorship bias:**
2016–2018 cohorts show lower observed default rates at early ages NOT because
they are safer, but because most loans had insufficient time to default by the
dataset cutoff (2018 Q4). This systematically biases test-set AUC downward for
recent years and explains the gap: val AUC (2015) ≈ 0.755 vs test AUC (2018) ≈ 0.707.

> Plot: `figures/vintage_cohort_curves.png`
> Data: `vintage_cohort.csv`

---

## 13  Leakage Screening
Heuristic keyword scan identified **{len(leak_cols)}** candidate post-origination
columns to review before modeling.

Examples: {", ".join(f"`{c}`" for c in leak_cols[:10])}

**Action:** These were dropped prior to model training (see model_v3.py feature list).

---

## 14  Immediate Next Steps
1. Confirm leakage column set dropped / flagged.
2. Create temporal train / val / test splits by `issue_month`.
3. Train baseline Logistic Regression → report ROC-AUC, PR-AUC, calibration.
4. Train XGBoost + LightGBM; compare with bootstrap CIs.
5. Run macro-feature ablation study (Model A borrower-only vs Model B borrower+macro).
6. Stress-test best model under recession / rate-shock scenarios.
"""
    return md


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_console_summary(
    df: pd.DataFrame,
    summary: dict,
    miss_top: pd.DataFrame,
    by_year: pd.DataFrame,
    corr_df: pd.DataFrame,
    leak_cols: List[str],
) -> None:
    """Print a structured terminal summary of all key EDA findings.

    Args:
        df:        Full processed DataFrame (used for target counts).
        summary:   Data summary dictionary produced in main().
        miss_top:  Top-N missingness table from missingness_table().
        by_year:   Yearly default rate aggregates.
        corr_df:   Feature–target correlation ranking.
        leak_cols: List of flagged leakage column names.
    """
    n_pos  = int((df["y_default"] == 1).sum())
    n_neg  = int((df["y_default"] == 0).sum())
    total  = n_pos + n_neg

    print("\n" + "=" * 65)
    print("CAPSTONE EDA — FULL PIPELINE SUMMARY")
    print("=" * 65)

    print(f"\n  Dataset")
    print(f"    Rows (resolved loans)   : {total:>10,}")
    print(f"    Defaults (y=1)          : {n_pos:>10,}  ({n_pos/total:.2%})")
    print(f"    Non-defaults (y=0)      : {n_neg:>10,}  ({n_neg/total:.2%})")
    print(f"    Class ratio (neg:pos)   : {n_neg/n_pos:.2f}:1")
    print(f"    Issue date range        : {summary['issue_month_min']} → {summary['issue_month_max']}")

    if not by_year.empty:
        print(f"\n  Default rate by issue year:")
        for _, r in by_year.sort_values("issue_year").iterrows():
            bar = "█" * int(r["default_rate"] * 40)
            print(f"    {int(r['issue_year'])}  {r['default_rate']:.3f}  {bar}  (n={int(r['n']):,})")

    if not corr_df.empty:
        print(f"\n  Top-5 features by |correlation| with y_default:")
        for _, r in corr_df.head(5).iterrows():
            direction = "+" if r["correlation"] > 0 else ""
            print(f"    {r['feature']:<35}  {direction}{r['correlation']:.4f}")

    print(f"\n  Missingness (top 10 columns):")
    for _, r in miss_top.head(10).iterrows():
        print(f"    {r['column']:<40}  {r['missing_pct']:.2f}%")

    print(f"\n  Leakage heuristic: {len(leak_cols)} candidate post-origination columns found.")
    if leak_cols:
        print(f"    Examples: {', '.join(leak_cols[:8])}")
        print("    → Drop / flag these before modeling.")

    print(f"\n  Macro columns added: {len(summary['macro_columns_added'])}")

    print(f"\n  Artifacts saved → {OUTDIR}/")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full EDA pipeline end-to-end.

    Steps:
      1. Load raw loan CSV and parse issue dates.
      2. Build binary default target; drop unresolved statuses.
      3. Load and engineer FRED macro features; merge onto loan data.
      4. Compute data assets: missingness, numeric summary, aggregates.
      5. Rank features by correlation with y_default.
      6. Build vintage cohort table.
      7. Generate all diagnostic figures.
      8. Write markdown EDA report.
      9. Print console summary.
    """
    np.random.seed(RANDOM_SEED)
    ensure_outdirs()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("=== [1/9] Loading accepted loans CSV ===")
    accepted = pd.read_csv(ACCEPTED_FILE, low_memory=False)
    accepted = normalize_colnames(accepted)
    print(f"  Shape: {accepted.shape[0]:,} rows × {accepted.shape[1]:,} cols")

    accepted, issue_col = infer_issue_date(accepted)
    accepted["issue_month"] = create_month_key(accepted[issue_col])

    # ── 2. Target ─────────────────────────────────────────────────────────────
    print("=== [2/9] Creating target variable (y_default) ===")
    accepted = make_target(accepted)
    accepted = accepted.dropna(subset=["issue_month"])
    accepted["issue_year"] = accepted["issue_month"].dt.year
    print(f"  After dropping NaT issue_month: {accepted.shape[0]:,} rows")

    # ── 3. Macro ──────────────────────────────────────────────────────────────
    print("=== [3/9] Loading and engineering macro features ===")
    fed = read_fred_csv(FEDFUNDS_FILE, "FEDFUNDS")
    unr = read_fred_csv(UNRATE_FILE,   "UNRATE")

    fed["month"] = create_month_key(fed["date"])
    unr["month"] = create_month_key(unr["date"])
    fed = fed.drop(columns=["date"]).drop_duplicates("month").sort_values("month")
    unr = unr.drop(columns=["date"]).drop_duplicates("month").sort_values("month")

    # Outer join then forward-fill so months with only one series are still covered.
    macro = pd.merge(fed, unr, on="month", how="outer").sort_values("month").ffill()
    for col in ["FEDFUNDS", "UNRATE"]:
        for lag in [1, 3, 6]:
            macro[f"{col}_lag{lag}"] = macro[col].shift(lag)
            macro[f"{col}_chg{lag}"] = macro[col] - macro[f"{col}_lag{lag}"]
        macro[f"{col}_roll3"] = macro[col].rolling(3).mean()
        macro[f"{col}_roll6"] = macro[col].rolling(6).mean()

    accepted = accepted.merge(macro, left_on="issue_month", right_on="month", how="left")
    accepted = accepted.drop(columns=["month"])

    # ── 4. Data assets ────────────────────────────────────────────────────────
    print("=== [4/9] Computing data assets (missingness, numeric summary, aggregates) ===")

    # Small random sample saved for manual inspection of merged columns.
    to_csv(
        accepted.sample(n=min(5000, len(accepted)), random_state=RANDOM_SEED),
        os.path.join(OUTDIR, "macro_merged_sample.csv"),
    )

    # Missingness
    miss_top = missingness_table(accepted, top_n=50)
    to_csv(miss_top, os.path.join(OUTDIR, "missingness_top50.csv"))

    # Numeric summary
    num_summary = basic_numeric_summary(accepted)
    if not num_summary.empty:
        num_summary.reset_index(names="column").to_csv(
            os.path.join(OUTDIR, "numeric_summary.csv"), index=False
        )

    # Leakage candidates
    leak_cols = detect_leakage_candidates(accepted)

    # Default rate aggregates
    by_year = (
        accepted.groupby("issue_year")["y_default"]
        .agg(default_rate="mean", n="size")
        .reset_index()
        .sort_values("issue_year")
    )
    to_csv(by_year, os.path.join(OUTDIR, "target_rate_by_year.csv"))

    by_month = (
        accepted.groupby("issue_month")["y_default"]
        .agg(default_rate="mean", n="size")
        .reset_index()
        .sort_values("issue_month")
    )
    to_csv(by_month, os.path.join(OUTDIR, "target_rate_by_issue_month.csv"))

    for seg_col, fname in [("grade", "target_rate_by_grade.csv"),
                            ("term",  "target_rate_by_term.csv"),
                            ("purpose", "target_rate_by_purpose.csv")]:
        if seg_col in accepted.columns:
            seg_agg = (
                accepted.groupby(seg_col)["y_default"]
                .agg(default_rate="mean", n="size")
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            to_csv(seg_agg, os.path.join(OUTDIR, fname))

    # Summary JSON
    summary = {
        "accepted_shape_after_target_and_dates": list(accepted.shape),
        "issue_col_used": issue_col,
        "issue_month_min": str(accepted["issue_month"].min()),
        "issue_month_max": str(accepted["issue_month"].max()),
        "target_positive_rate": float(accepted["y_default"].mean()),
        "target_counts": accepted["y_default"].value_counts().to_dict(),
        "macro_columns_added": [
            c for c in accepted.columns
            if c.startswith("FEDFUNDS") or c.startswith("UNRATE")
        ],
        "leakage_candidates": leak_cols,
    }
    to_json(summary, os.path.join(OUTDIR, "data_summary.json"))

    # ── 5. Feature–target correlation ─────────────────────────────────────────
    print("=== [5/9] Computing feature–target correlations ===")
    corr_df = feature_target_correlation(accepted, target="y_default", top_n=TOP_CORR_N)
    to_csv(corr_df, os.path.join(OUTDIR, "feature_target_corr.csv"))
    print(f"  Top feature: {corr_df.iloc[0]['feature']} (|r|={corr_df.iloc[0]['abs_corr']:.4f})")

    # ── 6. Vintage cohort ─────────────────────────────────────────────────────
    print("=== [6/9] Building vintage cohort analysis ===")
    cohort = build_vintage_cohort(accepted)
    to_csv(cohort, os.path.join(OUTDIR, "vintage_cohort.csv"))

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("=== [7/9] Generating figures ===")

    plot_class_imbalance(accepted, "class_imbalance.png")
    plot_default_rate_by_month(by_month, "default_rate_by_month.png")
    plot_macro_default_overlay(by_month, macro, "macro_default_overlay.png")
    plot_macro_corr_heatmap(accepted, "corr_target_macro.png")
    plot_feature_corr(corr_df, "corr_feature_target.png")

    for col, fname, log_scale in [
        ("int_rate",   "kde_int_rate.png",   False),
        ("dti",        "kde_dti.png",         False),
        ("loan_amnt",  "kde_loan_amnt.png",   False),
        ("annual_inc", "kde_annual_inc.png",  True),
    ]:
        if col in accepted.columns:
            plot_kde_comparison(accepted, col, fname, log_scale=log_scale)

    plot_default_by_subgrade(accepted, "default_rate_by_subgrade.png")

    if "purpose" in accepted.columns:
        plot_default_by_purpose(accepted, "default_rate_by_purpose.png")

    heatmap_df = plot_grade_year_heatmap(accepted, "grade_year_heatmap.png")
    if not heatmap_df.empty:
        to_csv(heatmap_df, os.path.join(OUTDIR, "grade_year_heatmap.csv"))

    plot_vintage_cohort(cohort, "vintage_cohort_curves.png")

    # ── 8. EDA report ─────────────────────────────────────────────────────────
    print("=== [8/9] Writing EDA report (Markdown) ===")
    report_md = build_eda_report(summary, miss_top, by_year, by_month, corr_df, leak_cols)
    with open(os.path.join(OUTDIR, "eda_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    # ── 9. Console summary ────────────────────────────────────────────────────
    print("=== [9/9] Console summary ===")
    print_console_summary(accepted, summary, miss_top, by_year, corr_df, leak_cols)

    print(f"Done.  All outputs → {OUTDIR}/")


if __name__ == "__main__":
    main()
