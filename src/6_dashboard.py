#!/usr/bin/env python3
"""
File 6 of 6 — Dashboard Generator (4 separate PNGs)

Produces 4 focused dashboard images:
  dashboard_1_overview.png     — Dataset & Macro
  dashboard_2_features.png     — Feature Analysis
  dashboard_3_segmentation.png — Credit Segmentation
  dashboard_4_model.png        — Model Results

Pipeline order:
  1_exploratory_data_analysis.py   ← must be run first
  4_model_v3.py                    ← must be run first
  6_dashboard.py                   ← THIS FILE (run after 1 and 4)

Usage:
  python 6_dashboard.py

Requirements:
  - outputs/eda/figures/   must exist (run file 1 first)
  - outputs/model/figures/ must exist (run file 4 first)
"""

from __future__ import annotations

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EDA_FIGS    = os.path.join(BASE_DIR, "..", "outputs", "eda",        "figures")
MODEL_FIGS  = os.path.join(BASE_DIR, "..", "outputs", "model",      "figures")
DASH_DIR    = os.path.join(BASE_DIR, "..", "outputs", "dashboards")

HEADER_COLOR  = "#1a1a2e"
SECTION_COLOR = "#16213e"
BG_COLOR      = "#f8f9fa"
DPI           = 150


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_img(folder: str, filename: str):
    """
    Load a PNG from disk and return it as a matplotlib image array.

    Returns None (instead of raising) when the file is missing so that
    dashboards can still render with placeholder tiles rather than crashing.

    Args:
        folder:   Directory containing the image file.
        filename: Bare filename (e.g. "class_imbalance.png").

    Returns:
        NumPy image array, or None if the file does not exist.
    """
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {filename}")
        return None
    return mpimg.imread(path)


def show_img(ax, img, title: str = "", caption: str = "") -> None:
    """
    Render an image (or a grey placeholder) onto a matplotlib Axes.

    Args:
        ax:      Axes object to draw into.
        img:     Image array from load_img(), or None for a placeholder tile.
        title:   Panel title displayed above the image.
        caption: Interpretive note displayed below the image in italic grey.
    """
    if img is None:
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, f"Missing\n{title}", ha="center", va="center",
                fontsize=7, color="#999999", transform=ax.transAxes)
    else:
        ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3, color="#222222")
    if caption:
        ax.text(0.5, -0.03, caption, ha="center", va="top", fontsize=6.5,
                color="#666666", style="italic", transform=ax.transAxes)


def add_header(fig, title: str, subtitle: str = "") -> None:
    """Dark header band at the top of each dashboard."""
    fig.text(0.5, 0.987, title, ha="center", va="top",
             fontsize=13, fontweight="bold", color="white",
             bbox=dict(boxstyle="square,pad=0.4", facecolor=HEADER_COLOR,
                       edgecolor="none"))
    if subtitle:
        fig.text(0.5, 0.968, subtitle, ha="center", va="top",
                 fontsize=8.5, color="#aaaaaa")


def save(fig, filename: str) -> None:
    """
    Save a figure to DASH_DIR and close it to free memory.

    Args:
        fig:      matplotlib Figure to save.
        filename: Output filename (e.g. "dashboard_1_overview.png").
    """
    os.makedirs(DASH_DIR, exist_ok=True)
    path = os.path.join(DASH_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved → outputs/dashboards/{filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 1 — Dataset & Macro Overview
# ─────────────────────────────────────────────────────────────────────────────
#
#  ┌──────────────────────────────────────┐
#  │  class_imbalance │ default_by_month  │
#  │       macro_default_overlay          │
#  └──────────────────────────────────────┘

def build_dashboard_1() -> None:
    """
    Build Dashboard 1: Dataset overview and macro context.

    Layout (2 rows × 4 cols):
      Row 0 — class imbalance bar chart (col 0) + default rate by month (cols 1-3)
      Row 1 — macro overlay (UNRATE & FEDFUNDS vs default rate, full width)

    Saves: dashboard_1_overview.png
    """
    fig = plt.figure(figsize=(18, 14), facecolor=BG_COLOR)
    add_header(fig,
               "Dashboard 1 — Dataset & Macro Overview",
               "Credit Risk Modeling Under Macroeconomic Conditions  |  FSE 570 Capstone")

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        left=0.02, right=0.98, top=0.94, bottom=0.03,
        hspace=0.32, wspace=0.06,
        height_ratios=[1, 1.1],
    )

    # Row 0: class imbalance (1 col) + default rate by month (3 cols)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    show_img(ax1, load_img(EDA_FIGS, "class_imbalance.png"),
             title="Class Imbalance",
             caption="~27% default rate — motivates\nscale_pos_weight & Platt calibration")
    show_img(ax2, load_img(EDA_FIGS, "default_rate_by_month.png"),
             title="Default Rate by Issue Month",
             caption="Rising post-2015 default rate reflects temporal distribution shift "
                     "and survivorship bias in recent cohorts")

    # Row 1: macro overlay (full width)
    ax3 = fig.add_subplot(gs[1, :])
    show_img(ax3, load_img(EDA_FIGS, "macro_default_overlay.png"),
             title="Monthly Default Rate vs. UNRATE & FEDFUNDS (Dual Axis)",
             caption="GFC window (Sep 2008 – Jun 2009) shaded  |  "
                     "Weak co-movement motivates macro ablation study in model_v3.py")

    save(fig, "dashboard_1_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 2 — Feature Analysis
# ─────────────────────────────────────────────────────────────────────────────
#
#  ┌──────────────────────────────────────────┐
#  │ corr_target_macro │ corr_feature_target  │
#  │  kde_int  │ kde_dti │ kde_loan │ kde_inc │
#  └──────────────────────────────────────────┘

def build_dashboard_2() -> None:
    """
    Build Dashboard 2: Feature correlations and distribution comparisons.

    Layout (2 rows × 4 cols):
      Row 0 — macro-target correlation heatmap (cols 0-1) +
               top-30 feature-target correlations (cols 2-3)
      Row 1 — KDE plots for int_rate, dti, loan_amnt, annual_inc (one per col)

    KDE plots compare the distribution of each feature for defaulted vs
    non-defaulted loans to show separability.

    Saves: dashboard_2_features.png
    """
    fig = plt.figure(figsize=(18, 16), facecolor=BG_COLOR)
    add_header(fig,
               "Dashboard 2 — Feature Analysis",
               "Correlation Rankings & Distribution Comparisons (Default vs Non-Default)")

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        left=0.02, right=0.98, top=0.94, bottom=0.03,
        hspace=0.32, wspace=0.06,
        height_ratios=[1.2, 1],
    )

    # Row 0: macro corr heatmap (left 2) + feature-target corr (right 2)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    show_img(ax1, load_img(EDA_FIGS, "corr_target_macro.png"),
             title="Correlation: y_default vs Engineered Macro Features",
             caption="Weak macro-target correlations foreshadow GAP 1 finding: "
                     "macro adds no AUC lift")
    show_img(ax2, load_img(EDA_FIGS, "corr_feature_target.png"),
             title="Top-30 Feature–Target Correlations (Point-Biserial)",
             caption="Red = positively correlated with default  |  "
                     "Blue = negatively correlated  |  |r| > 0.5 flagged for leakage review")

    # Row 1: 4x KDE plots
    kde_plots = [
        ("kde_int_rate.png",   "Interest Rate",        "Strong separation"),
        ("kde_dti.png",        "Debt-to-Income Ratio", "Moderate separation"),
        ("kde_loan_amnt.png",  "Loan Amount",          "Weak-to-moderate separation"),
        ("kde_annual_inc.png", "Annual Income (log)",  "Moderate separation"),
    ]
    for col, (fname, title, caption) in enumerate(kde_plots):
        ax = fig.add_subplot(gs[1, col])
        show_img(ax, load_img(EDA_FIGS, fname), title=title, caption=caption)

    save(fig, "dashboard_2_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 3 — Credit Segmentation
# ─────────────────────────────────────────────────────────────────────────────
#
#  ┌──────────────────────────────────────────┐
#  │        default_by_subgrade (full)        │
#  │ default_by_purpose │ grade_year_heatmap  │
#  │        vintage_cohort_curves (full)      │
#  └──────────────────────────────────────────┘

def build_dashboard_3() -> None:
    """
    Build Dashboard 3: Credit segmentation and temporal analysis.

    Layout (3 rows × 2 cols):
      Row 0 — default rate by sub-grade A1→G5 (full width)
      Row 1 — default rate by loan purpose (left) + grade×year heatmap (right)
      Row 2 — vintage cohort curves (full width)

    The vintage cohort plot is the key artifact explaining the val→test AUC
    drop (0.755 → 0.707): recent cohorts are right-truncated and appear safer
    only because most loans had insufficient time to default by dataset cutoff.

    Saves: dashboard_3_segmentation.png
    """
    fig = plt.figure(figsize=(18, 20), facecolor=BG_COLOR)
    add_header(fig,
               "Dashboard 3 — Credit Segmentation & Temporal Analysis",
               "Sub-Grade Risk Profile  |  Loan Purpose  |  Grade × Year  |  Vintage Cohorts")

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.02, right=0.98, top=0.96, bottom=0.02,
        hspace=0.28, wspace=0.06,
        height_ratios=[1, 1.1, 1.2],
    )

    # Row 0: sub-grade (full width)
    ax1 = fig.add_subplot(gs[0, :])
    show_img(ax1, load_img(EDA_FIGS, "default_rate_by_subgrade.png"),
             title="Default Rate by Sub-Grade (A1 → G5)",
             caption="Monotonic risk increase across sub-grades — "
                     "directly motivates sub_grade_enc ordinal feature engineering")

    # Row 1: purpose (left) + grade heatmap (right)
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    show_img(ax2, load_img(EDA_FIGS, "default_rate_by_purpose.png"),
             title="Default Rate by Loan Purpose (Ranked)",
             caption="Small business & renewable energy highest risk categories")
    show_img(ax3, load_img(EDA_FIGS, "grade_year_heatmap.png"),
             title="Default Rate Heatmap: Grade × Issue Year",
             caption="Post-2015 default rate rise is grade-uniform → confirms temporal drift")

    # Row 2: vintage cohort (full width)
    ax4 = fig.add_subplot(gs[2, :])
    show_img(ax4, load_img(EDA_FIGS, "vintage_cohort_curves.png"),
             title="Vintage Cohort Analysis: Default Rate by Loan Age per Issue-Year",
             caption="Right-truncation visible — 2016–2018 cohorts appear safer only because "
                     "most loans had insufficient time to default by dataset cutoff (2018 Q4)  |  "
                     "Explains test AUC decline: val 0.755 → test 0.707")

    save(fig, "dashboard_3_segmentation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 4 — Model Results
# ─────────────────────────────────────────────────────────────────────────────
#
#  ┌──────────────────────────────────────────┐
#  │      gap1_roc_comparison (full)          │
#  │  gap2_regime_auc  │  gap3_stress_sim     │
#  └──────────────────────────────────────────┘

def build_dashboard_4() -> None:
    """
    Build Dashboard 4: Model evaluation across GAP 1, 2, and 3.

    Layout (2 rows × 2 cols):
      Row 0 — GAP 1 ROC curve: Model A (borrower-only) vs Model B (borrower+macro),
               full width; bootstrap 95% CI shown
      Row 1 — GAP 2 AUC by year/regime (left) +
               GAP 3 stress simulation bar chart (right)

    Saves: dashboard_4_model.png
    """
    fig = plt.figure(figsize=(18, 18), facecolor=BG_COLOR)
    add_header(fig,
               "Dashboard 4 — Model Results",
               "GAP 1: Macro Comparison  |  GAP 2: Regime Analysis  |  GAP 3: Stress Simulation")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.02, right=0.98, top=0.95, bottom=0.02,
        hspace=0.28, wspace=0.06,
        height_ratios=[1.1, 1],
    )

    # Row 0: GAP 1 ROC (full width)
    ax1 = fig.add_subplot(gs[0, :])
    show_img(ax1, load_img(MODEL_FIGS, "gap1_roc_macro_comparison.png"),
             title="GAP 1 — Borrower-only vs Borrower+Macro ROC Curve (Test Set)",
             caption="Macro features do NOT improve AUC — borrower attributes already encode "
                     "macro conditions via LendingClub's interest-rate pricing mechanism  |  "
                     "Bootstrap 95% CI confirms result is statistically significant")

    # Row 1: GAP 2 (left) + GAP 3 (right)
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    show_img(ax2, load_img(MODEL_FIGS, "gap2_regime_auc_by_year.png"),
             title="GAP 2 — AUC by Year / Economic Regime (2015–2018)",
             caption="AUC declines 0.755 → 0.707 over evaluation period — "
                     "temporal drift + survivorship bias in recent cohorts")
    show_img(ax3, load_img(MODEL_FIGS, "gap3_stress_simulation.png"),
             title="GAP 3 — Portfolio Stress Simulation (Calibrated Probabilities)",
             caption="Parallel macro feature shifts applied to test set under 4 scenarios  |  "
                     "Platt scaling corrects scale_pos_weight probability inflation")

    save(fig, "dashboard_4_model.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Generate all four dashboard PNGs in sequence.

    Steps:
      1.  Warn if either upstream figures directory is missing (dashboards
          will still render, but affected tiles will show grey placeholders).
      2.  Build and save Dashboard 1 — Dataset & Macro Overview.
      3.  Build and save Dashboard 2 — Feature Analysis.
      4.  Build and save Dashboard 3 — Credit Segmentation.
      5.  Build and save Dashboard 4 — Model Results (GAP 1/2/3).
      6.  Print final path summary.
    """
    print("=== Building 4 Dashboards ===\n")

    for folder, name in [(EDA_FIGS,   "outputs/eda/figures"),
                          (MODEL_FIGS, "outputs/model/figures")]:
        if not os.path.exists(folder):
            print(f"  [WARN] {name} not found — figures will show placeholders.")
            print(f"         Run the corresponding script first.\n")

    build_dashboard_1()
    build_dashboard_2()
    build_dashboard_3()
    build_dashboard_4()

    print("\nDone.  4 dashboards saved to:")
    print(f"  outputs/dashboards/")
    print("    dashboard_1_overview.png")
    print("    dashboard_2_features.png")
    print("    dashboard_3_segmentation.png")
    print("    dashboard_4_model.png")


if __name__ == "__main__":
    main()
