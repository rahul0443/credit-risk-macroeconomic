#!/usr/bin/env python3
"""
File 5 of 6 — Full Pipeline Orchestrator

Runs the complete capstone pipeline end-to-end in a single command.
Files 2 and 3 (hyperparameter searches) are excluded — they are
one-time Colab runs whose best parameters are already embedded in
4_model_v3.py.

Pipeline order:
  1_exploratory_data_analysis.py   ← Stage 1 (EDA)
  2_model_hyperparameter_search.py ← Colab only — skipped here
  3_model_lgbm_search.py           ← Colab only — skipped here
  4_model_v3.py                    ← Stage 2 (Modeling)
  5_run_pipeline.py                ← THIS FILE
  6_dashboard.py                   ← Stage 3 (Dashboards) — run by this file

Usage:
  python 5_run_pipeline.py                    # run all stages
  python 5_run_pipeline.py --skip-eda        # skip EDA, run modeling + dashboards
  python 5_run_pipeline.py --skip-model      # run EDA only (skips model + dashboards)
  python 5_run_pipeline.py --skip-dashboard  # run EDA + model, skip dashboards

Outputs:
  outputs/eda/          ← EDA artifacts (figures, CSVs, report)
  outputs/model/        ← Model artifacts (metrics, figures, JSON)
  outputs/dashboards/   ← Dashboard PNGs (requires stages 1 & 2 first)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_run(filename: str) -> None:
    """
    Dynamically load a .py file by path and call its main() function.

    Uses importlib rather than a direct import because filenames that start
    with a digit (e.g. "1_exploratory_data_analysis.py") are not valid Python
    module names and cannot be imported with the standard import statement.

    Args:
        filename: Bare filename (no directory) relative to BASE_DIR.

    Raises:
        SystemExit: If the file does not exist or has no main() function.
    """
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [ERROR] File not found: {filepath}")
        sys.exit(1)

    spec   = importlib.util.spec_from_file_location("_stage_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        print(f"  [ERROR] {filename} has no main() function.")
        sys.exit(1)

    module.main()


def divider(title: str) -> None:
    """
    Print a full-width banner to visually separate pipeline stages in the log.

    Args:
        title: Stage label to display inside the banner.
    """
    width = 65
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def fmt_elapsed(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable "Xm Ys" string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        String of the form "2m 34s".
    """
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full capstone pipeline across three stages.

    Steps:
      1.  Parse --skip-eda / --skip-model / --skip-dashboard flags.
      2.  Print pipeline header with course and project metadata.
      3.  Stage 1 — EDA: load and run 1_exploratory_data_analysis.py
          (skipped if --skip-eda; existing outputs/eda/ are reused).
      4.  Stage 2 — Modeling: load and run 4_model_v3.py
          (skipped if --skip-model).
      5.  Stage 3 — Dashboards: load and run 6_dashboard.py
          (skipped if --skip-dashboard OR --skip-model, because dashboards
          depend on model outputs from Stage 2).
      6.  Print final summary: total runtime and artifact tree.
    """
    parser = argparse.ArgumentParser(
        description="Capstone pipeline: EDA → Credit Risk Modeling"
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip Stage 1 (EDA). Use existing outputs/eda/ artifacts.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip Stage 2 (Modeling). Run EDA only.",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip Stage 3 (Dashboards). Run EDA and/or Modeling only.",
    )
    args = parser.parse_args()

    pipeline_start = time.time()

    print("\n" + "=" * 65)
    print("  CAPSTONE PIPELINE — Credit Risk Under Macro Conditions")
    print("  FSE 570 Data Science Capstone | Arizona State University")
    print("=" * 65)
    print("\n  Note: Files 2 & 3 (hyperparameter searches) are excluded.")
    print("  They were one-time Colab runs; tuned params live in 4_model_v3.py.")
    if args.skip_model:
        print("  Note: --skip-model also skips Stage 3 (Dashboards) since")
        print("        dashboards require model outputs from Stage 2.\n")
    else:
        print()

    # ── Stage 1: EDA ─────────────────────────────────────────────────────────
    if args.skip_eda:
        divider("Stage 1 — EDA  [SKIPPED via --skip-eda]")
        print("  Using existing artifacts in outputs/eda/")
        print("  To regenerate: python 1_exploratory_data_analysis.py")
    else:
        divider("Stage 1 of 2 — Exploratory Data Analysis")
        print("  Running: 1_exploratory_data_analysis.py")
        print("  Estimated time: 5–15 minutes\n")
        t0 = time.time()
        load_and_run("1_exploratory_data_analysis.py")
        elapsed = time.time() - t0
        print(f"\n  Stage 1 complete in {fmt_elapsed(elapsed)}.")
        print("  Outputs → outputs/eda/")

    # ── Stage 2: Modeling ─────────────────────────────────────────────────────
    if args.skip_model:
        divider("Stage 2 — Modeling  [SKIPPED via --skip-model]")
        print("  To run modeling: python 4_model_v3.py")
    else:
        divider("Stage 2 of 3 — Credit Risk Modeling (v3)")
        print("  Running: 4_model_v3.py")
        print("  Estimated time: 20–40 minutes\n")
        t0 = time.time()
        load_and_run("4_model_v3.py")
        elapsed = time.time() - t0
        print(f"\n  Stage 2 complete in {fmt_elapsed(elapsed)}.")
        print("  Outputs → outputs/model/")

    # ── Stage 3: Dashboards ───────────────────────────────────────────────────
    # Skip dashboards if --skip-dashboard was set, OR if --skip-model was set
    # (dashboards depend on both EDA and model outputs)
    skip_dash = args.skip_dashboard or args.skip_model
    if skip_dash:
        if args.skip_model:
            divider("Stage 3 — Dashboards  [SKIPPED — model outputs unavailable]")
            print("  Run without --skip-model to generate dashboards.")
        else:
            divider("Stage 3 — Dashboards  [SKIPPED via --skip-dashboard]")
            print("  To generate dashboards: python 6_dashboard.py")
    else:
        divider("Stage 3 of 3 — Dashboard Generator")
        print("  Running: 6_dashboard.py")
        print("  Estimated time: < 1 minute\n")
        t0 = time.time()
        load_and_run("6_dashboard.py")
        elapsed = time.time() - t0
        print(f"\n  Stage 3 complete in {fmt_elapsed(elapsed)}.")
        print("  Outputs → outputs/dashboards/")

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    divider("Pipeline Complete")
    print(f"  Total runtime : {fmt_elapsed(total)}")
    print()
    print("  Artifacts produced:")
    if not args.skip_eda:
        print("    outputs/eda/")
        print("      ├── eda_report.md")
        print("      ├── feature_target_corr.csv")
        print("      ├── missingness_top50.csv")
        print("      ├── numeric_summary.csv")
        print("      ├── vintage_cohort.csv")
        print("      ├── grade_year_heatmap.csv")
        print("      ├── target_rate_by_*.csv")
        print("      └── figures/  (13 plots)")
    if not args.skip_model:
        print("    outputs/model/")
        print("      ├── metrics_v3.json")
        print("      └── figures/")
        print("          ├── gap1_roc_macro_comparison.png")
        print("          ├── gap2_regime_auc_by_year.png")
        print("          └── gap3_stress_simulation.png")
    if not skip_dash:
        print("    outputs/dashboards/")
        print("      ├── dashboard_1_overview.png")
        print("      ├── dashboard_2_features.png")
        print("      ├── dashboard_3_segmentation.png")
        print("      └── dashboard_4_model.png")
    print()
    print("  Full pipeline file order (for reference):")
    print("    1_exploratory_data_analysis.py   — local")
    print("    2_model_hyperparameter_search.py — Colab (one-time, done)")
    print("    3_model_lgbm_search.py           — Colab (one-time, done)")
    print("    4_model_v3.py                    — local")
    print("    5_run_pipeline.py                — this file")
    print("    6_dashboard.py                   — run by this file (Stage 3)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
