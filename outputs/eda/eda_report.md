# EDA Report — Capstone: Credit Risk Modeling Under Macroeconomic Conditions

## 1  Dataset Snapshot
| Metric | Value |
|---|---|
| Rows (after target + date filter) | 1,373,915 |
| Columns | 172 |
| Default-like rate (positive class) | 0.2148 (21.48%) |
| Issue date range | 2007-06-01 00:00:00 → 2018-12-01 00:00:00 |
| Target: y=1 | ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (16-30 days)', 'Late (31-120 days)'] |
| Target: y=0 | ['Does not meet the credit policy. Status:Fully Paid', 'Fully Paid'] |

---

## 2  Macroeconomic Enrichment (FRED)
- **Series merged:** UNRATE, FEDFUNDS
- **Engineered features per series:** lag1 / lag3 / lag6, chg1 / chg3, roll3 / roll6
- **Total macro columns added:** 18
- **Merge completeness:** all loans matched by issue month (FRED data covers 2007–2018)

---

## 3  Missingness (top 15 columns)
- **member_id**: 100.00% missing
- **orig_projected_additional_accrued_interest**: 99.60% missing
- **hardship_loan_status**: 99.44% missing
- **hardship_last_payment_amount**: 99.44% missing
- **hardship_end_date**: 99.44% missing
- **hardship_length**: 99.44% missing
- **hardship_dpd**: 99.44% missing
- **hardship_start_date**: 99.44% missing
- **payment_plan_start_date**: 99.44% missing
- **hardship_type**: 99.44% missing
- **hardship_reason**: 99.44% missing
- **hardship_status**: 99.44% missing
- **deferral_term**: 99.44% missing
- **hardship_amount**: 99.44% missing
- **hardship_payoff_balance_amount**: 99.44% missing

**Interpretation:** Columns with >60% missingness are typically post-origination
fields (e.g., hardship-related) and are excluded before modeling. Columns with
moderate missingness use median imputation + missing-indicator flags.

---

## 4  Target Behaviour Over Time
- Default rate by year  → `target_rate_by_year.csv`
- Default rate by month → `target_rate_by_issue_month.csv`
- Plot                  → `figures/default_rate_by_month.png`

Peak yearly default rate: **2017** at **0.272**

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
  - `recoveries` (+0.4761)
  - `collection_recovery_fee` (+0.4498)
  - `int_rate` (+0.2642)
  - `hardship_dpd` (+0.2295)
  - `out_prncp` (+0.2100)

**Top negatively correlated** (higher → less likely to default):
  - `last_fico_range_high` (-0.6650)
  - `last_fico_range_low` (-0.5680)
  - `total_rec_prncp` (-0.4473)
  - `last_pymnt_amnt` (-0.3584)
  - `total_pymnt` (-0.3233)

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
Bar chart confirming dataset imbalance (~21% defaults).
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
Heuristic keyword scan identified **39** candidate post-origination
columns to review before modeling.

Examples: `acc_now_delinq`, `collection_recovery_fee`, `collections_12_mths_ex_med`, `debt_settlement_flag`, `debt_settlement_flag_date`, `delinq_2yrs`, `delinq_amnt`, `funded_amnt_inv`, `hardship_amount`, `hardship_dpd`

**Action:** These were dropped prior to model training (see model_v3.py feature list).

---

## 14  Immediate Next Steps
1. Confirm leakage column set dropped / flagged.
2. Create temporal train / val / test splits by `issue_month`.
3. Train baseline Logistic Regression → report ROC-AUC, PR-AUC, calibration.
4. Train XGBoost + LightGBM; compare with bootstrap CIs.
5. Run macro-feature ablation study (Model A borrower-only vs Model B borrower+macro).
6. Stress-test best model under recession / rate-shock scenarios.
