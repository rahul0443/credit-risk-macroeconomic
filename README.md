# Credit Risk Modeling Under Macroeconomic Conditions

Graduate capstone research project — Arizona State University, FSE 570 (Spring 2026)

## Overview
Built a credit risk modeling pipeline on 1.37M LendingClub loans (2007–2018) 
augmented with Federal Reserve macroeconomic data (FRED). Investigated whether 
macro variables improve default prediction beyond borrower-level features.

## Key Findings
- Borrower-only LightGBM model achieved test AUC of 0.710
- Macro features (FEDFUNDS, UNRATE) did not improve predictive performance
- Stress testing under recession scenarios quantified portfolio risk shifts
- Temporal drift analysis revealed survivorship bias in recent cohorts

## Technical Stack
Python, LightGBM, FRED API, statsmodels, scikit-learn, Platt Scaling, Bootstrap Testing

## Data Source
LendingClub loan data (2007–2018) via Kaggle public dataset.
FRED macroeconomic series via St. Louis Fed API.

## Report
See `report.pdf` for full methodology, results, and discussion.
