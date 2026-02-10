# ============================================================
# feature_schema.py
# ------------------------------------------------------------
# Single Source of Truth for Feature Definitions
#
# - RAW_FEATURES: what the USER enters in Streamlit
# - LR_FEATURES : features used by Logistic Regression (WOE space)
# - XGB_FEATURES: features used by XGBoost (raw / engineered)
# ============================================================


# ------------------------------------------------------------
# 1. RAW FEATURES (USER INPUTS)
# ------------------------------------------------------------
# These are UNIQUE raw borrower attributes required to derive
# BOTH LR (18 WOE features) and XGB (24 raw features).
# The UI MUST collect all of these.
# ------------------------------------------------------------

RAW_FEATURES = [
    # Shared core borrower & loan attributes
    "loan_amnt",
    "term",
    "int_rate",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
    "verification_status",
    "fico",
    "dti",
    "inq_last_6mths",
    "revol_util",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "mort_acc",
    "total_bc_limit",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_tl",
    "delinq_2yrs",

    # XGBoost-only raw features
    "grade",
    "sub_grade",
    "fico_range_low",
    "num_actv_rev_tl",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "credit_age_months"
]


# ------------------------------------------------------------
# 2. LOGISTIC REGRESSION FEATURES (18 FEATURES – WOE SPACE)
# ------------------------------------------------------------
# These are the EXACT features the LR model was trained on,
# AFTER binning and WOE transformation.
# Order MUST match model training.
# ------------------------------------------------------------

LR_FEATURES = [
    "fico",
    "dti",
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
    "verification_status",
    "inq_last_6mths",
    "revol_util",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "mort_acc",
    "total_bc_limit",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_tl",
    "delinq_2yrs"
]


# ------------------------------------------------------------
# 3. XGBOOST FEATURES (24 FEATURES – RAW / ENGINEERED SPACE)
# ------------------------------------------------------------
# These MUST match xgb_features.json EXACTLY (same names & order).
# No WOE, no bins here.
# ------------------------------------------------------------

XGB_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "mort_acc",
    "total_bc_limit",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_tl",
    "num_actv_rev_tl",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "credit_age_months"
]
