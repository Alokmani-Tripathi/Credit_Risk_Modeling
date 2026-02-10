# ============================================================
# feature_pipeline.py
# ------------------------------------------------------------
# Converts RAW user inputs into:
# 1) WOE-transformed features for Logistic Regression
# 2) Raw feature matrix for XGBoost
# ============================================================

import json
import pandas as pd

from feature_schema import LR_FEATURES, XGB_FEATURES


# ============================================================
# Load WOE maps
# ============================================================

with open("woe_maps.json", "r") as f:
    WOE_MAPS = json.load(f)


# ============================================================
# BINNING FUNCTIONS (MUST MATCH LR TRAINING)
# ============================================================

def bin_fico(fico: float) -> str:
    if fico < 580:
        return "<580"
    elif fico < 670:
        return "580-669"
    elif fico < 740:
        return "670-739"
    else:
        return "740+"


def bin_dti(dti: float) -> str:
    if dti < 20:
        return "<20"
    elif dti < 35:
        return "20-34"
    else:
        return "35+"


def bin_loan_amnt(amt: float) -> str:
    if amt < 5000:
        return "<5k"
    elif amt < 10000:
        return "5k-10k"
    elif amt < 20000:
        return "10k-20k"
    else:
        return "20k+"


def bin_revol_util(util: float) -> str:
    if util < 30:
        return "<30%"
    elif util < 60:
        return "30-59%"
    else:
        return "60%+"


def bin_int_rate(rate: float) -> str:
    if rate < 10:
        return "<10%"
    elif rate < 15:
        return "10-15%"
    elif rate < 20:
        return "15-20%"
    else:
        return "20%+"


def bin_credit_age(months: float) -> str:
    if months < 24:
        return "<2y"
    elif months < 60:
        return "2-5y"
    elif months < 120:
        return "5-10y"
    else:
        return "10y+"


def bin_bc_util(util: float) -> str:
    if util < 30:
        return "<30%"
    elif util < 60:
        return "30-59%"
    elif util < 75:
        return "60-74%"
    else:
        return "75%+"


def bin_percent_bc_gt_75(pct: float) -> str:
    if pct == 0:
        return "0%"
    elif pct < 25:
        return "1-24%"
    elif pct < 50:
        return "25-49%"
    elif pct < 75:
        return "50-74%"
    else:
        return "75%+"


def safe_woe(feature: str, bin_label: str) -> float:
    """
    Safe WOE lookup with fallback
    """
    return WOE_MAPS.get(feature, {}).get(bin_label, 0.0)


# ============================================================
# LOGISTIC REGRESSION PIPELINE
# ============================================================

def prepare_lr_input(user_input: dict) -> pd.DataFrame:
    """
    RAW → BIN → WOE → LR input (exactly 18 features)
    """

    # Initialize ALL LR features with neutral WOE
    data = {f: 0.0 for f in LR_FEATURES}

    # ------------------------
    # CATEGORICAL
    # ------------------------
    data["emp_length"] = safe_woe("emp_length", user_input["emp_length"])
    data["home_ownership"] = safe_woe("home_ownership", user_input["home_ownership"])
    data["term"] = safe_woe("term", str(user_input["term"]))
    data["verification_status"] = safe_woe(
        "verification_status", user_input["verification_status"]
    )

    # purpose → purpose_group
    purpose = user_input["purpose"]
    if purpose in ["credit_card", "debt_consolidation"]:
        purpose_group = "debt"
    elif purpose in ["home_improvement", "major_purchase"]:
        purpose_group = "home"
    else:
        purpose_group = "other"

    data["purpose_group"] = safe_woe("purpose_group", purpose_group)

    # ------------------------
    # NUMERIC (BINNED)
    # ------------------------
    data["fico"] = safe_woe("fico", bin_fico(user_input["fico"]))
    data["dti"] = safe_woe("dti", bin_dti(user_input["dti"]))
    data["loan_amnt"] = safe_woe("loan_amnt", bin_loan_amnt(user_input["loan_amnt"]))
    data["revol_util"] = safe_woe("revol_util", bin_revol_util(user_input["revol_util"]))
    data["int_rate"] = safe_woe("int_rate", bin_int_rate(user_input["int_rate"]))

    # ------------------------
    # COUNT / RATIO FEATURES
    # ------------------------
    data["inq_last_6mths"] = safe_woe(
        "inq_last_6mths", str(user_input["inq_last_6mths"])
    )
    data["acc_open_past_24mths"] = safe_woe(
        "acc_open_past_24mths", str(user_input["acc_open_past_24mths"])
    )
    data["mo_sin_rcnt_tl"] = safe_woe(
        "mo_sin_rcnt_tl", str(user_input["mo_sin_rcnt_tl"])
    )
    data["mths_since_recent_inq"] = safe_woe(
        "mths_since_recent_inq", str(user_input["mths_since_recent_inq"])
    )

    # ------------------------
    # CREDIT AGE
    # ------------------------
    credit_age = user_input.get("credit_age_months", 0)
    data["credit_age"] = safe_woe("credit_age", bin_credit_age(credit_age))

    # ------------------------
    # BC FEATURES
    # ------------------------
    data["bc_util"] = safe_woe(
        "bc_util", bin_bc_util(user_input.get("bc_util", 0))
    )

    data["percent_bc_gt_75"] = safe_woe(
        "percent_bc_gt_75",
        bin_percent_bc_gt_75(user_input.get("percent_bc_gt_75", 0))
    )

    # Enforce exact feature order
    return pd.DataFrame([[data[f] for f in LR_FEATURES]], columns=LR_FEATURES)


# ============================================================
# XGBOOST PIPELINE
# ============================================================

def prepare_xgb_input(user_input: dict) -> pd.DataFrame:
    """
    RAW → XGB feature dataframe (24 columns)
    """

    row = {feature: user_input.get(feature, 0) for feature in XGB_FEATURES}

    return pd.DataFrame([row], columns=XGB_FEATURES)
