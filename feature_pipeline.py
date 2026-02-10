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
    RAW → BIN → WOE → LR feature dataframe (18 columns)
    """

    data = {}

    # Numeric with binning
    data["fico"] = safe_woe("fico", bin_fico(user_input["fico"]))
    data["dti"] = safe_woe("dti", bin_dti(user_input["dti"]))
    data["loan_amnt"] = safe_woe("loan_amnt", bin_loan_amnt(user_input["loan_amnt"]))
    data["revol_util"] = safe_woe("revol_util", bin_revol_util(user_input["revol_util"]))

    # Numeric without binning (already categorical-like)
    data["inq_last_6mths"] = safe_woe("inq_last_6mths", str(user_input["inq_last_6mths"]))
    data["acc_open_past_24mths"] = safe_woe(
        "acc_open_past_24mths", str(user_input["acc_open_past_24mths"])
    )
    data["avg_cur_bal"] = safe_woe("avg_cur_bal", str(user_input["avg_cur_bal"]))
    data["mort_acc"] = safe_woe("mort_acc", str(user_input["mort_acc"]))
    data["total_bc_limit"] = safe_woe("total_bc_limit", str(user_input["total_bc_limit"]))
    data["mo_sin_old_rev_tl_op"] = safe_woe(
        "mo_sin_old_rev_tl_op", str(user_input["mo_sin_old_rev_tl_op"])
    )
    data["mo_sin_rcnt_tl"] = safe_woe(
        "mo_sin_rcnt_tl", str(user_input["mo_sin_rcnt_tl"])
    )
    data["delinq_2yrs"] = safe_woe("delinq_2yrs", str(user_input["delinq_2yrs"]))

    # Pure categorical
    data["term"] = safe_woe("term", str(user_input["term"]))
    data["emp_length"] = safe_woe("emp_length", user_input["emp_length"])
    data["home_ownership"] = safe_woe("home_ownership", user_input["home_ownership"])
    data["annual_inc"] = safe_woe("annual_inc", str(user_input["annual_inc"]))
    data["purpose"] = safe_woe("purpose", user_input["purpose"])
    data["verification_status"] = safe_woe(
        "verification_status", user_input["verification_status"]
    )

    # Enforce correct feature order
    return pd.DataFrame([[data[f] for f in LR_FEATURES]], columns=LR_FEATURES)


# ============================================================
# XGBOOST PIPELINE
# ============================================================

def prepare_xgb_input(user_input: dict) -> pd.DataFrame:
    """
    RAW → XGB feature dataframe (24 columns)
    """

    row = {}

    for feature in XGB_FEATURES:
        # Direct passthrough
        row[feature] = user_input.get(feature, 0)

    return pd.DataFrame([row], columns=XGB_FEATURES)
