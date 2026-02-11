# ============================================================
# woe_transformer.py
# ------------------------------------------------------------
# Convert RAW user inputs into WOE space
# Used for:
# 1) LR prediction (via feature_pipeline)
# 2) Reason codes
# ============================================================

import json
import pandas as pd
from feature_schema import LR_FEATURES


# ============================================================
# Load WOE maps
# ============================================================

with open("woe_maps.json", "r") as f:
    WOE_MAPS = json.load(f)


# ============================================================
# SAFE WOE LOOKUP
# ============================================================

def safe_woe(feature: str, bin_label: str) -> float:
    return WOE_MAPS.get(feature, {}).get(bin_label, 0.0)


# ============================================================
# BINNING FUNCTIONS (MUST MATCH TRAINING)
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


# ============================================================
# MAIN TRANSFORM FUNCTION
# ============================================================

def transform_user_input_to_woe(user_input: dict) -> pd.DataFrame:
    """
    Convert RAW borrower input into EXACT WOE feature vector
    matching LR training.
    """

    data = {}

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    data["emp_length"] = safe_woe("emp_length", user_input["emp_length"])
    data["home_ownership"] = safe_woe("home_ownership", user_input["home_ownership"])
    data["term"] = safe_woe("term", str(user_input["term"]))
    data["verification_status"] = safe_woe(
        "verification_status", user_input["verification_status"]
    )

    # PURPOSE → PURPOSE_GROUP
    purpose = user_input["purpose"]

    if purpose in ["credit_card", "debt_consolidation"]:
        purpose_group = "debt"
    elif purpose in ["home_improvement", "major_purchase"]:
        purpose_group = "home"
    else:
        purpose_group = "other"

    data["purpose_group"] = safe_woe("purpose_group", purpose_group)

    # --------------------------------------------------------
    # NUMERIC (BINNED)
    # --------------------------------------------------------

    data["fico"] = safe_woe("fico", bin_fico(user_input["fico"]))
    data["dti"] = safe_woe("dti", bin_dti(user_input["dti"]))
    data["loan_amnt"] = safe_woe("loan_amnt", bin_loan_amnt(user_input["loan_amnt"]))
    data["revol_util"] = safe_woe(
        "revol_util", bin_revol_util(user_input["revol_util"])
    )
    data["int_rate"] = safe_woe(
        "int_rate", bin_int_rate(user_input["int_rate"])
    )

    # --------------------------------------------------------
    # CREDIT AGE
    # --------------------------------------------------------

    credit_age = user_input.get("credit_age_months", 0)
    data["credit_age"] = safe_woe("credit_age", bin_credit_age(credit_age))

    # --------------------------------------------------------
    # COUNT / RATIO FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BC FEATURES
    # --------------------------------------------------------

    data["bc_util"] = safe_woe(
        "bc_util", bin_bc_util(user_input["bc_util"])
    )

    data["percent_bc_gt_75"] = safe_woe(
        "percent_bc_gt_75",
        bin_percent_bc_gt_75(user_input["percent_bc_gt_75"])
    )

    # --------------------------------------------------------
    # ANNUAL INCOME (as string category)
    # --------------------------------------------------------

    data["annual_inc"] = safe_woe(
        "annual_inc", str(user_input["annual_inc"])
    )

    # --------------------------------------------------------
    # FINAL – ENFORCE EXACT FEATURE ORDER
    # --------------------------------------------------------

    return pd.DataFrame([[data[f] for f in LR_FEATURES]], columns=LR_FEATURES)
