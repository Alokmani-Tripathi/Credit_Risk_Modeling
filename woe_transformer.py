

# import json
# import pandas as pd



# # Load WOE maps once (global)
# with open("woe_maps.json", "r") as f:
#     WOE_MAPS = json.load(f)


# def get_woe(feature_name, bin_label):
#     """
#     Safely fetch WOE value for a feature-bin pair.
#     """
#     try:
#         return WOE_MAPS[feature_name][bin_label]
#     except KeyError:
#         # fallback to Missing if unseen
#         return WOE_MAPS[feature_name].get("Missing", 0.0)



# def bin_int_rate(rate):
#     if rate <= 7:
#         return "<=7%"
#     elif rate <= 10:
#         return "7–10%"
#     elif rate <= 13:
#         return "10–13%"
#     elif rate <= 16:
#         return "13–16%"
#     elif rate <= 20:
#         return "16–20%"
#     else:
#         return "20%+"



# def bin_fico(fico):
#     if fico < 660:
#         return "<660"
#     elif fico < 680:
#         return "660–679"
#     elif fico < 700:
#         return "680–699"
#     elif fico < 720:
#         return "700–719"
#     elif fico < 760:
#         return "720–759"
#     else:
#         return "760+"


# def bin_loan_amnt(amount):
#     if amount <= 5000:
#         return "<=5k"
#     elif amount <= 10000:
#         return "5k–10k"
#     elif amount <= 15000:
#         return "10k–15k"
#     elif amount <= 20000:
#         return "15k–20k"
#     elif amount <= 30000:
#         return "20k–30k"
#     else:
#         return "30k+"

# def bin_dti(dti):
#     if dti < 10:
#         return "<10"
#     elif dti < 20:
#         return "10–20"
#     elif dti < 30:
#         return "20–30"
#     elif dti < 40:
#         return "30–40"
#     else:
#         return "40+"



# def bin_inq_6mths(x):
#     if x <= 1:
#         return "0–1"
#     elif x == 2:
#         return "1–2"
#     else:
#         return "2+"


# #def get_purpose_woe(purpose_raw):
#     #purpose_raw = purpose_raw.lower()
#     #return get_woe("purpose_group", purpose_raw)

# def get_purpose_woe(purpose_raw):
#     if purpose_raw is None:
#         return get_woe("purpose_group", "Missing")

#     purpose_raw = purpose_raw.lower().strip()
#     return get_woe("purpose_group", purpose_raw)

# def transform_user_input_to_woe(user_input):
#     """
#     Convert raw borrower input into WOE-transformed dataframe
#     """

#     data = {}

#     data["emp_length"] = get_woe("emp_length", user_input["emp_length"])
#     data["home_ownership"] = get_woe("home_ownership", user_input["home_ownership"])
#     data["purpose_group"] = get_purpose_woe(user_input["purpose"])
#     data["term"] = get_woe("term", user_input["term"])
#     data["verification_status"] = get_woe("verification_status", user_input["verification_status"])
#     data["credit_age"] = get_woe("credit_age", user_input["credit_age_bin"])

#     data["fico"] = get_woe("fico", bin_fico(user_input["fico"]))
#     data["int_rate"] = get_woe("int_rate", bin_int_rate(user_input["int_rate"]))
#     data["loan_amnt"] = get_woe("loan_amnt", bin_loan_amnt(user_input["loan_amnt"]))
#     data["annual_inc"] = get_woe("annual_inc", user_input["annual_inc_bin"])
#     data["inq_last_6mths"] = get_woe("inq_last_6mths", bin_inq_6mths(user_input["inq_last_6mths"]))
#     data["dti"] = get_woe("dti", bin_dti(user_input["dti"]))

#     data["revol_util"] = get_woe("revol_util", user_input["revol_util_bin"])
#     data["bc_util"] = get_woe("bc_util", user_input["bc_util_bin"])
#     data["percent_bc_gt_75"] = get_woe("percent_bc_gt_75", user_input["percent_bc_gt_75_bin"])
#     data["acc_open_past_24mths"] = get_woe("acc_open_past_24mths", user_input["acc_open_past_24mths_bin"])
#     data["mo_sin_rcnt_tl"] = get_woe("mo_sin_rcnt_tl", user_input["mo_sin_rcnt_tl_bin"])
#     data["mths_since_recent_inq"] = get_woe("mths_since_recent_inq", user_input["mths_since_recent_inq_bin"])


#     EXPECTED_FEATURE_ORDER = [
#     'emp_length',
#     'home_ownership',
#     'purpose_group',
#     'term',
#     'verification_status',
#     'credit_age',
#     'int_rate',
#     'loan_amnt',
#     'fico',
#     'annual_inc',
#     'inq_last_6mths',
#     'dti',
#     'revol_util',
#     'bc_util',
#     'percent_bc_gt_75',
#     'acc_open_past_24mths',
#     'mo_sin_rcnt_tl',
#     'mths_since_recent_inq'
# ]
#     return pd.DataFrame([data], columns=EXPECTED_FEATURE_ORDER)













# ============================================================
# woe_transformer.py
# ------------------------------------------------------------
# Purpose:
# Convert RAW user inputs into WOE space
# ONLY for explainability / reason codes (LR model)
# ------------------------------------------------------------
# This file is NOT used for prediction.
# Prediction uses feature_pipeline.prepare_lr_input()
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
# BINNING FUNCTIONS (MUST MATCH feature_pipeline.py)
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
    Safe WOE lookup with fallback for unseen bins
    """
    return WOE_MAPS.get(feature, {}).get(bin_label, 0.0)


# ============================================================
# PUBLIC FUNCTION (USED BY reason_codes.py)
# ============================================================

def transform_user_input_to_woe(user_input: dict) -> pd.DataFrame:
    """
    Convert RAW user input into WOE DataFrame
    for LR explainability only.
    """

    data = {}

    # Numeric (binned)
    data["fico"] = safe_woe("fico", bin_fico(user_input["fico"]))
    data["dti"] = safe_woe("dti", bin_dti(user_input["dti"]))
    data["loan_amnt"] = safe_woe("loan_amnt", bin_loan_amnt(user_input["loan_amnt"]))
    data["revol_util"] = safe_woe("revol_util", bin_revol_util(user_input["revol_util"]))

    # Count / balance style (treated as categorical in WOE)
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

    return pd.DataFrame([[data[f] for f in LR_FEATURES]], columns=LR_FEATURES)






