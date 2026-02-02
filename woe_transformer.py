

import json
import pandas as pd



# Load WOE maps once (global)
with open("woe_maps.json", "r") as f:
    WOE_MAPS = json.load(f)


def get_woe(feature_name, bin_label):
    """
    Safely fetch WOE value for a feature-bin pair.
    """
    try:
        return WOE_MAPS[feature_name][bin_label]
    except KeyError:
        # fallback to Missing if unseen
        return WOE_MAPS[feature_name].get("Missing", 0.0)



def bin_int_rate(rate):
    if rate <= 7:
        return "<=7%"
    elif rate <= 10:
        return "7–10%"
    elif rate <= 13:
        return "10–13%"
    elif rate <= 16:
        return "13–16%"
    elif rate <= 20:
        return "16–20%"
    else:
        return "20%+"



def bin_fico(fico):
    if fico < 660:
        return "<660"
    elif fico < 680:
        return "660–679"
    elif fico < 700:
        return "680–699"
    elif fico < 720:
        return "700–719"
    elif fico < 760:
        return "720–759"
    else:
        return "760+"


def bin_loan_amnt(amount):
    if amount <= 5000:
        return "<=5k"
    elif amount <= 10000:
        return "5k–10k"
    elif amount <= 15000:
        return "10k–15k"
    elif amount <= 20000:
        return "15k–20k"
    elif amount <= 30000:
        return "20k–30k"
    else:
        return "30k+"

def bin_dti(dti):
    if dti < 10:
        return "<10"
    elif dti < 20:
        return "10–20"
    elif dti < 30:
        return "20–30"
    elif dti < 40:
        return "30–40"
    else:
        return "40+"



def bin_inq_6mths(x):
    if x <= 1:
        return "0–1"
    elif x == 2:
        return "1–2"
    else:
        return "2+"


#def get_purpose_woe(purpose_raw):
    #purpose_raw = purpose_raw.lower()
    #return get_woe("purpose_group", purpose_raw)

def get_purpose_woe(purpose_raw):
    if purpose_raw is None:
        return get_woe("purpose_group", "Missing")

    purpose_raw = purpose_raw.lower().strip()
    return get_woe("purpose_group", purpose_raw)

def transform_user_input_to_woe(user_input):
    """
    Convert raw borrower input into WOE-transformed dataframe
    """

    data = {}

    data["emp_length"] = get_woe("emp_length", user_input["emp_length"])
    data["home_ownership"] = get_woe("home_ownership", user_input["home_ownership"])
    data["purpose_group"] = get_purpose_woe(user_input["purpose"])
    data["term"] = get_woe("term", user_input["term"])
    data["verification_status"] = get_woe("verification_status", user_input["verification_status"])
    data["credit_age"] = get_woe("credit_age", user_input["credit_age_bin"])

    data["fico"] = get_woe("fico", bin_fico(user_input["fico"]))
    data["int_rate"] = get_woe("int_rate", bin_int_rate(user_input["int_rate"]))
    data["loan_amnt"] = get_woe("loan_amnt", bin_loan_amnt(user_input["loan_amnt"]))
    data["annual_inc"] = get_woe("annual_inc", user_input["annual_inc_bin"])
    data["inq_last_6mths"] = get_woe("inq_last_6mths", bin_inq_6mths(user_input["inq_last_6mths"]))
    data["dti"] = get_woe("dti", bin_dti(user_input["dti"]))

    data["revol_util"] = get_woe("revol_util", user_input["revol_util_bin"])
    data["bc_util"] = get_woe("bc_util", user_input["bc_util_bin"])
    data["percent_bc_gt_75"] = get_woe("percent_bc_gt_75", user_input["percent_bc_gt_75_bin"])
    data["acc_open_past_24mths"] = get_woe("acc_open_past_24mths", user_input["acc_open_past_24mths_bin"])
    data["mo_sin_rcnt_tl"] = get_woe("mo_sin_rcnt_tl", user_input["mo_sin_rcnt_tl_bin"])
    data["mths_since_recent_inq"] = get_woe("mths_since_recent_inq", user_input["mths_since_recent_inq_bin"])


    EXPECTED_FEATURE_ORDER = [
    'emp_length',
    'home_ownership',
    'purpose_group',
    'term',
    'verification_status',
    'credit_age',
    'int_rate',
    'loan_amnt',
    'fico',
    'annual_inc',
    'inq_last_6mths',
    'dti',
    'revol_util',
    'bc_util',
    'percent_bc_gt_75',
    'acc_open_past_24mths',
    'mo_sin_rcnt_tl',
    'mths_since_recent_inq'
]
    return pd.DataFrame([data], columns=EXPECTED_FEATURE_ORDER)

