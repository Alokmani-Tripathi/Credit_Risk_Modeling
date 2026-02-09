

import pandas as pd
import json

with open("woe_maps.json") as f:
    WOE_MAPS = json.load(f)

with open("xgb_features.json") as f:
    XGB_FEATURES = json.load(f)


# -------------------------------
# Logistic Regression pipeline
# -------------------------------
def prepare_lr_input(user_input: dict) -> pd.DataFrame:
    data = {}

    data["emp_length"] = WOE_MAPS["emp_length"].get(user_input["emp_length"], 0)
    data["home_ownership"] = WOE_MAPS["home_ownership"].get(user_input["home_ownership"], 0)
    data["term"] = WOE_MAPS["term"].get(user_input["term"], 0)
    data["verification_status"] = WOE_MAPS["verification_status"].get(
        user_input["verification_status"], 0
    )

    data["fico"] = WOE_MAPS["fico"][str(user_input["fico"])]
    data["credit_age"] = WOE_MAPS["credit_age"][str(user_input["credit_age_months"])]
    data["dti"] = WOE_MAPS["dti"][str(user_input["dti"])]

    return pd.DataFrame([data])


# -------------------------------
# XGBoost pipeline
# -------------------------------
def prepare_xgb_input(user_input: dict) -> pd.DataFrame:
    row = {f: 0 for f in XGB_FEATURES}

    row["loan_amnt"] = user_input["loan_amnt"]
    row["int_rate"] = user_input["int_rate"]
    row["dti"] = user_input["dti"]
    row["fico_range_low"] = user_input["fico_range_low"]

    row[f"home_ownership_{user_input['home_ownership']}"] = 1
    row[f"verification_status_{user_input['verification_status']}"] = 1
    row[f"purpose_{user_input['purpose']}"] = 1

    return pd.DataFrame([row])[XGB_FEATURES]
