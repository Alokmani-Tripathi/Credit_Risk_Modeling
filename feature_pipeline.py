# ============================================================
# feature_pipeline.py
# ------------------------------------------------------------
# Unified feature transformation layer
# - LR uses original WOE transformer
# - XGB rebuilds exact training feature space
# ============================================================

import pandas as pd
from woe_transformer import transform_user_input_to_woe


# ============================================================
# LOGISTIC REGRESSION PIPELINE
# ============================================================

def prepare_lr_input(user_input: dict) -> pd.DataFrame:
    """
    Use EXACT same WOE transformer as standalone LR app.
    This guarantees identical PD results.
    """
    return transform_user_input_to_woe(user_input)


# ============================================================
# XGBOOST PIPELINE
# ============================================================

def prepare_xgb_input(user_input: dict) -> pd.DataFrame:
    """
    Rebuild EXACT XGBoost training feature space.
    Must match xgb_model.feature_names exactly.
    """

    XGB_TRAIN_FEATURES = [
        'grade',
        'sub_grade',
        'fico_range_low',
        'term',
        'int_rate',
        'loan_amnt',
        'annual_inc',
        'dti',
        'emp_length',
        'verification_status_Source Verified',
        'home_ownership_MORTGAGE',
        'home_ownership_RENT',
        'mort_acc',
        'acc_open_past_24mths',
        'num_actv_rev_tl',
        'delinq_2yrs',
        'mths_since_recent_bc',
        'mths_since_recent_inq',
        'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_tl',
        'avg_cur_bal',
        'tot_cur_bal',
        'total_bc_limit',
        'purpose_small_business'
    ]

    row = {f: 0 for f in XGB_TRAIN_FEATURES}

    # -------------------------
    # NUMERIC FEATURES
    # -------------------------
    numeric_fields = [
        'fico_range_low', 'term', 'int_rate', 'loan_amnt',
        'annual_inc', 'dti', 'mort_acc', 'acc_open_past_24mths',
        'num_actv_rev_tl', 'delinq_2yrs', 'mths_since_recent_bc',
        'mths_since_recent_inq', 'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_tl', 'avg_cur_bal', 'tot_cur_bal',
        'total_bc_limit'
    ]

    for f in numeric_fields:
        row[f] = user_input.get(f, 0)

    # -------------------------
    # ORDINAL ENCODING
    # -------------------------
    grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    row["grade"] = grade_map.get(user_input.get("grade"), 0)

    sub_grade_map = {
        f"{g}{i}": idx
        for idx, (g, i) in enumerate(
            [(g, i) for g in "ABCDEFG" for i in range(1,6)], start=1
        )
    }
    row["sub_grade"] = sub_grade_map.get(user_input.get("sub_grade"), 0)

    emp_length_map = {
        "<1":0,"1-3":1,"3-5":2,"5-10":3,"10+":4,"Missing":0
    }
    row["emp_length"] = emp_length_map.get(user_input.get("emp_length"), 0)

    # -------------------------
    # ONE-HOT FEATURES
    # -------------------------
    if user_input.get("home_ownership") == "MORTGAGE":
        row["home_ownership_MORTGAGE"] = 1

    if user_input.get("home_ownership") == "RENT":
        row["home_ownership_RENT"] = 1

    if user_input.get("verification_status") == "Source Verified":
        row["verification_status_Source Verified"] = 1

    if user_input.get("purpose") == "small_business":
        row["purpose_small_business"] = 1

    return pd.DataFrame([row], columns=XGB_TRAIN_FEATURES)
