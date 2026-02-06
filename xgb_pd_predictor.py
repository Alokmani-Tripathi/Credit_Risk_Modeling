
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# Load trained XGBoost model + feature order
# ============================================================

MODEL_PATH = "xgb_model.joblib"
FEATURE_PATH = "xgb_features.json"

xgb_model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    FEATURE_ORDER = json.load(f)


# ============================================================
# PD Prediction Function (XGBoost)
# ============================================================

def predict_pd_xgb(user_input: dict) -> float:
    """
    Predict Probability of Default (PD) for a single borrower
    using XGBoost model.

    Parameters
    ----------
    user_input : dict
        Model-ready borrower inputs (numeric / encoded)

    Returns
    -------
    float
        PD value between 0 and 1
    """

    # --------------------------------------------------------
    # Step 1: Convert input to DataFrame
    # --------------------------------------------------------
    df_input = pd.DataFrame([user_input])

    # --------------------------------------------------------
    # Step 2: Defensive check for missing features
    # --------------------------------------------------------
    missing_features = set(FEATURE_ORDER) - set(df_input.columns)
    if missing_features:
        raise ValueError(
            f"Missing features for XGBoost model: {missing_features}"
        )

    # --------------------------------------------------------
    # Step 3: Enforce training feature order
    # --------------------------------------------------------
    df_input = df_input[FEATURE_ORDER]

    # --------------------------------------------------------
    # Step 4: Predict PD
    # predict_proba -> [P(non-default), P(default)]
    # --------------------------------------------------------
    pd_value = xgb_model.predict_proba(df_input)[:, 1][0]

    # --------------------------------------------------------
    # Step 5: Numerical safety
    # --------------------------------------------------------
    pd_value = np.clip(pd_value, 1e-6, 1 - 1e-6)

    return float(pd_value)


# ============================================================
# Local Test (Optional – development only)
# ============================================================

if __name__ == "__main__":

    sample_borrower = {
        'grade': 3,
        'sub_grade': 12,
        'fico_range_low': 710,
        'term': 36,
        'int_rate': 13.5,
        'loan_amnt': 15000,
        'annual_inc': 85000,
        'dti': 18.4,
        'emp_length': 6,
        'verification_status_Source Verified': 1,
        'home_ownership_MORTGAGE': 1,
        'home_ownership_RENT': 0,
        'mort_acc': 2,
        'acc_open_past_24mths': 4,
        'num_actv_rev_tl': 6,
        'delinq_2yrs': 0,
        'mths_since_recent_bc': 12,
        'mths_since_recent_inq': 5,
        'mo_sin_old_rev_tl_op': 120,
        'mo_sin_rcnt_tl': 6,
        'avg_cur_bal': 9000,
        'tot_cur_bal': 120000,
        'total_bc_limit': 45000,
        'purpose_small_business': 0
    }

    pd = predict_pd_xgb(sample_borrower)
    print(f"Predicted XGB PD: {pd:.6f}")
