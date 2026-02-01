
import joblib
import numpy as np
import pandas as pd

from woe_transformer import transform_user_input_to_woe


# ============================================================
# Load trained model bundle (model + feature order)
# ============================================================

MODEL_BUNDLE_PATH = "model.joblib"

model_bundle = joblib.load(MODEL_BUNDLE_PATH)
model = model_bundle["model"]
FEATURE_ORDER = model_bundle["features"]


# ============================================================
# PD Prediction Function
# ============================================================

def predict_pd(user_input: dict) -> float:
    """
    Predict Probability of Default (PD) for a single borrower.

    Parameters
    ----------
    user_input : dict
        Raw borrower inputs (UI-friendly)

    Returns
    -------
    float
        PD value between 0 and 1
    """

    # --------------------------------------------------------
    # Step 1: Raw input -> WOE transformation
    # --------------------------------------------------------
    woe_df = transform_user_input_to_woe(user_input)

    # --------------------------------------------------------
    # Step 2: Defensive check for missing features
    # --------------------------------------------------------
    missing_features = set(FEATURE_ORDER) - set(woe_df.columns)
    if missing_features:
        raise ValueError(
            f"Missing features after WOE transformation: {missing_features}"
        )

    # --------------------------------------------------------
    # Step 3: Enforce training feature order
    # --------------------------------------------------------
    woe_df = woe_df[FEATURE_ORDER]

    # --------------------------------------------------------
    # Step 4: Predict PD
    # predict_proba -> [P(non-default), P(default)]
    # --------------------------------------------------------
    pd_value = model.predict_proba(woe_df)[:, 1][0]

    # --------------------------------------------------------
    # Step 5: Numerical safety (optional but recommended)
    # --------------------------------------------------------
    pd_value = np.clip(pd_value, 1e-6, 1 - 1e-6)

    return float(pd_value)


# ============================================================
# Local Test (Optional – for development only)
# ============================================================

if __name__ == "__main__":

    sample_borrower = {
        "emp_length": "5-9",
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "term": "36 months",
        "verification_status": "Verified",
        "credit_age_bin": "5-10y",

        "fico": 710,
        "int_rate": 13.5,
        "loan_amnt": 12000,
        "annual_inc_bin": "60-80k",
        "inq_last_6mths": 1,
        "dti": 22,

        "revol_util_bin": "40-60%",
        "bc_util_bin": "60-80%",
        "percent_bc_gt_75_bin": "34-66%",
        "acc_open_past_24mths_bin": "2-3",
        "mo_sin_rcnt_tl_bin": "7-12",
        "mths_since_recent_inq_bin": "4-6"
    }

    pd = predict_pd(sample_borrower)
    print(f"Predicted PD: {pd:.6f}")
