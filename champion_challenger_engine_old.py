

# import joblib
# from feature_pipeline import prepare_lr_input, prepare_xgb_input
# from scorecard import pd_to_score
# from decision_engine import make_decision

# lr_model = joblib.load("model.joblib")
# xgb_model = joblib.load("xgb_model.joblib")


# def run_champion_challenger(user_input):
#     lr_X = prepare_lr_input(user_input)
#     xgb_X = prepare_xgb_input(user_input)

#     pd_lr = lr_model.predict_proba(lr_X)[0, 1]
#     pd_xgb = xgb_model.predict_proba(xgb_X)[0, 1]

#     score_lr = pd_to_score(pd_lr)
#     score_xgb = pd_to_score(pd_xgb)

#     return {
#         "logistic": make_decision(pd_lr, score_lr),
#         "xgboost": make_decision(pd_xgb, score_xgb),
#         "pd_lr": pd_lr,
#         "pd_xgb": pd_xgb,
#         "score_lr": score_lr,
#         "score_xgb": score_xgb
#     }



# ============================================================
# champion_challenger_engine.py
# ------------------------------------------------------------
# Orchestrates Logistic Regression (Champion) vs
# XGBoost (Challenger) on the same borrower
# ============================================================

import joblib

from feature_pipeline import prepare_lr_input, prepare_xgb_input
from scorecard import pd_to_score
from decision_engine import make_decision


# ============================================================
# Load models once (module-level)
# ============================================================

# Logistic Regression model bundle
lr_bundle = joblib.load("model.joblib")
lr_model = lr_bundle["model"]

# XGBoost model
xgb_model = joblib.load("xgb_model.joblib")


# ============================================================
# Champion–Challenger Runner
# ============================================================

def run_champion_challenger(user_input: dict) -> dict:
    """
    Run both Logistic Regression (Champion) and
    XGBoost (Challenger) for a single borrower.

    Parameters
    ----------
    user_input : dict
        Raw borrower inputs (must match RAW_FEATURES)

    Returns
    -------
    dict
        Results for LR, XGB, and agreement flag
    """

    # ----------------------------
    # Logistic Regression (Champion)
    # ----------------------------
    X_lr = prepare_lr_input(user_input)
    pd_lr = lr_model.predict_proba(X_lr)[0, 1]
    score_lr = pd_to_score(pd_lr)

    lr_result = make_decision(
        score=score_lr,
        pd=pd_lr,
        model_name="Logistic Regression"
    )

    lr_result["X_lr"] = X_lr

    # ----------------------------
    # XGBoost (Challenger)
    # ----------------------------
    X_xgb = prepare_xgb_input(user_input)
    pd_xgb = xgb_model.predict_proba(X_xgb)[0, 1]
    score_xgb = pd_to_score(pd_xgb)

    xgb_result = make_decision(
        score=score_xgb,
        pd=pd_xgb,
        model_name="XGBoost"
    )

    # ----------------------------
    # Agreement check
    # ----------------------------
    agreement = lr_result["decision"] == xgb_result["decision"]

    return {
        "logistic": lr_result,
        "xgboost": xgb_result,
        "agreement": agreement
    }













