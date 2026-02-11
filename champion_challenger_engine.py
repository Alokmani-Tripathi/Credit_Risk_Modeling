# ============================================================
# champion_challenger_engine.py
# ------------------------------------------------------------
# Champion–Challenger Framework
# Logistic Regression (Champion)
# XGBoost (Challenger)
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
    Run both Logistic Regression (Champion)
    and XGBoost (Challenger) for a borrower.

    Returns
    -------
    dict with:
        logistic
        xgboost
        agreement
        debug_vectors (optional inspection)
    """

    # ========================================================
    # 1️⃣ Logistic Regression (Champion)
    # ========================================================

    X_lr = prepare_lr_input(user_input)

    pd_lr = lr_model.predict_proba(X_lr)[0, 1]
    score_lr = pd_to_score(pd_lr)

    lr_result = make_decision(
        score=score_lr,
        pd=pd_lr,
        model_name="Logistic Regression"
    )

    # Attach model input vector for debugging
    lr_result["X_lr"] = X_lr


    # ========================================================
    # 2️⃣ XGBoost (Challenger)
    # ========================================================

    X_xgb = prepare_xgb_input(user_input)

    pd_xgb = xgb_model.predict_proba(X_xgb)[0, 1]
    score_xgb = pd_to_score(pd_xgb)

    xgb_result = make_decision(
        score=score_xgb,
        pd=pd_xgb,
        model_name="XGBoost"
    )

    # Attach XGB input for debugging
    xgb_result["X_xgb"] = X_xgb


    # ========================================================
    # 3️⃣ Agreement Logic
    # ========================================================

    agreement = lr_result["decision"] == xgb_result["decision"]


    # ========================================================
    # Final Output
    # ========================================================

    return {
        "logistic": lr_result,
        "xgboost": xgb_result,
        "agreement": agreement
    }
