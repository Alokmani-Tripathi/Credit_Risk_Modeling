
# ============================================================
# Champion – Challenger Orchestration Engine
# ============================================================

from pd_predictor import predict_pd              # Logistic PD
from xgb_pd_predictor import predict_pd_xgb      # XGBoost PD

from scorecard import pd_to_score
from decision_engine import decision_with_reason


# ============================================================
# Unified Prediction Function
# ============================================================

def run_champion_challenger(borrower_input: dict) -> dict:
    """
    Run Logistic (Champion) and XGBoost (Challenger)
    for the same borrower input.

    Parameters
    ----------
    borrower_input : dict
        Raw borrower information from UI

    Returns
    -------
    dict
        Side-by-side model results
    """

    # -------------------------------
    # Logistic Regression (Champion)
    # -------------------------------
    pd_lr = predict_pd(borrower_input)
    score_lr = pd_to_score(pd_lr)

    result_lr = decision_with_reason(
        score=score_lr,
        pd=pd_lr,
        model_name="Logistic Regression"
    )

    # -------------------------------
    # XGBoost (Challenger)
    # -------------------------------
    pd_xgb = predict_pd_xgb(borrower_input)
    score_xgb = pd_to_score(pd_xgb)

    result_xgb = decision_with_reason(
        score=score_xgb,
        pd=pd_xgb,
        model_name="XGBoost"
    )

    # -------------------------------
    # Agreement / Disagreement Flag
    # -------------------------------
    agreement = result_lr["decision"] == result_xgb["decision"]

    # -------------------------------
    # Final Output
    # -------------------------------
    return {
        "logistic": result_lr,
        "xgboost": result_xgb,
        "agreement": agreement
    }
