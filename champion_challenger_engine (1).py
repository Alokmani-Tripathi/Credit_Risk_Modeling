

import joblib
from feature_pipeline import prepare_lr_input, prepare_xgb_input
from scorecard import pd_to_score
from decision_engine import make_decision

lr_model = joblib.load("model.joblib")
xgb_model = joblib.load("xgb_model.joblib")


def run_champion_challenger(user_input):
    lr_X = prepare_lr_input(user_input)
    xgb_X = prepare_xgb_input(user_input)

    pd_lr = lr_model.predict_proba(lr_X)[0, 1]
    pd_xgb = xgb_model.predict_proba(xgb_X)[0, 1]

    score_lr = pd_to_score(pd_lr)
    score_xgb = pd_to_score(pd_xgb)

    return {
        "logistic": make_decision(pd_lr, score_lr),
        "xgboost": make_decision(pd_xgb, score_xgb),
        "pd_lr": pd_lr,
        "pd_xgb": pd_xgb,
        "score_lr": score_lr,
        "score_xgb": score_xgb
    }
