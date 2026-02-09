# ============================================================
# Credit Decision Engine
# ============================================================

def score_to_risk_band(score: float) -> str:
    """
    Map credit score to risk band
    """
    if score >= 720:
        return "VERY_LOW"
    elif score >= 680:
        return "LOW"
    elif score >= 640:
        return "MEDIUM"
    elif score >= 600:
        return "HIGH"
    else:
        return "VERY_HIGH"


def score_to_decision(score: float) -> str:
    """
    Map credit score to final decision
    """
    if score >= 680:
        return "APPROVE"
    elif score >= 620:
        return "REVIEW"
    else:
        return "REJECT"


def decision_with_reason(
    score: float,
    pd: float | None = None,
    model_name: str = "MODEL"
) -> dict:
    """
    Final credit decision with business explanation
    """

    risk_band = score_to_risk_band(score)
    decision = score_to_decision(score)

    if decision == "APPROVE":
        reason = "Low credit risk. Eligible for approval under standard policy."
    elif decision == "REVIEW":
        reason = "Moderate credit risk. Requires manual review or adjusted terms."
    else:
        reason = "High default risk. Application does not meet credit policy."

    result = {
        "model": model_name,
        "decision": decision,
        "risk_band": risk_band,
        "score": round(score, 0),
        "reason": reason
    }

    if pd is not None:
        result["pd_percent"] = round(pd * 100, 2)

    return result


# ============================================================
# REQUIRED WRAPPER (THIS FIXES YOUR ERROR)
# ============================================================

def make_decision(pd: float, score: float, model_name: str = "MODEL") -> dict:
    """
    Wrapper used by Champion–Challenger engine.
    Converts PD + score into final decision dict.
    """
    return decision_with_reason(
        score=score,
        pd=pd,
        model_name=model_name
    )


