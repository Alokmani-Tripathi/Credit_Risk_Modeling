
# decision_engine.py

def score_to_decision(score: float) -> str:
    if score >= 750:
        return "APPROVE"
    elif score >= 650:
        return "REVIEW"
    else:
        return "REJECT"


def decision_with_reason(score: float) -> dict:
    """
    Final credit decision with explanation
    """

    if score >= 750:
        return {
            "decision": "APPROVE",
            "risk_level": "LOW",
            "reason": "Low credit risk. Eligible for instant approval."
        }

    elif score >= 650:
        return {
            "decision": "REVIEW",
            "risk_level": "MEDIUM",
            "reason": "Moderate risk. Requires manual review or tighter terms."
        }

    else:
        return {
            "decision": "REJECT",
            "risk_level": "HIGH",
            "reason": "High default risk. Loan not approved."
        }

