
# decision_engine.py

def score_to_decision(score: float) -> str:
    """
    Convert credit score into final lending decision
    """

    if score >= 750:
        return "APPROVE"
    elif score >= 650:
        return "REVIEW"
    else:
        return "REJECT"


def decision_with_reason(score: float) -> dict:
    """
    Decision with explanation (for UI / audit)
    """

    if score >= 750:
        return {
            "decision": "APPROVE",
            "risk_level": "LOW",
            "comment": "Low credit risk. Eligible for instant approval."
        }

    elif score >= 650:
        return {
            "decision": "REVIEW",
            "risk_level": "MEDIUM",
            "comment": "Moderate risk. Requires manual review or tighter terms."
        }

    else:
        return {
            "decision": "REJECT",
            "risk_level": "HIGH",
            "comment": "High default risk. Loan not approved."
        }
