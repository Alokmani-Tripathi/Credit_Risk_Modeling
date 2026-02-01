
# reason_codes.py

import pandas as pd
import numpy as np

# Human-readable explanations
REASON_MAP = {
    "fico": "Low credit score",
    "int_rate": "High interest rate",
    "dti": "High debt-to-income ratio",
    "revol_util": "High revolving credit utilization",
    "bc_util": "High bankcard utilization",
    "percent_bc_gt_75": "Many bankcards over 75% utilization",
    "annual_inc": "Low annual income",
    "loan_amnt": "High loan amount requested",
    "inq_last_6mths": "Recent credit inquiries",
    "acc_open_past_24mths": "Many recently opened accounts",
    "mo_sin_rcnt_tl": "Very recent credit activity",
    "mths_since_recent_inq": "Very recent credit inquiry",
    "credit_age": "Short credit history",
    "emp_length": "Short employment history",
    "home_ownership": "Risky home ownership profile",
    "purpose_group": "High-risk loan purpose",
    "term": "Longer loan tenure",
    "verification_status": "Income not fully verified"
}


def get_reason_codes(
    woe_df: pd.DataFrame,
    model,
    top_n: int = 5
):
    """
    Generate top positive & negative reason codes
    """

    # Model coefficients
    coef_df = pd.DataFrame({
        "feature": woe_df.columns,
        "woe": woe_df.iloc[0].values,
        "coef": model.coef_[0]
    })

    # Contribution = coef * woe
    coef_df["contribution"] = coef_df["coef"] * coef_df["woe"]

    # Sort by absolute impact
    coef_df["abs_contribution"] = coef_df["contribution"].abs()
    coef_df = coef_df.sort_values("abs_contribution", ascending=False)

    # Split drivers
    negative_drivers = coef_df[coef_df["contribution"] > 0].head(top_n)
    positive_drivers = coef_df[coef_df["contribution"] < 0].head(top_n)

    def explain(row):
        return REASON_MAP.get(row["feature"], row["feature"])

    return {
        "risk_increasing_factors": [
            explain(row) for _, row in negative_drivers.iterrows()
        ],
        "risk_reducing_factors": [
            explain(row) for _, row in positive_drivers.iterrows()
        ]
    }
