# ============================================================
# reason_codes.py
# ------------------------------------------------------------
# Purpose:
# Generate reason codes for Logistic Regression (WOE-based)
# ------------------------------------------------------------
# Input  : WOE-transformed dataframe (1 row)
# Model  : Trained Logistic Regression
# Output : Top risk-increasing & risk-reducing factors
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def get_reason_codes(
    woe_df: pd.DataFrame,
    lr_model,
    top_n: int = 3
) -> dict:
    """
    Generate reason codes for a Logistic Regression scorecard.

    Parameters
    ----------
    woe_df : pd.DataFrame
        Single-row dataframe containing WOE-transformed features
        (columns must match LR_FEATURES order)
    lr_model : sklearn LogisticRegression
        Trained LR model
    top_n : int
        Number of top positive / negative contributors

    Returns
    -------
    dict
        {
          "risk_increasing_factors": [...],
          "risk_reducing_factors": [...]
        }
    """

    # ----------------------------
    # Sanity checks
    # ----------------------------
    if woe_df.shape[0] != 1:
        raise ValueError("woe_df must contain exactly one row")

    if len(lr_model.coef_[0]) != woe_df.shape[1]:
        raise ValueError(
            "Mismatch between LR coefficients and WOE features"
        )

    # ----------------------------
    # Compute contributions
    # contribution = coef * WOE
    # ----------------------------
    coefs = lr_model.coef_[0]
    values = woe_df.iloc[0].values

    contributions = coefs * values

    contrib_df = pd.DataFrame({
        "feature": woe_df.columns,
        "woe_value": values,
        "coefficient": coefs,
        "contribution": contributions
    })

    # ----------------------------
    # Sort contributions
    # ----------------------------
    contrib_df = contrib_df.sort_values(
        by="contribution",
        ascending=False
    )

    # ----------------------------
    # Risk interpretation
    # Positive contribution → higher log-odds of default
    # Negative contribution → lower log-odds of default
    # ----------------------------

    risk_increasing = contrib_df[
        contrib_df["contribution"] > 0
    ].head(top_n)

    risk_reducing = contrib_df[
        contrib_df["contribution"] < 0
    ].tail(top_n)

    # ----------------------------
    # Format outputs (human-readable)
    # ----------------------------
    risk_increasing_factors = [
        f"{row.feature} (impact: +{row.contribution:.3f})"
        for row in risk_increasing.itertuples()
    ]

    risk_reducing_factors = [
        f"{row.feature} (impact: {row.contribution:.3f})"
        for row in risk_reducing.itertuples()
    ]

    return {
        "risk_increasing_factors": risk_increasing_factors,
        "risk_reducing_factors": risk_reducing_factors
    }
