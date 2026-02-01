
import numpy as np

# ============================================================
# Scorecard configuration (business constants)
# ============================================================

BASE_SCORE = 600        # score at base odds
BASE_ODDS = 50          # good:bad odds at base score
PDO = 50                # points to double odds


# ============================================================
# PD -> Score conversion
# ============================================================

def pd_to_score(pd: float) -> float:
    """
    Convert Probability of Default (PD) into a credit score.

    Parameters
    ----------
    pd : float
        Probability of Default (0 < pd < 1)

    Returns
    -------
    float
        Credit score
    """

    # Safety check
    pd = np.clip(pd, 1e-6, 1 - 1e-6)

    # Convert PD to odds
    odds = (1 - pd) / pd

    # Score formula
    score = BASE_SCORE + (PDO / np.log(2)) * np.log(odds / BASE_ODDS)

    return round(score, 0)

