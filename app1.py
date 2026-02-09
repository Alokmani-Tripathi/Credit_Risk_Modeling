
import streamlit as st
import joblib

from champion_challenger_engine import run_champion_challenger
from reason_codes import get_reason_codes
from woe_transformer import transform_user_input_to_woe


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Decision Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HEADER
# ============================================================

st.title("🏦 Credit Risk Decision Engine")
st.caption(
    "Champion–Challenger Framework | Logistic Regression vs XGBoost"
)

st.markdown(
    """
    This application evaluates a borrower using **two credit risk models**:
    - **Logistic Regression** (Champion – policy friendly & interpretable)
    - **XGBoost** (Challenger – non-linear & higher predictive power)

    Both models generate **PD → Credit Score → Decision**  
    on a **common scorecard scale**.
    """
)

st.divider()

# ============================================================
# LOAD LOGISTIC MODEL (FOR REASON CODES)
# ============================================================

model_bundle = joblib.load("model.joblib")
logit_model = model_bundle["model"]

# ============================================================
# SIDEBAR – BORROWER INPUTS
# ============================================================

st.sidebar.header("📋 Borrower Inputs")

with st.sidebar.expander("👤 Borrower Profile", expanded=True):
    emp_length = st.selectbox("Employment Length", ["<5", "5-9", "10+", "Missing"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    annual_inc_bin = st.selectbox(
        "Annual Income",
        ["<40k", "40-60k", "60-80k", "80-120k", "120+", "Missing"]
    )

with st.sidebar.expander("💳 Loan Details", expanded=True):
    loan_amnt = st.number_input("Loan Amount", 1000, 50000, 15000)
    term = st.selectbox("Loan Term", ["36 months", "60 months"])
    int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)
    purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "small_business", "home_improvement", "other"]
    )
    verification_status = st.selectbox(
        "Verification Status",
        ["Not Verified", "Source Verified", "Verified"]
    )

with st.sidebar.expander("📊 Credit Behavior", expanded=True):
    fico = st.number_input("FICO Score", 300, 850, 700)
    dti = st.number_input("Debt-to-Income Ratio (%)", 0.0, 60.0, 20.0)
    inq_last_6mths = st.number_input("Inquiries (Last 6 Months)", 0, 10, 1)
    revol_util_bin = st.selectbox(
        "Revolving Utilization",
        ["<20%", "20-40%", "40-60%", "60-80%", "80+", "Missing"]
    )

# ============================================================
# BUILD BORROWER INPUT
# ============================================================

borrower = {
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc_bin": annual_inc_bin,
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "purpose": purpose,
    "verification_status": verification_status,
    "fico": fico,
    "dti": dti,
    "inq_last_6mths": inq_last_6mths,
    "revol_util_bin": revol_util_bin
}

# ============================================================
# MAIN ACTION
# ============================================================

st.divider()

if st.button("🚀 Evaluate Borrower", use_container_width=True):

    results = run_champion_challenger(borrower)

    # ========================================================
    # SUMMARY BANNER
    # ========================================================

    if results["agreement"]:
        st.success("✅ Both models agree on the final decision")
    else:
        st.warning("⚠️ Models disagree – requires closer review")

    st.divider()

    # ========================================================
    # CHAMPION vs CHALLENGER
    # ========================================================

    col1, col2 = st.columns(2)

    # -------------------------------
    # Logistic Regression
    # -------------------------------
    with col1:
        st.subheader("📘 Logistic Regression (Champion)")

        lr = results["logistic"]
        st.metric("PD (%)", lr["pd_percent"])
        st.metric("Credit Score", lr["score"])
        st.metric("Risk Band", lr["risk_band"])
        st.markdown(f"**Decision:** `{lr['decision']}`")
        st.caption(lr["reason"])

        st.markdown("**🧠 Reason Codes**")
        woe_df = transform_user_input_to_woe(borrower)
        reasons = get_reason_codes(woe_df, logit_model)
        st.write("❌ Risk Increasing:", reasons["risk_increasing_factors"])
        st.write("✅ Risk Reducing:", reasons["risk_reducing_factors"])

    # -------------------------------
    # XGBoost
    # -------------------------------
    with col2:
        st.subheader("📗 XGBoost (Challenger)")

        xgb = results["xgboost"]
        st.metric("PD (%)", xgb["pd_percent"])
        st.metric("Credit Score", xgb["score"])
        st.metric("Risk Band", xgb["risk_band"])
        st.markdown(f"**Decision:** `{xgb['decision']}`")
        st.caption(xgb["reason"])
