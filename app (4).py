
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
st.caption("Champion–Challenger Framework | Logistic Regression vs XGBoost")

st.markdown(
    """
    This application evaluates a borrower using **two credit risk models**:
    - **Logistic Regression** (Champion – interpretable & policy-friendly)
    - **XGBoost** (Challenger – non-linear & higher predictive power)

    Both models produce:
    **PD → Credit Score → Risk Band → Decision**
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
# SIDEBAR – BORROWER INPUTS (RAW FEATURES ONLY)
# ============================================================

st.sidebar.header("📋 Borrower Inputs")

with st.sidebar.expander("👤 Borrower Profile", expanded=True):
    emp_length = st.selectbox("Employment Length", ["<5", "5-9", "10+", "Missing"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    annual_inc = st.number_input("Annual Income", 0, 1_000_000, 60000)

with st.sidebar.expander("💳 Loan Details", expanded=True):
    loan_amnt = st.number_input("Loan Amount", 1000, 50000, 15000)
    term = st.selectbox("Loan Term", [36, 60])
    int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)
    purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "small_business",
         "home_improvement", "other"]
    )
    verification_status = st.selectbox(
        "Verification Status",
        ["Not Verified", "Source Verified", "Verified"]
    )

with st.sidebar.expander("📊 Credit Behaviour", expanded=True):
    fico = st.number_input("FICO Score", 300, 850, 700)
    fico_range_low = fico  # XGB needs this explicitly
    dti = st.number_input("Debt-to-Income Ratio (%)", 0.0, 60.0, 20.0)
    inq_last_6mths = st.number_input("Inquiries (Last 6 Months)", 0, 10, 1)
    revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 40.0)

with st.sidebar.expander("📊 Account History", expanded=False):
    acc_open_past_24mths = st.number_input("Accounts Opened (Last 24 Months)", 0, 20, 2)
    mort_acc = st.number_input("Mortgage Accounts", 0, 10, 1)
    num_actv_rev_tl = st.number_input("Active Revolving Trades", 0, 20, 4)
    delinq_2yrs = st.number_input("Delinquencies (Last 2 Years)", 0, 10, 0)
    avg_cur_bal = st.number_input("Average Current Balance", 0, 500000, 12000)
    tot_cur_bal = st.number_input("Total Current Balance", 0, 1_000_000, 45000)
    total_bc_limit = st.number_input("Total Bankcard Limit", 0, 200000, 20000)
    mths_since_recent_bc = st.number_input("Months Since Recent Bankcard", 0, 300, 18)
    mths_since_recent_inq = st.number_input("Months Since Recent Inquiry", 0, 300, 6)
    mo_sin_old_rev_tl_op = st.number_input("Months Since Oldest Revolving Trade", 0, 500, 120)
    mo_sin_rcnt_tl = st.number_input("Months Since Recent Trade", 0, 300, 9)
    credit_age_months = st.number_input("Credit Age (Months)", 0, 600, 180)

# ============================================================
# BUILD RAW BORROWER INPUT (SINGLE SOURCE OF TRUTH)
# ============================================================

borrower = {
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "purpose": purpose,
    "verification_status": verification_status,
    "fico": fico,
    "fico_range_low": fico_range_low,
    "dti": dti,
    "inq_last_6mths": inq_last_6mths,
    "revol_util": revol_util,
    "acc_open_past_24mths": acc_open_past_24mths,
    "mort_acc": mort_acc,
    "num_actv_rev_tl": num_actv_rev_tl,
    "delinq_2yrs": delinq_2yrs,
    "avg_cur_bal": avg_cur_bal,
    "tot_cur_bal": tot_cur_bal,
    "total_bc_limit": total_bc_limit,
    "mths_since_recent_bc": mths_since_recent_bc,
    "mths_since_recent_inq": mths_since_recent_inq,
    "mo_sin_old_rev_tl_op": mo_sin_old_rev_tl_op,
    "mo_sin_rcnt_tl": mo_sin_rcnt_tl,
    "credit_age_months": credit_age_months
}

# ============================================================
# MAIN ACTION
# ============================================================

st.divider()

if st.button("🚀 Evaluate Borrower", use_container_width=True):

    results = run_champion_challenger(borrower)

    if results["agreement"]:
        st.success("✅ Both models agree on the final decision")
    else:
        st.warning("⚠️ Models disagree – requires closer review")

    st.divider()

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

        st.markdown("### 🧠 Reason Codes")
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
