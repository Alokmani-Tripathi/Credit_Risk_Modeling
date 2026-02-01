
import streamlit as st
import joblib

from woe_transformer import transform_user_input_to_woe
from pd_predictor import predict_pd
from scorecard import pd_to_score
from decision_engine import decision_with_reason
from reason_codes import get_reason_codes

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk PD Scoring Engine",
    layout="centered"
)

st.title("🏦 Credit Risk PD Scoring System")
st.caption("End-to-end Probability of Default model with explainability")

# --------------------------------------------------
# Load model
# --------------------------------------------------
model_bundle = joblib.load("model.joblib")
model = model_bundle["model"]

# --------------------------------------------------
# Sidebar: Borrower Inputs
# --------------------------------------------------
st.sidebar.header("📋 Borrower Information")

emp_length = st.sidebar.selectbox(
    "Employment Length",
    ["<5", "5-9", "10+", "Missing"]
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "car", "vacation", "wedding",
        "credit_card", "home_improvement",
        "major_purchase", "house",
        "debt_consolidation", "medical",
        "moving", "renewable_energy",
        "small_business", "other", "educational"
    ]
)

term = st.sidebar.selectbox(
    "Loan Term",
    ["36 months", "60 months"]
)

verification_status = st.sidebar.selectbox(
    "Verification Status",
    ["Not Verified", "Source Verified", "Verified"]
)

credit_age_bin = st.sidebar.selectbox(
    "Credit History Length",
    ["<5y", "5-10y", "10-20y", "20y+"]
)

fico = st.sidebar.number_input(
    "FICO Score", min_value=300, max_value=850, value=700
)

int_rate = st.sidebar.number_input(
    "Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0
)

loan_amnt = st.sidebar.number_input(
    "Loan Amount", min_value=1000, max_value=50000, value=15000
)

annual_inc_bin = st.sidebar.selectbox(
    "Annual Income",
    ["<40k", "40-60k", "60-80k", "80-120k", "120+", "Missing"]
)

inq_last_6mths = st.sidebar.number_input(
    "Inquiries in Last 6 Months", min_value=0, max_value=10, value=1
)

dti = st.sidebar.number_input(
    "Debt-to-Income Ratio (%)", min_value=0.0, max_value=60.0, value=20.0
)

revol_util_bin = st.sidebar.selectbox(
    "Revolving Utilization",
    ["<20%", "20-40%", "40-60%", "60-80%", "80+", "Missing"]
)

bc_util_bin = st.sidebar.selectbox(
    "Bankcard Utilization",
    ["<20%", "20-40%", "40-60%", "60-80%", "80+", "Missing"]
)

percent_bc_gt_75_bin = st.sidebar.selectbox(
    "% Bankcards > 75%",
    ["0%", "1-33%", "34-66%", "67+", "Missing"]
)

acc_open_past_24mths_bin = st.sidebar.selectbox(
    "Accounts Opened (24 Months)",
    ["0-1", "2-3", "4-6", "7+", "Missing"]
)

mo_sin_rcnt_tl_bin = st.sidebar.selectbox(
    "Months Since Recent Trade",
    ["≤3", "4-6", "7-12", "13+", "Missing"]
)

mths_since_recent_inq_bin = st.sidebar.selectbox(
    "Months Since Recent Inquiry",
    ["≤3", "4-6", "7-12", "13+", "Missing"]
)

# --------------------------------------------------
# Assemble input
# --------------------------------------------------
borrower = {
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "purpose": purpose,
    "term": term,
    "verification_status": verification_status,
    "credit_age_bin": credit_age_bin,
    "fico": fico,
    "int_rate": int_rate,
    "loan_amnt": loan_amnt,
    "annual_inc_bin": annual_inc_bin,
    "inq_last_6mths": inq_last_6mths,
    "dti": dti,
    "revol_util_bin": revol_util_bin,
    "bc_util_bin": bc_util_bin,
    "percent_bc_gt_75_bin": percent_bc_gt_75_bin,
    "acc_open_past_24mths_bin": acc_open_past_24mths_bin,
    "mo_sin_rcnt_tl_bin": mo_sin_rcnt_tl_bin,
    "mths_since_recent_inq_bin": mths_since_recent_inq_bin
}

# --------------------------------------------------
# Main Action
# --------------------------------------------------
if st.button("🔍 Evaluate Borrower"):
    pd_value = predict_pd(borrower)
    score = pd_to_score(pd_value)
    decision = decision_with_reason(score)

    woe_df = transform_user_input_to_woe(borrower)
    reasons = get_reason_codes(woe_df, model)

    st.subheader("📊 Model Output")
    st.metric("Probability of Default (PD)", f"{pd_value:.2%}")
    st.metric("Credit Score", f"{score}")

    if decision["decision"] == "APPROVE":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.caption(decision["reason"])

    st.subheader("🧠 Reason Codes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ❌ Risk Increasing Factors")
        for r in reasons["risk_increasing_factors"]:
            st.write(f"- {r}")

    with col2:
        st.markdown("### ✅ Risk Mitigating Factors")
        for r in reasons["risk_reducing_factors"]:
            st.write(f"- {r}")
