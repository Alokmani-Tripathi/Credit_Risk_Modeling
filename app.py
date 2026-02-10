# ============================================================
# app.py
# ------------------------------------------------------------
# Streamlit UI for Credit Risk Decision Engine
# Champion (Logistic Regression) vs Challenger (XGBoost)
# ============================================================

import streamlit as st
import joblib

from champion_challenger_engine import run_champion_challenger
from woe_transformer import transform_user_input_to_woe
from reason_codes import get_reason_codes


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

    • **Logistic Regression (Champion)** – Interpretable, policy-friendly  
    • **XGBoost (Challenger)** – Non-linear, higher predictive power  

    Both models generate:
    **PD → Credit Score → Risk Band → Final Decision**
    on a **common scorecard scale**.
    """
)

st.divider()


# ============================================================
# LOAD LR MODEL (FOR REASON CODES ONLY)
# ============================================================

lr_bundle = joblib.load("model.joblib")
lr_model = lr_bundle["model"]


# ============================================================
# SIDEBAR – RAW BORROWER INPUTS
# ============================================================

st.sidebar.header("📋 Borrower Profile")

borrower = {}

def numeric_input(label, default=0, min_val=0, max_val=1_000_000):
    return st.sidebar.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default
    )

# -----------------------------
# Loan & Employment
# -----------------------------

borrower["loan_amnt"] = numeric_input("Loan Amount", 15000, 1000, 50000)
borrower["term"] = st.sidebar.selectbox("Loan Term (Months)", [36, 60])
borrower["int_rate"] = st.sidebar.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)

borrower["emp_length"] = st.sidebar.selectbox(
    "Employment Length",
    ["<1", "1-3", "3-5", "5-10", "10+", "Missing"]
)

borrower["home_ownership"] = st.sidebar.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

borrower["annual_inc"] = numeric_input("Annual Income", 60000, 0, 1_000_000)

borrower["purpose"] = st.sidebar.selectbox(
    "Loan Purpose",
    ["debt_consolidation", "credit_card", "small_business",
     "home_improvement", "other"]
)

borrower["verification_status"] = st.sidebar.selectbox(
    "Verification Status",
    ["Not Verified", "Source Verified", "Verified"]
)

# -----------------------------
# Credit Profile
# -----------------------------

borrower["fico"] = numeric_input("FICO Score", 700, 300, 850)
borrower["fico_range_low"] = borrower["fico"]  # for XGB

borrower["dti"] = st.sidebar.number_input("Debt-to-Income (%)", 0.0, 60.0, 20.0)
borrower["inq_last_6mths"] = numeric_input("Inquiries (Last 6 Months)", 1, 0, 10)

borrower["revol_util"] = st.sidebar.number_input(
    "Revolving Utilization (%)", 0.0, 150.0, 40.0
)

# ✅ MISSING LR FEATURES (NOW FIXED)
borrower["bc_util"] = st.sidebar.number_input(
    "Bankcard Utilization (%)", 0.0, 150.0, 35.0
)

borrower["percent_bc_gt_75"] = st.sidebar.number_input(
    "% Bankcards with Utilization > 75%", 0.0, 100.0, 20.0
)

borrower["acc_open_past_24mths"] = numeric_input(
    "Accounts Opened (Last 24 Months)", 2, 0, 20
)

borrower["mo_sin_rcnt_tl"] = numeric_input(
    "Months Since Recent Trade", 9, 0, 300
)

borrower["mths_since_recent_inq"] = numeric_input(
    "Months Since Recent Inquiry", 6, 0, 300
)

borrower["credit_age_months"] = numeric_input(
    "Credit Age (Months)", 180, 0, 600
)

# -----------------------------
# XGB-only Inputs
# -----------------------------

borrower["grade"] = st.sidebar.selectbox(
    "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
)

borrower["sub_grade"] = st.sidebar.selectbox(
    "Loan Sub-Grade",
    ["A1","A2","A3","A4","A5",
     "B1","B2","B3","B4","B5",
     "C1","C2","C3","C4","C5",
     "D1","D2","D3","D4","D5"]
)

borrower["num_actv_rev_tl"] = numeric_input(
    "Active Revolving Trades", 4, 0, 20
)

borrower["mths_since_recent_bc"] = numeric_input(
    "Months Since Recent Bankcard", 18, 0, 300
)

borrower["avg_cur_bal"] = numeric_input(
    "Average Current Balance", 12000, 0, 500_000
)

borrower["mort_acc"] = numeric_input(
    "Mortgage Accounts", 1, 0, 10
)

borrower["total_bc_limit"] = numeric_input(
    "Total Bankcard Limit", 20000, 0, 200_000
)

borrower["delinq_2yrs"] = numeric_input(
    "Delinquencies (Last 2 Years)", 0, 0, 10
)


# ============================================================
# MAIN ACTION
# ============================================================

st.divider()

if st.button("🚀 Evaluate Borrower", use_container_width=True):

    results = run_champion_challenger(borrower)

    if results["agreement"]:
        st.success("✅ Both models agree on the final decision")
    else:
        st.warning("⚠️ Models disagree – manual review recommended")

    st.divider()

    col1, col2 = st.columns(2)

    # ----------------------------
    # Logistic Regression
    # ----------------------------
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
        reasons = get_reason_codes(woe_df, lr_model)

        st.write("❌ Risk Increasing Factors:", reasons["risk_increasing_factors"])
        st.write("✅ Risk Reducing Factors:", reasons["risk_reducing_factors"])

    # ----------------------------
    # XGBoost
    # ----------------------------
    with col2:
        st.subheader("📗 XGBoost (Challenger)")
        xgb = results["xgboost"]

        st.metric("PD (%)", xgb["pd_percent"])
        st.metric("Credit Score", xgb["score"])
        st.metric("Risk Band", xgb["risk_band"])
        st.markdown(f"**Decision:** `{xgb['decision']}`")
        st.caption(xgb["reason"])
