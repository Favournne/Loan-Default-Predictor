import streamlit as st
import requests

# Set up clean web page configuration
st.set_page_config(
    page_title="Automated Credit Risk Underwriter",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Automated Credit Risk Underwriter")
st.markdown("Enter the applicant's financial and credit profile below to query the production risk engine.")
st.markdown("---")

# 1. Layout the Input Form using columns for scannability
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Profile")
    person_age = st.number_input("Applicant Age", min_value=18, max_value=90, value=28, step=1)
    
    # Accept raw months in the UI to make onboarding seamless for your users
    emp_months = st.number_input("Employment Length (Total Months)", min_value=0, max_value=720, value=60, step=1)
    
    person_income = st.number_input("Annual Income ($)", min_value=1000, max_value=5000000, value=45000, step=500)
    
    person_home_ownership = st.selectbox(
        "Home Ownership Status",
        options=["MORTGAGE", "RENT", "OWN", "OTHER"],
        index=0
    )

with col2:
    st.subheader("💰 Loan Requests")
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=100, max_value=500000, value=5000, step=100)
    loan_int_rate = st.number_input("Offered Interest Rate (%)", min_value=0.5, max_value=35.0, value=10.5, step=0.1)
    
    loan_intent = st.selectbox(
        "Loan Purpose / Intent",
        options=["BUSINESS", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
        index=1
    )
    
    cb_person_default_on_file = st.selectbox(
        "Historical Default on File?",
        options=["N", "Y"],
        index=0
    )

st.markdown("---")

# 2. Trigger calculations and backend API calls on button click
if st.button("Evaluate Credit Application", type="primary"):
    
    # Conversions: Convert raw months input into decimal year values for the model
    person_emp_length = round(float(emp_months / 12.0), 2)
    
    # Construct the exact data payload expected by FastAPI main.py
    # (Notice we preserve uppercase formatting implicitly via selection options)
    payload = {
        "person_age": int(person_age),
        "person_emp_length": person_emp_length,
        "loan_amnt": int(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "person_income": float(person_income),
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "cb_person_default_on_file": cb_person_default_on_file
    }
    
    # Target path for your local running Uvicorn server application
    backend_url = "http://127.0.0.1:8000/predict"
    
    with st.spinner("Querying production underwriting ledger..."):
        try:
            response = requests.post(backend_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                status = result.get("status")
                prob = result.get("probability_of_default")
                dti = result.get("internal_calculations", {}).get("calculated_dti", 0.0)
                
                # Metric display blocks
                st.subheader("📋 Underwriting Verdict")
                
                if status == "APPROVED":
                    st.success(f"🎉 Application **APPROVED**")
                else:
                    st.error(f"❌ Application **REJECTED**")
                
                # Visualizing key underlying metrics side-by-side
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Risk Probability", f"{round(prob * 100, 2)}%")
                m_col2.metric("Calculated DTI", f"{round(dti * 100, 2)}%")
                m_col3.metric("Decision Threshold", "31.0%")
                
            else:
                st.error(f"Backend Server Error: Received Status Code {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Failed! Make sure your FastAPI backend is running on port 8000 (`python -m uvicorn main:app --reload`).")