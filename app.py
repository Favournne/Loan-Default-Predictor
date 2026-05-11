import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os

base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, 'credit_risk_model.pkl'))
model_columns = joblib.load(os.path.join(base_path, 'model_columns.pkl'))

st.set_page_config(page_title="Credit Risk Ai", layout="centered")

# Set page title
st.title("🏦 AI Loan Officer")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", value=10000, step=500)
    loan_grade = st.selectbox("Loan Grade (A=1, G=7)", [1, 2, 3, 4, 5, 6, 7])

with col2:
    emp_length = st.number_input("Years of Employment", value=5, min_value=0)
    int_rate = st.number_input("Interest Rate (%)", value=11.0)
    age = st.number_input("Applicant Age", value=25, min_value=18)

# Calculate the features the model needs
loan_percent_income = loan_amount / income
import numpy as np
log_income = np.log1p(income)

if st.button("Analyze Risk", use_container_width=True):
    # Data to send to your FastAPI (main.py)
    payload = {
        "person_age": age,
        "person_emp_length": float(emp_length),
        "loan_amnt": loan_amount,
        "loan_int_rate": int_rate,
        "loan_grade": loan_grade,
        "log_person_income": log_income,
        "loan_percent_income": loan_percent_income
    }
    
    try:
        # Link to your FastAPI running on port 8000
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        prob = result['probability_of_default']
        status = result['status']
        
        st.markdown("### Result:")
        if status == "APPROVED":
            st.success(f"✅ **APPROVED** (Risk Score: {prob:.2%})")
        else:
            st.error(f"❌ **REJECTED** (Risk Score: {prob:.2%})")
            
    except Exception as e:
        st.warning("Ensure your FastAPI server (main.py) is running on port 8000!")


if st.button("Analyze Risk"):
    # Create the DataFrame directly here
    input_df = pd.DataFrame([payload]) # Using the payload dictionary you made
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prob = model.predict_proba(input_df)[:, 1][0]
    # (Show your Approved/Rejected messages here)
