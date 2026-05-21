import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# 1. Load the model and columns (Streamlit Cloud uses these files from your GitHub)
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, 'credit_risk_model.pkl'))
model_columns = joblib.load(os.path.join(base_path, 'model_columns.pkl'))

st.set_page_config(page_title="Credit Risk AI", layout="centered")

st.title("🏦 AI Loan Officer")
st.markdown("---")

# 2. Input Fields
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", value=10000, step=500)
    loan_grade = st.selectbox("Loan Grade (A=1, G=7)", [1, 2, 3, 4, 5, 6, 7])

with col2:
    emp_length = st.number_input("Years of Employment", value=5, min_value=0)
    int_rate = st.number_input("Interest Rate (%)", value=11.0)
    age = st.number_input("Applicant Age", value=25, min_value=18)

# 3. Prediction Logic
if st.button("Analyze Risk", use_container_width=True):
    # Prepare the features
    loan_percent_income = loan_amount / income
    log_income = np.log1p(income)
    
    # Create the DataFrame the model expects
    payload = {
        "person_age": age,
        "person_emp_length": float(emp_length),
        "loan_amnt": loan_amount,
        "loan_int_rate": int_rate,
        "loan_grade": loan_grade,
        "log_person_income": log_income,
        "loan_percent_income": loan_percent_income
    }
    
    input_df = pd.DataFrame([payload])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Run prediction directly from the loaded .pkl file
    prob = model.predict_proba(input_df)[:, 1][0]
    
    st.markdown("### Result:")
    # Using 0.31 as the threshold (adjust based on your best model performance)
    if prob > 0.31:
        st.error(f"❌ **REJECTED** (Risk Score: {prob:.2%})")
        st.write("This applicant has a high probability of defaulting on the loan.")
    else:
        st.success(f"✅ **APPROVED** (Risk Score: {prob:.2%})")
        st.write("This applicant meets the credit safety requirements.")
