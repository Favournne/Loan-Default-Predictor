from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, 'loan_risk_pipeline.pkl')

print("=== STARTING ROBUST PRODUCTION PIPELINE ENGINE ===")
# This will now load perfectly because it contains only standard scikit-learn objects!
pipeline = joblib.load(pipeline_path)

app = FastAPI(title="Credit Risk API - Production")

class LoanApplication(BaseModel):
    person_age: int
    person_emp_length: float
    loan_amnt: int
    loan_int_rate: float
    person_income: float
    person_home_ownership: str       # e.g., "RENT", "MORTGAGE", "OWN"
    loan_intent: str                 # e.g., "PERSONAL", "EDUCATION"
    cb_person_default_on_file: str   # e.g., "Y", "N"

@app.post("/predict")
def predict(data: LoanApplication):
    #  Safely parse incoming data into a python dictionary
    input_data = data.model_dump()

    input_data['person_home_ownership'] = input_data['person_home_ownership'].upper()
    input_data['loan_intent'] = input_data['loan_intent'].upper()
    input_data['cb_person_default_on_file'] = input_data['cb_person_default_on_file'].upper()     

    #  Perform the feature engineering calculations safely on-the-fly
    raw_income = input_data['person_income']
    raw_loan_amount = input_data['loan_amnt']
    
    input_data['log_person_income'] = np.log(raw_income)
    input_data['loan_percent_income'] = float(raw_loan_amount / raw_income)
    
    # 3. Convert to DataFrame (The pipeline transformer will pass numericals and handle one-hot categories)
    input_df = pd.DataFrame([input_data])
    
    # 4. Hand data off to the pipeline securely
    prob = pipeline.predict_proba(input_df)[:, 1][0]
    
    # 5. Apply operational decision threshold
    decision = "REJECTED" if prob >= 0.31 else "APPROVED"
    
    return { 
        "status": decision,
        "probability_of_default": round(float(prob), 4),
        "threshold_used": 0.31,
        "internal_calculations": {
            "annual_income_used": raw_income,
            "calculated_dti": round(input_data['loan_percent_income'], 4)
        }
    }