from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Load the model and columns we saved earlier
model = joblib.load('credit_risk_model.pkl')
model_columns = joblib.load('model_columns.pkl')

app = FastAPI(title="Credit Risk API")

# 2. Define the "Shape" of the incoming data
class LoanApplication(BaseModel):
    person_age: int
    person_emp_length: float
    loan_amnt: int
    loan_int_rate: float
    loan_grade: int
    log_person_income: float
    loan_percent_income: float
    # Add any other columns used in your X_train here

@app.post("/predict")
def predict(data: LoanApplication):
    # Convert incoming JSON to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Ensure columns match exactly what the model saw during training
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Get probability
    prob = model.predict_proba(input_df)[:, 1][0]
    
    # Apply your tuned 0.31 threshold
    decision = "REJECTED" if prob >= 0.31 else "APPROVED"
    
    return {
        "status": decision,
        "probability_of_default": round(prob, 4),
        "threshold_used": 0.31
    }