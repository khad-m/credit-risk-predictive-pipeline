from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Initialise application and load serialised artifacts
app = FastAPI(
    title="Credit Risk Predictive Engine",
    description="Production microservice API serving machine learning risk classifications.",
    version="1.0.0"
)

try:
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('credit_feature_scaler.pkl')
except FileNotFoundError:
    raise RuntimeError("Production model artifacts missing. Execute train.py prior to server launch.")

# Enforce rigorous datatype validation boundaries at the network edge
class CustomerDataInput(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Customer age in years.")
    annual_income: float = Field(..., ge=0, description="Gross annual income in GBP.")
    credit_score: int = Field(..., ge=300, le=850, description="Standardised credit scoring value.")
    debt_to_income_ratio: float = Field(..., ge=0.0, le=1.0, description="Proportional debt service ratio.")
    missed_payments_last_2_years: int = Field(..., ge=0, description="Frequency of historic default flags.")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 34,
                "annual_income": 45000.0,
                "credit_score": 620,
                "debt_to_income_ratio": 0.28,
                "missed_payments_last_2_years": 0
            }
        }

@app.get("/")
def health_check():
    """Verify endpoint availability and subsystem state."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/api/v1/predict", status_code=200)
def predict_credit_risk(payload: CustomerDataInput):
    """
    Ingest customer profiles, perform inference routines, and return predictive risks.
    """
    try:
        # 1. Transform Pydantic payload into structured array
        raw_features = np.array([[
            payload.age,
            payload.annual_income,
            payload.credit_score,
            payload.debt_to_income_ratio,
            payload.missed_payments_last_2_years
        ]])

        # 2. Execute pipeline transformations using the saved scaling parameters
        scaled_features = scaler.transform(raw_features)

        # 3. Compute predictive classification probabilities
        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        
        high_risk_probability = float(probabilities[1])

        # 4. Generate structured analytical payload
        return {
            "is_high_risk_assessment": prediction,
            "risk_probability_score": round(high_risk_probability, 4),
            "risk_classification": "HIGH_RISK" if prediction == 1 else "LOW_RISK"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(e)}")
    