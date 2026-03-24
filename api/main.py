from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

app = FastAPI(
    title="Insurance Risk Scoring API",
    description="MLOps pipeline prediction endpoint for insurance risk scoring",
    version="1.0.0"
)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models")
model = None
model_name = None
prediction_log = []


def load_best_model():
    """Load the best available model."""
    global model, model_name
    priority = ["lightgbm", "xgboost", "random_forest"]
    for name in priority:
        path = os.path.join(MODEL_PATH, f"{name}.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            model_name = name
            print(f"Loaded model: {name}")
            return
    print("No saved model found. Run train.py first.")


load_best_model()


# ── Request/Response Schemas ──────────────────────────────────────────────────
class InsuranceFeatures(BaseModel):
    age: int = Field(..., ge=18, le=85, description="Customer age")
    gender: str = Field(..., description="Male or Female")
    marital_status: str = Field(..., description="Single/Married/Divorced/Widowed")
    annual_income: int = Field(..., ge=10000, description="Annual income in currency")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    num_dependents: int = Field(..., ge=0, le=10)
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    years_with_insurer: int = Field(..., ge=0)
    previous_claims: int = Field(..., ge=0)
    claim_amount_history: int = Field(..., ge=0)
    policy_type: str = Field(..., description="Health/Auto/Life/Home/Travel")
    coverage_amount: int = Field(..., ge=0)
    policy_duration_years: int = Field(..., ge=1)
    num_policies: int = Field(..., ge=1)
    bmi: float = Field(..., ge=15, le=50)
    smoker: int = Field(..., ge=0, le=1)
    exercise_frequency: str = Field(..., description="Never/Rarely/Sometimes/Often/Daily")
    chronic_conditions: int = Field(..., ge=0, le=3)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45, "gender": "Male", "marital_status": "Married",
                "annual_income": 75000, "credit_score": 680,
                "num_dependents": 2, "debt_to_income": 0.3,
                "years_with_insurer": 5, "previous_claims": 1,
                "claim_amount_history": 15000, "policy_type": "Health",
                "coverage_amount": 500000, "policy_duration_years": 10,
                "num_policies": 2, "bmi": 26.5, "smoker": 0,
                "exercise_frequency": "Sometimes", "chronic_conditions": 1
            }
        }


class PredictionResponse(BaseModel):
    risk_score: float
    risk_category: str
    high_risk: bool
    confidence: float
    model_used: str
    timestamp: str
    recommendation: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Insurance Risk Scoring API",
        "version": "1.0.0",
        "model": model_name or "not loaded",
        "status": "healthy" if model else "model not loaded"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "total_predictions": len(prediction_log)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: InsuranceFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first."
        )

    try:
        # Convert to DataFrame
        data = pd.DataFrame([features.dict()])

        # Predict
        proba = model.predict_proba(data)[0]
        risk_score = float(proba[1])
        high_risk = risk_score >= 0.5
        confidence = float(max(proba))

        # Risk category
        if risk_score < 0.3:
            risk_category = "Low Risk"
            recommendation = "Standard premium. No additional review needed."
        elif risk_score < 0.5:
            risk_category = "Moderate Risk"
            recommendation = "Apply standard premium with moderate loading factor."
        elif risk_score < 0.7:
            risk_category = "High Risk"
            recommendation = "Apply high-risk premium loading. Request medical records."
        else:
            risk_category = "Very High Risk"
            recommendation = "Escalate for manual underwriter review. Consider denial or very high premium."

        result = {
            "risk_score": round(risk_score, 4),
            "risk_category": risk_category,
            "high_risk": high_risk,
            "confidence": round(confidence, 4),
            "model_used": model_name,
            "timestamp": datetime.now().isoformat(),
            "recommendation": recommendation
        }

        # Log prediction
        log_entry = {**features.dict(), **result}
        prediction_log.append(log_entry)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/history")
def get_history(limit: int = 100):
    """Get recent prediction history."""
    return {
        "total": len(prediction_log),
        "predictions": prediction_log[-limit:]
    }


@app.get("/model/info")
def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_name,
        "model_type": str(type(model.named_steps["model"]).__name__),
        "features": {
            "numeric": ["age", "annual_income", "credit_score", "num_dependents",
                        "debt_to_income", "years_with_insurer", "previous_claims",
                        "claim_amount_history", "coverage_amount",
                        "policy_duration_years", "num_policies", "bmi",
                        "smoker", "chronic_conditions"],
            "categorical": ["gender", "marital_status", "policy_type",
                            "exercise_frequency"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
