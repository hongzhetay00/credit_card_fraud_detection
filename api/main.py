from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraudulent credit card transactions.",
    version="1.0"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "Models" / "xgboost_best.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "Models" / "xgboost_threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API"
    }

# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    # Convert incoming JSON into a one-row DataFrame
    input_data = pd.DataFrame([
        transaction.model_dump()
    ])

    # Get probability of Class 1 = Fraud
    fraud_probability = model.predict_proba(
        input_data
    )[:, 1][0]

    # Apply the threshold selected in Notebook 05
    prediction = int(
        fraud_probability >= threshold
    )

    # Make result easier to read
    label = "Fraud" if prediction == 1 else "Legitimate"

    return {
        "fraud_probability": float(fraud_probability),
        "prediction": prediction,
        "label": label
    }