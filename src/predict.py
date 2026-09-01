"""Prediction helper for a saved fraud-detection model."""

from pathlib import Path

import joblib
import pandas as pd


def predict_fraud(feature_data: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    """Return a fraud prediction and probability for each feature row."""
    artifact = joblib.load(model_path)
    expected_columns = artifact["feature_columns"]
    missing_columns = set(expected_columns) - set(feature_data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing expected columns: {sorted(missing_columns)}")
    features = feature_data.reindex(columns=expected_columns)

    probabilities = artifact["model"].predict_proba(features)[:, 1]
    return pd.DataFrame({
        "fraud_probability": probabilities,
        "fraud_prediction": (probabilities >= artifact["threshold"]).astype(int),
    }, index=feature_data.index)
