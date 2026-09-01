"""Reusable training and saving functions for the selected fraud model."""

from pathlib import Path

import joblib
import pandas as pd

from src.data_utils import clean_data, load_data, split_data, split_features_target


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and split data with the project's protected holdout rule."""
    clean_df = clean_data(load_data())
    X, y = split_features_target(clean_df)
    return split_data(X, y)


def train_and_save_model(model, threshold: float, output_path: str | Path) -> None:
    """Train an already-selected model on training data and save it with metadata."""
    X_train, _, y_train, _ = load_training_data()
    model.fit(X_train, y_train)
    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_columns": X_train.columns.tolist(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
