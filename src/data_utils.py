"""Small, reusable helpers for preparing the credit-card fraud dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_features_target(df):
    """Separate between X and y"""
    return df.drop("Class", axis=1), df["Class"]