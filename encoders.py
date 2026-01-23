# encoders.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_ = None
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        # Handle numpy input
        if isinstance(X, np.ndarray):
            self.columns_ = [f"col_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.columns_)
        else:
            self.columns_ = X.columns.tolist()

        for col in self.columns_:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)

        X_out = X.copy()
        for col in self.columns_:
            X_out[col] = X_out[col].map(self.freq_maps_[col]).fillna(0)

        return X_out
