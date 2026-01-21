import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Custom Transformer (REQUIRED)
# ===============================
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map_ = {}

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        for col in X_df.columns:
            self.freq_map_[col] = X_df[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].map(self.freq_map_[col]).fillna(0)
        return X_df

    def _to_dataframe(self, X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return X


# ===============================
# App Config
# ===============================
st.set_page_config(
    page_title="SLA-aware Delivery ETA Prediction System",
    layout="wide"
)

st.title("📦 SLA-aware Delivery ETA Prediction System")
st.caption(
    "Final Model: XGBoost Quantile Regression (p94) — output merepresentasikan estimasi ETA yang SLA-safe"
)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_quantile_p94.pkl")

model = load_model()

# ===============================
# Feature Contract (EXACT)
# ===============================
MODEL_FEATURES = [
    "customer_state",
    "order_item_id",
    "price",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "product_category_name_english",
    "is_weekend",
    "same_state",
    "distance_km",
]

FEATURE_DTYPES = {
    "customer_state": "object",
    "order_item_id": "int64",
    "price": "float64",
    "product_weight_g": "float64",
    "product_length_cm": "float64",
    "product_height_cm": "float64",
    "product_width_cm": "float64",
    "product_category_name_english": "object",
    "is_weekend": "int64",
    "same_state": "int64",
    "distance_km": "float64",
}

# Default values MUST exist in training data
DEFAULT_FEATURE_VALUES = {
    "customer_state": "SP",
    "product_category_name_english": "bed_bath_table",
    "order_item_id": 1,
    "is_weekend": 0,
    "same_state": 1,
    "distance_km": 300.0,
    "product_length_cm": 30.0,
}

# ===============================
# UI — User Inputs
# ===============================
st.subheader("🔮 SLA-safe Delivery Time Prediction (p94)")

col1, col2 = st.columns(2)

with col1:
    price = st.slider("Product Price", 0.0, 5000.0, 300.0)
    product_weight_g = st.slider("Product Weight (g)", 0.0, 30000.0, 7000.0)

with col2:
    product_height_cm = st.slider("Product Height (cm)", 0.0, 200.0, 50.0)
    product_width_cm = st.slider("Product Width (cm)", 0.0, 200.0, 20.0)

# ===============================
# Prediction
# ===============================
if st.button("Predict SLA-safe ETA"):
    # user-provided inputs
    input_dict = {
        "price": price,
        "product_weight_g": product_weight_g,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
    }

    # fill missing features with dataset defaults
    for col in MODEL_FEATURES:
        if col not in input_dict:
            input_dict[col] = DEFAULT_FEATURE_VALUES[col]

    # build dataframe in correct order
    input_df = pd.DataFrame([input_dict])[MODEL_FEATURES]

    # enforce dtypes (CRITICAL)
    for col, dtype in FEATURE_DTYPES.items():
        input_df[col] = input_df[col].astype(dtype)

    # predict
    p94_eta = model.predict(input_df)[0]

    st.success(f"📦 Predicted SLA-safe ETA (p94): **{p94_eta:.1f} days**")
    st.caption(
        "Artinya: ~94% pesanan dengan karakteristik serupa diperkirakan akan tiba "
        "pada atau sebelum estimasi waktu ini."
    )
