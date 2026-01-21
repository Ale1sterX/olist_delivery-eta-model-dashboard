import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================================================
# Custom Transformer (WAJIB ADA, IDENTIK SAAT TRAINING)
# ======================================================
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map_ = {}

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        for col in X_df.columns:
            self.freq_map_[col] = X_df[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_df = self._to_df(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].map(self.freq_map_[col]).fillna(0)
        return X_df

    def _to_df(self, X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return X


# ======================================================
# Streamlit Config
# ======================================================
st.set_page_config(
    page_title="SLA-aware Delivery ETA Prediction System",
    layout="wide"
)

st.title("📦 SLA-aware Delivery ETA Prediction System")
st.caption(
    "Final Model: XGBoost Quantile Regression (p94). "
    "Output merepresentasikan estimasi ETA yang bersifat SLA-safe."
)

# ======================================================
# Load Model
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_quantile_p94.pkl")

model = load_model()

# ======================================================
# Feature Contract (HARUS IDENTIK)
# ======================================================
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

# Default values (PASTI ADA di training data)
DEFAULT_FEATURE_VALUES = {
    "customer_state": "SP",
    "product_category_name_english": "bed_bath_table",
    "order_item_id": 1,
    "is_weekend": 0,
    "same_state": 1,
    "distance_km": 300.0,
    "product_length_cm": 30.0,
}

# ======================================================
# UI Inputs (User-facing)
# ======================================================
st.subheader("🔮 SLA-safe Delivery Time Prediction (p94)")

col1, col2 = st.columns(2)

with col1:
    price = st.slider("Product Price", 0.0, 5000.0, 300.0)
    product_weight_g = st.slider("Product Weight (g)", 0.0, 30000.0, 7000.0)

with col2:
    product_height_cm = st.slider("Product Height (cm)", 0.0, 200.0, 50.0)
    product_width_cm = st.slider("Product Width (cm)", 0.0, 200.0, 20.0)

# ======================================================
# Prediction Logic
# ======================================================
if st.button("Predict SLA-safe ETA"):
    # 1. Input dari user
    input_dict = {
        "price": price,
        "product_weight_g": product_weight_g,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
    }

    # 2. Isi semua fitur lain pakai default dataset
    for col in MODEL_FEATURES:
        if col not in input_dict:
            input_dict[col] = DEFAULT_FEATURE_VALUES[col]

    # 3. Bangun DataFrame SESUAI URUTAN TRAINING
    input_df = pd.DataFrame([input_dict])[MODEL_FEATURES]

    # (DEBUG OPTIONAL – boleh dihapus kalau sudah yakin)
    # st.write("Input to model:", input_df)

    # 4. Predict (pipeline handle semuanya)
    p94_eta = model.predict(input_df)[0]

    st.success(f"📦 Predicted SLA-safe ETA (p94): **{p94_eta:.1f} days**")
    st.caption(
        "Interpretasi: sekitar 94% pesanan dengan karakteristik serupa "
        "diperkirakan akan tiba pada atau sebelum waktu ini."
    )
