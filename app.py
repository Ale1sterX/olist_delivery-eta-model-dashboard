import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.freq_map_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in X.columns:
            X_out[col] = X[col].map(self.freq_map_[col]).fillna(0)
        return X_out

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="SLA-aware Delivery ETA Prediction System",
    layout="wide"
)

st.title("📦 SLA-aware Delivery ETA Prediction System")
st.caption(
    "Final Model: XGBoost Quantile Regression (α = 0.94) · "
    "Output represents an SLA-safe delivery estimate (p94)"
)

# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists("xgb_quantile_p94.pkl"):
        st.error("Model file xgb_quantile_p94.pkl tidak ditemukan di repository.")
        st.stop()
    return joblib.load("xgb_quantile_p94.pkl")

model = load_model()

# =====================
# EXACT feature schema (from feature_names_in_)
# =====================
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
    "distance_km"
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
    "distance_km": "float64"
}

# =====================
# Prediction section
# =====================
st.subheader("🔮 SLA-safe Delivery Time Prediction (p94)")

st.markdown(
    """
    Masukkan karakteristik pesanan untuk memperoleh estimasi waktu pengiriman
    yang **bersifat SLA-safe**, yaitu estimasi waktu sehingga sekitar **94% pesanan
    dengan karakteristik serupa diperkirakan selesai pada atau sebelum waktu ini**.
    """
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.slider("Distance (km)", 0, 2000, 300)
        price = st.slider("Product Price", 0.0, 500.0, 100.0)
        product_weight_g = st.slider("Product Weight (g)", 0, 10000, 1000)

    with col2:
        product_length_cm = st.slider("Product Length (cm)", 0, 100, 30)
        product_height_cm = st.slider("Product Height (cm)", 0, 100, 20)
        product_width_cm = st.slider("Product Width (cm)", 0, 100, 20)

    submitted = st.form_submit_button("Predict SLA-safe ETA")

# =====================
# Inference
# =====================
if submitted:
    # Default-safe input template
    input_dict = {
        "customer_state": "SP",                     # default state
        "order_item_id": 1,                         # dummy identifier
        "price": price,
        "product_weight_g": product_weight_g,
        "product_length_cm": product_length_cm,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
        "product_category_name_english": "others",  # safe default
        "is_weekend": 0,                            # assume weekday
        "same_state": 1,                            # assume same state
        "distance_km": distance_km
    }

    # Ensure all required features exist
    input_df = pd.DataFrame([input_dict])[MODEL_FEATURES]

    # Force correct dtypes (IMPORTANT)
    for col, dtype in FEATURE_DTYPES.items():
        input_df[col] = input_df[col].astype(dtype)

    p94_eta = model.predict(input_df)[0]

    st.success(
        f"📦 SLA-safe delivery estimate (p94): **{p94_eta:.1f} days**"
    )

    st.caption(
        "Interpretation: sekitar 94% pesanan historis dengan karakteristik serupa "
        "diperkirakan akan terkirim pada atau sebelum waktu ini."
    )

    st.caption(
        "Catatan: fitur yang tidak diisi pengguna diasumsikan bernilai default "
        "sesuai konfigurasi sistem."
    )

# =====================
# Footer
# =====================
st.markdown("---")
st.caption("Delivery ETA Prediction System · Quantile Regression p94 · Streamlit")
