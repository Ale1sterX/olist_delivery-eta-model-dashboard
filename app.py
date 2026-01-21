import streamlit as st
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# Custom Transformer (WAJIB untuk unpickle model)
# =====================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            freq_map = self.freq_maps_.get(col, {})
            X[col] = X[col].map(freq_map).fillna(0)
        return X


# =====================================================
# Load model (Pipeline)
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_quantile_p94.pkl")

model = load_model()


# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(
    page_title="SLA-aware Delivery ETA Prediction",
    layout="wide"
)

st.title("📦 SLA-aware Delivery ETA Prediction System")
st.caption(
    "Final Model: XGBoost Quantile Regression (p94) — "
    "Output merepresentasikan estimasi ETA yang SLA-safe"
)

st.markdown("---")

# =====================================================
# Input Form (HARUS SESUAI feature_names_in_)
# =====================================================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Product Price", value=300.0)
        product_weight_g = st.number_input("Product Weight (g)", value=7000.0)
        product_length_cm = st.number_input("Product Length (cm)", value=30.0)
        product_height_cm = st.number_input("Product Height (cm)", value=50.0)
        product_width_cm = st.number_input("Product Width (cm)", value=20.0)

    with col2:
        customer_state = st.number_input("Customer State (encoded)", value=10)
        order_item_id = st.number_input("Order Item ID", value=1)
        product_category_name_english = st.number_input(
            "Product Category Frequency",
            value=0.05,
            min_value=0.0,
            max_value=1.0
        )
        is_weekend = st.selectbox("Is Weekend?", [0, 1])
        same_state = st.selectbox("Same State (Seller & Customer)", [0, 1])
        distance_km = st.number_input("Distance (km)", value=1000.0)

    submit = st.form_submit_button("🚀 Predict SLA-safe ETA")


# =====================================================
# Prediction Logic
# =====================================================
if submit:
    input_df = pd.DataFrame([{
        "customer_state": customer_state,
        "order_item_id": order_item_id,
        "price": price,
        "product_weight_g": product_weight_g,
        "product_length_cm": product_length_cm,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
        "product_category_name_english": product_category_name_english,
        "is_weekend": is_weekend,
        "same_state": same_state,
        "distance_km": distance_km,
    }])

    try:
        eta_p94 = model.predict(input_df)[0]

        st.success(
            f"📦 **Predicted SLA-safe ETA (p94): {eta_p94:.2f} days**"
        )

        st.caption(
            "Interpretasi: sekitar 94% pesanan dengan karakteristik serupa "
            "diperkirakan akan selesai **pada atau sebelum** waktu ini."
        )

    except Exception as e:
        st.error("Terjadi error saat melakukan prediksi.")
        st.exception(e)
