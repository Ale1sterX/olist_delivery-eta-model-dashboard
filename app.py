import joblib
import streamlit as st
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
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.freq_map_[col]).fillna(0)
        return X_transformed

@st.cache_resource
def load_model():
    return joblib.load("xgb_quantile_p94.pkl")

model = load_model()

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="Delivery ETA Model Dashboard",
    layout="wide"
)

st.title("📦 Delivery ETA Model Dashboard")
st.caption("Baseline vs ML Model • Tail Risk (p94) • Cost Implication")

# =====================
# Sidebar
# =====================
st.sidebar.header("⚙️ Configuration")

view_mode = st.sidebar.radio(
    "Pilih sistem",
    ["Baseline (Existing)", "ML Model (XGBoost Asymmetric)"]
)

cost_per_failed_delivery = st.sidebar.slider(
    "Biaya per late / failed delivery (USD)",
    min_value=5.0,
    max_value=40.0,
    value=17.2,
    step=0.5
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Test Set")

# =====================
# Metrics from notebook
# =====================
mae_baseline = 12.8
mae_model = 12.2

late_rate_baseline = 0.067
late_rate_model = 0.066

p94_model = 24.507785

# =====================
# Select metrics based on toggle
# =====================
if view_mode == "Baseline (Existing)":
    mae = mae_baseline
    late_rate = late_rate_baseline
    label = "Baseline (Existing System)"
else:
    mae = mae_model
    late_rate = late_rate_model
    label = "ML Model (XGBoost Asymmetric)"

# =====================
# Performance section
# =====================
st.subheader(f"📊 Performance Overview — {label}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE (days)", f"{mae:.2f}")

with col2:
    st.metric("Late Delivery Rate", f"{late_rate*100:.2f}%")

with col3:
    if view_mode == "ML Model (XGBoost Asymmetric)":
        st.metric("p94 Absolute Error (days)", f"{p94_model:.2f}")
    else:
        st.metric("p94 Absolute Error (days)", "N/A")

# =====================
# Cost simulation
# =====================
st.subheader("💰 Cost Simulation (Direct Operational Cost)")

n_orders = 200
late_orders = int(n_orders * late_rate)
total_cost = late_orders * cost_per_failed_delivery

st.markdown(
    f"""
    **Asumsi simulasi:**
    - Jumlah order: **{n_orders}**
    - Late delivery rate: **{late_rate*100:.2f}%**
    - Jumlah order terlambat: **{late_orders}**
    - Biaya per late delivery: **USD {cost_per_failed_delivery:.2f}**
    """
)

st.metric(
    "Estimated Direct Cost of Late Deliveries",
    f"USD {total_cost:,.2f}"
)

# =====================
# Interpretation
# =====================
st.subheader("🧠 Interpretation")

if view_mode == "Baseline (Existing)":
    st.markdown(
        """
        - Sistem existing menggunakan estimasi statis tanpa mekanisme adaptif.
        - MAE relatif lebih tinggi dan estimasi cenderung konservatif.
        - Tidak tersedia evaluasi tail risk eksplisit (p94) pada level observasi.
        """
    )
else:
    st.markdown(
        f"""
        - Model berhasil **menurunkan MAE** dibandingkan baseline.
        - **Late delivery rate tetap stabil**, menunjukkan tidak ada degradasi SLA.
        - Nilai **p94 ≈ {p94_model:.1f} hari** menunjukkan **tail risk tetap terkendali**.
        - Penggunaan asymmetric loss membantu mengelola risiko pada kasus ekstrem.
        """
    )

# =====================
# Footer
# =====================
st.caption("Delivery ETA ML Project • Streamlit Dashboard")
