import streamlit as st
import pandas as pd
import joblib

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
    "Output represents an SLA-safe delivery estimate"
)

# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    return joblib.load("xgb_quantile_p94.pkl")

model = load_model()

# =====================
# Sidebar: system settings
# =====================
st.sidebar.header("⚙️ System Settings")

cost_per_failed_delivery = st.sidebar.slider(
    "Biaya per late / failed delivery (USD)",
    min_value=5.0,
    max_value=40.0,
    value=17.2,
    step=0.5
)

st.sidebar.markdown("---")
st.sidebar.caption("Model output = p94 (upper bound / SLA-safe)")

# =====================
# Section 1: Prediction system
# =====================
st.subheader("🔮 SLA-safe Delivery Time Prediction (p94)")

st.markdown(
    """
    Masukkan karakteristik pesanan untuk memperoleh estimasi waktu pengiriman
    **yang bersifat SLA-safe**, yaitu estimasi waktu sehingga sekitar **94% pesanan
    dengan karakteristik serupa diperkirakan selesai pada atau sebelum waktu ini**.
    """
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.slider("Distance (km)", 0, 2000, 300)
        freight_value = st.slider("Freight Value", 0.0, 200.0, 40.0)

    with col2:
        seller_avg_delay = st.slider("Seller Average Delay (days)", 0.0, 10.0, 2.0)
        customer_state_encoded = st.number_input(
            "Customer State (encoded)",
            min_value=0,
            max_value=30,
            value=5
        )

    submitted = st.form_submit_button("Predict SLA-safe ETA")

if submitted:
    input_df = pd.DataFrame({
        "distance_km": [distance_km],
        "freight_value": [freight_value],
        "seller_avg_delay": [seller_avg_delay],
        "customer_state_encoded": [customer_state_encoded]
    })

    p94_eta = model.predict(input_df)[0]

    st.success(
        f"📦 SLA-safe delivery estimate (p94): **{p94_eta:.1f} days**"
    )

    st.caption(
        "Interpretation: sekitar 94% pesanan historis dengan karakteristik serupa "
        "diperkirakan akan terkirim pada atau sebelum waktu ini."
    )

# =====================
# Section 2: Historical benchmark
# =====================
st.subheader("📊 Historical Benchmark (Contextual Comparison)")

late_rate_baseline = 0.067
late_rate_model = 0.066

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Late Delivery Rate — Existing System",
        f"{late_rate_baseline*100:.2f}%"
    )

with col2:
    st.metric(
        "Late Delivery Rate — Quantile p94 Model",
        f"{late_rate_model*100:.2f}%",
        delta=f"{(late_rate_baseline - late_rate_model)*100:.2f}%"
    )

st.caption(
    "Benchmark ini dihitung dari data historis dan ditampilkan sebagai konteks. "
    "Nilai ini **tidak merepresentasikan prediksi individual** untuk input di atas."
)

# =====================
# Section 3: Cost implication
# =====================
st.subheader("💰 Cost Implication (Illustrative Simulation)")

n_orders = st.slider(
    "Jumlah order (simulasi periode tertentu)",
    min_value=50,
    max_value=500,
    value=200,
    step=10
)

late_orders_baseline = int(n_orders * late_rate_baseline)
late_orders_model = int(n_orders * late_rate_model)

cost_baseline = late_orders_baseline * cost_per_failed_delivery
cost_model = late_orders_model * cost_per_failed_delivery

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Late Orders (Baseline)", late_orders_baseline)

with col2:
    st.metric("Late Orders (Model)", late_orders_model)

with col3:
    st.metric(
        "Estimated Cost Reduction",
        f"USD {cost_baseline - cost_model:,.2f}"
    )

st.caption(
    "Biaya dihitung menggunakan estimasi biaya langsung per keterlambatan "
    "(mis. redelivery, handling, dan customer service)."
)

# =====================
# Section 4: System interpretation
# =====================
st.subheader("🧠 System Interpretation")

st.markdown(
    """
    - Sistem ini menggunakan **Quantile Regression (α = 0.94)** untuk menghasilkan
      estimasi waktu pengiriman yang berorientasi pada SLA, bukan estimasi rata-rata.
    - Pendekatan ini dirancang untuk **mengendalikan risiko keterlambatan**, terutama
      pada kasus ekstrem.
    - Perbandingan dengan sistem existing dilakukan pada level historis untuk menjaga
      interpretasi tetap valid dan tidak menyesatkan pada level individual.
    """
)

# =====================
# Footer
# =====================
st.caption(
    "Delivery ETA Prediction System · Quantile Regression p94 · Streamlit"
)
