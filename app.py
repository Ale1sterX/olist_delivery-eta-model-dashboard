import streamlit as st
import pandas as pd
import numpy as np
import pickle
from math import radians, sin, cos, sqrt, asin
from encoders import FrequencyEncoder 

# Load model & data
with open("models/xgb_p94_model.pkl", "rb") as f:
    model = pickle.load(f)

zip_df = pd.read_csv(
    "olist_geolocation_dataset.csv",
    dtype={"geolocation_zip_code_prefix": str}
)

zip_df = (
    zip_df
    .groupby("geolocation_zip_code_prefix", as_index=False)
    .agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
        "geolocation_state": "first"
    })
    .set_index("geolocation_zip_code_prefix")
)

state_freq_df = pd.read_csv("state_frequency.csv")


state_freq_map = dict(
    zip(state_freq_df.customer_state, state_freq_df.freq_customer_state)
)

# Distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))

category_list = (
    pd.read_csv("olist_combined_dataset.csv")
    ["product_category_name_english"]
    .dropna()
    .unique()
)

# UI
st.title("Smart ETA Prediction")
st.caption("Risk-aware delivery estimation")

st.subheader("Customer & Seller")

customer_zip = st.text_input("Customer ZIP Code")
seller_zip = st.text_input("Seller ZIP Code")
customer_state = st.selectbox(
    "Customer State",
    state_freq_df.customer_state
)

st.subheader("Product Info")
product_category = st.selectbox(
    "Product Category",
    sorted(category_list)
)
price = st.number_input("Price (R$)", min_value=0.0)
product_weight_g = st.number_input("Product Weight (g)", min_value=0.0)
product_length_cm = st.number_input("Length (cm)", min_value=0.0)
product_height_cm = st.number_input("Height (cm)", min_value=0.0)
product_width_cm = st.number_input("Width (cm)", min_value=0.0)
order_item_id = st.number_input("Product Quantity", min_value=1)

is_weekend = st.checkbox("Order on Weekend")

# Prediction
if st.button("Predict ETA"):

    if customer_zip not in zip_df.index or seller_zip not in zip_df.index:
        st.error("ZIP code not found")
    else:
        cust = zip_df.loc[customer_zip]
        sell = zip_df.loc[seller_zip]

        distance_km = haversine(
            cust["geolocation_lat"],
            cust["geolocation_lng"],
            sell["geolocation_lat"],
            sell["geolocation_lng"]
        )

        same_state = int(
            cust["geolocation_state"] == sell["geolocation_state"]
        )

        input_df = pd.DataFrame([{
            "customer_state": customer_state,
            "product_category_name_english": product_category,
            "order_item_id": order_item_id,
            "price": price,
            "product_weight_g": product_weight_g,
            "product_length_cm": product_length_cm,
            "product_height_cm": product_height_cm,
            "product_width_cm": product_width_cm,
            "is_weekend": int(is_weekend),
            "same_state": same_state,
            "distance_km": distance_km,
        }])

        pred = model.predict(input_df)[0]

        st.success(f"Estimated Delivery Time: **{pred:.1f} days**")
