
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("landslide_risk_model.pkl")
encoder = joblib.load("landslide_encoder.pkl")

st.set_page_config(
    page_title="AI Landslide Risk Monitoring",
    page_icon="🌏",
    layout="centered"
)

st.title("🌏 AI-Based Landslide Risk Monitoring")
st.write("Northeast Region of India")

st.divider()

states = [
    "Assam",
    "Arunachal Pradesh",
    "Meghalaya",
    "Manipur",
    "Mizoram",
    "Nagaland",
    "Tripura",
    "Sikkim"
]

triggers = [
    "downpour",
    "rain",
    "monsoon",
    "unknown"
]

settings = [
    "unknown",
    "above road",
    "below road",
    "natural slope",
    "urban area"
]

state = st.selectbox("State", states)
trigger = st.selectbox("Landslide Trigger", triggers)
setting = st.selectbox("Landslide Setting", settings)

latitude = st.number_input(
    "Latitude",
    value=26.1548,
    format="%.4f"
)

longitude = st.number_input(
    "Longitude",
    value=91.7351,
    format="%.4f"
)

if st.button("🚨 Predict Risk"):

    input_data = pd.DataFrame([{
        "admin_division_name": state,
        "landslide_trigger": trigger,
        "landslide_setting": setting,
        "latitude": latitude,
        "longitude": longitude
    }])

    categorical_features = [
        "admin_division_name",
        "landslide_trigger",
        "landslide_setting"
    ]

    numeric_features = [
        "latitude",
        "longitude"
    ]

    encoded_cat = encoder.transform(
        input_data[categorical_features]
    )

    encoded_num = input_data[numeric_features].to_numpy()

    encoded_input = np.hstack([
        encoded_cat,
        encoded_num
    ])

    prediction = model.predict(encoded_input)[0]
    probabilities = model.predict_proba(encoded_input)[0]

    st.subheader("Risk Prediction")

    if prediction == "High":
        st.error("🔴 HIGH RISK")
    elif prediction == "Medium":
        st.warning("🟠 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")

    st.subheader("Risk Probability")

    probability_df = pd.DataFrame({
        "Risk Level": model.classes_,
        "Probability": probabilities
    })

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.bar_chart(
        probability_df.set_index("Risk Level")
    )

    st.dataframe(
        probability_df,
        use_container_width=True
    )

st.divider()

st.caption(
    "Prototype model trained on historical landslide data from the Northeast Region of India."
)
