
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("landslide_risk_model.pkl")
preprocessor = joblib.load("landslide_preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Landslide Risk Monitor",
    page_icon="🌏",
    layout="wide"
)

# Title
st.title("🌏 AI-Based Landslide Risk Monitoring")
st.write("AI-powered landslide risk prediction for the Northeast Region of India")

st.divider()

# Input section
st.subheader("🔍 Predict Landslide Risk")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox(
        "State",
        [
            "Assam",
            "Arunachal Pradesh",
            "Meghalaya",
            "Manipur",
            "Mizoram",
            "Nagaland",
            "Sikkim",
            "Tripura"
        ]
    )

    trigger = st.selectbox(
        "Landslide Trigger",
        [
            "downpour",
            "rain",
            "continuous_rain",
            "monsoon",
            "unknown"
        ]
    )

    setting = st.selectbox(
        "Landslide Setting",
        [
            "unknown",
            "above_road",
            "natural_slope"
        ]
    )

with col2:
    latitude = st.number_input(
        "Latitude",
        value=25.0,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        value=93.0,
        format="%.6f"
    )

# Prediction button
if st.button("🚨 Predict Risk", use_container_width=True):

    new_data = pd.DataFrame({
        "admin_division_name": [state],
        "landslide_trigger": [trigger],
        "landslide_setting": [setting],
        "latitude": [latitude],
        "longitude": [longitude]
    })

    encoded_data = preprocessor.transform(new_data)

    prediction = model.predict(encoded_data)[0]
    probabilities = model.predict_proba(encoded_data)[0]

    st.divider()

    st.subheader("📊 AI Prediction Result")

    if prediction == "High":
        st.error(f"🔴 HIGH RISK — {prediction}")
    elif prediction == "Medium":
        st.warning(f"🟡 MEDIUM RISK — {prediction}")
    else:
        st.success(f"🟢 LOW RISK — {prediction}")

    st.write("### Prediction Probability")

    probability_data = pd.DataFrame({
        "Risk Level": model.classes_,
        "Probability": probabilities
    })

    probability_data["Probability"] = (
        probability_data["Probability"] * 100
    ).round(2)

    st.bar_chart(
        probability_data.set_index("Risk Level")
    )

    st.dataframe(
        probability_data,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.caption(
    "AI-Based Landslide Risk Monitoring System | Northeast Region of India"
)
