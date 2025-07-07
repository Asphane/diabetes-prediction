import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load pipeline and features list
model = joblib.load("diabetes_model_pipeline.pkl")
features = joblib.load("features_used.pkl")

st.write("📋 Features used for prediction:")
st.write(features)


# Title and intro
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's medical information below:")

# Set realistic defaults
default_values = {
    "Pregnancies": 1,
    "Glucose": 100,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}

# Create input fields dynamically
input_data = []
for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, value=float(default_values.get(feature, 0.0)), format="%.2f")
    input_data.append(value)

# Predict button
if st.button("Predict"):
    # Create DataFrame with input features
    input_df = pd.DataFrame([input_data], columns=features)

    st.write("🧾 Input DataFrame")
    st.dataframe(input_df)

    # Make prediction
    st.write("🔍 Raw prediction probabilities:")
    st.write(model.predict_proba(input_df))

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display results
    if prediction == 1:
        st.error(f"🩸 High risk of diabetes.\n\nProbability: **{probability:.2%}**")
    else:
        st.success(f"✅ Low risk of diabetes.\n\nProbability: **{probability:.2%}**")
