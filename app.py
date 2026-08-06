
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title
st.title("Diabetes Prediction using Logistic Regression")

st.write("Enter the patient details below:")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)

glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)

blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)

bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction Button
if st.button("Predict"):

    # Create dataframe from user input
    input_data = pd.DataFrame({
        "Pregnancies":[pregnancies],
        "Glucose":[glucose],
        "BloodPressure":[blood_pressure],
        "SkinThickness":[skin_thickness],
        "Insulin":[insulin],
        "BMI":[bmi],
        "DiabetesPedigreeFunction":[diabetes_pedigree],
        "Age":[age]
    })

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    # Display Result
    if prediction[0] == 1:
        st.error("Patient is likely to have Diabetes.")
    else:
        st.success("Patient is unlikely to have Diabetes.")

    st.write("Probability of Diabetes:", round(probability[0][1]*100,2), "%")
