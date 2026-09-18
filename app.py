
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model and preprocessing objects
# -----------------------------
model = joblib.load("osteoporosis_xgb_model.pkl")
scaler = joblib.load("osteoporosis_scaler.pkl")
model_features = joblib.load("osteoporosis_features.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Osteoporosis Risk Assessment",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Osteoporosis Risk Assessment")
st.write(
    "A prototype clinical decision-support tool for estimating "
    "osteoporosis risk based on patient characteristics."
)

st.info(
    "This is an AI/ML prototype for educational and demonstration purposes. "
    "It is not a diagnostic tool and should not replace clinical assessment."
)


# -----------------------------
# Patient Information
# -----------------------------
st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=None,
    placeholder="Enter age"
)

gender = st.selectbox(
    "Gender",
    ["", "Female", "Male"]
)

hormonal_changes = st.selectbox(
    "Hormonal Changes",
    ["", "Premenopausal", "Postmenopausal"]
)

family_history = st.selectbox(
    "Family History of Osteoporosis",
    ["", "No", "Yes"]
)

body_weight = st.selectbox(
    "Body Weight",
    ["", "Normal", "Underweight", "Overweight"]
)

calcium = st.selectbox(
    "Calcium Intake",
    ["", "Adequate", "Low"]
)

vitamin_d = st.selectbox(
    "Vitamin D Intake",
    ["", "Insufficient", "Sufficient"]
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["", "Active", "Sedentary"]
)

smoking = st.selectbox(
    "Smoking",
    ["", "No", "Yes"]
)

alcohol = st.selectbox(
    "Alcohol Consumption",
    ["", "No", "Yes", "Not Reported"]
)

medical_conditions = st.selectbox(
    "Medical Conditions",
    ["", "None", "Rheumatoid Arthritis", "Not Reported"]
)

medications = st.selectbox(
    "Medications",
    ["", "None", "Not Reported"]
)

prior_fractures = st.selectbox(
    "Prior Fractures",
    ["", "No", "Yes"]
)


# -----------------------------
# Risk Assessment
# -----------------------------
if st.button("Assess Osteoporosis Risk"):

    if age is None or "" in [
        gender,
        hormonal_changes,
        family_history,
        body_weight,
        calcium,
        vitamin_d,
        physical_activity,
        smoking,
        alcohol,
        medical_conditions,
        medications,
        prior_fractures
    ]:
        st.warning("Please complete all patient fields before assessing risk.")

    else:

        # Create patient dataframe
        patient = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Hormonal Changes": hormonal_changes,
            "Family History": family_history,
            "Body Weight": body_weight,
            "Calcium Intake": calcium,
            "Vitamin D Intake": vitamin_d,
            "Physical Activity": physical_activity,
            "Smoking": smoking,
            "Alcohol Consumption": alcohol,
            "Medical Conditions": medical_conditions,
            "Medications": medications,
            "Prior Fractures": prior_fractures
        }])

        # Match missing-value handling used during training
        patient = patient.fillna("Not Reported")

        # One-hot encode categorical variables
        categorical_cols = patient.select_dtypes(
            include="object"
        ).columns

        patient_encoded = pd.get_dummies(
            patient,
            columns=categorical_cols,
            drop_first=True
        )

      # Add any model features that are missing
      for feature in model_features:
      if feature not in patient_encoded.columns:
        patient_encoded[feature] = 0.0

      # Keep exactly the features used by the model
      patient_processed = patient_encoded[model_features].copy()

       # Convert all model inputs to numeric values
       patient_processed = patient_processed.astype(float)

        # Scale Age using the same scaler used during training
        patient_processed["Age"] = scaler.transform(
            patient_processed[["Age"]]
        )

        # Prediction
        probability = model.predict_proba(
            patient_processed
        )[0, 1]

        prediction = model.predict(
            patient_processed
        )[0]

        # -----------------------------
        # Display result
        # -----------------------------
        st.header("Risk Assessment")

        st.metric(
            "Estimated Osteoporosis Probability",
            f"{probability * 100:.1f}%"
        )

        if probability < 0.30:
            risk_category = "Low Risk"
        elif probability < 0.60:
            risk_category = "Moderate Risk"
        else:
            risk_category = "High Risk"

        if risk_category == "Low Risk":
            st.success(risk_category)
        elif risk_category == "Moderate Risk":
            st.warning(risk_category)
        else:
            st.error(risk_category)

        st.write(
            "**Model classification:**",
            "Osteoporosis" if prediction == 1 else "No Osteoporosis"
        )

        st.caption(
            "Risk categories are prototype thresholds for demonstration "
            "and are not clinical diagnostic thresholds."
        )
