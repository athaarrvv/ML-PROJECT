import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. Page Setup
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💰 Data Science Salary Predictor")
st.write("Enter your details below to estimate your salary.")

# 2. Load the Model Assets
@st.cache_resource
def load_artifacts():
    file_path = "model/salary_model_artifacts.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

artifacts = load_artifacts()

# Stop if model is missing
if artifacts is None:
    st.error("❌ Error: 'salary_model_artifacts.pkl' not found in 'model/' folder.")
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_names = artifacts["features"]

# 3. Input Form (UI)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        work_year = st.number_input("Work Year", min_value=2020, max_value=2030, value=2024)
        experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"], 
                                      format_func=lambda x: {"EN": "Entry Level", "MI": "Mid Level", "SE": "Senior Level", "EX": "Executive"}[x])
        employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"], 
                                     format_func=lambda x: {"FT": "Full Time", "PT": "Part Time", "CT": "Contract", "FL": "Freelance"}[x])
        job_title = st.text_input("Job Title", "Data Scientist")

    with col2:
        employee_residence = st.text_input("Employee Residence (ISO Code)", "US")
        remote_ratio = st.selectbox("Remote Ratio", [0, 50, 100], 
                                  format_func=lambda x: {0: "On-Site", 50: "Hybrid", 100: "Remote"}[x])
        company_location = st.text_input("Company Location (ISO Code)", "US")
        company_size = st.selectbox("Company Size", ["S", "M", "L"], 
                                  format_func=lambda x: {"S": "Small", "M": "Medium", "L": "Large"}[x])

    # Submit Button
    submitted = st.form_submit_button("Predict Salary")

# 4. Prediction Logic (Runs only when button is clicked)
if submitted:
    try:
        # Initialize all 577 features to 0
        input_dict = {col: 0 for col in feature_names}

        # Set Numerical Values
        input_dict['work_year'] = work_year
        input_dict['remote_ratio'] = remote_ratio

        # Set Categorical Values (One-Hot Encoding Logic)
        categories = [
            ('experience_level', experience_level),
            ('employment_type', employment_type),
            ('job_title', job_title),
            ('employee_residence', employee_residence),
            ('company_location', company_location),
            ('company_size', company_size)
        ]

        for prefix, value in categories:
            col_name = f"{prefix}_{value}"
            if col_name in input_dict:
                input_dict[col_name] = 1
            # If col_name not found, it's ignored (handled by base case)

        # Create DataFrame & Scale
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_names]  # Strict column ordering
        
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        salary = prediction[0]

        # Show Result
        st.success(f"### 💵 Estimated Salary: ${salary:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
