import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- LOAD THE ARTIFACTS ---
# We load the model, scaler, and feature list you just sent.
file_path = "model/salary_model_artifacts.pkl"

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        artifacts = pickle.load(f)
    
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["features"]
    print(f"✅ Success! Loaded {len(feature_names)} features.")
else:
    print(f"❌ Error: File not found at {file_path}")
    model, scaler, feature_names = None, None, []

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    work_year: int = Form(...),
    experience_level: str = Form(...),
    employment_type: str = Form(...),
    job_title: str = Form(...),
    employee_residence: str = Form(...),
    remote_ratio: int = Form(...),
    company_location: str = Form(...),
    company_size: str = Form(...)
):
    if model is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "Model file not found. Please check the 'model' folder."
        })

    try:
        # 1. Prepare input dictionary with all 577 features set to 0
        input_dict = {col: 0 for col in feature_names}

        # 2. Assign Numerical Values
        input_dict['work_year'] = work_year
        input_dict['remote_ratio'] = remote_ratio

        # 3. Assign Categorical Values (One-Hot Encoding)
        # We manually construct the column name (e.g., "experience_level_SE")
        # and set it to 1 if it exists in our feature list.
        
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
            # Note: If the column isn't found, it usually means it was dropped 
            # during training to avoid dummy variable trap, or the input value is new.

        # 4. Convert to DataFrame and align columns
        input_df = pd.DataFrame([input_dict])
        
        # vital: ensure columns are in the exact same order as training
        input_df = input_df[feature_names]

        # 5. Scale the input
        input_scaled = scaler.transform(input_df)

        # 6. Predict
        prediction = model.predict(input_scaled)
        
        # 7. Format Result
        salary_value = round(prediction[0], 2)
        formatted_salary = f"${salary_value:,.2f}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction_text": f"Estimated Salary: {formatted_salary}"
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Prediction Failed: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)