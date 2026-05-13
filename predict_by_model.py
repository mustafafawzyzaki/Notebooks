# pyrefly: ignore [missing-import]
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

# Load the trained bundle (dict with 'pipeline', 'target_encoder', etc.)
bundle = joblib.load("obesity_xgb_top10.joblib")
pipeline = bundle["pipeline"]
target_encoder = bundle["target_encoder"]

app = FastAPI(title="Obesity Prediction API")

# The 10 features the model expects
TOP_10_FEATURES = bundle.get("feature_cols", [
    "Age", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI",
    "family_history_with_overweight", "FAVC", "CALC",
])

# Optional: map numeric output back to labels if your model returns ints
LABEL_MAP = {
    0: "Underweight",
    1: "Normal",
    2: "Overweight",
    3: "Obesity",
}


class PatientData(BaseModel):
    Height: float
    Weight: float
    Age: float
    Vegetable_consumption_frequency: float
    main_meals_count: float
    water_consumption: float
    Physical_activity_frequency: float
    Tech_device_time: float
    family_history_with_overweight: str  # "yes" / "no"
    High_calorie_food: str               # "yes" / "no"
    Alcohol_consumption: str             # "no" / "Sometimes" / "Frequently" / "Always"


def predict_one(patient_dict: dict) -> dict:
    """Predict obesity class for a single patient."""
    df = pd.DataFrame([patient_dict])

    # Auto-calculate BMI from Height and Weight if not provided
    if "BMI" not in df.columns and {"Height", "Weight"}.issubset(df.columns):
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    missing = [col for col in TOP_10_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[TOP_10_FEATURES]
    raw_pred = pipeline.predict(X)[0]

    # Make it JSON-serializable
    if isinstance(raw_pred, (np.integer, np.floating)):
        raw_pred = raw_pred.item()

    label = LABEL_MAP.get(raw_pred, raw_pred) if isinstance(raw_pred, int) else raw_pred

    return {"prediction": label, "bmi": float(df["BMI"].iloc[0])}


@app.get("/healthy")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict")
def obesity_prediction(request_data: PatientData):
    patient_data = {
        "Height": request_data.Height,
        "Weight": request_data.Weight,
        "Age": request_data.Age,
        "FCVC": request_data.Vegetable_consumption_frequency,
        "NCP": request_data.main_meals_count,
        "CH2O": request_data.water_consumption,
        "FAF": request_data.Physical_activity_frequency,
        "TUE": request_data.Tech_device_time,
        "family_history_with_overweight": request_data.family_history_with_overweight,
        "FAVC": request_data.High_calorie_food,
        "CALC": request_data.Alcohol_consumption,
    }
    return predict_one(patient_data)


# ===========================================================================
# Run the server: `python main.py`
# ===========================================================================
if __name__ == "__main__":
    uvicorn.run(
        "predict_by_model:app",        # change "main" if your file has a different name
        host="0.0.0.0",    # accessible on your local network; use "127.0.0.1" for localhost only
        port=8008,
        reload=True,       # auto-reload on code changes; set False in production
    )