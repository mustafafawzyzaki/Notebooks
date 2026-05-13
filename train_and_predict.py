"""
Obesity Classification — Train, Save & Predict
================================================

This script replicates the full pipeline from Obesity_Last.ipynb:
  1. Loads train.csv and engineers BMI
  2. Groups the 7 target classes into 4
  3. Builds the same preprocessing + XGBoost pipeline
  4. Trains on the full dataset (or 80/20 split for evaluation)
  5. Saves the trained model to disk via joblib
  6. Provides a predict_obesity() function for inference

Selected Model : XGBoost (baseline from notebook)
Selected Features: All 15 (7 numerical + 8 categorical)
Target Classes : Underweight, Normal, Overweight, Obesity
Primary Metric : Macro F1
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier


# ──────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────

try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running inside Jupyter where __file__ is undefined
    _BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(_BASE_DIR, "train.csv")
MODEL_PATH = os.path.join(_BASE_DIR, "obesity_xgb_model.joblib")


# ──────────────────────────────────────────────────────────
# 2. FEATURE DEFINITIONS
#    Matches the notebook exactly (Obesity_Last.ipynb)
# ──────────────────────────────────────────────────────────

# --- Numerical features (7) ---
NUMERICAL_COLS = [
    "Age",   # Age in years
    "FCVC",  # Frequency of Consumption of Vegetables (1–3)
    "NCP",   # Number of main meals per day (1–4)
    "CH2O",  # Daily water consumption level (1–3)
    "FAF",   # Frequency of Physical Activity (0–3)
    "TUE",   # Time Using Technology devices (0–2 hours)
    "BMI",   # Body Mass Index = Weight / Height² (engineered)
]

# --- Categorical features (8) ---
CATEGORICAL_COLS = [
    "Gender",                         # Male / Female
    "family_history_with_overweight",  # yes / no
    "FAVC",                           # Frequent high-calorie food (yes / no)
    "CAEC",                           # Snacking between meals (ordinal)
    "SMOKE",                          # Smoker (yes / no)
    "SCC",                            # Calorie-consumption monitoring (yes / no)
    "CALC",                           # Alcohol consumption (ordinal)
    "MTRANS",                         # Main transportation mode (nominal)
]

# All features used for training
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# --- Ordinal encoding order (matches notebook) ---
ORDINAL_COLS = ["CAEC", "CALC"]
ORDINAL_CATEGORIES = [
    ["no", "Sometimes", "Frequently", "Always"],  # CAEC
    ["no", "Sometimes", "Frequently", "Always"],  # CALC — "Always" pre-registered
]

# --- Nominal columns = all categorical except ordinal ---
NOMINAL_COLS = [c for c in CATEGORICAL_COLS if c not in ORDINAL_COLS]

# --- Target ---
TARGET_COL = "NObeyesdad"
TARGET_CLASSES = ["Underweight", "Normal", "Overweight", "Obesity"]

# --- Target grouping map (7 → 4) ---
TARGET_GROUP_MAP = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal",
    "Overweight_Level_I": "Overweight",
    "Overweight_Level_II": "Overweight",
    "Obesity_Type_I": "Obesity",
    "Obesity_Type_II": "Obesity",
    "Obesity_Type_III": "Obesity",
}


# ──────────────────────────────────────────────────────────
# 3. BUILD PREPROCESSING + MODEL PIPELINE
# ──────────────────────────────────────────────────────────

def build_pipeline():
    """
    Build the exact preprocessing + XGBoost pipeline from the notebook.

    Preprocessing:
        - RobustScaler   → 7 numerical features
        - OrdinalEncoder  → CAEC, CALC (with explicit category order)
        - OneHotEncoder   → 6 nominal features

    Model:
        - XGBClassifier with the notebook's baseline hyperparameters
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler()),
    ])

    ordinal_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERICAL_COLS),
        ("ord", ordinal_transformer, ORDINAL_COLS),
        ("nom", nominal_transformer, NOMINAL_COLS),
    ])

    # XGBoost — exact hyperparameters from the notebook's baseline model
    # (the baseline XGBoost beat the tuned version on validation Macro F1)
    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


# ──────────────────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ──────────────────────────────────────────────────────────

def load_and_prepare_data(data_path=DATA_PATH):
    """Load train.csv, engineer BMI, group the target, return X and y."""
    df = pd.read_csv(data_path)

    # Drop id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Engineer BMI
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # Group target (idempotent — safe to re-run)
    if df[TARGET_COL].isin(TARGET_GROUP_MAP.keys()).any():
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_GROUP_MAP)

    # Drop Height and Weight (replaced by BMI)
    df = df.drop(columns=["Height", "Weight"])

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    return X, y


def compute_sample_weights(y_train_series):
    """Compute balanced class weights: total / (n_classes × class_count)."""
    class_counts = y_train_series.value_counts()
    n_classes = len(class_counts)
    total = len(y_train_series)
    weight_map = {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }
    return y_train_series.map(weight_map).values


def train_model(evaluate=True, save_path=MODEL_PATH):
    """
    Train the XGBoost pipeline.

    Parameters
    ----------
    evaluate : bool
        If True, do an 80/20 split and print metrics before
        retraining on the full dataset.
    save_path : str
        Where to save the .joblib artifact.

    Returns
    -------
    dict with keys: "pipeline", "target_encoder", "path"
    """
    X, y = load_and_prepare_data()

    # Encode target labels
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(TARGET_CLASSES)
    y_encoded = target_encoder.transform(y)

    if evaluate:
        # ---- Evaluation split (80/20) ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded,
        )

        sample_weights = compute_sample_weights(
            pd.Series(target_encoder.inverse_transform(y_train))
        )

        pipeline = build_pipeline()
        pipeline.fit(
            X_train, y_train,
            model__sample_weight=sample_weights,
        )

        y_pred = pipeline.predict(X_test)
        y_pred_labels = target_encoder.inverse_transform(y_pred)
        y_test_labels = target_encoder.inverse_transform(y_test)

        acc = accuracy_score(y_test_labels, y_pred_labels)
        macro_f1 = f1_score(y_test_labels, y_pred_labels, average="macro")
        weighted_f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")

        print("=" * 55)
        print("  EVALUATION (80/20 split)")
        print("=" * 55)
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Macro F1     : {macro_f1:.4f}")
        print(f"  Weighted F1  : {weighted_f1:.4f}")
        print()
        print(classification_report(
            y_test_labels, y_pred_labels,
            target_names=TARGET_CLASSES,
        ))

    # ---- Retrain on full dataset ----
    print("Retraining on the full dataset...")
    sample_weights_full = compute_sample_weights(
        pd.Series(target_encoder.inverse_transform(y_encoded))
    )

    pipeline = build_pipeline()
    pipeline.fit(
        X, y_encoded,
        model__sample_weight=sample_weights_full,
    )

    # ---- Save ----
    artifact = {
        "pipeline": pipeline,
        "target_encoder": target_encoder,
        "feature_cols": FEATURE_COLS,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_classes": TARGET_CLASSES,
    }
    joblib.dump(artifact, save_path)
    print(f"\nModel saved to: {save_path}")
    print(f"File size    : {os.path.getsize(save_path) / 1024:.1f} KB")

    return artifact


# ──────────────────────────────────────────────────────────
# 5. LOAD MODEL & PREDICT
# ──────────────────────────────────────────────────────────

def load_model(path=MODEL_PATH):
    """Load the saved .joblib model artifact."""
    artifact = joblib.load(path)
    print(f"Model loaded from: {path}")
    return artifact


def predict_obesity(
    artifact,
    Age: float,
    Height: float,
    Weight: float,
    FCVC: float,
    NCP: float,
    CH2O: float,
    FAF: float,
    TUE: float,
    Gender: str,
    family_history_with_overweight: str,
    FAVC: str,
    CAEC: str,
    SMOKE: str,
    SCC: str,
    CALC: str,
    MTRANS: str,
):
    """
    Predict obesity class for a single person.

    Parameters
    ----------
    artifact : dict
        Output of load_model() or train_model().
    Age : float
        Age in years.
    Height : float
        Height in meters (used to compute BMI).
    Weight : float
        Weight in kilograms (used to compute BMI).
    FCVC : float
        Frequency of Vegetable Consumption (1–3).
    NCP : float
        Number of main meals per day (1–4).
    CH2O : float
        Daily water consumption level (1–3).
    FAF : float
        Frequency of Physical Activity (0–3).
    TUE : float
        Time Using Technology devices (0–2 hours).
    Gender : str
        "Male" or "Female".
    family_history_with_overweight : str
        "yes" or "no".
    FAVC : str
        Frequent high-calorie food consumption — "yes" or "no".
    CAEC : str
        Snacking between meals — "no", "Sometimes", "Frequently", "Always".
    SMOKE : str
        Smoker — "yes" or "no".
    SCC : str
        Calorie-consumption monitoring — "yes" or "no".
    CALC : str
        Alcohol consumption — "no", "Sometimes", "Frequently", "Always".
    MTRANS : str
        Main transportation — e.g. "Public_Transportation", "Automobile",
        "Walking", "Motorbike", "Bike".

    Returns
    -------
    dict with "prediction" (str), "probabilities" (dict), "bmi" (float)
    """
    BMI = Weight / (Height ** 2)

    row = pd.DataFrame([{
        "Age": Age,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "BMI": BMI,
        "Gender": Gender,
        "family_history_with_overweight": family_history_with_overweight,
        "FAVC": FAVC,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "CALC": CALC,
        "MTRANS": MTRANS,
    }])

    pipeline = artifact["pipeline"]
    target_encoder = artifact["target_encoder"]

    pred_encoded = pipeline.predict(row)[0]
    prediction = target_encoder.inverse_transform([pred_encoded])[0]

    probas = pipeline.predict_proba(row)[0]
    prob_dict = {
        cls: round(float(p), 4)
        for cls, p in zip(target_encoder.classes_, probas)
    }

    return {
        "prediction": prediction,
        "probabilities": prob_dict,
        "bmi": round(BMI, 2),
    }


# ──────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- Train and save ---
    artifact = train_model(evaluate=True, save_path=MODEL_PATH)

    # --- Load and predict (demo) ---
    print("\n" + "=" * 55)
    print("  DEMO PREDICTION")
    print("=" * 55)

    loaded = load_model(MODEL_PATH)

    result = predict_obesity(
        loaded,
        Age=24.0,
        Height=1.70,           # meters
        Weight=81.7,           # kg
        FCVC=2.0,              # Frequency of Vegetable Consumption
        NCP=3.0,               # Number of main meals
        CH2O=2.8,              # Daily water consumption
        FAF=0.0,               # Frequency of Physical Activity
        TUE=1.0,               # Time Using Technology
        Gender="Male",
        family_history_with_overweight="yes",
        FAVC="yes",
        CAEC="Sometimes",
        SMOKE="no",
        SCC="no",
        CALC="Sometimes",
        MTRANS="Public_Transportation",
    )

    print(f"\n  Predicted class : {result['prediction']}")
    print(f"  Computed BMI    : {result['bmi']}")
    print(f"  Probabilities   :")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 40)
        print(f"    {cls:>12s}  {prob:.4f}  {bar}")
