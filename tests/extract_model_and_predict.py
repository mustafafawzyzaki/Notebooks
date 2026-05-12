"""
Obesity Classification - Model Extraction & Prediction App
==========================================================

This script:
1. Loads the training data and re-trains the best model (Random Forest).
2. Saves the trained model + preprocessing pipeline to disk using joblib.
3. Provides a predict() function that accepts raw user input and returns
   the predicted obesity class.

Final Features Used for Input
-----------------------------
NUMERICAL (7 features):
  - Age          : Age in years (float)
  - FCVC         : Frequency of vegetable consumption (float, typically 1-3)
  - NCP          : Number of main meals (float, typically 1-4)
  - CH2O         : Daily water consumption level (float, typically 1-3)
  - FAF          : Physical activity frequency (float, typically 0-3)
  - TUE          : Time using technology devices (float, typically 0-2)
  - BMI          : Body Mass Index = Weight / Height² (float)

CATEGORICAL (8 features):
  - Gender                         : 'Male' or 'Female'
  - family_history_with_overweight : 'yes' or 'no'
  - FAVC                           : 'yes' or 'no' (frequent high-calorie food)
  - CAEC                           : 'no', 'Sometimes', 'Frequently', 'Always'
  - SMOKE                          : 'yes' or 'no'
  - SCC                            : 'yes' or 'no' (calorie monitoring)
  - CALC                           : 'no', 'Sometimes', 'Frequently', 'Always'
  - MTRANS                         : 'Automobile', 'Bike', 'Motorbike',
                                     'Public_Transportation', 'Walking'

Output
------
  One of four obesity-level classes:
    - Underweight
    - Normal
    - Overweight
    - Obesity
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ──────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────
try:
    _BASE_DIR = os.path.dirname(__file__)
except NameError:
    # Running inside a Jupyter notebook where __file__ is not defined
    _BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(_BASE_DIR, "train.csv")
MODEL_PATH = os.path.join(_BASE_DIR, "obesity_rf_model.joblib")

TARGET_COL = "NObeyesdad"

TARGET_MAPPING = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal",
    "Overweight_Level_I": "Overweight",
    "Overweight_Level_II": "Overweight",
    "Obesity_Type_I": "Obesity",
    "Obesity_Type_II": "Obesity",
    "Obesity_Type_III": "Obesity",
}

NUMERICAL_COLS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]

CATEGORICAL_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

# Split categorical into binary vs multi-class
BINARY_CAT_COLS = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
MULTI_CAT_COLS = ["CAEC", "CALC", "MTRANS"]

# Ordinal mappings for ordinal categoricals
ORDINAL_CAEC = ["no", "Sometimes", "Frequently", "Always"]
ORDINAL_CALC = ["no", "Sometimes", "Frequently", "Always"]

RANDOM_STATE = 42
TARGET_CLASSES = ["Underweight", "Normal", "Overweight", "Obesity"]


# ──────────────────────────────────────────────────────────
# 2. TRAINING & SAVING
# ──────────────────────────────────────────────────────────
def train_and_save_model():
    """
    Train the Random Forest pipeline on train.csv and save it to disk.
    Returns the fitted pipeline and target-label encoder.
    """
    # --- Load data -------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"])

    # --- Feature engineering: BMI ----------------------------------------
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # --- Target grouping -------------------------------------------------
    original_classes = set(TARGET_MAPPING.keys())
    current_classes = set(df[TARGET_COL].dropna().unique())
    if current_classes.issubset(original_classes):
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)

    # --- Separate features & target --------------------------------------
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # --- Encode target ---------------------------------------------------
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # --- Train / test split ----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # --- Build preprocessing pipeline -----------------------------------
    numerical_transformer = Pipeline(steps=[
        ("scaler", RobustScaler()),
    ])

    binary_transformer = Pipeline(steps=[
        ("ordinal", OrdinalEncoder()),
    ])

    multi_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_COLS),
        ("bin", binary_transformer, BINARY_CAT_COLS),
        ("multi", multi_transformer, MULTI_CAT_COLS),
    ])

    # --- Full pipeline: preprocessing + Random Forest --------------------
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # --- Fit -------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # --- Evaluate --------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    y_pred_labels = target_encoder.inverse_transform(y_pred)
    y_test_labels = target_encoder.inverse_transform(y_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro F1  : {macro_f1:.4f}")
    print()
    print(classification_report(y_test_labels, y_pred_labels,
                                labels=TARGET_CLASSES))

    # --- Save model + encoder to disk ------------------------------------
    artifact = {
        "pipeline": pipeline,
        "target_encoder": target_encoder,
        "feature_cols": feature_cols,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_classes": TARGET_CLASSES,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    return artifact


# ──────────────────────────────────────────────────────────
# 3. PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH):
    """Load the saved model artifact from disk."""
    return joblib.load(path)


def predict(
    artifact: dict,
    *,
    Age: float,
    FCVC: float,
    NCP: float,
    CH2O: float,
    FAF: float,
    TUE: float,
    BMI: float,
    Gender: str,
    family_history_with_overweight: str,
    FAVC: str,
    CAEC: str,
    SMOKE: str,
    SCC: str,
    CALC: str,
    MTRANS: str,
) -> str:
    """
    Predict the obesity class for a single observation.

    Parameters
    ----------
    artifact : dict
        The loaded model artifact (from load_model()).
    All other parameters match the feature names described at the top of
    this file.

    Returns
    -------
    str
        Predicted class – one of: Underweight, Normal, Overweight, Obesity.
    """
    pipeline = artifact["pipeline"]
    target_encoder = artifact["target_encoder"]
    feature_cols = artifact["feature_cols"]

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

    row = row[feature_cols]  # ensure correct column order
    y_pred_encoded = pipeline.predict(row)
    y_pred_label = target_encoder.inverse_transform(y_pred_encoded)
    return y_pred_label[0]


# ──────────────────────────────────────────────────────────
# 4. MAIN – run from command line to train & demo
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Training Random Forest Obesity Classifier")
    print("=" * 60)
    art = train_and_save_model()

    print("\n" + "=" * 60)
    print("  Demo prediction")
    print("=" * 60)
    result = predict(
        art,
        Age=24.0,
        FCVC=2.0,
        NCP=3.0,
        CH2O=2.8,
        FAF=0.0,
        TUE=1.0,
        BMI=28.3,
        Gender="Male",
        family_history_with_overweight="yes",
        FAVC="yes",
        CAEC="Sometimes",
        SMOKE="no",
        SCC="no",
        CALC="Sometimes",
        MTRANS="Public_Transportation",
    )
    print(f"\nPredicted class: {result}")
