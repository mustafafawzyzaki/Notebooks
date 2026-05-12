"""
============================================================
  PASTE THE CODE BELOW INTO NEW CELLS IN YOUR NOTEBOOK
  (obesity_final_final_final_final.ipynb)
============================================================

CELL 1  ──  Extract & Save the Best Model (Random Forest)
CELL 2  ──  Load Model & Run a Prediction Demo

─────────────────────────────────────────────
FINAL INPUT FEATURES  (what the user enters)
─────────────────────────────────────────────
  NUMERICAL (7):
    Age   – age in years (float)
    FCVC  – frequency of vegetable consumption (float, 1-3)
    NCP   – number of main meals (float, 1-4)
    CH2O  – daily water consumption level (float, 1-3)
    FAF   – physical activity frequency (float, 0-3)
    TUE   – time using technology devices (float, 0-2)
    BMI   – Body Mass Index = Weight / Height² (float)

  CATEGORICAL (8):
    Gender                          – 'Male' or 'Female'
    family_history_with_overweight  – 'yes' or 'no'
    FAVC                            – 'yes' or 'no'
    CAEC   – 'no', 'Sometimes', 'Frequently', 'Always'
    SMOKE  – 'yes' or 'no'
    SCC    – 'yes' or 'no'
    CALC   – 'no', 'Sometimes', 'Frequently', 'Always'
    MTRANS – 'Automobile','Bike','Motorbike',
             'Public_Transportation','Walking'

─────────────────────────────────────────────
OUTPUT  (what the model returns)
─────────────────────────────────────────────
  One of four obesity-level classes:
    Underweight | Normal | Overweight | Obesity
============================================================
"""

# %%  ──────────────── CELL 1: EXTRACT & SAVE ────────────────
# =========================
# EXTRACT & SAVE THE BEST MODEL (Random Forest)
# =========================
import joblib, os

# ---- build a sklearn Pipeline that bundles preprocessing + classifier ----
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import (
    RobustScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# The final feature lists (already defined earlier in the notebook)
final_numerical_cols = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
final_categorical_cols = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]

binary_cat_cols  = ["Gender", "family_history_with_overweight",
                    "FAVC", "SMOKE", "SCC"]
multi_cat_cols   = ["CAEC", "CALC", "MTRANS"]

# Preprocessing transformers
num_transformer   = SKPipeline([("scaler", RobustScaler())])
bin_transformer   = SKPipeline([("ordinal", OrdinalEncoder())])
multi_transformer = SKPipeline([("onehot",
                                 OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))])

preprocessor = ColumnTransformer(transformers=[
    ("num",   num_transformer,   final_numerical_cols),
    ("bin",   bin_transformer,   binary_cat_cols),
    ("multi", multi_transformer, multi_cat_cols),
])

# Full pipeline: preprocessing → Random Forest
rf_pipeline = SKPipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# ---- Encode target ----
target_enc = LabelEncoder()
y_encoded = target_enc.fit_transform(train[target_col])

# ---- Prepare feature matrix ----
feature_cols = final_numerical_cols + final_categorical_cols
X_full = train[feature_cols].copy()

# ---- Train / test split ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---- Fit ----
rf_pipeline.fit(X_tr, y_tr)

# ---- Quick evaluation ----
y_pred = rf_pipeline.predict(X_te)
y_pred_labels = target_enc.inverse_transform(y_pred)
y_te_labels   = target_enc.inverse_transform(y_te)

target_classes = ["Underweight", "Normal", "Overweight", "Obesity"]
print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
print(f"Macro F1 : {f1_score(y_te, y_pred, average='macro'):.4f}")
print()
print(classification_report(y_te_labels, y_pred_labels, labels=target_classes))

# ---- Save everything needed for prediction ----
MODEL_PATH = os.path.join(os.getcwd(), "obesity_rf_model.joblib")

artifact = {
    "pipeline":         rf_pipeline,
    "target_encoder":   target_enc,
    "feature_cols":     feature_cols,
    "numerical_cols":   final_numerical_cols,
    "categorical_cols": final_categorical_cols,
    "target_classes":   target_classes,
}
joblib.dump(artifact, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")


# %%  ──────────────── CELL 2: LOAD & PREDICT ────────────────
# =========================
# LOAD MODEL & PREDICT (demo)
# =========================
import joblib, os, pandas as pd

MODEL_PATH = os.path.join(os.getcwd(), "obesity_rf_model.joblib")
art = joblib.load(MODEL_PATH)


def predict_obesity(art, **kwargs):
    """
    Predict the obesity class for a single observation.

    Required keyword arguments (the final input features):
        Age, FCVC, NCP, CH2O, FAF, TUE, BMI,
        Gender, family_history_with_overweight, FAVC,
        CAEC, SMOKE, SCC, CALC, MTRANS

    Returns
    -------
    str : one of  Underweight | Normal | Overweight | Obesity
    """
    row = pd.DataFrame([kwargs])
    row = row[art["feature_cols"]]          # correct column order
    y_pred = art["pipeline"].predict(row)
    return art["target_encoder"].inverse_transform(y_pred)[0]


# ---- Example call ----
result = predict_obesity(
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
print(f"Predicted class: {result}")
