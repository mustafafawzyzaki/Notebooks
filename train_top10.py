"""
Obesity Classification — Train on Top 10 Features
================================================

This script trains the XGBoost model on the Top 10 Features selected in Obesity_Last.ipynb:
['BMI', 'CH2O', 'Age', 'TUE', 'FAF', 'NCP', 'FCVC', 'family_history_with_overweight', 'FAVC', 'CALC']
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
    _BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(_BASE_DIR, "train.csv")
MODEL_PATH = os.path.join(_BASE_DIR, "obesity_xgb_top10.joblib")

# ──────────────────────────────────────────────────────────
# 2. FEATURE DEFINITIONS (TOP 10 ONLY)
# ──────────────────────────────────────────────────────────

NUMERICAL_COLS = [
    "Age",   
    "FCVC",  
    "NCP",   
    "CH2O",  
    "FAF",   
    "TUE",   
    "BMI",   
]

CATEGORICAL_COLS = [
    "family_history_with_overweight",  
    "FAVC",                           
    "CALC",                           
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

ORDINAL_COLS = ["CALC"]
ORDINAL_CATEGORIES = [
    ["no", "Sometimes", "Frequently", "Always"],  # CALC
]

NOMINAL_COLS = [c for c in CATEGORICAL_COLS if c not in ORDINAL_COLS]

TARGET_COL = "NObeyesdad"
TARGET_CLASSES = ["Underweight", "Normal", "Overweight", "Obesity"]

TARGET_GROUP_MAP = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal",
    "Overweight_Level_I": "Overweight",
    "Overweight_Level_II": "Overweight",
    "Obesity_Type_I": "Obesity",
    "Obesity_Type_II": "Obesity",
    "Obesity_Type_III": "Obesity",
}

def build_pipeline():
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

def load_and_prepare_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    if df[TARGET_COL].isin(TARGET_GROUP_MAP.keys()).any():
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_GROUP_MAP)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    return X, y

def compute_sample_weights(y_train_series):
    class_counts = y_train_series.value_counts()
    n_classes = len(class_counts)
    total = len(y_train_series)
    weight_map = {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }
    return y_train_series.map(weight_map).values

def train_model(evaluate=True, save_path=MODEL_PATH):
    X, y = load_and_prepare_data()

    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(TARGET_CLASSES)
    y_encoded = target_encoder.transform(y)

    if evaluate:
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
        print("  EVALUATION (80/20 split) - Top 10 Features")
        print("=" * 55)
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Macro F1     : {macro_f1:.4f}")
        print(f"  Weighted F1  : {weighted_f1:.4f}")
        print()
        print(classification_report(
            y_test_labels, y_pred_labels,
            target_names=TARGET_CLASSES,
        ))

    print("Retraining on the full dataset with Top 10 features...")
    sample_weights_full = compute_sample_weights(
        pd.Series(target_encoder.inverse_transform(y_encoded))
    )

    pipeline = build_pipeline()
    pipeline.fit(
        X, y_encoded,
        model__sample_weight=sample_weights_full,
    )

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

if __name__ == "__main__":
    artifact = train_model(evaluate=True, save_path=MODEL_PATH)
