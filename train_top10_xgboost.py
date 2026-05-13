"""
train_top10_xgboost.py
======================

Train the final XGBoost obesity classifier using ONLY the top 10 features
selected by the notebook's permutation importance step, then save the
fitted pipeline with joblib.

Usage
-----
    python train_top10_xgboost.py --input train.csv --output obesity_top10_xgb.joblib

What this script does
---------------------
1. Loads train.csv (the Kaggle obesity dataset used in the notebook).
2. Drops the id column.
3. Engineers BMI = Weight / Height**2.
4. Groups the 7 original NObeyesdad labels into 4 classes
   (Underweight, Normal, Overweight, Obesity).
5. Keeps only the top 10 features.
6. Builds the same preprocessing pipeline as the notebook
   (RobustScaler + OrdinalEncoder for CAEC/CALC + OneHotEncoder for nominals)
   but restricted to the columns that survive in TOP10_FEATURES.
7. Splits 60/20/20 train/val/test with stratification (same as the notebook).
8. Applies class-balanced sample weights on the training set.
9. Trains XGBoost on (train + val) for the final fit.
10. Reports test-set Macro F1, accuracy, and the classification report.
11. Saves a joblib bundle that predict_obesity_top10.py can load.

If you want to change the top 10 features, edit TOP10_FEATURES below.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Top 10 features — edit this list to match the exact set chosen in the
# notebook by permutation_importance_df.head(10)["Feature"].tolist().
# ---------------------------------------------------------------------------

TOP10_FEATURES = [
    "BMI",
    "Age",
    "Gender",
    "family_history_with_overweight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
    "CAEC",
]

# Full schema metadata from the notebook (used to know how to treat each col)
ALL_NUMERICAL_COLS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
ALL_CATEGORICAL_COLS = [
    "Gender", "family_history_with_overweight", "FAVC", "CAEC",
    "SMOKE", "SCC", "CALC", "MTRANS",
]
ALL_ORDINAL_COLS = ["CAEC", "CALC"]
ALL_ORDINAL_CATEGORIES = {
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
}

TARGET_COL = "NObeyesdad"
TARGET_MAPPING = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight":       "Normal",
    "Overweight_Level_I":  "Overweight",
    "Overweight_Level_II": "Overweight",
    "Obesity_Type_I":      "Obesity",
    "Obesity_Type_II":     "Obesity",
    "Obesity_Type_III":    "Obesity",
}

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, engineer BMI, and group the target into 4 classes."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {csv_path}")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if {"Height", "Weight"}.issubset(df.columns):
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    elif "BMI" not in df.columns:
        raise ValueError("Need either BMI, or Height + Weight to compute BMI.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING).fillna(df[TARGET_COL])

    expected_classes = {"Underweight", "Normal", "Overweight", "Obesity"}
    actual_classes = set(df[TARGET_COL].unique())
    unexpected = actual_classes - expected_classes
    if unexpected:
        raise ValueError(f"Unexpected target labels after grouping: {unexpected}")

    return df


def select_top10(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only the top 10 feature columns and the target."""
    missing = [c for c in TOP10_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"TOP10_FEATURES missing from data: {missing}")
    return df[TOP10_FEATURES].copy(), df[TARGET_COL].copy()


# ---------------------------------------------------------------------------
# Preprocessing pipeline (restricted to the columns in TOP10_FEATURES)
# ---------------------------------------------------------------------------

def build_preprocessor() -> tuple[ColumnTransformer, dict]:
    """Build a ColumnTransformer that handles only the columns present in
    TOP10_FEATURES. Returns the transformer and a metadata dict that the
    prediction-time bundle should carry."""
    numerical_cols = [c for c in TOP10_FEATURES if c in ALL_NUMERICAL_COLS]
    categorical_cols = [c for c in TOP10_FEATURES if c in ALL_CATEGORICAL_COLS]
    ordinal_cols = [c for c in categorical_cols if c in ALL_ORDINAL_COLS]
    nominal_cols = [c for c in categorical_cols if c not in ordinal_cols]
    ordinal_categories = [ALL_ORDINAL_CATEGORIES[c] for c in ordinal_cols]

    transformers = []
    if numerical_cols:
        transformers.append((
            "num",
            Pipeline([("scaler", RobustScaler())]),
            numerical_cols,
        ))
    if ordinal_cols:
        transformers.append((
            "ord",
            Pipeline([("encoder", OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ))]),
            ordinal_cols,
        ))
    if nominal_cols:
        transformers.append((
            "nom",
            Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]),
            nominal_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    meta = {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
        "ordinal_categories": ordinal_categories,
        "nominal_cols": nominal_cols,
    }
    return preprocessor, meta


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def class_balanced_weights(y: pd.Series) -> np.ndarray:
    """Same formula used in the notebook."""
    counts = y.value_counts()
    return y.map(lambda c: len(y) / (len(counts) * counts[c])).to_numpy()


def train(input_csv: str | Path, output_path: str | Path) -> None:
    df = load_and_prepare(input_csv)
    X, y = select_top10(df)

    print(f"\nUsing top 10 features: {TOP10_FEATURES}")
    print(f"X shape: {X.shape}    y shape: {y.shape}")
    print("\nTarget class distribution:")
    print(y.value_counts().to_string())

    # Same 60/20/20 split as the notebook
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_train_val,
    )
    print(f"\nSplit sizes  -- train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # Encode target (XGBoost needs integer labels)
    target_encoder = LabelEncoder()
    y_train_enc = target_encoder.fit_transform(y_train)

    preprocessor, meta = build_preprocessor()

    xgb_params = dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # ---- Step 1: train on X_train only, evaluate on X_val (sanity check)
    pipe_train_only = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(**xgb_params)),
    ])
    pipe_train_only.fit(
        X_train, y_train_enc,
        model__sample_weight=class_balanced_weights(y_train),
    )
    y_val_pred = target_encoder.inverse_transform(pipe_train_only.predict(X_val))

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")

    # ---- Step 2: final fit on (train + val), evaluate once on X_test
    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)
    y_dev_enc = target_encoder.transform(y_dev)

    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()[0]),
        ("model", XGBClassifier(**xgb_params)),
    ])
    final_pipeline.fit(
        X_dev, y_dev_enc,
        model__sample_weight=class_balanced_weights(y_dev),
    )

    y_test_pred = target_encoder.inverse_transform(final_pipeline.predict(X_test))

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\nFinal test results (model trained on train+val):")
    print(f"  Test Accuracy:    {test_acc:.4f}")
    print(f"  Test Macro F1:    {test_f1:.4f}")
    print(f"  Test Weighted F1: {test_f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    labels = ["Underweight", "Normal", "Overweight", "Obesity"]
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    # ---- Save the bundle for the prediction script
    bundle = {
        "model": final_pipeline,
        "model_name": "XGBoost (Top 10 Features)",
        "uses_encoded_target": True,
        "target_encoder": target_encoder,
        "target_classes": list(target_encoder.classes_),

        "feature_order": TOP10_FEATURES,
        "numerical_cols": meta["numerical_cols"],
        "categorical_cols": meta["categorical_cols"],
        "ordinal_cols": meta["ordinal_cols"],
        "ordinal_categories": meta["ordinal_categories"],
        "nominal_cols": meta["nominal_cols"],

        "bmi_formula": "Weight / (Height ** 2)" if "BMI" in TOP10_FEATURES else None,

        "validation_macro_f1": float(val_f1),
        "validation_accuracy": float(val_acc),
        "test_macro_f1": float(test_f1),
        "test_accuracy": float(test_acc),
        "test_weighted_f1": float(test_f1_weighted),

        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    joblib.dump(bundle, output_path)
    print(f"\nSaved model bundle to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost obesity classifier on the top 10 features."
    )
    parser.add_argument(
        "--input", "-i", default="train.csv",
        help="Path to the training CSV (default: train.csv).",
    )
    parser.add_argument(
        "--output", "-o", default="obesity_top10_xgb.joblib",
        help="Path for the saved joblib bundle (default: obesity_top10_xgb.joblib).",
    )
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    train(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
