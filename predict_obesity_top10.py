"""
predict_obesity_top10.py
========================

Standalone prediction script for the top-10-feature XGBoost obesity
classifier trained by train_top10_xgboost.py.

The saved joblib bundle carries the fitted pipeline (preprocessor + XGBoost),
the target encoder, and the exact feature order, so this script does not
duplicate any preprocessing logic.

Three ways to call it
---------------------

1) Single patient as a Python dict:

       from predict_obesity_top10 import predict_one

       patient = {
           "Gender": "Female",
           "Age": 27,
           "Height": 1.62,   # optional: used to compute BMI if BMI not given
           "Weight": 64.0,   # optional: used to compute BMI if BMI not given
           "family_history_with_overweight": "yes",
           "FCVC": 2.0,
           "NCP": 3.0,
           "CAEC": "Sometimes",
           "CH2O": 2.0,
           "FAF": 1.0,
           "TUE": 1.0,
       }
       print(predict_one(patient))

2) Batch DataFrame:

       from predict_obesity_top10 import predict
       predict(df_new_patients, return_proba=True)

3) Command line:

       python predict_obesity_top10.py --input new_patients.csv --output preds.csv --proba
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
import pandas as pd


DEFAULT_BUNDLE_PATH = "obesity_top10_xgb.joblib"


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def load_bundle(bundle_path: str | Path = DEFAULT_BUNDLE_PATH) -> dict:
    """Load the saved joblib bundle and check the keys it must carry."""
    bundle = joblib.load(bundle_path)
    required_keys = {
        "model", "feature_order", "numerical_cols",
        "categorical_cols", "target_classes",
    }
    missing = required_keys - bundle.keys()
    if missing:
        raise ValueError(
            f"Bundle '{bundle_path}' is missing required keys: {sorted(missing)}"
        )
    return bundle


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _ensure_bmi(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """If the model needs BMI but the user supplied only Height + Weight,
    compute BMI from them. Leaves df unchanged if BMI is already present."""
    if "BMI" not in bundle["feature_order"]:
        return df
    if "BMI" in df.columns and df["BMI"].notna().all():
        return df
    if {"Height", "Weight"}.issubset(df.columns):
        df = df.copy()
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        return df
    raise ValueError(
        "Model expects a BMI column. Either pass BMI directly, "
        "or pass both Height (m) and Weight (kg) so it can be computed."
    )


def prepare_input(
    data: pd.DataFrame | Mapping | Iterable[Mapping],
    bundle: dict,
) -> pd.DataFrame:
    """Turn user input into a DataFrame with the exact column order the
    pipeline expects. Accepts a DataFrame, a single dict, or a list of dicts."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, Mapping):
        df = pd.DataFrame([dict(data)])
    else:
        df = pd.DataFrame(list(data))

    df = _ensure_bmi(df, bundle)

    expected = bundle["feature_order"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required input columns: {missing}. "
            f"Expected (in order): {expected}"
        )

    return df[expected]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _decode(predictions: np.ndarray, bundle: dict) -> np.ndarray:
    """XGBoost was trained on label-encoded targets, so decode back to
    original class names. Returns predictions unchanged for other models."""
    if bundle.get("uses_encoded_target") and bundle.get("target_encoder") is not None:
        return bundle["target_encoder"].inverse_transform(predictions)
    return predictions


def predict(
    data: pd.DataFrame | Mapping | Iterable[Mapping],
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    return_proba: bool = False,
) -> pd.DataFrame:
    """Predict the obesity class for one or many records. Returns a
    DataFrame with a 'prediction' column (and one probability column per
    class when return_proba=True)."""
    bundle = load_bundle(bundle_path)
    X = prepare_input(data, bundle)
    pipe = bundle["model"]

    preds = _decode(pipe.predict(X), bundle)
    out = pd.DataFrame({"prediction": preds}, index=X.index)

    if return_proba and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if bundle.get("uses_encoded_target") and bundle.get("target_encoder") is not None:
            class_names = bundle["target_encoder"].classes_
        else:
            class_names = pipe.classes_
        proba_df = pd.DataFrame(
            proba,
            columns=[f"proba_{c}" for c in class_names],
            index=X.index,
        )
        out = pd.concat([out, proba_df], axis=1)

    return out


def predict_one(
    patient: Mapping,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    return_proba: bool = True,
) -> dict:
    """Convenience wrapper: predict for a single patient dict.
    Returns a flat dict with the predicted class and class probabilities."""
    return predict(patient, bundle_path=bundle_path, return_proba=return_proba).iloc[0].to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict obesity class from a CSV using the top-10 XGBoost model."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input CSV. Must contain the top-10 columns "
             "(or Height + Weight if BMI is not supplied).",
    )
    parser.add_argument(
        "--output", "-o", default="predictions.csv",
        help="Path for the output CSV (default: predictions.csv).",
    )
    parser.add_argument(
        "--bundle", "-b", default=DEFAULT_BUNDLE_PATH,
        help=f"Path to the joblib bundle (default: {DEFAULT_BUNDLE_PATH}).",
    )
    parser.add_argument(
        "--proba", action="store_true",
        help="Also include per-class probabilities in the output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    df_in = pd.read_csv(input_path)
    bundle = load_bundle(args.bundle)

    print(f"Loaded model bundle: {args.bundle}")
    print(f"  Model: {bundle.get('model_name', 'unknown')}")
    print(f"  Expects {len(bundle['feature_order'])} features: {bundle['feature_order']}")
    print(f"  Read {len(df_in)} rows from {input_path}")

    result = predict(df_in, bundle_path=args.bundle, return_proba=args.proba)
    output = pd.concat(
        [df_in.reset_index(drop=True), result.reset_index(drop=True)],
        axis=1,
    )
    output.to_csv(args.output, index=False)

    print(f"\nWrote predictions to: {args.output}")
    print("\nPrediction distribution:")
    print(result["prediction"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
