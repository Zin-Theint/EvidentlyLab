#!/usr/bin/env python3
"""
CPE393 – ML Monitoring Lab (Evidently 0.7.x)

STRICT MODE (default):
- By default, LAB_STRICT=1 so this script raises NotImplementedError at each TODO.
- Students must complete the marked TODOs for the code to run.

OPTIONAL (disable strict to demo the pipeline):
    LAB_STRICT=0 python lab_fairnessdrift_student.py
"""

from __future__ import annotations
from pathlib import Path
import json
import os
import warnings

import numpy as np
import pandas as pd

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Evidently 0.7.x
from evidently import Report
from evidently.presets import DataDriftPreset

# ---------------- Config ----------------
# CHANGED: Strict ON by default. Set LAB_STRICT=0 to allow fallbacks.
STRICT = os.getenv("LAB_STRICT", "1") == "1"

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
TARGET = "target"

# Keep student runs separate
EXP_NAME = "EvidentlyLab"
mlflow.set_experiment(EXP_NAME)

def _todo_fail(msg: str):
    raise NotImplementedError(f"TODO REQUIRED: {msg} — complete this section to proceed.")

def _todo_warn(msg: str):
    # Only used when STRICT=0 to keep the lab runnable for demos
    print(f"⚠ TODO (demo fallback): {msg}")

# ---------------- Provided utilities ----------------
def load_local_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs and coerce `target` to {0,1} ints."""
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            f"CSV files not found.\nExpected:\n  {TRAIN_CSV}\n  {TEST_CSV}\n"
            "Put CSVs under data/ with a binary 'target' column."
        )
    ref = pd.read_csv(TRAIN_CSV)
    cur = pd.read_csv(TEST_CSV)

    if TARGET not in ref.columns or TARGET not in cur.columns:
        raise ValueError(f"Both CSVs must have a '{TARGET}' column.")

    # Normalize target to 0/1 if needed
    for df in (ref, cur):
        if df[TARGET].dtype == object:
            s = df[TARGET].astype(str).str.strip().str.lower()
            df[TARGET] = s.isin([">50k", "1", "yes", "true", "positive", "y"]).astype(int)
        else:
            df[TARGET] = df[TARGET].astype(int)
    return ref, cur


def log_confusion_and_roc(y_true, y_pred, y_proba, tag: str) -> None:
    """Log confusion matrix and optional ROC points as JSON artifacts."""
    cm = confusion_matrix(y_true, y_pred)
    cm_path = REPORTS_DIR / f"{tag}_confusion_matrix.json"
    cm_path.write_text(json.dumps({"labels": [0, 1], "matrix": cm.tolist()}, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(cm_path))

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_path = REPORTS_DIR / f"{tag}_roc_curve_points.json"
        roc_path.write_text(json.dumps({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(roc_path))


def save_report(report: Report, current_df: pd.DataFrame, reference_df: pd.DataFrame | None,
                html_name: str, json_name: str) -> None:
    """Run Evidently report (0.7.x API) and save HTML/JSON artifacts."""
    eval_result = report.run(current_data=current_df, reference_data=reference_df)
    html_path = REPORTS_DIR / html_name
    json_path = REPORTS_DIR / json_name
    eval_result.save_html(str(html_path))
    eval_result.save_json(str(json_path))
    mlflow.log_artifact(str(html_path))
    mlflow.log_artifact(str(json_path))


# ---------------- STUDENT TODOs (strict by default) ----------------
def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    >>> STUDENT TODO:
    Return (numeric_cols, categorical_cols), excluding TARGET.

    Hints:
      numeric -> df.select_dtypes(include=[np.number])
      categorical -> everything else excluding TARGET
    """
    if STRICT:
        _todo_fail("Implement split_columns(df) to return (numeric_cols, categorical_cols)")

    # ---- demo fallback when LAB_STRICT=0 ----
    _todo_warn("Implement split_columns(df)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore").tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [TARGET]]
    return numeric_cols, categorical_cols


def _one_hot_encoder_compatible():
    """
    Keep compatibility across sklearn versions:
    - new: OneHotEncoder(..., sparse_output=False)
    - old: OneHotEncoder(..., sparse=False)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], model_name: str) -> Pipeline:
    """
    >>> STUDENT TODO:
    - ColumnTransformer:
        numeric -> StandardScaler(with_mean=False)
        categorical -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    - Append classifier:
        'LogReg' -> LogisticRegression(max_iter=1000, solver='lbfgs')
        'RandomForest' -> RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    """
    if STRICT:
        _todo_fail("Implement build_pipeline(numeric_cols, categorical_cols, model_name)")

    # ---- demo fallback when LAB_STRICT=0 ----
    _todo_warn("Implement build_pipeline(numeric_cols, categorical_cols, model_name)")
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),
            ("cat", _one_hot_encoder_compatible(), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if model_name == "LogReg":
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")  # safe default in demo mode
        _todo_warn(f"Unknown model_name '{model_name}', defaulting to LogisticRegression")
    return Pipeline([("pre", pre), ("clf", clf)])


def run_model(model_name: str, ref: pd.DataFrame, cur: pd.DataFrame) -> None:
    """
    >>> STUDENT TODO inside this function:
    - Fit pipeline
    - Predict on test; compute y_proba if supported
    - Compute metrics: accuracy, precision, recall, f1 (+ roc_auc if y_proba)
    - Log metrics & params
    - Log confusion matrix and ROC points
    - Generate Evidently Data Drift report
    """
    num_cols, cat_cols = split_columns(ref)
    pipe = build_pipeline(num_cols, cat_cols, model_name)

    X_train, y_train = ref.drop(columns=[TARGET]), ref[TARGET].astype(int)
    X_test,  y_test  = cur.drop(columns=[TARGET]), cur[TARGET].astype(int)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("role", "STUDENT")
        mlflow.log_params({
            "model": model_name,
            "numeric_cols": ",".join(num_cols),
            "categorical_cols": ",".join(cat_cols),
        })

        if STRICT:
            _todo_fail("Complete training, metrics logging, artifacts, and drift report in run_model")

        # ---- demo fallback when LAB_STRICT=0 ----
        _todo_warn("Complete training/metrics/artifacts/drift in run_model (demo fallback executing)")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                pass
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Best-effort signature + model
        try:
            sig = infer_signature(X_train, pipe.predict(X_train))
        except Exception:
            sig = None
        try:
            mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_train.head(1), signature=sig)
        except Exception:
            pass

        # Artifacts
        log_confusion_and_roc(y_test, y_pred, y_proba, tag=model_name.lower())

        # Evidently (Data Drift)
        save_report(
            Report([DataDriftPreset()]),
            current_df=cur, reference_df=ref,
            html_name=f"{model_name}_data_drift.html",
            json_name=f"{model_name}_data_drift.json",
        )

        print("[DEMO FALLBACK] "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
              + "   (Enable LAB_STRICT=1 to enforce TODO completion)")

# ---------------- Main ----------------
def main() -> None:
    warnings.filterwarnings("default")
    print(f"STRICT MODE: {'ON (TODOs required)' if STRICT else 'OFF (demo fallbacks enabled)'}")
    ref, cur = load_local_csvs()
    print(f"Loaded: train={ref.shape}, test={cur.shape}. Columns: {list(ref.columns)}")

    for model in ["LogReg", "RandomForest"]:
        run_model(model, ref, cur)

    print("\nDone. When STRICT=1, this message is unreachable until TODOs are completed.")
    print("Artifacts saved under ./reports (demo mode only).")


if __name__ == "__main__":
    main()
