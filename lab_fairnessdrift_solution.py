#!/usr/bin/env python3
"""
CPE393 – Evidently Monitoring Lab (Evidently 0.7.x minimal, drift-focused)

Uses your data/train.csv and data/test.csv with a binary `target` column (0/1).

- Trains 2 models: Logistic Regression, RandomForest
- Logs metrics & lightweight artifacts to MLflow (accuracy, precision, recall, F1, ROC AUC; confusion matrix JSON; ROC points JSON)
- Generates Evidently report for Data Drift (0.7.x API: Report + DataDriftPreset)

Run:
    python lab_fairness_drift.py

(optional) MLflow UI:
    mlflow server --host 127.0.0.1 --port 8080
"""

from __future__ import annotations
from pathlib import Path
import json
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

# Evidently 0.7.x API
from evidently import Report
from evidently.presets import DataDriftPreset

# -------- Paths & Experiment --------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
TARGET = "target"

EXP_NAME = "Evidently"
mlflow.set_experiment(EXP_NAME)


# -------- Data helpers --------
def load_local_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            f"CSV files not found.\nExpected:\n  {TRAIN_CSV}\n  {TEST_CSV}\n"
            "Please put your CSVs under data/ with a binary 'target' column."
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


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore").tolist()
    cat_cols = [c for c in df.columns if c not in num_cols + [TARGET]]
    return num_cols, cat_cols


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], model_name: str) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if model_name == "LogReg":
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    else:
        raise ValueError("model_name must be 'LogReg' or 'RandomForest'")
    return Pipeline([("pre", pre), ("clf", clf)])


# -------- Plot/artifact helpers --------
def log_confusion_and_roc(y_true, y_pred, y_proba, tag: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_path = REPORTS_DIR / f"{tag}_confusion_matrix.json"
    cm_path.write_text(json.dumps({"labels": [0, 1], "matrix": cm.tolist()}, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(cm_path))

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_path = REPORTS_DIR / f"{tag}_roc_curve_points.json"
        roc_path.write_text(json.dumps({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(roc_path))


# -------- Evidently helpers (new API) --------
def save_report(report: Report, current_df: pd.DataFrame, reference_df: pd.DataFrame | None,
                html_name: str, json_name: str) -> None:
    """
    With 0.7.x, report.run(...) returns an evaluation object that you save.
    """
    eval_result = report.run(current_data=current_df, reference_data=reference_df)
    html_path = REPORTS_DIR / html_name
    json_path = REPORTS_DIR / json_name
    eval_result.save_html(str(html_path))
    eval_result.save_json(str(json_path))
    mlflow.log_artifact(str(html_path))
    mlflow.log_artifact(str(json_path))


# -------- Train/Eval one model --------
def run_model(model_name: str, ref: pd.DataFrame, cur: pd.DataFrame) -> None:
    num_cols, cat_cols = split_columns(ref)
    pipe = build_pipeline(num_cols, cat_cols, model_name)

    X_train, y_train = ref.drop(columns=[TARGET]), ref[TARGET].astype(int)
    X_test,  y_test  = cur.drop(columns=[TARGET]), cur[TARGET].astype(int)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model": model_name,
            "numeric_cols": ",".join(num_cols),
            "categorical_cols": ",".join(cat_cols),
        })

        # Fit + predict
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        # Metrics
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

        # Log model (MLflow warns about artifact_path deprecation; still works)
        sig = None
        try:
            sig = infer_signature(X_train, pipe.predict(X_train))
        except Exception:
            pass
        try:
            mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_train.head(1), signature=sig)
        except Exception:
            pass

        # Lightweight artifacts
        log_confusion_and_roc(y_test, y_pred, y_proba, tag=model_name.lower())

        # ---- Evidently Report: Data Drift only (0.7.x) ----
        # Works on raw ref/cur data; will include 'target' drift if present.
        save_report(
            Report([DataDriftPreset()]),
            current_df=cur,
            reference_df=ref,
            html_name=f"{model_name}_data_drift.html",
            json_name=f"{model_name}_data_drift.json",
        )

        print(f"[{model_name}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


# -------- Main --------
def main() -> None:
    warnings.filterwarnings("default")
    ref, cur = load_local_csvs()
    print(f"Loaded: train={ref.shape}, test={cur.shape}. Columns: {list(ref.columns)}")

    for model in ["LogReg", "RandomForest"]:
        run_model(model, ref, cur)

    print("\nDone ✅  Open MLflow UI to inspect runs and Evidently reports.")
    print("Artifacts saved under ./reports")


if __name__ == "__main__":
    main()
