# MLflow + Evidently AI: Student Lab

### **Objective**

Train and track a classification model with **MLflow**, then evaluate and monitor data/model drift using **Evidently AI**.  
Students will implement key parts of the monitoring pipeline, log Evidently reports as MLflow artifacts, and interpret the results.

## What You’ll Learn

- Set up an experiment in **MLflow** and log parameters, metrics, plots, and models.
- Use **Evidently AI** to generate **Data Drift**, **Target Drift**, and **Classification Performance** reports.
- Log Evidently HTML and JSON reports to MLflow runs.
- Interpret drift and performance results to understand model stability and fairness.

## 📁 Repo Structure

CPE393-FAIRNESS-DRIFT-LAB/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── lab_fairnessdrift_student.py # student (STRICT mode enabled)
├── requirements.txt
└── README.md

## Setup

```bash
# 1) Create and activate a virtual environment (example with venv)
python3.12 -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

Part 1 — Run the Script

python lab_fairnessdrift_student.py
What it does
Loads data/train.csv (reference) and data/test.csv (current).
Trains two classifiers (Logistic Regression and Random Forest).
Logs model parameters, metrics, and artifacts to MLflow.
Generates Evidently reports (Data Drift, Target Drift) and logs them as HTML + JSON.
By default, LAB_STRICT=1 → the file raises NotImplementedError until all TODOs are implemented.
To demonstrate the pipeline with fallback code:
LAB_STRICT=0 python lab_fairnessdrift_student.py

Part 2 — Explore MLflow UI

Start the MLflow tracking UI locally:
mlflow server --host 127.0.0.1 --port 8080
```
