# Customer-Churn-Predictor

This repository implements a **config-driven, reproducible machine
learning pipeline** for customer churn prediction. The project is
structured to mirror production-style ML systems, emphasizing
engineering practices such as CLI-based execution, configuration
management, deterministic runs, and artifact tracking.

---

## Engineering Focus

Rather than focusing only on model accuracy, this project is designed to demonstrate:

-   Config-driven experiment management\
-   Deterministic training via seed control\
-   Structured artifact output (models, metrics, logs)\
-   Clean CLI entrypoints for training and evaluation\
-   Separation of configuration and application logic

This mirrors how batch ML systems are designed in real production environments.

---

## Problem Statement

Given historical customer account, billing, and service usage data, design a reproducible machine learning pipeline that predicts customer churn, the loss of customers who stop using a company's products or services. The system must support configurable training runs, consistent feature preprocessing, artifact tracking, and deterministic evaluation suitable for production-style workflows.

---

## Current Implemenation

### CLI Entrypoint

Run the pipeline using:

python -m src.cli train --config configs/base.yaml

The CLI:

-   Parses commands (`train`, `eval`, `predict`)\
-   Loads YAML configuration files\
-   Supports hierarchical config inheritance\
-   Initializes logging\
-   Sets deterministic seeds\
-   Creates structured artifact directories\
-   Saves the exact run configuration used

### Configuration System

Configurations are defined in YAML files:

configs/ ├── base.yaml └── xgb.yaml

Features:

-   Supports `inherit:` for experiment overrides\
-   Deep-merges configuration dictionaries\
-   Validates required config keys\
-   Keeps experiments reproducible and version-controlled

Example:

inherit: base.yaml

model: name: xgboost params: n_estimators: 400

### Logging & Reproducibility

Each run:

-   Logs to console\
-   Writes `artifacts/run.log`\
-   Stores `run_config.json`\
-   Sets global random seed for reproducibility

### Artifact Structure

Successful runs generate structured outputs:

artifacts/ ├── run.log └── metrics/ └── run_config.json

Future steps will add:

-   Trained model artifacts\
-   Evaluation metrics\
-   ROC and confusion matrix plots

---

## Project Structure

customer-churn-predictor/ ├── configs/ ├── data/ │ └── raw/ ├── src/ │
├── cli.py │ ├── config.py │ └── utils/ ├── artifacts/ ├──
requirements.txt └── README.md

---

## Data

This project uses the **Telco Customer Churn** dataset.

Due to licensing and size considerations, raw data is not committed to
the repository.

To run locally:

1.  Download the dataset\
2.  Place the CSV at:

data/raw/telco_churn.csv

---

## Next Steps

Upcoming development phases:

-   Data ingestion and schema validation\
-   Feature preprocessing pipeline (ColumnTransformer)\
-   Model training and registry\
-   Evaluation metrics and visualization\
-   Basic test coverage

---

## Why This Project Matters

This repository demonstrates the ability to:

-   Structure ML systems beyond notebooks\
-   Separate configuration from logic\
-   Implement reproducible training workflows\
-   Design for traceability and maintainability

It focuses on building systems, not just models.