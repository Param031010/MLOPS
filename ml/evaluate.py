"""Standalone evaluation: scores the current Production model on the
chronological test split and logs recall, macro-F1, and a confusion matrix
image as MLflow artifacts under a fresh 'evaluation' run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))
from train import EXPERIMENT_NAME, MODEL_NAME, chronological_split, default_tracking_uri, load_training_data  # noqa: E402
from reporting import log_split_report  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    client = mlflow.tracking.MlflowClient()
    mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]

    df = load_training_data()
    _, _, test_df = chronological_split(df)
    X_test = test_df  # the registered pipeline does its own feature engineering
    y_true = test_df["severity_id"].tolist()

    proba = np.asarray(model.predict(X_test))
    classes = sorted(set(y_true))
    y_pred = [classes[i] for i in np.argmax(proba, axis=1)]

    print(f"[evaluate] Production v{mv.version}")
    print(classification_report(y_true, y_pred, zero_division=0))

    with mlflow.start_run(run_name=f"evaluate-v{mv.version}"):
        metrics = log_split_report("test", y_true, y_pred, classes,
                                   f"{MODEL_NAME} v{mv.version}", ARTIFACTS_DIR)
        mlflow.set_tag("evaluated_model_version", mv.version)

    for k, v in metrics.items():
        print(f"  {k:32s} {v:.4f}")
    print(f"[evaluate] artifacts saved to {ARTIFACTS_DIR} and logged to MLflow")


if __name__ == "__main__":
    main()
